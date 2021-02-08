from collections import OrderedDict
import logging
import torch
import os
import abc
import shutil
import time
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch import nn

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class BaseModel(object):
  def __init__(self, nets, process_idx, model_dir, optimizer, learning_rate,
               max_steps, tensorboard_log_dir=None, replica_rank=0, reload=None,
               **kwargs):
    self.nets = nets
    self.model_dir = model_dir
    self.optimizer = self.get_optimizer(optimizer, learning_rate)
    self.scheduler = get_linear_schedule_with_warmup(
      self.optimizer,
      num_warmup_steps=500,
      num_training_steps=max_steps
    )
    self.best_valid_score = float("-inf")
    self.best_val_loss = float("inf")

    if reload:
      self.reload(reload)

    # setup distributed training
    if torch.cuda.is_available():
      self.nets.cuda(process_idx)
      self.nets = nn.parallel.DistributedDataParallel(
        nets, device_ids=[process_idx], find_unused_parameters=True
      )
    self.replica_rank = replica_rank

    # save parameters
    self.parameters = kwargs
    self.parameters.update({'optimizer': optimizer,
                            'learning_rate': learning_rate})

    if self.replica_rank == 0 and tensorboard_log_dir:
      shutil.rmtree(tensorboard_log_dir)
      os.makedirs(tensorboard_log_dir)
      self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

  def reload(self, reload):
    logger.info('reloading pre-trained model from: %s' % reload)
    #state = torch.load(os.path.join(reload, 'best_val_model.pth.tar'))
    state = torch.load(reload, map_location='cpu')
    state_dict = state['state_dict']

    # the reloaded model should be trained via distributed multi-gpu module
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      name = k[len('module.'):]
      new_state_dict[name] = v
    state_dict = new_state_dict

    self.nets.load_state_dict(state_dict)

  @abc.abstractmethod
  def build_nets(self, **kwargs):
    raise NotImplementedError("build_nets")

  def get_optimizer(self, optimizer, learning_rate):
    """
    parse optimization method parameters, and initialize optimizer function
    """
    # initialize optimizer function
    model_params = filter(lambda p: p.requires_grad, self.nets.parameters())
    if optimizer == 'sgd':
      optimizer_ft = torch.optim.SGD(
        model_params, lr=learning_rate, momentum=0.9
      )
    elif optimizer == 'adagrad':
      optimizer_ft = torch.optim.Adagrad(
        model_params, lr=learning_rate
      )
    elif optimizer == 'adam':
      optimizer_ft = torch.optim.Adam(
        model_params, lr=learning_rate
      )
    elif optimizer == 'bert_adam':
      optimizer_ft = AdamW(
        model_params, lr=learning_rate, weight_decay=0.01,
      )
    else:
      raise Exception('unknown optimization method.')

    return optimizer_ft

  def train(self,
            train_loader=None,
            validation_loader=None,
            epochs=10,
            validation_freq=-1,
            log_freq=200,
            ):

    # Iterate over data.
    self.current_step = 0
    for epoch in range(epochs):
      num_batches = 0
      epoch_loss = []
      for x in train_loader:
        self.nets.train(True)  # Set nets to training mode

        # forward
        preds, prob, loss = self.nets.forward(**x)
        if self.replica_rank == 0:
          self.writer.add_scalar('Loss/train', loss, self.current_step)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem
        # in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.nets.parameters(), 5)
        self.optimizer.step()
        self.scheduler.step()

        # log current training status
        num_batches += 1
        self.current_step += 1

        epoch_loss.append(loss.item())

        if num_batches % log_freq == 0:
          logger.info('{} batches processed, current epoch loss: {}'.format(
            num_batches, sum(epoch_loss) / len(epoch_loss)
          ))
        if validation_freq > 0 and num_batches % validation_freq == 0:
          self.evaluate(validation_loader=validation_loader,
                        train_loss=sum(epoch_loss) / len(epoch_loss),
                        epoch=epoch,
                        mini_batch=num_batches)

      # print epoch loss
      if validation_freq == -1:
        self.evaluate(validation_loader=validation_loader,
                      train_loss=sum(epoch_loss) / len(epoch_loss),
                      epoch=epoch,
                      mini_batch=num_batches)
      logger.info(
        '===== Epoch {:d} finished ====='.format(epoch)
      )

  def evaluate(self,
               validation_loader=None,
               train_loss=None,
               epoch=0,
               mini_batch=0):
    # Iterate over data.
    sys_preds = []
    epoch_loss = []
    for x in validation_loader:
      self.nets.train(False)  # Set nets to eval mode

      # forward
      preds, probs, loss = self.nets.forward(**x)

      # compute the mean of a list of loss from multi gpu
      loss = torch.mean(loss)

      epoch_loss.append(loss.data.cpu())

      sys_preds.append(preds.data.cpu())

    epoch_loss = float(sum(epoch_loss) / len(epoch_loss))
    score = self.scorer(sys_preds, validation_loader)
    if self.replica_rank == 0:
      self.writer.add_scalar('Loss/eval', epoch_loss, self.current_step)

    if score > self.best_valid_score and train_loss > epoch_loss * 0.5:
      self.best_valid_score = score
      logger.info('new best validation score: %.4f' % score)

    else:
      logger.info('validation score: {:.4f}'.format(score))

    if epoch_loss < self.best_val_loss and train_loss > epoch_loss * 0.5:
      self.best_val_loss = epoch_loss
      logger.info('new best validation loss: %.4f' % epoch_loss)

      logger.info('saving the current model to disk...')
      state = {
        'parameters': self.parameters,
        'state_dict': self.nets.state_dict(),
        'best_valid_score': self.best_valid_score,
        'optimizer': self.optimizer.state_dict()
      }
      if self.replica_rank == 0:
        torch.save(
          state, os.path.join(self.model_dir, f"epoch{epoch}_{mini_batch}_best_val_model.pth.tar")
        )

    else:
      logger.info('validation loss: {:.4f}'.format(epoch_loss))

  def predict(self,
              eval_loader=None,
              log_freq=1000):
    # Iterate over data.
    sys_preds = []
    epoch_loss = []
    num_batches = 0
    for x in eval_loader:
      self.nets.train(False)  # Set nets to training mode

      # forward
      preds, probs, loss = self.nets.forward(**x)

      # compute the mean of a list of loss from multi gpu
      loss = torch.mean(loss)

      epoch_loss.append(loss.data.cpu())

      sys_preds.append(preds.data.cpu())

      # log current training status
      num_batches += 1
      if num_batches % log_freq == 0:
        logger.info('%d batches processed' % num_batches)
      yield preds.data.cpu().numpy(), \
            probs.data.cpu().numpy(), \
            loss.data.cpu().numpy()

  @abc.abstractmethod
  def scorer(self, sys_preds, target):
    raise NotImplementedError("scorer")
