import argparse
import logging
import shutil
import os
import re
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import netifaces as ni

from end2end.soloist.methods.baseline.dataset.dataset import SoloistDataset
from end2end.soloist.methods.baseline.nn.gpt import GPT2Model


logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


def get_arg_parser():
  """
  build argument parser for training.
  """
  parser = argparse.ArgumentParser()
  # IO config
  parser.add_argument(
    "--train", required=True,
    help="training set directory path"
  )
  parser.add_argument(
    "--dev", required=True,
    help="dev set directory path"
  )
  parser.add_argument(
    "--model_dir", required=True,
    help="model directory path"
  )
  parser.add_argument(
    "--tensorboard_log_dir", default=None,
    help="directory to store tensorboard log."
  )
  parser.add_argument(
    "--pretrained_gpt", required=True,
    help="GPT2 pretrained model directory path"
  )
  parser.add_argument(
    "--reload", type=str,
    help="pre-trained model dir to reload parameters from."
  )
  parser.add_argument(
    "--max_turn", type=int
  )
  parser.add_argument(
    "--max_seq_len", type=int, default=1024,
  )
  parser.add_argument(
    "--max_resp_len", type=int, default=512,
  )
  # train config
  parser.add_argument(
    "--optimizer", default="sgd",
    help="Learning method (sgd, adagrad, adam, bert_adam)"
  )
  parser.add_argument(
    "--learning_rate", default=0.01, type=float,
    help="learning rate"
  )
  parser.add_argument(
    "--num_epochs", default="10",
    type=int, help="Number of training epochs"
  )
  parser.add_argument(
    "--batch_size_per_gpu", default="20",
    type=int, help="Batch size per node."
  )
  parser.add_argument(
    "--gpu_num", default="1", type=int,
    help="number of gpus in a node."
  )
  parser.add_argument(
    "--node_num", default=1, type=int,
    help='number of nodes.'
  )
  parser.add_argument(
    "--rank", default=0, type=int,
    help='ranking within the nodes.'
  )
  parser.add_argument(
    "--master_addr", default='localhost',
    help='address of the master node. set to localhost if node_num is 1.'

  )
  parser.add_argument(
    "--master_port", default='3300',
    help='port of the master node.'

  )
  parser.add_argument(
    "--net_interface", default='eth0',
    help='network interface to use to communicate between nodes. use ifconfig '
         'to check interfaces. (eth0 on luban and eno1 on mtv server.)'
  )
  parser.add_argument(
    "--job_name", default='train_dist',
    help='job name for distributed training.'
  )
  # display config
  parser.add_argument(
    "--validation_freq", default=None, type=int,
    help="frequency to validate model. it's the number of batches."
  )
  parser.add_argument(
    "--log_freq", default=1000, type=int,
    help="frequency to log. it's the number of batches."
  )
  return parser


def main():
  parser = get_arg_parser()
  args, _ = parser.parse_known_args()

  parameters = vars(args)

  # copy pre-trained gpt vocab and config to model dir
  shutil.copy(
    os.path.join(parameters['pretrained_gpt'], 'config.json'),
    parameters['model_dir']
  )
  shutil.copy(
    os.path.join(parameters['pretrained_gpt'], 'vocab.json'),
    parameters['model_dir']
  )
  shutil.copy(
    os.path.join(parameters['pretrained_gpt'], 'merges.txt'),
    parameters['model_dir']
  )

  # setup distributed training
  parameters['world_size'] = parameters['node_num'] * parameters['gpu_num']
  os.environ['NCCL_DEBUG'] = 'INFO'
  os.environ['NCCL_SOCKET_IFNAME'] = parameters['net_interface']
  if parameters['node_num'] == 1:
    self_addr = ni.ifaddresses(parameters['net_interface'])[ni.AF_INET][0]['addr']
    os.environ['MASTER_ADDR'] = self_addr
  else:
    os.environ['MASTER_ADDR'] = parameters['master_addr']
  os.environ['MASTER_PORT'] = parameters['master_port']
  kill_process_by_port(os.environ['MASTER_PORT'])

  logger.info('spawning processes for distributed training, this may take a while.')
  mp.spawn(train_dist, nprocs=args.gpu_num, args=(parameters,))


def train_dist(process_idx, parameters):
  logger.info('initializing process group...')
  replica_rank = parameters['rank'] * parameters['gpu_num'] + process_idx
  logger.info('current replica rank {}'.format(replica_rank))
  dist.init_process_group(backend='nccl',
                          init_method='env://',
                          world_size=parameters['world_size'],
                          rank=replica_rank)
  torch.manual_seed(0)

  # set train logger
  if replica_rank == 0:  # only master process logs to file
    log_file = os.path.join(parameters['model_dir'], 'train.log')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(
      logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    )
    logger.addHandler(fileHandler)
  elif parameters['node_num'] == 1:
    # prevent non-master process logging to stdout in single node setting.
    logger.setLevel(logging.WARN)

  # print parameters
  logger.info('training parameters:')
  for k, v in parameters.items():
    logger.info('  %s=%s' % (str(k), str(v)))

  # data pipeline
  train_dataset = SoloistDataset(
    data_in_json=parameters['train'],
    **parameters
  )

  # export ml instances for debugging
  if replica_rank == 0:
    export_ml_instances(parameters, train_dataset)

  validation_dataset = SoloistDataset(
    data_in_json=parameters['dev'],
    **parameters
  )

  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=parameters['world_size'],
    rank=replica_rank,
    shuffle=True
  )

  train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=parameters['batch_size_per_gpu'],
    shuffle=False,  # must be false when used with distributed sampler
    num_workers=0,
    pin_memory=True,
    sampler=train_sampler
  )

  validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset,
    batch_size=parameters['batch_size_per_gpu'],
    shuffle=False,
    num_workers=0,
    pin_memory=True
  )

  # init or reload model
  max_steps = len(train_loader) * parameters['num_epochs']
  model = GPT2Model(**parameters,
                    process_idx=process_idx,
                    replica_rank=replica_rank,
                    tokenizer=train_dataset.tokenizer,
                    max_steps=max_steps)

  # start training
  logger.info('training starts...')
  model.train(train_loader=train_loader,
              validation_loader=validation_loader,
              validation_freq=parameters['validation_freq'],
              epochs=parameters['num_epochs'],
              log_freq=parameters['log_freq'],
              )


def kill_process_by_port(port):
  cmd = "lsof -i tcp:'{}' | awk 'NR!=1 {{print $2}}' | xargs kill -9" \
    .format(str(port))
  subprocess.call(cmd, shell=True)


def export_ml_instances(parameters, train_dataset):
  out_file = os.path.join(parameters['model_dir'], 'ml_instances.txt')
  with open(out_file, 'w') as f_out:
    for i in range(5000):
      model_inputs = train_dataset[i]

      f_out.write('=' * 80 + '\n')
      positive_ml_tokens = train_dataset.tokenizer.convert_ids_to_tokens(
        model_inputs['positive_input_ids'].tolist()
      )
      positive_ml_string = ''.join(positive_ml_tokens).replace('Ġ', ' ')
      f_out.write('{:<20} {}\n\n'.format(
        'positive ml string:', positive_ml_string.replace('[PAD]', ''))
      )

      # get token index span of belief state
      belief_state_start = model_inputs['belief_state_start'].tolist()[0]
      belief_state_end = model_inputs['belief_state_end'].tolist()[0]
      response_start = model_inputs['response_start'].tolist()[0]
      response_end = model_inputs['response_end'].tolist()[0]

      f_out.write('{:<20}\n'.format('dialog history:'))
      dialog_history = ''.join(
        positive_ml_tokens[:belief_state_start]
      ).replace('Ġ', ' ').strip()
      dialog_history = re.split('User: |System: ', dialog_history)[1:]
      for i in range(0, len(dialog_history), 2):
        f_out.write('  {} {}\n'.format('User:  ', dialog_history[i]))
        try:
          f_out.write('  {} {}\n'.format('System:', dialog_history[i + 1]))
        except IndexError:
          pass

      f_out.write('{:<20} {}\n'.format(
        'belief str:',
        ''.join(
          positive_ml_tokens[belief_state_start:belief_state_end + 1]).replace(
          'Ġ', ' ').strip()
      ))
      f_out.write('{:<20} {}\n'.format(
        'response str:',
        ''.join(positive_ml_tokens[response_start:response_end + 1]).replace(
          'Ġ', ' ').strip()
      ))
      f_out.write('\n')

      # negative instance
      negative_ml_tokens = train_dataset.tokenizer.convert_ids_to_tokens(
        model_inputs['negative_input_ids'].tolist())
      negative_ml_string = ''.join(negative_ml_tokens).replace('Ġ', ' ')
      f_out.write('{:<20} {}\n'.format(
        'negative ml string:', negative_ml_string.replace('[PAD]', ''))
      )
      negative_eos_start = model_inputs['negative_eos_start'].tolist()[0]
      f_out.write(
        'negative eos: {}\n'.format(negative_ml_tokens[negative_eos_start])
      )

      negative_sampling_choice = model_inputs['neg_sample_choice']
      f_out.write(
        'negative sampling choice: {}\n'.format(negative_sampling_choice)
      )

      belief_state_start = negative_ml_tokens.index('Ġ<SOB>Ġ')
      belief_state_end = negative_ml_tokens.index('Ġ<EOB>Ġ')
      response_start = negative_ml_tokens.index('Ġ<EOKB>Ġ')
      response_end = negative_ml_tokens.index('Ġ<EOS>Ġ')

      f_out.write('{:<20} {}\n'.format(
        'belief str:',
        ''.join(
          negative_ml_tokens[belief_state_start:belief_state_end + 1]).replace(
          'Ġ', ' ').strip()
      ))
      f_out.write('{:<20} {}\n'.format(
        'response str:',
        ''.join(negative_ml_tokens[response_start:response_end + 1]).replace(
          'Ġ', ' ').strip()
      ))
      f_out.write('\n')


if __name__ == "__main__":
  main()
