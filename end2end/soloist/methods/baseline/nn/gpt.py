import logging
import os
import itertools
import torch
import collections
import torch.nn as nn
from end2end.soloist.methods.baseline.nn.base_model import BaseModel

from transformers import (
  GPT2Config,
  GPT2LMHeadModel,
  GPT2Tokenizer,
)

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class GPT2Nets(nn.Module):
  def __init__(self, pretrained_gpt, tokenizer=None, is_test=False, **kwargs):
    super(GPT2Nets, self).__init__()

    #
    # GPT2
    #
    self.tokenizer = tokenizer

    self.config = GPT2Config.from_json_file(
      os.path.join(pretrained_gpt, 'config.json')
    )

    if is_test:
      self.gpt2 = GPT2LMHeadModel(self.config)
    else:
      self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_gpt)
    self.gpt2.resize_token_embeddings(len(self.tokenizer))

    #
    # init contrastive prediction linear
    #
    self.contrastive_linear = nn.Linear(self.config.n_embd, 2)

    #
    # initialize weights of each layer
    #
    self.init_params()

  def init_params(self):
    for p in self.named_parameters():
      if 'gpt' in p[0]:
        continue
      elif 'bias' in p[0]:
        p[1].data.zero_()
      elif p[1].dim() == 1:
        nn.init.uniform_(p[1].data)
      else:
        nn.init.xavier_uniform_(p[1].data)

  def param_size(self):
    num_params = 0
    for p in self.parameters():
      num_params += p.numel()
    return num_params

  def forward(self,
              positive_input_ids,
              positive_attention_mask,
              positive_labels,
              negative_input_ids,
              negative_attention_mask,
              belief_state_start,
              belief_state_end,
              response_start,
              response_end,
              positive_eos_start,
              negative_eos_start,
              contrastive_positive_labels,
              contrastive_negative_labels,
              **kwargs):
    batch_size = positive_input_ids.size(0)

    # encode positive sample:
    # language model crossentropy loss is reduced in GPT2LMHeadModel,
    # as we needs the loss of each token to compute the loss of the span of
    # belief state and response, so we write forward ourselves.
    positive_transformer_outputs = self.gpt2.transformer(
      positive_input_ids,
      attention_mask=positive_attention_mask
    )
    # [batch_size, seq_len, hidden_dim]
    positive_hidden_states = positive_transformer_outputs[0]
    # print('positive_hidden_states', positive_hidden_states.size())

    # [batch_size, seq_len, vocab_size]
    positive_lm_logits = self.gpt2.lm_head(positive_hidden_states)
    # print('positive_lm_logits', positive_lm_logits.size())

    loss = None
    if positive_labels is not None \
        and contrastive_positive_labels is not None \
        and contrastive_negative_labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = positive_lm_logits[..., :-1, :].contiguous()
      shift_labels = positive_labels[..., 1:].contiguous()
      # Flatten the tokens
      loss_fct = nn.CrossEntropyLoss(reduce=False)  # set reduce to false

      # [batch_size, seq_len - 1]
      language_model_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
      ).view(batch_size, -1)
      # print('language_model_loss', language_model_loss.size())

      # belief prediction loss
      # [batch_size, ?] ? is ambiguous length according to belief state span
      # length
      belief_prediction_loss = []
      for i, element in enumerate(language_model_loss):
        belief_prediction_loss.append(
          element[belief_state_start.view(-1)[i]:belief_state_end.view(-1)[i]+1]
        )
      # print('belief_prediction_loss',
      #       [item.size() for item in belief_prediction_loss])

      # response generation loss
      # [batch_size, ?] ? is ambiguous length according to response span length
      response_generation_loss = []
      for i, element in enumerate(language_model_loss):
        response_generation_loss.append(
          element[response_start.view(-1)[i]:response_end.view(-1)[i]+1]
        )
      # print('response_generation_loss',
      #       [item.size() for item in response_generation_loss])

      #
      # contrastive loss
      #
      # positive sample part
      # [batch_size, seq_len, 2]
      positive_sample_logits = self.contrastive_linear(positive_hidden_states)
      # print('positive_sample_logits', positive_sample_logits.size())
      # print('positive_eos_start', positive_eos_start.size())

      # [batch_size, 2]
      positive_sample_eos_logits = []
      for i, element in enumerate(positive_sample_logits):
        positive_sample_eos_logits.append(element[positive_eos_start.view(-1)[i]])
      positive_sample_eos_logits = torch.stack(positive_sample_eos_logits)
      # print('positive_sample_eos_logits', positive_sample_eos_logits.size())

      # [batch_size]
      contrastive_loss_positive = nn.CrossEntropyLoss(reduce=False)(
        positive_sample_eos_logits, contrastive_positive_labels.view(-1)
      )
      # print('contrastive_loss_positive', contrastive_loss_positive.size())

      # negative sample part
      # encode negative sample
      negative_transformer_outputs = self.gpt2.transformer(
        negative_input_ids,
        attention_mask=negative_attention_mask
      )

      # [batch_size, seq_len, hidden_dim]
      negative_hidden_states = negative_transformer_outputs[0]

      # [batch_size, seq_len, 2]
      negative_sample_logits = self.contrastive_linear(negative_hidden_states)

      # [batch_size, 2]
      negative_sample_eos_logits = []
      for i, element in enumerate(negative_sample_logits):
        negative_sample_eos_logits.append(element[negative_eos_start.view(-1)[i]])
      negative_sample_eos_logits = torch.stack(negative_sample_eos_logits, dim=0)
      # print('negative_sample_eos_logits', negative_sample_eos_logits.size())

      # [batch_size]
      contrastive_loss_negative = nn.CrossEntropyLoss(reduce=False)(
        negative_sample_eos_logits, contrastive_negative_labels.view(-1)
      )
      # print('contrastive_loss_negative', contrastive_loss_negative.size())

      # avg loss by batch size
      loss = (sum([sum(element) for element in belief_prediction_loss])
              + sum([sum(element) for element in response_generation_loss])
              + sum(contrastive_loss_positive)
              + sum(contrastive_loss_negative)) / batch_size

    preds = loss
    probs = loss

    return preds, probs, loss


class GPT2Model(BaseModel):
  def __init__(self, pretrained_gpt, process_idx, max_steps=1000, tokenizer=None,
               is_test=False, **kwargs):
    nets = GPT2Nets(pretrained_gpt, tokenizer, is_test)

    kwargs.update({'pretrained_gpt': pretrained_gpt})
    super(GPT2Model, self).__init__(
      max_steps=max_steps, nets=nets, process_idx=process_idx, **kwargs
    )

  def scorer(self, sys_preds, data_loader):
    return 0


def safe_division(n, d):
  return n / d if d else 0
