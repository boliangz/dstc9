import torch
import logging
import json
from transformers import GPT2Tokenizer
import random

from end2end.soloist.methods.baseline.dataset.dialog import Dialog

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.root.setLevel(level=logging.INFO)


class SoloistDataset(torch.utils.data.dataset.Dataset):
  def __init__(self, data_in_json, pretrained_gpt, max_seq_len, max_resp_len,
               is_test=False, max_turn=10, **kwargs):
    # parameters
    self.is_test = is_test
    self.max_seq_len = max_seq_len
    self.max_resp_len = max_resp_len
    self.max_turn = max_turn

    # number of corrupted turns where the length of belief + db + response
    # string is already larger than max sequence length
    self.num_corrupted_turn = 0

    # number of turns that fit the max_seq_len after max_turn adjusted.
    self.num_adjusted_turn = 0

    # load data
    with open(data_in_json) as f:
      data = json.load(f)

    # load pretrained gpt2 tokenizer
    self.tokenizer = self.load_tokenizer(pretrained_gpt)

    # load dialogs
    self.dialogs = []
    for dialog_id, dialog_json in data.items():
        self.dialogs.append(Dialog(dialog_id, dialog_json, max_resp_len))
    self.dialogs = self.dialogs

    # make ml instances
    self.instances = []
    for dialog in self.dialogs:
        self.instances += dialog.ml_instances

    logger.info(
      '{} dialogs are loaded from: {}'.format(len(self.dialogs), data_in_json)
    )
    logger.info(
      '{} ML instances are loaded.'.format(len(self.instances))
    )

  def get_dialog(self, dialog_id):
    for d in self.dialogs:
      if d.id == dialog_id:
        return d

  @ staticmethod
  def load_tokenizer(pretrained_gpt):
    tokenizer = GPT2Tokenizer.from_pretrained(
      pretrained_gpt, do_lower_case=False
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # add soloist special tokens
    tokens_to_add = ['Ġ<SOB>Ġ', 'Ġ<EOB>Ġ', 'Ġ<EOKB>Ġ', 'Ġ<EOS>Ġ', 'Ġ<DMN>Ġ']
    tokenizer.add_tokens(tokens_to_add)
    logger.info('tokens added to the tokenizer:')
    logger.info(', '.join(tokens_to_add))

    return tokenizer

  def __getitem__(self, idx):
    instance = self.instances[idx]

    # dynamically adjust max turn to fit the max_seq_len
    max_turn = min(self.max_turn, len(instance.history))
    while max_turn >= 0:
      try:
        # get positive and negative ML instance strings
        positive_ml_string, negative_ml_string, neg_sample_choice = instance.get_ml_string(
          max_turn=max_turn, all_dialogs=self.dialogs
        )

        positive_inputs = self.tokenizer(positive_ml_string,
                                         max_length=self.max_seq_len,
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors="pt")
        positive_ml_tokens = self.tokenizer.convert_ids_to_tokens(
          positive_inputs['input_ids'].tolist()[0]
        )
        # get token index span of belief state
        belief_state_start = positive_ml_tokens.index('Ġ<SOB>Ġ')
        belief_state_end = positive_ml_tokens.index('Ġ<EOB>Ġ')
        response_start = positive_ml_tokens.index('Ġ<EOKB>Ġ')
        response_end = positive_ml_tokens.index('Ġ<EOS>Ġ')

        negative_inputs = self.tokenizer(negative_ml_string,
                                         max_length=self.max_seq_len,
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors="pt")
        negative_ml_tokens = self.tokenizer.convert_ids_to_tokens(
          negative_inputs['input_ids'].tolist()[0])
        negative_eos_start = negative_ml_tokens.index('Ġ<EOS>Ġ')

        model_inputs = {
          # positive sample
          'positive_input_ids': positive_inputs['input_ids'][0],
          'positive_attention_mask': positive_inputs['attention_mask'][0],
          'positive_labels': positive_inputs['input_ids'][0],
          # negative sample
          'negative_input_ids': negative_inputs['input_ids'][0],
          'negative_attention_mask': negative_inputs['attention_mask'][0],
          # belief state span
          'belief_state_start': torch.LongTensor([belief_state_start]),
          'belief_state_end': torch.LongTensor([belief_state_end]),
          # response span
          'response_start': torch.LongTensor([response_start]),
          'response_end': torch.LongTensor([response_end]),
          # positive eos
          'positive_eos_start': torch.LongTensor([response_end]),
          # negative eos
          'negative_eos_start': torch.LongTensor([negative_eos_start]),
          # contrastive positive labels
          'contrastive_positive_labels': torch.LongTensor([1]),
          # contrastive negatie labels
          'contrastive_negative_labels': torch.LongTensor([0]),
          # neg_sample_choice
          'neg_sample_choice': torch.LongTensor([neg_sample_choice])
        }
        if max_turn != min(self.max_turn, len(instance.history)):
          self.num_adjusted_turn += 1
          # print(positive_ml_string)
          # print(negative_ml_string)

        break
      # special tokens, e.g. "Ġ<EOS>Ġ", are truncated if string length is
      # larger than max_seq_len, so a ValueError will be thrown when doing
      # string indexing.
      except ValueError:
        # Reduce the max turn until the string fits max_seq_len.
        max_turn -= 1

    # when string can't fix max_seq_len even when max_turn=0, randomly
    # select another instance from the dataset.
    if max_turn < 0:
        self.num_corrupted_turn += 1
        # print(positive_ml_string)
        # print(negative_ml_string)

        return self.__getitem__(random.choice(range(len(self))))

    return model_inputs

  def __len__(self):
    return len(self.instances)
