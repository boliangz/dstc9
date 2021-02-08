import argparse
import logging
import shutil
import os
import re
import random
import torch
import csv
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import netifaces as ni
import collections
import traceback

from end2end.soloist.methods.baseline.dataset.dataset import SoloistDataset
from end2end.soloist.methods.baseline.nn.gpt import GPT2Model
from end2end.kb_search import get_db_state, belief_state_to_dict


logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()




def main():
  parser = argparse.ArgumentParser()
  # IO config
  parser.add_argument(
    "--test", required=True,
    help="multiwoz test set json file."
  )
  parser.add_argument(
    "--model_dir", required=True,
    help="model directory path"
  )
  parser.add_argument(
    "--output", help='output csv file.'
  )
  parser.add_argument(
    "--log_freq", default=50, type=int,
    help="frequency to log. it's the number of batches."
  )
  args, _ = parser.parse_known_args()

  parameters = vars(args)

  model, model_parameters = load_model(parameters['model_dir'])
  parameters.update(model_parameters)

  # data pipeline
  test_dataset = SoloistDataset(
    data_in_json=parameters['test'],
    **parameters
  )

  test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
  )

  with open(args.output, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"')
    field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'bspn', 'resp_gen',
             'resp', 'aspn_gen', 'aspn', 'dspn_gen', 'dspn', 'pointer']

    csv_writer.writerow(field)

    current_dialog_id = None

    # start inference
    for x in test_loader:
      model.nets.train(False)  # Set nets to eval mode

      ml_instance_tokens = model.nets.module.tokenizer.convert_ids_to_tokens(
        x['positive_input_ids'].tolist()[0]
      )
      ml_instance_str = ''.join(ml_instance_tokens).replace('Ġ', ' ')

      # get dialog history
      dialog_history = ml_instance_tokens[:x['belief_state_start'][0]]
      dialog_history = ''.join(dialog_history).replace('Ġ', ' ')
      # print(dialog_history)

      # predict belief state
      num_beams = 1
      while True:
        # belief state prediction
        belief_state_eos = 'Ġ<EOB>Ġ'
        belief_state_str = model_step(
          model=model,
          input_str=dialog_history + ' Ġ<SOB>Ġ',
          stop_token=belief_state_eos,
          num_beams=num_beams
        )
        # print('belief_state_str: ', belief_state_str)

        # retrieve db state
        try:
          db_state = get_db_state(belief_state_str, dataset='multiwoz')
          if not db_state:
            db_state = 'EMPTY'
          db_state_str = '{} DB: {} Ġ<EOKB>Ġ'.format(
            belief_state_str.replace('<EOB>', 'Ġ<EOB>Ġ').replace('<SOB>', 'Ġ<SOB>Ġ'),
            db_state
          )
          break
        except Exception as e:
          print('belief_state_str', belief_state_str)
          traceback.print_exc()
          num_beams += 1
          continue

      # response prediction
      response_eos = 'Ġ<EOS>Ġ'
      response_template = model_step(
        model=model,
        input_str=db_state_str,
        stop_token=response_eos
      )
      # print('response_template:', response_template)

      # parse results and generate results in DAMB scorer format
      dialog_id = x['dialog_id'][0]
      turn_num = x['turn_num'][0]

      user_utterance = dialog_history.split('User:')[-1].strip()

      # parse system belief state
      bs_sys = belief_state_to_dict(belief_state_str)
      belief_span_sys = ' '.join(
        ['[{}] {}'.format(k, ' '.join([' '.join([_k, _v]) for _k, _v in v.items()]))
         for k, v in bs_sys.items()]
      )

      # parse gold belief state
      belief_state_gold = belief_state_to_dict(ml_instance_str)
      belief_span_gold = ' '.join(
        ['[{}] {}'.format(k, ' '.join([' '.join([_k, _v]) for _k, _v in v.items()]))
         for k, v in belief_state_gold.items()]
      )

      # parse system response template
      response_template_sys = re.search('<EOKB>(.*)<EOS>', response_template).group(1).strip()
      ###
      # Postprocess for reference number to be 8-digits if not delexed.
      ###
      if "reference number is : " in response_template_sys \
          and "[value_reference]" not in response_template_sys:
        generated_ref = re.search('reference number is : (.*) \.',
                                  response_template_sys).group(1)
        response_template_sys = response_template_sys.replace(generated_ref, "[value_reference]")

      # parse gold response template
      response_template_gold = re.search('<EOKB>(.*)<EOS>', ml_instance_str).group(1).strip()

      # save the first empty row
      if dialog_id.lower() != current_dialog_id:
        csv_writer.writerow(
          [dialog_id.lower(),
           len(test_dataset.get_dialog(dialog_id).turns), '', '', '', '',
           '', '', '', '', '', '']
        )
        current_dialog_id = dialog_id.lower()

      # save row
      csv_writer.writerow(
        [dialog_id.lower(),
         turn_num,
         user_utterance,
         belief_span_sys,
         belief_span_gold,
         response_template_sys,
         response_template_gold,
         '',
         '',
         '',
         '',
         '']
      )


def load_model(model_path):
  # load tokenizer
  tokenizer = SoloistDataset.load_tokenizer(model_path)

  # load model
  print('loading model from: %s' % model_path)
  checkpoint = os.path.join(model_path, 'best_val_model.pth.tar')
  if not torch.cuda.is_available():
    state = torch.load(checkpoint, map_location=torch.device('cpu'))
  else:
    state = torch.load(checkpoint)

  # get model parameters
  state_parameters = state['parameters']
  state_parameters['pretrained_gpt'] = model_path
  parameters = state_parameters

  # init process group
  parameters['node_num'] = 1
  parameters['rank'] = 0

  # setup distributed learning env
  parameters['world_size'] = parameters['node_num'] * parameters['gpu_num']
  os.environ['NCCL_DEBUG'] = 'INFO'
  os.environ['NCCL_SOCKET_IFNAME'] = parameters['net_interface']
  self_addr = ni.ifaddresses(parameters['net_interface'])[ni.AF_INET][0]['addr']
  os.environ['MASTER_ADDR'] = self_addr
  port = random.randint(3300, 61000)
  os.environ['MASTER_PORT'] = str(port)

  print('initializing process group...')
  dist.init_process_group(backend='nccl',
                          init_method='env://',
                          world_size=1,
                          rank=0)
  torch.manual_seed(0)

  # init model
  print('model parameters:')
  for k, v in parameters.items():
    print('  {}={}'.format(k, v))

  model = GPT2Model(**parameters,
                    model_dir=model_path,
                    process_idx=0,
                    replica_rank=0,
                    is_test=True,
                    tokenizer=tokenizer)

  # load pre-trained weigths
  if not torch.cuda.is_available():
    new_state_dict = collections.OrderedDict()
    for k, v in state['state_dict'].items():
      name = k[7:]  # remove `module.`
      new_state_dict[name] = v
    state['state_dict'] = new_state_dict

  model.nets.load_state_dict(state['state_dict'])
  model.nets.train(False)

  return model, parameters


def model_step(model, input_str, stop_token, num_beams=1):
  # suppress transformers tokenization warning
  import logging
  from transformers import logger
  logger.setLevel(logging.ERROR)

  inputs = model.nets.module.tokenizer(input_str, return_tensors="pt")
  input_ids = inputs['input_ids'].cuda()
  attention_mask = inputs['attention_mask'].cuda()
  token_type_ids = inputs['attention_mask'].cuda()

  output_sequences = model.nets.module.gpt2.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    max_length=1024,
    # stop at stop_token
    eos_token_id=model.nets.module.tokenizer.convert_tokens_to_ids([stop_token])[0],
    num_return_sequences=1,
    num_beams=num_beams,
    do_sample=False
  )

  output_tokens = model.nets.module.tokenizer.convert_ids_to_tokens(
    output_sequences.tolist()[0]
  )

  rtn = ''.join(output_tokens).replace('Ġ', ' ').strip()

  return rtn


if __name__ == "__main__":
  main()
