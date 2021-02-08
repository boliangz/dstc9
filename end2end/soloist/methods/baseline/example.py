import os
import random
import torch
import torch.distributed as dist
import netifaces as ni

from transformers import GPT2Tokenizer
from end2end.soloist.methods.baseline.nn.gpt import GPT2Model


def load_model(parameters):
  # load model
  model_dir = parameters['model']
  print('loading model from: %s' % model_dir)
  model_path = os.path.join(model_dir, 'best_val_model.pth.tar')
  state = torch.load(model_path)

  # get model parameters
  state_parameters = state['parameters']
  state_parameters.update(parameters)
  state_parameters['pretrained_gpt'] = model_dir
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
                    model_dir=model_dir,
                    process_idx=0,
                    replica_rank=0,
                    is_test=True,
                    tokenizer=load_tokenizer(parameters))

  # load pre-trained weigths
  model.nets.load_state_dict(state['state_dict'])
  model.nets.train(False)

  return model


def load_tokenizer(parameter):
  tokenizer = GPT2Tokenizer.from_pretrained(
    parameter['model'], do_lower_case=False
  )
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  # add soloist special tokens
  tokens_to_add = ['Ġ<SOB>Ġ', 'Ġ<EOB>Ġ', 'Ġ<EOKB>Ġ', 'Ġ<EOS>Ġ']
  tokenizer.add_tokens(tokens_to_add)
  print('tokens added to the tokenizer:')
  print(', '.join(tokens_to_add))

  return tokenizer


def model_step(model, input_str, stop_token):
  inputs = model.nets.module.tokenizer(input_str, return_tensors="pt")
  output_sequences = model.nets.module.gpt2.generate(
    input_ids=inputs['input_ids'].cuda(),
    attention_mask=inputs['attention_mask'].cuda(),
    token_type_ids=inputs['attention_mask'].cuda(),
    max_length=1024,
    eos_token_id=model.nets.module.tokenizer.convert_tokens_to_ids([stop_token])[0],  # stop at [EOS]
    num_return_sequences=1,
    num_beams=1,
    do_sample=False
  )
  output_tokens = model.nets.module.tokenizer.convert_ids_to_tokens(output_sequences.tolist()[0])

  print(''.join(output_tokens).replace('Ġ', ' '))


def main():
  ground_truth = "User: i would like to book a 5 star , or closest to it , in the east part of town please . " \
                 "Ġ<SOB>Ġ hotel { area = east, stars = 5 } restaurant { area = east } Ġ<EOB>Ġ " \
                 "DB: hotel 0 match Ġ<EOKB>Ġ " \
                 "there are several hotel -s with those characteristics . could you narrow down by price range and hotel type you want ? Ġ<EOS>Ġ"

  belief_state_prediction_input = "User: i would like to book a 5 star , or closest to it , in the east part of town please . " \
                                  "Ġ<SOB>Ġ"

  response_prediction_input = "User: i am planning a trip in cambridge System: OK , what type of information are you looking for ? " \
                              "User: i need to find a train to arrive on saturday by 11:45 going to cambridge from bishops stortford . " \
                              "System: Train number TR6163 leaves Bishops Stortford at 05:29 and arrives in Cambridge by 06:07 . Would you like a ticket for this one ? " \
                              "User: yes , i would like a ticket for 1 please . " \
                              "Ġ<SOB>Ġ train { destination = cambridge, day = saturday, arrive = 11:45, departure = bishops stortford, people = 1 } " \
                              "Ġ<EOB>Ġ DB: attraction 7 match Ġ<EOKB>Ġ"

  parameters = {
    'model': '/nfs/project/users/boliangzhang/workspace/dstc9/end2end_soloist/experiment_management/model/baseline'
  }



  model = load_model(parameters)

  # model_step(model,
  #            "User: i 'd like a sports place in the centre please . a boat type of attraction . System: there are 2 boat attractions in the centre . 1 is scudamores punting co and the other is scudamores punting co . User: oh , and what is their postcode , please ? what is the address ? System: the address is granta place , cb21rs . User: i ' m departing from peterborough . i need a train going to leicester . this will be for tuesday . i want to get there by 18:30 at the latest . Ġ<SOB>Ġ",
  #            stop_token='Ġ<EOB>Ġ')

  model_step(model, response_prediction_input, stop_token='Ġ<EOS>Ġ')


if __name__ == "__main__":
  main()