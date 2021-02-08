# experiment management system
import argparse
import logging
import os
import subprocess
import json
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


def train(config, config_name):
  model_dir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'model', config_name
  )
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  tensorboard_log_dir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'tensorboard_log', config_name
  )
  if not os.path.exists(tensorboard_log_dir):
    os.makedirs(tensorboard_log_dir)

  #
  # run command
  #
  script = config['global_config']['python_path'] + '/end2end/soloist/methods/{}/train_dist.py'.format(config['method'])
  cmd = [
    'python',
    script,
    '--train', config['data']['train'],
    '--dev', config['data']['dev'],
    '--model_dir', model_dir,
    '--tensorboard_log_dir', tensorboard_log_dir,
    '--job_name', config_name,
    '--python_path', config['global_config']['python_path']
  ]
  for k, v in config['train'].items():
    cmd += ['--{}'.format(k), v]

  os.environ.update(
    {'CUDA_VISIBLE_DEVICES': config['train']['cuda_device'],
     'PYTHONPATH': config['global_config']['python_path']}
  )
  logger.info('train script:')
  logger.info(' '.join(cmd))
  subprocess.call(' '.join(cmd), shell=True)


def eval(config, config_name, output_dir):
  # to-do
  pass


def luban_dist_inference(test_set, num_tasks, config, config_name, output_dir):
  # to-do
  pass


def ems(args, config, config_name):
  if args.mode in ['train', 'all']:
    train(config, config_name)

  if args.mode in ['eval', 'all']:
    # to-do
    pass


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('config')
  parser.add_argument('mode', default='all', type=str,
                      help='model: train, eval, score, all')
  args = parser.parse_args()

  # load config
  with open(args.config) as f:
    config = json.load(f)
  config_name = os.path.basename(args.config).replace('.json', '')

  ems(args, config, config_name)


if __name__ == "__main__":
  main()
