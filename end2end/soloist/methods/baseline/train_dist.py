import logging
import os
import subprocess
import netifaces as ni
from end2end.soloist.methods.baseline.train import get_arg_parser

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


def main():
  parser = get_arg_parser()
  parser.add_argument('--on_luban', default=0, type=int)
  parser.add_argument('--python_path')
  args, _ = parser.parse_known_args()
  parameters = vars(args)

  # convert nargs parameters to string
  for k, v in parameters.items():
    if type(v) == list:
      parameters[k] = ' '.join([str(ele) for ele in v])

  if parameters['node_num'] == 1:
    single_node_train(parameters)
  else:
    dist_train(parameters)


def single_node_train(parameters):
  # create luban tasks
  train_script = os.path.join(os.path.dirname(__file__), 'train.py')
  cmd = """
        python {} {}
        """.format(
    train_script,
    ' '.join(['--{} {}'.format(k, v) for k, v in parameters.items() if v]),
  )
  logger.info('single node train cmd:\n{}'.format(cmd))
  subprocess.call(cmd, shell=True)


def dist_train(parameters):
  # if currently not on luban, submit luban offline task to start distributed
  # training.
  if parameters['on_luban'] == 0:
    parameters['on_luban'] = 1
    cmd = """
    lubano submit {} '{}' --gpus 1 --job_name {}.node-{} --python_path {}
    """.format(
      os.path.abspath(__file__),
      ' '.join(['--{} {}'.format(k, v) for k, v in parameters.items() if v]),
      parameters['job_name'],
      0,
      parameters['python_path']
    )
    subprocess.call(cmd, shell=True)
  # if executed on luban, first submit children tasks and then execute master
  # task
  else:
    parameters.pop('on_luban')

    master_addr = ni.ifaddresses(parameters['net_interface'])[ni.AF_INET][0]['addr']
    parameters['master_addr'] = master_addr
    logger.info('MASTER_ADDE={}'.format(master_addr))

    # create luban tasks
    train_script = os.path.join(os.path.dirname(__file__), 'train.py')
    for i in range(parameters['node_num'] - 1):
      parameters['rank'] = i + 1
      cmd = """
      lubano submit {} '{}' --gpus 1 --job_name {}.node-{} --python_path {}
      """.format(
        train_script,
        ' '.join(['--{} {}'.format(k, v) for k, v in parameters.items() if v]),
        parameters['job_name'],
        i + 1,
        parameters['python_path']
      )
      logger.info('cmd for node {}:\n{}'.format(i+1, cmd))
      subprocess.call(cmd, shell=True)

    # create master task
    parameters['rank'] = 0
    cmd = """
    python {} {}
    """.format(
      train_script,
      ' '.join(['--{} {}'.format(k, v) for k, v in parameters.items() if v])
    )
    logger.info('cmd for master node:\n{}'.format(cmd))
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
  main()
