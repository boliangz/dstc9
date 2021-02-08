import os
from end2end.soloist.methods.baseline.SoloistAgent import Soloist

from convlab2.task.multiwoz.goal_generator import GoalGenerator


def get_goal():
  # get goal
  goal_generator = GoalGenerator()

  goal = goal_generator.get_user_goal()
  goal_message, _ = goal_generator.build_message(goal)
  return goal_message


def init_dialog():
  goal_message = get_goal()
  print('=' * 80)
  print('Your goal: (type "success" or "fail" to finish the dialog. )')
  for i, m in enumerate(goal_message):
    print('{}. {}'.format(i + 1, m))
  print('-' * 80)


def main():
  # init agent
  model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submission1/model')
  sys_agent = Soloist(model_dir, user_simulator=False)

  init_dialog()

  while True:
    user_utter = input('USER:  ')
    if user_utter in ['success', 'fail']:
      print('')
      sys_agent.init_session()
      init_dialog()
    else:
      sys_reponse = sys_agent.response(user_utter)[0]
      print('SYS:  ', sys_reponse)


if __name__ == "__main__":
  main()