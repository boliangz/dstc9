
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.nlu.milu.multiwoz import MILU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import random
import numpy as np
import torch
from convlab2.e2e.sequicity.multiwoz import Sequicity

# #  BERT nlu
# sys_nlu = BERTNLU()
# # sys_nlu = MILU()
# # simple rule DST
# sys_dst = RuleDST()
# # rule policy
# sys_policy = RulePolicy()
# # template NLG
# sys_nlg = TemplateNLG(is_user=False)
# # assemble
# sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')


sys_agent = Sequicity()

# print(sys_agent.response("I want to find a moderate hotel"))
# print(sys_agent.response("Which type of hotel is it ?"))
# print(sys_agent.response("OK , where is its address ?"))
# print(sys_agent.response("Thank you !"))
#
# a=input()

''' MILU
I have 3 options for you. How about cityroomz ? Fits your request perfectly .
It is a hotel .
It is located at 74 chesterton road.
You are welcome.
---- Sequicity
there are 4 guesthouses in the north . what type of cuisine are you looking for ?
jinling noodle bar is a nice restaurant and the postcode is cb23pp .
the address is , postcode cb21uf .
you are welcome , goodbye .
'''

# MILU
user_nlu = MILU()
# not use dst
user_dst = None
# rule policy
user_policy = RulePolicy(character='usr')
# template NLG
user_nlg = TemplateNLG(is_user=True)
# assemble
user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')



evaluator = MultiWozEvaluator()
sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)

set_seed(20200131)

sys_response = ''
sess.init_session()
print('init goal:')
pprint(sess.evaluator.goal)
print('-'*50)
for i in range(20):
    sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
    print('user:', user_response)
    print('sys:', sys_response)
    print()
    if session_over is True:
        break
print('task success:', sess.evaluator.task_success())
print('book rate:', sess.evaluator.book_rate())
print('inform precision/recall/f1:', sess.evaluator.inform_F1())
print('-'*50)
print('final goal:')
pprint(sess.evaluator.goal)
print('='*100)

'''
pipeline
task success: 1
book rate: 1
inform precision/recall/f1: (1, 1.0, 1)


Sequicity
task success: 0
book rate: None
inform precision/recall/f1: (0, 0.0, 0)
'''
