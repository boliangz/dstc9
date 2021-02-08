from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent
from convlab2.util.analysis_tool.analyzer import Analyzer

from end2end.soloist.methods.baseline.SoloistAgent import Soloist

import random
import numpy as np
import torch
import os


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end():
    # soloist model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    sys_agent = Soloist(model_file=model_dir)

    # BERT nlu trained on sys utterance
    user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                       model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
    # The max input length of BERTNLU is 512.
    sys_agent.max_resp_token = int(512 / 2)

    # not use dst
    user_dst = None
    # rule policy
    user_policy = RulePolicy(character='usr')
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    # assemble
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='SOLOIST', total_dialog=10)


if __name__ == '__main__':
    test_end2end()
