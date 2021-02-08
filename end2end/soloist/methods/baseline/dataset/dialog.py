import itertools
import random
import re
from end2end.kb_search import get_db

DB = get_db(dataset='multiwoz')

class Dialog(object):
  def __init__(self, dialog_id, json_object, max_resp_len):
    self.id = dialog_id
    self.turns = []
    self.ml_instances = []

    self.from_json(json_object, max_resp_len)

    self.make_machine_learning_instances()

  def from_json(self, json_object, max_resp_len):
    log = json_object['log']
    for turn in log:
      self.turns.append(Turn(turn, max_resp_len))

  def make_machine_learning_instances(self):
    # convert dialog into soloist format
    for i in range(len(self.turns)):
      # ignore booking-nobook turn
      if 'booking-nobook' in self.turns[i].dialog_act:
        continue
      # ignore number of domains in one turn > 1
      if len(self.turns[i].turn_domain.split()) > 1:
        continue

      # ignore turns that have duplicated placeholders
      # placeholders = [
      #   item.replace('[value_', '').replace(']', '')
      #   for item in re.findall(r'\[.*?\]', self.turns[i].resp_delex)
      #   if 'reference' not in item
      # ]
      # if len(set(placeholders)) != len(placeholders):
      #   continue

      self.ml_instances.append(MLInstance(self.turns[:i+1], self.id))


class Turn(object):
  def __init__(self, json_object, max_resp_len):
    self.user = None
    self.user_delex = None
    self.resp = None
    self.resp_delex = None
    self.match = None
    self.constraint = None
    self.turn_id = None
    self.turn_domain = None
    self.dialog_act = None

    self.from_json(json_object, max_resp_len)

  def from_json(self, json_object, max_resp_len):
    # truncate response based on max_resp_len
    def truncate(string):
      return ' '.join(string.split()[:max_resp_len])

    self.user = truncate(json_object['user'])
    self.user_delex = truncate(json_object['user_delex'])
    self.resp = truncate(json_object['resp'])
    self.resp_delex = truncate(json_object['resp_delex'])
    self.match = json_object['match']
    self.constraint = json_object['constraint']
    self.turn_id = json_object['turn_num']
    self.turn_domain = json_object['turn_domain'].replace('[', '').replace(']', '')
    self.dialog_act = json_object['dialog_act']


class MLInstance(object):
  def __init__(self, turns, dialog_id):
    self.current_turn = turns[-1]
    self.history = turns[:-1]
    self.dialog_id = dialog_id
    self.turn_num = len(turns) - 1

  def get_dialog_history(self, max_turn):
    dialog_history = list(itertools.chain.from_iterable(
      [['User: {}'.format(t.user), 'System: {}'.format(t.resp)]
       for t in self.history[-max_turn:]]  # 历史轮user/agent消息
      + [['User: {}'.format(self.current_turn.user)]]  # 当前轮用户消息
    ))
    dialog_history = ' '.join(dialog_history)
    return dialog_history

  def get_belief_state(self):
    belief_state = []
    for k, v in self.current_turn.constraint.items():
      b_s = []
      for _k, _v in v['inform'].items():
        b_s.append('{} = {}'.format(_k, _v))
      for _k, _v in v.items():
        if _k not in ['request', 'inform']:
          b_s.append('{} = {}'.format(_k, _v))

      # ignore slots that have "none" slot value
      b_s = [s for s in b_s if '= none' not in s]

      if b_s:  # ignore empty state
        belief_state.append('{} {{ {} }}'.format(k, ', '.join(b_s)))

    belief_state = 'Ġ<SOB>Ġ {} Ġ<DMN>Ġ {} Ġ<EOB>Ġ'.format(
      self.current_turn.turn_domain,
      ' '.join(belief_state)
    )

    return belief_state

  def get_db_match(self):
    db_match = self.current_turn.match
    match_str = []
    for k, v in db_match.items():
      if k in self.current_turn.constraint and self.current_turn.constraint[k]['inform'] == {}:
        continue
      # if only 1 attraction match, include entrance fee info
      if k == 'attraction' and v == 1:
        attraction_constraint = self.current_turn.constraint['attraction']
        matched_attraction = DB.queryJsons(domain='attraction', constraints=attraction_constraint)
        assert len(matched_attraction) == 1, (matched_attraction, attraction_constraint, '\n', self.current_turn.user)
        entrance_fee = matched_attraction[0]['price']
        match_str.append(
          '{} {} match ( entrance fee = {} )'.format(k, v, entrance_fee)
        )
      else:
        match_str.append('{} {} match'.format(k, v))
    if not match_str:
      match_str = 'EMPTY'  # if match is emtpy
    else:
      match_str = ', '.join(match_str)
    db_string = 'DB: {} Ġ<EOKB>Ġ'.format(match_str)

    return db_string

  def get_response(self):
    response = self.current_turn.resp_delex
    response = '{} Ġ<EOS>Ġ'.format(response)
    return response

  def get_ml_string(self, max_turn, all_dialogs):
    # prepare positive ml string
    dialog_history = self.get_dialog_history(max_turn)
    belief_state = self.get_belief_state()
    db_match = self.get_db_match()
    response = self.get_response()
    positive_ml_string = ' '.join(
      [dialog_history, belief_state, db_match, response]
    )

    # sample a different belief state from the whole dataset
    negative_belief_state = belief_state
    while negative_belief_state == belief_state:
      tmp = random.choice(all_dialogs).ml_instances
      while not tmp:
          tmp = random.choice(all_dialogs).ml_instances
      negative_belief_state = random.choice(tmp).get_belief_state()
      #negative_belief_state = random.choice(
      #  random.choice(all_dialogs).ml_instances
      #).get_belief_state()

    # sample a different response from the whole dataset
    negative_response = response
    while negative_response == response:
      tmp = random.choice(all_dialogs).ml_instances
      while not tmp:
          tmp = random.choice(all_dialogs).ml_instances
      negative_response = random.choice(tmp).get_response()
      #negative_response = random.choice(
      #  random.choice(all_dialogs).ml_instances
      #).get_response()

    # according to the paper, consider three types of negative samples with
    # probability 1/3: (i) negative belief, where only the belief item is
    # replaced (ii) negative response, where only the response item is
    # replaced (iii) negative belief + response, where both the belief and
    # response items are replaced.
    negative_candidats = [
      ' '.join([dialog_history, negative_belief_state, db_match, response]),
      ' '.join([dialog_history, belief_state, db_match, negative_response]),
      ' '.join([dialog_history, negative_belief_state, db_match, negative_response])
    ]
    negative_sampling_choice = random.choice(range(len(negative_candidats)))
    negative_ml_string = negative_candidats[negative_sampling_choice]

    return positive_ml_string, negative_ml_string, negative_sampling_choice
