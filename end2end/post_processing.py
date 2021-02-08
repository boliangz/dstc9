import re
import json
import random
import copy
import string
import traceback
from end2end.kb_search import get_db, belief_state_to_dict


'''
Post-process for the system output of response prediction.
@model_res: system output of response prediction (delexed).
@dataset: dataset used in the model
@output: final system response.
'''
def get_final_response(model_res, dataset= 'multiwoz'):
  # get db
  db = get_db(dataset)
  # get system delexed response
  delexed_response = re.search('<EOKB>(.*)<EOS>', model_res).group(1)
  try:
    # get turn domain
    turn_domain = re.search('<SOB>(.*)<DMN>', model_res).group(1).strip()
    # get belief constraints
    constraints = belief_state_to_dict(model_res)

    # fix reference number placeholder in the template
    delexed_response = process_reference_number(delexed_response)

    # replace placeholder with db entity info.
    response = replace_placeholders(
      delexed_response, turn_domain, db, constraints
    )

    # prepare delexed response for humane evaluation
    human_eval_delexed_response = prepare_human_eval_delexed_response(
      db, delexed_response, turn_domain, constraints
    )
    # replace placeholder for human eval delexed response
    human_eval_response = replace_placeholders(
      human_eval_delexed_response, turn_domain, db, constraints
    )

    # global replacement
    response = global_replacement(response)
    human_eval_response = global_replacement(human_eval_response)

    return response, human_eval_response

  except Exception as e:
    traceback.print_exc()
    print('model_res:', model_res)
    return delexed_response, delexed_response


def prepare_human_eval_delexed_response(db, delexed_response, turn_domain, constraints):
  # replace template that includes [value_reference] by a customized template
  # that shows all booking related information, e.g. departure/destination,
  # leave/arrive time, # of people, # of days, etc.
  delexed_response = process_booking_response(
    delexed_response, turn_domain, constraints
  )
  delexed_response = list_examples_recommendation(
    db, delexed_response, turn_domain, constraints
  )

  return delexed_response


def post_process_constraints(belief_state_str):
  #
  # fix type=hotel constraints
  #
  rtn = belief_state_str.replace(', type = hotel', '')
  rtn = rtn.replace('type = hotel, ', '')
  rtn = rtn.replace('hotel { type = hotel } ', '')
  if rtn != belief_state_str:
    print('--> fix_hotel_constraints triggered.')
  belief_state_str = rtn

  #
  # fix name constraints
  #
  def dict_to_belief_str(cons_dict):
    # dict to string
    constraints_str = []
    for domain, cons in cons_dict.items():
      constraints_str.append(
        '{} {{ {} }}'.format(
          domain,
          ', '.join(['{} = {}'.format(k, v) for k, v in cons.items()])
        )
      )
    constraints_str = ' '.join(constraints_str)
    return constraints_str

  db = get_db()
  constraints = belief_state_to_dict(belief_state_str)
  new_constraints = copy.deepcopy(constraints)
  for domain, cons in new_constraints.items():
    if 'name' in cons:
      match_result = db.queryJsons(domain, {'inform': cons})
      if not match_result:
        del cons['name']

  if constraints != new_constraints:
    print('--> fix name constraints triggered.')
    constraints_str = dict_to_belief_str(constraints)
    new_constraints_str = dict_to_belief_str(new_constraints)
    belief_state_str = belief_state_str.replace(
      constraints_str, new_constraints_str
    )
    # print(constraints_str)
    # print(new_constraints_str)

  return belief_state_str


# def fix_hotel_constraints(belief_state_str, dialog_history):
#   constraints = belief_state_to_dict(belief_state_str)
#   hotel_type_valid = False
#   for domain, cons in constraints.items():
#     if domain == 'hotel' and cons.get('type', '') == 'hotel':
#       # check the dialog history, for turns that are labeled hotel, if any
#       # user utterances contain keyword "type", we mark the type=hotel slot
#       # valid, otherwise, remove type=hotel constraint
#       for i in range(1, len(dialog_history), step=2):
#         if dialog_history[i][1] == 'hotel' and 'type' in \
#             dialog_history[i - 1][0]:
#           hotel_type_valid = True
#           print('-' * 80)
#           print(i)
#           print(dialog_history)
#           print(constraints)
#           print('')
#           break
#       if hotel_type_valid is False:
#         print(constraints)
#         del constraints[domain]['type']
#         print(constraints)
#         break
#
#   # dict to string
#   constraints_str = []
#   for domain, cons in constraints.items():
#     constraints_str.append(
#       '{} {{ {} }}'.format(
#         domain,
#         ', '.join(['{} = {}'.format(k, v) for k, v in cons.items()])
#       )
#     )
#   constraints_str = ' '.join(constraints_str)
#
#   new_belief_state_str = '{} <DMN> {} <EOB>'.format(
#     belief_state_str.split('<DMN>')[0].strip(),
#     constraints_str.strip()
#   )
#
#   if hotel_type_valid is False:
#     assert belief_state_str == new_belief_state_str
#   else:
#     print(belief_state_str)
#     print(new_belief_state_str)
#
#   return new_belief_state_str


def replace_placeholders(delexed_response, turn_domain, db, constraints):
  if turn_domain == 'general':
    return delexed_response
  elif turn_domain == 'taxi':
    response = process_taxi_domain(db, delexed_response, constraints)
  elif turn_domain == 'police':
    response = process_police_domain(db, delexed_response)
  else:
    response = process_other_domains(
      db, delexed_response, turn_domain, constraints
    )

  # if some placeholders are not replaced, replace the placeholders by belief
  # slots
  response = process_remaining_placeholders(response, constraints, turn_domain)

  return response


def process_taxi_domain(db, delexed_response, constraints):
  final_response = delexed_response

  # Process for taxi domain specifically.
  # assume each turn involves only one domain.
  taxi_attributes = db.dbs['taxi'][0]
  # place_holders for taxi: [value_car], [value_phone]
  all_place_holders = re.findall(r'\[.*?\]', delexed_response)
  while (len(all_place_holders) > 0):
    place_holders = set(all_place_holders)
    for h in place_holders:
      all_place_holders.remove(h)

    for slot in place_holders:
      slot_name = re.search('_(.*)]', slot).group(1)
      if slot_name == 'phone':
        # phone pattern: "^[0-9]{10}$"
        slot_value_str = "".join(
          [random.choice(string.digits) for i in range(11)])
        final_response = final_response.replace(slot, slot_value_str, 1)
      elif slot_name == 'car':
        car_color = random.choice(taxi_attributes['taxi_colors'])
        car_type = random.choice(taxi_attributes['taxi_types'])
        slot_value_str = car_color + ' ' + car_type
        final_response = final_response.replace(slot, slot_value_str, 1)
      elif slot_name == 'name':
        if 'departure' in constraints['taxi'] and 'destination' not in constraints['taxi']:
          slot_value_str = constraints['taxi']['departure']
          final_response = final_response.replace(slot, slot_value_str, 1)
        elif 'departure' not in constraints['taxi'] and 'destination' in constraints['taxi']:
          slot_value_str = constraints['taxi']['destination']
          final_response = final_response.replace(slot, slot_value_str, 1)
        elif 'departure' in constraints['taxi'] and 'destination' in constraints['taxi']:
          resp_list = final_response.split()
          if slot in resp_list and resp_list.index(slot) - 1 >= 0:
            prepos = resp_list[resp_list.index(slot) - 1]
            if prepos == 'from':
              resp_list[resp_list.index(slot)] = constraints['taxi']['departure']
              final_response = ' '.join(resp_list)
            elif prepos == 'to':
              resp_list[resp_list.index(slot)] = constraints['taxi']['destination']
              final_response = ' '.join(resp_list)

  return final_response


def process_police_domain(db, delexed_response):
  final_response = delexed_response
  # Process for police domain specifically.
  # assume each turn involves only one domain.
  police_entity = db.dbs['police'][0]
  place_holders = re.findall(r'\[.*?\]', delexed_response)
  for slot in place_holders:
    slot_name = re.search('_(.*)]', slot).group(1)
    slot_value_str = cast_to_str(police_entity[slot_name])
    final_response = final_response.replace(slot, slot_value_str)
  return final_response


def process_other_domains(db, delexed_response, turn_domain, constraints):
  final_response = delexed_response

  # # let turn_domain be the first domain to be searched.
  # all_domains = copy.deepcopy(db.ontology.all_domains)
  # all_domains.remove(turn_domain)
  # all_domains.insert(0, turn_domain)

  # for domain in all_domains:
  if turn_domain in db.ontology.db_domains and \
      turn_domain in constraints.keys():
    # to be compatible with the new format
    domain_constraints = {'inform': constraints[turn_domain]}
    query_results = db.queryJsons(turn_domain, domain_constraints)
    all_place_holders = re.findall(r'\[.*?\]', delexed_response)
    # skip non-db entity slots
    tmp_placeholders = []
    for slot in all_place_holders:
      slot_name = re.search('_(.*)]', slot).group(1)
      if slot_name in ['people', 'stay'] or \
          (turn_domain == 'hotel' and slot_name == 'day') or \
          (turn_domain == 'restaurant' and slot_name in ['day', 'time']):
        continue
      tmp_placeholders.append(slot)
    all_place_holders = tmp_placeholders

    all_place_holders_copy = copy.deepcopy(all_place_holders)
    # a dictionary of sets of slot values that have been replaced.
    # a unique slot/place_holder cannot be replaced with duplicated values.
    replaced_slot_value_recorder = dict()

    query_results = sort_by_time_for_train(query_results, turn_domain, constraints)

    for item in query_results:
      # get plance holders for one entity
      place_holders = set(all_place_holders)

      match = True
      ### to filter out ineligible entities.
      # 1) the entity is able to replace all the placeholders.
      for slot in all_place_holders_copy:
        slot_name = re.search('_(.*)]', slot).group(1)
        if not slot_name in item:
          match = False
          break
      # 2) the slot-value to be replaced of the entity has not been replaced yet.
      if match == True:
        for slot in place_holders:
          slot_name = re.search('_(.*)]', slot).group(1)
          slot_value_str = cast_to_str(item[slot_name])
          if slot_name == 'stars':
            slot_value_str = slot_value_str + ' star'
          if slot in replaced_slot_value_recorder and slot_value_str in replaced_slot_value_recorder[slot]:
            match = False
            break

      # right now, we can guaranttee that the entity can be used to replace.
      if match == True:
        for slot in place_holders:
          slot_name = re.search('_(.*)]', slot).group(1)
          slot_value_str = cast_to_str(item[slot_name])
          if slot_name == 'stars':
            slot_value_str = slot_value_str + ' star'
          if slot_name == 'price' and turn_domain == 'train' and \
              'people' in domain_constraints['inform'] and \
              domain_constraints['inform']['people'] != 'dontcare' and \
              int(domain_constraints['inform']['people']) > 1:
            slot_value_str = '{} pounds'.format(str(float(slot_value_str.split()[0]) * int(domain_constraints['inform']['people'])))
          # final_response = final_response.replace(slot, slot_value_str) # replace all for the slot
          final_response = final_response.replace(slot, slot_value_str, 1) # replace only one slot with one entity
          if slot not in replaced_slot_value_recorder:
            replaced_slot_value_recorder[slot] = set()
          replaced_slot_value_recorder[slot].add(slot_value_str)
        for h in place_holders:
          all_place_holders.remove(h)

        ######
        # replace util all the place_holders have been replaced.
        if len(all_place_holders)==0:
          break

  return final_response


def process_booking_response(delexed_response, turn_domain, constraints):
  if '[value_reference]' in delexed_response:
    template = 'booking was successful . {}, reference number: [' \
               'value_reference] . is there anything else i can help with ? '
    if turn_domain == 'restaurant':
      info = ['restaurant: [value_name]']
      if 'day' in constraints[turn_domain]:
        info.append('day: [value_day]')
      if 'time' in constraints[turn_domain]:
        info.append('time: [value_time]')
      if 'people' in constraints[turn_domain]:
        info.append('people: [value_people]')
      return template.format(', '.join(info))
    if turn_domain == 'hotel':
      info = ['hotel: [value_name]']
      if 'day' in constraints[turn_domain]:
        info.append('day: [value_day]')
      if 'stay' in constraints[turn_domain]:
        info.append('stay: [value_stay] day(s)')
      if 'people' in constraints[turn_domain]:
        info.append('people: [value_people]')
      return template.format(', '.join(info))
    if turn_domain == 'train':
      info = ['train id: [value_id]']
      info.append('departure: [value_departure]')
      info.append('destination: [value_destination]')
      info.append('leave time: [value_leave]')
      info.append('arrive time: [value_arrive]')
      if 'people' in constraints[turn_domain]:
        info.append('people: [value_people]')
      else:
        info.append('people: 1')
      info.append('price: [value_price]')
      return template.format(', '.join(info))
    else:
      return template
  elif turn_domain == 'taxi' and \
      ('[value_car]' in delexed_response or
       '[value_phone]' in delexed_response):
    info = ['car type: [value_car]', 'phone number: [value_phone]']
    if 'departure' in constraints[turn_domain]:
      info.append('departure: [value_departure]')
    if 'destination' in constraints[turn_domain]:
      info.append('destination: [value_destination]')
    if 'leave' in constraints[turn_domain]:
      info.append('pick up: [value_leave]')
    if 'arrive' in constraints[turn_domain]:
      info.append('arrive: [value_arrive]')

    return 'i have booked a taxi for you. {}. is there anything else i can ' \
           'help with ?'.format(', '.join(info))
  else:
    return delexed_response



def list_examples_recommendation(db, delexed_response, turn_domain, constraints):
  keywords = ['what', 'where', 'which', 'is there', 'do you have', 'did you have', 'particular', 'specific', 'preference']
  domain_slots = {"taxi": [],
                  "police": [],
                  "hospital": ["department"],
                  "hotel": ["pricerange", "area", "stars"],
                  "attraction": ["area", "type"],
                  "train": ["day"],
                  "restaurant": ["food", "pricerange", "area"]
                  }
  if turn_domain in db.ontology.db_domains and \
      turn_domain in constraints.keys():
    domain_constraints = {'inform': constraints[turn_domain]}
    query_results = db.queryJsons(turn_domain, domain_constraints)
    if len(query_results) >= 3 \
       and '?' in delexed_response \
       and any(w in delexed_response for w in keywords) \
       and turn_domain in domain_slots:
       choices = {}
       for slot in domain_slots[turn_domain]:
           if slot in delexed_response:
               choices[slot] = set()
               for e in query_results:
                   if slot in e.keys():
                       choices[slot].add(e[slot])
       recommendation_info = ''
       for slot, choice_values in choices.items():
           if len(choice_values) > 1:
               recommendation_info += f", {slot} choices: " + ', '.join(list(choice_values)[:3])
       if recommendation_info != '':
           delexed_response = delexed_response + 'for example' + recommendation_info
  return delexed_response


def sort_by_time_for_train(query_results, turn_domain, constraints):
  if turn_domain != 'train':
      return query_results
  else:
      if 'arrive' in constraints[turn_domain] and 'leave' not in constraints[turn_domain]:
          query_results.sort(key=lambda x: int(x['arrive'].split(':')[0])*60+int(x['arrive'].split(':')[1]), reverse=True)
      else:
          query_results.sort(key=lambda x: int(x['leave'].split(':')[0])*60+int(x['leave'].split(':')[1]), reverse=False)
      return query_results


def process_remaining_placeholders(delexed_response, constraints, turn_domain):
  final_response = delexed_response
  # if some placeholders are not replaced, replace the placeholders by belief
  # slots
  remaining_place_holders = re.findall(r'\[.*?\]', final_response)
  if len(remaining_place_holders) > 0:
    domain_constraints = constraints[turn_domain]
    for p in remaining_place_holders:
      slot_name = re.search('_(.*)]', p).group(1)
      if slot_name in domain_constraints:
        slot_value_str = domain_constraints[slot_name]
        if slot_name == 'stars':
          slot_value_str += ' star'
      else:
        if turn_domain == 'attraction' and slot_name == 'price':
          slot_value_str = 'not listed'
        else:
          slot_value_str = p

      final_response = final_response.replace(p, slot_value_str)

  # replace [value_type] by domain name for remaining placeholders
  remaining_place_holders = re.findall(r'\[.*?\]', final_response)
  if len(remaining_place_holders) > 0:
    for p in remaining_place_holders:
      slot_name = re.search('_(.*)]', p).group(1)
      if slot_name == 'type':
        slot_value_str = turn_domain
        final_response = final_response.replace(p, slot_value_str)

  return final_response


def process_reference_number(delexed_response):
  ###
  # Postprocess for reference number to be 8-digits if not delexed.
  ###
  for ref_str in ["reference number is :", "reference number is"]:
    if ref_str in delexed_response \
        and "[value_reference]" not in delexed_response:
      generated_ref = re.search('{} (.*)\s'.format(ref_str),
                                delexed_response).group(1)
      delexed_response = delexed_response.replace(
        generated_ref, '[value_reference]'
      )
      return delexed_response
  return delexed_response


def cast_to_str(uncertain_type):
  ret_str = ''
  if isinstance(uncertain_type, str):
    ret_str = uncertain_type
  else:
    if isinstance(uncertain_type, list):
      ret_str = ', '.join(uncertain_type)
    else:
      if isinstance(uncertain_type, dict):
        ret_str = json.dumps(uncertain_type)
  return ret_str


def global_replacement(response):
  if response:
    response = response.replace('entrance fee is ?', 'entrance fee is not listed')

    # fix_suffix
    # not check whether a valid plural word yet
    response = response.replace(' -s ', 's ')
    response = response.replace(' -ly ', 'ly ')
    response = response.replace(' -er ', 'er ')

    response = response.strip()

  return response


if __name__ == '__main__':
  res = get_final_response("User: arrive by 08:30 <SOB> train <DMN> hotel { type = hotel, area = east, pricerange = expensive, name = huntingdon marriott hotel, people = 5, day = thursday, stay = 4 } train { day = friday, departure = cambridge, destination = leicester, arrive = 08:30 } <EOB> DB: hotel 1 match, train 3 match <EOKB> [value_id] arrives at [value_arrive] . would you like me to book it for you ? <EOS>")
  print(res)


  # res = get_final_response("User: i need a hotel close to downtown cambridge please . i need 1 with wifi . are there any 4 stars available ? it must have free parking . System: i have 2 hotels that meet your criteria . 1 is expensive and the other is cheap . do you have a preference ? User: please pick 1 for me . i need a hotel close to downtown cambridge please . System: university arms hotel is a expensive hotel in the centre . would you like me to book it for you ? User: i am not looking to make a booking at the moment. System: okay , how many days will you be staying ? User: i am not looking to make a booking at the moment . i do n't care about the stay of the hotel. System: how many days will you be staying ? User: i am not looking to make a booking at the moment . i do n't care about the stay of the hotel. System: okay , how many people will be staying ? User: i am not looking to make a booking at the moment . i do n't care about the people of the hotel. System: okay , i have booked you at the university arms hotel . your reference number is : 00000030 User: i am not looking to make a booking at the moment. <SOB> general <DMN> hotel { area = centre, parking = yes, stars = 4, internet = yes, name = university arms hotel } <EOB> DB: hotel 1 match <EOKB> is there anything else i can help you with today ? <EOS>")
  # print(res)

  # res = get_final_response("User: I want to find a tourist attraction in the centre. <SOB> attraction <DMN> attraction { area = centre } <EOB> DB: attraction 44 match, attraction 1 match ( entrance fee = free ) <EOKB> there are 44 attractions in the [value_area] . is there a specific type of attraction you are interested in ? <EOS>")
  # print(res)


  #bug
  # res = get_final_response("User: hi System:  okay , is there anything else i can help you with ?  User: i'm looking for some place to go to for the weekend System:  what type of attraction are you looking for ?  User: natural meseums, like where they teach science System:  there are 2 swimming pools in the north . 1 is jesus green outdoor pool and the other is kings hedges learner pool .  User: no i want natural museum System:  i am sorry , there are no museums in the north . would you like to try another area ?  User: sure System:  i am sorry , there are no museums in the north .  User: museum in other places? System:  there are 2 museums in the north . unknown and unknown .  User: museum in east? System:  there are 4 museums in the east . cambridge artworks is free .  User: what are the other ones System:  cambridge artworks is free .  User: where is cambridge artworks System:  it is located at 5 greens road .  User: their number is? System:  their phone number is 01223902168 .  User: ok i'll go to cambridge artworks System:  okay , what information do you need about the cambridge artworks ?  User: what's their postcode System:  their postcode is cb13ef.  User: now i need a place for the night System:  what area would you like to stay in ?  User: do n't care, as long as price is good <SOB> hotel <DMN> attraction { type = museum, area = east, name = cambridge artworks } <EOB> DB: attraction 1 match ( entrance fee = free ) <EOKB> i would recommend [value_name] . <EOS>")
  # print(res)


  # res = get_final_response("<SOB> restaurant <DMN> attraction { type = college, name = christ college } restaurant { food = chinese, area = centre, pricerange = expensive, name = ugly duckling } <EOB> restaurant 1 match, attraction 1 match ( entrance fee = free ) <EOKB> the phone number is [value_phone] and the postcode is [value_postcode]. <EOS>")
  # print(res)
  # exit()
  #
  #
  # res = get_final_response("User: ok, could you tell the address and the entrance fee <SOB> attraction <DMN> attraction { area = centre, type = theatre, name = adc theatre } train { destination = kings lynn, departure = cambridge, day = wednesday, arrive = 08:15, people = 1 } <EOB> DB: attraction 1 match ( entrance fee = ? ), train 3 match <EOKB> the address is [value_address] and the entrance fee is [value_price] . <EOS>")
  # print(res)
  exit()


  # res = get_final_response("User: On saturday please. There will be 5 people . We 'd like to stay for 5 nights . <SOB> hotel <DMN> hotel { parking = dontcare, pricerange = moderate, type = guest house, area = dontcare, name = a and b guest house, people = 5, day = saturday, stay = 5 } attraction { type = college, area = centre, name = christ college } <EOB> hotel 1 match, attraction 13 match <EOKB> booking was successful . reference number is : zq5v8g2 . <EOS>")
  # print(res)

  # random select entity for taxi domain.
  # res = get_final_response("User: I also want to book a taxi to commute between the two places . I need a taxi to pick me up from leverton house please. The taxi should go to maharajah tandoori restaurant . I need to arrive by 15:30 . System: i have booked you a [value_car] . the contact number is [value_phone] . <SOB> taxi <DMN> taxi { destination = maharajah tandoori restaurant, departure = leverton house, arrive = 15:30 } <EOB> <EOKB> i have booked you a [value_car] . the contact number is [value_phone] . <EOS>")
  # print(res)


  # duplicated place_holders in response
  # solution: instead of replace all place_holders in one time with one entity, every time replace all the unique place_holders only with one entity. Entities that are eligible to be used to replace the place_holders should satisfy that all the place_holders are included in the slots of the entities. The old-solution assumes that all the place_holders in the response template belong to one entity. The new-solution assumes that one entity has unique slots.
  res = get_final_response("<SOB> hotel <DMN> hotel { area = north, parking = yes, internet = no } attraction { name = the man on the moon } <EOB> hotel 0 match, attraction 1 match ( entrance fee = ? ) <EOKB> there are no [value_type] -s in the [value_area] . would you like to try a different area ? <EOS>")
  print(res)



  res = get_final_response("User: i do n't need anything booked . i just need to get the arrival time , travel time and price of a train from norwich to cambridge leaving after 21:15 . <SOB> train <DMN> train { leave = 21:15, departure = norwich, destination = cambridge, day = monday } <EOB> train 16 match <EOKB> there are trains arriving at [value_arrive] , [value_arrive] and [value_arrive] . they are [value_price] and [value_time] each . <EOS>")
  print(res)


  res = get_final_response("User: i would prefer the hotel be in the north part of town . <SOB> hotel <DMN> hotel { area = north, pricerange = cheap, parking = yes } <EOB> hotel 2 match, attraction 13 match <EOKB> [value_name] and [value_name] are both [value_type] and located in the [value_area] . would you like me to book 1 for you ? <EOS>")
  print(res)
  # city centre north b and b and worth house are both guesthouses and located in the north . would you like me to book one for you ?
  # "city centre north b and b" was cleaned as "city centre north bed and breakfast"



  res = get_final_response("User: please find me an expensive place to dine on the south side of town <SOB> restaurant <DMN> restaurant { area = south , pricerange = expensive } <EOB> restaurant 5 match <EOKB> sure , we have [value_food] , [value_food] , [value_food] or [value_food] that you could choose from . <EOS>")
  print(res)


  res = get_final_response("User: yes . i need a place to stay in the same part of town . it must have free parking . <SOB> hotel <DMN> hotel { area = centre , parking = yes } <EOB> hotel 4 match <EOKB> we have 2 [value_type] and 2 [value_type] -s that meet your needs . the [value_type] are [value_pricerange] , and the [value_type] -s are [value_pricerange] . do you have a preference ? <EOS>")
  print(res)
  # not fix yet.
  # current: we have 2 guest house and 2 hotels that meet your needs . the hotel are cheap , and the [value_type]s are expensive . do you have a preference ?


  res = get_final_response("User: i do n't have 1 in particular . can you give me a few examples ? <SOB> attraction <DMN> attraction { type = college , area = centre } <EOB> DB: attraction 13 match <EOKB> sure i have [value_name] in the city [value_area] and [value_name] , which is very famous in the [value_area] . <EOS>")
  print(res)
  # sure i have corpus christi in the city centre and christ 's college , which is very famous in the centre .
  # not fix yet.
  # current: sure i have christ 's college in the city centre and [value_name] , which is very famous in the [value_area] . ### all the 13 attractions are in the centre.


  res = get_final_response("User: now i need a taxi . <SOB> taxi <DMN> taxi { destination = dojo noodle bar , arrive = 19:15 , departure = kirkwood house } hotel { name = kirkwood house } restaurant {area = centre , food = asian oriental , pricerange = dontcare } <EOB> DB: restaurant 4 match, hotel 33 match <EOKB> i have booked a taxi from [value_name] to the [value_name] for you . a [value_car] will pick you up . the contact number is [value_phone] . is there anything else i can help you with ? <EOS>")
  print(res)


  res = get_final_response("User: not in terms of that , but do they have free parking and have a 3 star rating ? <SOB> hotel <DMN> hotel { area = centre , type = hotel } <EOB> DB: hotel 3 match <EOKB> the [value_name] has [value_stars] and parking , and the [value_name] has [value_stars] and parking . they are both [value_pricerange] . would you like more details ? <EOS>")
  print(res)
  # the gonville hotel has 3 stars and parking , and the university arms hotel has 4 stars and parking . they are both expensive . would you like more details ?





  #
  #
  # # police domain
  # res = get_final_response("User: Help I was just robbed ! Could you please help me contact the police ? Help , I need to find the nearest police station . System: the nearest police station is [value_address] . <SOB> police <DMN>  <EOB> <EOKB>the nearest police station is [value_address] . <EOS>")
  # print(res)
  #
  #
  # # merge -s
  # res = get_final_response("User: I need a restaurant to dine at in Cambridge on my upcoming trip . How about a british restaurant ? I also would like information on a place to eat in the centre . System: there are several british restaurant -s in the centre . do you have a price range ? <SOB> restaurant <DMN> restaurant { food = british, area = centre } <EOB> DB: restaurant 7 match <EOKB> there are several [value_food] restaurant -s in the [value_area] . do you have a price range ? <EOS>")
  # print(res)
  #
  #
  # # [value_stars]
  # res = get_final_response("User: What type of hotels are they ? System: university arms hotel is a 4 hotel in the centre . <SOB> hotel <DMN> hotel { area = centre, internet = yes, parking = yes, stars = 4, type = hotel } <EOB> DB: hotel 1 match <EOKB> [value_name] is a [value_stars] [value_type] in the [value_area] . <EOS>")
  # print(res)
  #
  #
  # # replace to a wrong domain
  # res = get_final_response("User: Can you help me plan a trip to see a particular attraction ? I also need a place to go in the west . System: i have 13 hotel -s in the centre . do you have a preference for type of attraction ? <SOB> attraction <DMN> hotel { area = centre, internet = yes, parking = yes, stars = 4, type = hotel } attraction { area = west } <EOB> DB: hotel 1 match, attraction 13 match <EOKB> i have 13 [value_type] -s in the [value_area] . do you have a preference for type of attraction ? <EOS>")
  # print(res)
  # final_response: i have 13 museums in the west . do you have a preference for type of attraction ?



  ### get_db_state ###

  # a = get_db_state("db_state_str User: i 'd like a sports place in the centre please . a boat type of attraction . System: there are 2 boat attractions in the centre . 1 is scudamores punting co and the other is scudamores punting co . User: oh , and what is their postcode , please ? what is the address ? System: the address is granta place , cb21rs . User: i ' m departing from peterborough . i need a train going to leicester . this will be for tuesday . i want to get there by 18:30 at the latest . <SOB> attraction { type = boat, area = centre, name = scudamore } train { destination = leicester, day = tuesday, arrive = 18:30, departure = peterborough } <EOB>")
  # print(a)
  # exit()

  # res = get_db_state('i would like to book a 5 star , or closest to it , in the east part of town please . <SOB> hotel { area = east, stars = 4, type = guest house } <EOB>', 'multiwoz')
  # print(res)

  # res = get_final_response('i would like to book a 5 star , or closest to it , in the east part of town please . <SOB> hotel { area = east, stars = 5 } restaurant { area = east } <EOB> DB: hotel 0 match <EOKB> i have booked you a table at the [value_name] at [value_address] . would you like to book a table for me ? <EOS>', 'multiwoz')
  # print(res)

  # res = get_final_response("User: I 'd like something in the north . System:  i have 13 guest house and 2 hotel -s in the north . do you have a price range preference ?  User: Can I have the address of a good one ? Can you recommend one and give me the entrance fee ? System:  i recommend the acorn guest house . it is a 4 guest house in the north . it is moderate -ly priced and has free parking and internet .  User: Is there an exact address , like a street number ? Thanks !. Can you just recommend one and tell me the entrance fee ? <SOB> hotel { area = north, pricerange = dontcare, name = acorn guest house } attraction { area = north } <EOB> DB: hotel 1 match attraction 4 match <EOKB> i recommend the riverboat georgina . it is located at cambridge passenger cruisers , jubilee house . the entrance fee is [value_price] . <EOS>")
  # print(res)


  # res = get_final_response("User: I 'd like something in the north . System:  i have 13 guest house and 2 hotel -s in the north . do you have a price range preference ?  User: Can I have the address of a good one ? Can you recommend one and give me the entrance fee ? System:  i recommend the acorn guest house . it is a 4 guest house in the north . it is moderate -ly priced and has free parking and internet .  User: Is there an exact address , like a street number ? Thanks !. Can you just recommend one and tell me the entrance fee ? <SOB> hotel { area = north, pricerange = dontcare, name = acorn guest house } <EOB> DB: hotel 1 match attraction 4 match <EOKB> i have booked you a table for 4 at [value_address] . your reference number is : [value_reference] . <EOS>")
  # print(res)
