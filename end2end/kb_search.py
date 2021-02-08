import re
import collections

from end2end.db_ops import MultiWozDB
from end2end.utils.common_utils import *

'''
Get db information
'''
def get_db(dataset= 'multiwoz'):
    if dataset == 'multiwoz':
        from end2end.data.multiwoz.ontology import global_ontology as ontology
        data_config_name = os.path.join(
            os.path.dirname(__file__), f"configs/data_configs/{dataset}.json"
        )
        data_config = load_config(data_config_name)
        db = MultiWozDB(data_config.processed_dbs, ontology)
    else:
        if dataset == 'camrest':
            from end2end.data.camrest.CamRest676.ontology import global_ontology as ontology
            data_config_name = os.path.join(
                os.path.dirname(__file__),
                f"configs/data_configs/{dataset}.json"
            )
            data_config = load_config(data_config_name)
            db = MultiWozDB(data_config.processed_dbs, ontology)
    return db


# '''
# example 1: <EOB> DB: attraction 44 match, attraction 1 match ( entrance fee = free ) <EOKB>
# '''
# def db_state_to_dict(model_res):
#   try:
#       db_state_str = re.search('<EOB> DB:(.*)<EOKB>', bs_str).group(1).strip()
#       db_state_dict = {}
#       for dom_db_state in db_state_str.split(','):
#         dom_db_state_list = dom_db_state.split()
#         if len(dom_db_state_list) >= 2:
#             domain = dom_db_state_list[0].strip()
#             match_num = dom_db_state_list[1].strip()
#             db_state_dict[domain] = int(match_num)
#       return db_state_dict
#   except Exception as e:
#       print(e)
#       print(model_res)
#       return {}


'''
Convert belief state string to dict format.
belief state string:
example 1: i would like to book a 5 star , or closest to it , in the east part of town please . <SOB> hotel { area = east, stars = 5, type = hotel } <EOB>

example 2: i would like to book a 5 star , or closest to it , in the east part of town please . <SOB> hotel { area = east, stars = 5 } restaurant { area = east } <EOB> DB: hotel 0 match <EOKB> i have booked you a table at the [value_name] at [value_address] . would you like to book a table for me ? <EOS>

output belief state dict:
example 1: {hotel: {area: east, stars: 5, type: hotel}}
example 2: {{hotel: {area: east, stars: 5}, {restaurant: {area: east}}
'''
def belief_state_to_dict(bs_str):
    try:
        turn_domain = re.search('<SOB>(.*)<DMN>', bs_str).group(1).strip()
        bs = re.search('<DMN>(.*)<EOB>', bs_str).group(1)
        bs_dict = collections.OrderedDict()
        for dom_const in re.findall(r'[a-z]+ \{.*?\}', bs):
            domain = re.search('(.*)\{', dom_const).group(1).strip()
            constraints_str = re.search('\{(.*)\}', dom_const).group(1)
            constraints = {}
            for sv in constraints_str.split(', '):
                if sv.strip() and '=' in sv:
                    s, v = sv.split('=')
                    constraints[s.strip()] = v.strip()
                else:
                    print("bs format error: ", bs_str)
            bs_dict[domain] = constraints
        if turn_domain not in bs_dict:
            bs_dict[turn_domain] = {}

        return bs_dict
    except Exception as e:
        print(e)
        print(bs_str)
        return {}


'''
Get DB State based on the belief state (domain, constraints)
@param input_belief_state: system output of belief state prediction
@param dataset: dataset used in the model
@output: DB state
For single domain in input_belief_state, return "domain x match"
For multi domain in input_belief_state,  return "domain1 x1 match domain2 x2 match ..."
'''
def get_db_state(input_belief_state, dataset= 'multiwoz'):
    db = get_db(dataset)
    constraints = belief_state_to_dict(input_belief_state)
    res = ""
    for domain in db.ontology.all_domains:
        if domain in db.ontology.db_domains and domain in constraints.keys():
            # to be compatible with the new format
            constraints[domain] = {'inform': constraints[domain]}
            query_results = db.queryJsons(domain, constraints[domain])
            if res == "":
                res = f"{domain} {len(query_results)} match"
            else:
                res = f"{res}, {domain} {len(query_results)} match"
            # if number of matched attraction is 1, append entrance fee info
            if domain == 'attraction' and len(query_results) == 1:
                res = '{} ( entrance fee = {} )'.format(
                    res, query_results[0]['price']
                )
    # print(res)
    return res




if __name__ == '__main__':
    # pass
    res = belief_state_to_dict("<DMN> travel { category = dontcare, location = london , england, free_entry = true } <EOB>")
    print(res)

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
