import json,  os, re, copy, zipfile
import spacy
from collections import OrderedDict
from tqdm import tqdm
from db_ops import MultiWozDB
from clean_dataset import clean_text, clean_slot_values
from utils.common_utils import *
from utils.vocab_utils import *
import argparse
from transformers import GPT2Tokenizer

import sys
import os
absolute_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(absolute_path)

### To be configured ###
from end2end.data.multiwoz.ontology import global_ontology as ontology
# from end2end.data.camrest.CamRest676.ontology import global_ontology as ontology
# from end2end.data.smd.kvret_dataset_public.ontology import global_ontology as ontology
# from end2end.data.schema.dstc8_schema.ontology import global_ontology as ontology
# from end2end.data.taskmaster2020.data.ontology import global_ontology as ontology
# from end2end.data.taskmaster2019_self.data.ontology import global_ontology as ontology
# from end2end.data.taskmaster2019_woz.data.ontology import global_ontology as ontology
# from end2end.data.msr_e2e.data.ontology import global_ontology as ontology

def get_db_values(data_config):
    processed = {}
    bspn_word = []
    nlp = spacy.load('en_core_web_sm')

    value_set_path = data_config.value_set
    with open(value_set_path, 'r') as f:
        value_set = json.loads(f.read().lower())

    with open(data_config.ontology_path, 'r') as f:
        otlg = json.loads(f.read().lower())

    for domain, slots in value_set.items():
        processed[domain] = {}
        bspn_word.append('['+domain+']')
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot)
            if s_p in ontology.informable_slots[domain]:
                bspn_word.append(s_p)
                processed[domain][s_p] = []

    for domain, slots in value_set.items():
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot)
            if s_p in ontology.informable_slots[domain]:
                for v in values:
                    _, v_p = clean_slot_values(domain, slot, v, ontology)
                    v_p = ' '.join([token.text for token in nlp(v_p)]).strip()
                    processed[domain][s_p].append(v_p)
                    for x in v_p.split():
                        if x not in bspn_word:
                            bspn_word.append(x)

    for domain_slot, values in otlg.items():
        domain, slot = domain_slot.split('-')
        if domain == 'bus':
            domain = 'taxi'
        if slot == 'price range':
            slot = 'pricerange'
        if slot == 'book stay':
            slot = 'stay'
        if slot == 'book day':
            slot = 'day'
        if slot == 'book people':
            slot = 'people'
        if slot == 'book time':
            slot = 'time'
        if slot == 'arrive by':
            slot = 'arrive'
        if slot == 'leave at':
            slot = 'leave'
        if slot == 'leaveat':
            slot = 'leave'
        if slot not in processed[domain]:
            processed[domain][slot] = []
            bspn_word.append(slot)
        for v in values:
            _, v_p = clean_slot_values(domain, slot, v, ontology)
            v_p = ' '.join([token.text for token in nlp(v_p)]).strip()
            if v_p not in processed[domain][slot]:
                processed[domain][slot].append(v_p)
                for x in v_p.split():
                    if x not in bspn_word:
                        bspn_word.append(x)

    with open(value_set_path.replace('.json', '_processed.json'), 'w') as f:
        json.dump(processed, f, indent=2)
    if not os.path.exists(data_config.processed_data_dir):
        os.mkdir(data_config.processed_data_dir)
    with open(data_config.processed_data_dir + '/bspn_word_collection.json', 'w') as f:
        json.dump(bspn_word, f, indent=2)

    print('DB value set processed! ')



def preprocess_db(db_paths):
    dbs = {}
    nlp = spacy.load('en_core_web_sm')
    for domain in ontology.all_domains:
        with open(db_paths[domain], 'r') as f:
            dbs[domain] = json.loads(f.read().lower())
            if domain != 'taxi':
                for idx, entry in enumerate(dbs[domain]):
                    new_entry = copy.deepcopy(entry)
                    for key, value in entry.items():
                        if type(value) is not str:
                            continue
                        del new_entry[key]
                        key, value = clean_slot_values(domain, key, value, ontology)
                        tokenize_and_back = ' '.join([token.text for token in nlp(value)]).strip()
                        new_entry[key] = tokenize_and_back
                    dbs[domain][idx] = new_entry
            else:
                dbs[domain] = [dbs[domain]]
        with open(db_paths[domain].replace('.json', '_processed.json'), 'w') as f:
            json.dump(dbs[domain], f, indent=2)
        print('[%s] DB processed! '%domain)


class DataPreprocessor(object):
    def __init__(self, data_config):
        self.nlp = spacy.load('en_core_web_sm')
        # self.db = MultiWozDB('/nfs/volume-242-1/yinglu/dstc9/ConvLab-2/data/multiwoz', cfg.dbs)
        # dbs = {
        #     'restaurant': 'CamRestDB.json',
        # }
        self.db = MultiWozDB(data_config.processed_dbs, ontology)
        # data_path = '/nfs/volume-242-1/yinglu/dstc9/ConvLab-2/data/multiwoz/val.json'
        # data_path = '/nfs/volume-242-1/yinglu/dstc9/ConvLab-2/data/multiwoz/train.json'
        # data_path = '/nfs/volume-242-1/yinglu/dstc9/ConvLab-2/end2end/data/CamRest/CamRest676/CamRest676_formated.json'
        with open(data_config.data_path, 'r') as f:
            self.convlab_data = json.loads(f.read().lower())
        # self.processed_data_dir = '/nfs/volume-242-1/yinglu/dstc9/ConvLab-2/end2end/data/multi-woz-processed/'
        self.processed_data_dir = data_config.processed_data_dir
        if not os.path.exists(self.processed_data_dir):
            os.mkdir(self.processed_data_dir)
        self.delex_sg_valdict_path = self.processed_data_dir + 'delex_single_valdict.json'
        self.delex_mt_valdict_path = self.processed_data_dir + 'delex_multi_valdict.json'
        self.ambiguous_val_path = self.processed_data_dir + 'ambiguous_values.json'
        if not os.path.exists(self.delex_sg_valdict_path):
            self.delex_sg_valdict, self.delex_mt_valdict, self.ambiguous_vals = self.get_delex_valdict()
        else:
            self.delex_sg_valdict = json.loads(open(self.delex_sg_valdict_path, 'r').read())
            self.delex_mt_valdict = json.loads(open(self.delex_mt_valdict_path, 'r').read())
            self.ambiguous_vals = json.loads(open(self.ambiguous_val_path, 'r').read())

        self.vocab = Vocab(ontology, data_config.vocab_size)


    def delex_by_annotation(self, dial_turn):
        u = dial_turn['text'].split()
        span = dial_turn['span_info']
        for s in span:
            slot = s[1]
            if slot == 'open':
                continue
            if ontology.da_abbr_to_slot_name.get(slot):
                slot = ontology.da_abbr_to_slot_name[slot]
            for idx in range(s[3], s[4]+1):
                u[idx] = ''
            try:
                u[s[3]] = '[value_'+slot+']'
            except:
                u[5] = '[value_'+slot+']'
        u_delex = ' '.join([t for t in u if t is not ''])
        u_delex = u_delex.replace('[value_address] , [value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_name] [value_name]', '[value_name]')
        u_delex = u_delex.replace('[value_name]([value_phone] )', '[value_name] ( [value_phone] )')
        return u_delex


    def delex_by_valdict(self, text):
        text = clean_text(text)

        text = re.sub(r'\d{5}\s?\d{5,7}|\+\d{1} \d{3}-\d{3}-\d{4}', '[value_phone]', text)
        text = re.sub(r'\d[\s-]stars?', '[value_stars]', text)
        text = re.sub(r'\$\d+|\$?\d+.?(\d+)?\s(pounds?|gbps?)', '[value_price]', text)
        text = re.sub(r'tr[\d]{4}', '[value_id]', text)
        # text = re.sub(r' ([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', ' [value_postcode] ', text)
        text = re.sub(r' ([a-z]{2}\d{2}[a-z]{2}) ',' [value_postcode]', text)

        for value, slot in self.delex_mt_valdict.items():
            text = text.replace(value, '[value_%s]'%slot)

        for value, slot in self.delex_sg_valdict.items():
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value:
                    tokens[idx] = '[value_%s]'%slot
            text = ' '.join(tokens)

        for ambg_ent in self.ambiguous_vals:
            start_idx = text.find(' '+ambg_ent)   # ely is a place, but appears in words like moderately
            if start_idx == -1:
                continue
            front_words = text[:start_idx].split()
            ent_type = 'time' if ':' in ambg_ent else 'place'

            for fw in front_words[::-1]:
                if fw in ['arrive', 'arrives', 'arrived', 'arriving', 'arrival', 'destination', 'there', 'reach',  'to', 'by', 'before']:
                    slot = '[value_arrive]' if ent_type=='time' else '[value_destination]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)
                elif fw in ['leave', 'leaves', 'leaving', 'depart', 'departs', 'departing', 'departure',
                                'from', 'after', 'pulls']:
                    slot = '[value_leave]' if ent_type=='time' else '[value_departure]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)

        text = text.replace('[value_car] [value_car]', '[value_car]')
        return text


    def get_delex_valdict(self):
        skip_entry_type = {
            'taxi': ['taxi_phone'],
            'police': ['id'],
            'hospital': ['id'],
            'hotel': ['id', 'location', 'internet', 'parking', 'takesbookings', 'stars', 'price', 'n', 'postcode', 'phone'],
            'attraction': ['id', 'location', 'pricerange', 'price', 'openhours', 'postcode', 'phone'],
            'train': ['price', 'id'],
            'restaurant': ['id', 'location', 'introduction', 'signature', 'type', 'postcode', 'phone'],
        }
        entity_value_to_slot= {}
        ambiguous_entities = []
        for domain, db_data in self.db.dbs.items():
            print('Processing entity values in [%s]'%domain)
            if domain != 'taxi' or (domain == 'taxi' and args.dataset == 'msr_e2e'):
                for db_entry in db_data:
                    for slot, value in db_entry.items():
                        if slot not in skip_entry_type[domain]:
                            if type(value) is not str:
                                raise TypeError("value '%s' in domain '%s' should be rechecked"%(slot, domain))
                            else:
                                slot, value = clean_slot_values(domain, slot, value, ontology)
                                value = ' '.join([token.text for token in self.nlp(value)]).strip()
                                if value in entity_value_to_slot and entity_value_to_slot[value] != slot:
                                    # print(value, ": ",entity_value_to_slot[value], slot)
                                    ambiguous_entities.append(value)
                                entity_value_to_slot[value] = slot
            else:   # taxi db specific
                db_entry = db_data[0]
                for slot, ent_list in db_entry.items():
                    if slot not in skip_entry_type[domain]:
                        for ent in ent_list:
                            entity_value_to_slot[ent] = 'car'
        ambiguous_entities = set(ambiguous_entities)
        if 'cambridge' in ambiguous_entities:
            ambiguous_entities.remove('cambridge')
        ambiguous_entities = list(ambiguous_entities)
        for amb_ent in ambiguous_entities:   # departure or destination? arrive time or leave time?
            entity_value_to_slot.pop(amb_ent)
        entity_value_to_slot['parkside'] = 'address'
        entity_value_to_slot['parkside, cambridge'] = 'address'
        entity_value_to_slot['cambridge belfry'] = 'name'
        entity_value_to_slot['hills road'] = 'address'
        entity_value_to_slot['hills rd'] = 'address'
        entity_value_to_slot['Parkside Police Station'] = 'name'

        single_token_values = {}
        multi_token_values = {}
        for val, slt in entity_value_to_slot.items():
            if val in ['cambridge']:
                continue
            if len(val.split())>1:
                multi_token_values[val] = slt
            else:
                single_token_values[val] = slt

        with open(self.delex_sg_valdict_path, 'w') as f:
            single_token_values = OrderedDict(sorted(single_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(single_token_values, f, indent=2)
            print('single delex value dict saved!')
        with open(self.delex_mt_valdict_path, 'w') as f:
            multi_token_values = OrderedDict(sorted(multi_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(multi_token_values, f, indent=2)
            print('multi delex value dict saved!')
        with open(self.ambiguous_val_path, 'w') as f:
            json.dump(ambiguous_entities, f, indent=2)
            print('ambiguous value dict saved!')

        return single_token_values, multi_token_values, ambiguous_entities


    def preprocess_main(self, save_path=None, is_test=False):
        """
        """
        data = {}
        count=0
        count_no_da=0
        self.unique_da = {}
        ordered_sysact_dict = {}
        pretrained_gpt = "/nfs/volume-242-1/yinglu/dstc9/ConvLab-2/end2end/resources/pretrained_models/gpt2_en"
        tokenizer = GPT2Tokenizer.from_pretrained(
          pretrained_gpt, do_lower_case=False
        )
        for fn, raw_dial in tqdm(list(self.convlab_data.items())):
            count +=1

            # for i, dt in enumerate(raw_dial['log']):
            #     if 'dialog_act' not in dt:
            #         count_no_da += 1
            #         continue
            # continue

            compressed_goal = {}
            dial_domains, dial_reqs = [], []
            for dom, g in raw_dial['goal'].items():
                if dom != 'topic' and dom != 'message' and g:
                    if g.get('reqt'):
                        for i, req_slot in enumerate(g['reqt']):
                            if ontology.normlize_slot_names.get(req_slot):
                                g['reqt'][i] = ontology.normlize_slot_names[req_slot]
                                dial_reqs.append(g['reqt'][i])
                    compressed_goal[dom] = g
                    if dom in ontology.all_domains:
                        dial_domains.append(dom)

            # dial_reqs = list(set(dial_reqs))
            # dial = {'goal': compressed_goal, 'log': []}
            dial = {'log': []}

            single_turn = {}
            constraint_dict = OrderedDict()
            prev_constraint_dict = {}
            prev_turn_domain = ['general']

            for turn_num, dial_turn in enumerate(raw_dial['log']):
                dial_state = dial_turn['metadata']
                if not dial_state:   # user
                    u = ' '.join(clean_text(dial_turn['text']).split())
                    if False and dial_turn['span_info']: ### todo
                        u_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        u_delex = self.delex_by_valdict(dial_turn['text'])

                    single_turn['user'] = u
                    single_turn['user_delex'] = u_delex

                else:   #system
                    if False and dial_turn['span_info']: ### todo
                        s_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        if not dial_turn['text']:
                            print(fn)
                        s_delex = self.delex_by_valdict(dial_turn['text'])
                    single_turn['resp'] = dial_turn['text']
                    single_turn['resp_delex'] = s_delex
                    single_turn['resp_delex_reference'] = single_turn['resp_delex']

                    # get belief state
                    for domain in dial_domains:
                        if not constraint_dict.get(domain):
                            constraint_dict[domain] = OrderedDict()
                            constraint_dict[domain]["request"] = []
                            constraint_dict[domain]["inform"] = OrderedDict()
                        info_sv = dial_state[domain]['semi']  ## here request/inform
                        # for multiwoz
                        if "request" not in info_sv and "inform" not in info_sv:
                            info_sv_request = []
                            info_sv_inform = info_sv
                        else:
                            info_sv_request = info_sv["request"]
                            info_sv_inform = info_sv["inform"]
                        for r_slot in info_sv_request:
                            constraint_dict[domain]["request"].append(r_slot)
                        for s, v in info_sv_inform.items():
                            s, v = clean_slot_values(domain, s, v, ontology)
                            if len(v.split()) > 1:
                                self.nlp.tokenizer.add_special_case("<sep>", [{"ORTH": "<sep>"}])
                                self.nlp.tokenizer.add_special_case("<or>", [{"ORTH": "<or>"}])
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain]["inform"][s] = v
                        book_sv = dial_state[domain]['book']
                        for s,v in book_sv.items():
                            if s == 'booked':
                                # continue ### ignore for damd
                                ### delex sys_resp if there is a reference number for soloist_baseline
                                reference_num = ""
                                for booked_item in v:
                                    if "reference" in booked_item:
                                        reference_num = booked_item['reference']
                                if reference_num != "":
                                    single_turn['resp_delex_reference'] = single_turn['resp_delex_reference'].replace(reference_num.lower(), '[value_reference]')
                                continue

                            s,v = clean_slot_values(domain, s,v, ontology)
                            if len(v.split())>1:
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v

                    constraints = []
                    # cons_delex = []
                    turn_dom_bs = []
                    for domain, info_slots in constraint_dict.items():
                        if info_slots and info_slots['inform']:
                            constraints.append('['+domain+']')
                            # cons_delex.append('['+domain+']')
                            constraints.extend(info_slots["request"])
                            for slot, value in info_slots["inform"].items():
                                constraints.append(slot + ' =')
                                constraints.extend(value.split())
                                # cons_delex.append(slot)
                            if domain not in prev_constraint_dict:
                                turn_dom_bs.append(domain)
                            elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                turn_dom_bs.append(domain)

                    sys_act_dict = {}
                    turn_dom_da = set()
                    for act in dial_turn['dialog_act']:
                        d, a = act.split('-')
                        turn_dom_da.add(d)
                    turn_dom_da = list(turn_dom_da)
                    if len(turn_dom_da) != 1 and 'general' in turn_dom_da:
                        turn_dom_da.remove('general')
                    if len(turn_dom_da) != 1 and 'booking' in turn_dom_da:
                        turn_dom_da.remove('booking')

                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]


                    # get system action
                    sys_act_dict = {}
                    if 'dialog_act' in dial_turn:
                        sys_act_dict = dial_turn['dialog_act']


                    # get db pointers
                    matnums = self.db.get_match_num(constraint_dict)
                    matnums = {k: v for k, v in matnums.items() if v != ""}
                    single_turn['match'] = json.dumps(matnums)
                    single_turn['match'] = json.loads(single_turn['match'])
                    # single_turn['constraint'] = ' '.join(constraints)
                    single_turn['constraint'] = json.dumps(constraint_dict)
                    single_turn['constraint'] = json.loads(single_turn['constraint'])
                    # single_turn['cons_delex'] = ' '.join(cons_delex)
                    single_turn['turn_num'] = len(dial['log'])
                    single_turn['turn_domain'] = ' '.join(['['+d+']' for d in turn_domain])
                    single_turn['dialog_act'] = json.dumps(sys_act_dict)
                    single_turn['dialog_act'] = json.loads(single_turn['dialog_act'])

                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)

                    if 'user' in single_turn:
                        dial['log'].append(single_turn)
                        # for t in single_turn['user'].split() + single_turn['resp'].split() + constraints + sys_act: ### todo
                        for t in single_turn['user'].split() + single_turn['resp'].split() + constraints:
                            self.vocab.add_word(t)
                        for t in single_turn['user_delex'].split():
                            if '[' in t and ']' in t and not t.startswith('[') and not t.endswith(']'):
                                single_turn['user_delex'].replace(t, t[t.index('['): t.index(']')+1])
                            elif not self.vocab.has_word(t):
                                self.vocab.add_word(t)

                    single_turn = {}


            # data[fn] = dial

            dialog_resp_token_length = 0
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokens_to_add = ['Ġ<SOB>Ġ', 'Ġ<EOB>Ġ', 'Ġ<EOKB>Ġ', 'Ġ<EOS>Ġ']
            tokenizer.add_tokens(tokens_to_add)
            for turn_i in dial['log']:
                turn_resp_length = len(tokenizer(turn_i['resp'])['input_ids'])
                if dialog_resp_token_length < turn_resp_length:
                    dialog_resp_token_length = turn_resp_length

            if len(dial['log'])>0 and dialog_resp_token_length < 100:
                data[fn] = dial
            else:
                print(f"dialogid {fn}, dialog_resp_token_length = {dialog_resp_token_length}")
            # pprint(dial)
            # if count == 20:
            #     break
        print("count_no_da: ", count_no_da)
        self.vocab.construct()
        self.vocab.save_vocab(self.processed_data_dir + 'vocab')
        with open(self.processed_data_dir + 'data_for_soloist.json', 'w') as f:
            json.dump(data, f, indent=2)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multiwoz",
                        help="multiwoz, default multiwoz")
    args = parser.parse_args()

    data_config_name = f"end2end/configs/data_configs/{args.dataset}.json"
    data_config = load_config(data_config_name)

    ### DB Value Processing/Cleanning (currently process for multiwoz only) ###
    if args.dataset == 'multiwoz':
        get_db_values(data_config)
        preprocess_db(data_config.dbs)

    dh = DataPreprocessor(data_config)
    data = dh.preprocess_main(data_config)
