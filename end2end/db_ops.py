import json, random, os
from fuzzywuzzy import fuzz
import re

class MultiWozDB(object):
    def __init__(self, db_paths, ontology):
        self.dbs = {}
        self.sql_dbs = {}
        self.ontology = ontology
        for domain in self.ontology.all_domains:
            if db_paths[domain]:
                with open(
                    os.path.join(os.path.dirname(__file__),
                                 os.path.pardir,
                                 db_paths[domain]),
                    'r'
                ) as f:
                    self.dbs[domain] = json.loads(f.read().lower())
            else:
                self.dbs[domain] = {}

        self.sanity_check()

    def sanity_check(self):
        for domain, db in self.dbs.items():
            if domain in ['taxi', 'police']:
                continue
            # get unique slot name
            db_slot_names = set()
            for entity in db:
                for k, v in entity.items():
                    db_slot_names.add(k)

            # set the slot value to 'not listed' if entity doesn't have the slot
            for s_name in db_slot_names:
                for i, entity in enumerate(db):
                    if s_name not in entity.keys():
                        self.dbs[domain][i][s_name] = 'not listed'
                        # print(s_name)

            # add reference number to db entity
            for i, entity in enumerate(db):
                self.dbs[domain][i]['reference'] = f'{i:08d}'

    def oneHotVector(self, domain, num):
        """Return number of available entities for particular domain."""
        vector = [0,0,0,0]
        if num == '':
            return vector
        if domain != 'train':
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num == 1:
                vector = [0, 1, 0, 0]
            elif num <=3:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        else:
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num <= 5:
                vector = [0, 1, 0, 0]
            elif num <=10:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        return vector


    def addBookingPointer(self, constraint, domain, book_state):
        """Add information about availability of the booking option."""
        # Booking pointer
        # Do not consider booking two things in a single turn.

        if domain not in self.ontology.booking_slots:
            return [0, 0]

        if book_state[domain]:
            return [0, 0]

        booking_requirement = self.ontology.booking_slots[domain]
        success = True
        for slot in booking_requirement:
            if slot not in constraint:
                success = False

        if success:
            return [0, 1]
            # TODO: when is booking not available?
        else:
            return [0, 0]


    def addDBPointer(self, domain, match_num, return_num=False):
        """Create database pointer for all related domains."""
        # if turn_domains is None:
        #     turn_domains = db_domains
        if domain in self.ontology.db_domains:
            vector = self.oneHotVector(domain, match_num)
        else:
            vector = [0, 0, 0 ,0]
        return vector

    def get_match_num(self, constraints, return_entry=False):
        """Create database pointer for all related domains."""
        match = {'general': ''}
        entry = {}
        # if turn_domains is None:
        #     turn_domains = db_domains
        for domain in self.ontology.all_domains:
            match[domain] = ''
            if domain in self.ontology.db_domains and constraints.get(domain):
                matched_ents = self.queryJsons(domain, constraints[domain])
                match[domain] = len(matched_ents)
                if return_entry :
                    entry[domain] = matched_ents
        if return_entry:
            return entry
        return match


    def pointerBack(self, vector, domain):
        # multi domain implementation
        # domnum = cfg.domain_num
        if domain.endswith(']'):
            domain = domain[1:-1]
        if domain != 'train':
            nummap = {
                0: '0',
                1: '1',
                2: '2-3',
                3: '>3'
            }
        else:
            nummap = {
                0: '0',
                1: '1-5',
                2: '6-10',
                3: '>10'
            }
        if vector[:4] == [0,0,0,0]:
            report = ''
        else:
            num = vector.index(1)
            report = domain+': '+nummap[num] + '; '

        if vector[-2] == 0 and vector[-1] == 1:
            report += 'booking: ok'
        if vector[-2] == 1 and vector[-1] == 0:
            report += 'booking: unable'

        return report

    def queryJsons(self,
                   domain,
                   constraints,
                   exactly_match=True,
                   return_name=False):
        """Returns the list of entities for a given domain
        based on the annotation of the belief state
        constraints: dict e.g. {'pricerange': 'cheap', 'area': 'west'}
        """
        # query the db
        # import pdb; pdb.set_trace()
        constraints = constraints["inform"]  # db search only for informable slots
        if domain == 'taxi':
            return [{'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']),
                     'taxi_types': random.choice(self.dbs[domain]['taxi_types']),
                     'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
        if domain == 'police':
            return self.dbs['police']
        if domain == 'hospital':
            if constraints.get('department'):
                for entry in self.dbs['hospital']:
                    if entry.get('department') == constraints.get('department'):
                        return [entry]
            else:
                return []

        if 'name' in constraints and constraints['name'] != 'dontcare':
            match_result = self._query_JsonsByName(
                domain, constraints, self.dbs[domain], exactly_match,
                return_name
            )
        else:
            match_result = self._queryJsonsBySlots(
                domain, constraints, self.dbs[domain], exactly_match,
                return_name
            )

        return match_result

    def _queryJsonsBySlots(self,
                           domain,
                           constraints,
                           db_entities,
                           exactly_match,
                           return_name):
        match_result = []
        for index, db_ent in enumerate(db_entities):
            match = True
            for s, v in constraints.items():
                if s == 'name':
                    continue
                if s in ['people', 'stay'] or(domain == 'hotel' and s == 'day') or \
                    (domain == 'restaurant' and s in ['day', 'time']):
                    continue

                skip_case = {
                    "don't care": 1,
                    "do n't care": 1,
                    "dont care": 1,
                    "not mentioned": 1,
                    "dontcare": 1,
                    "": 1
                }
                if skip_case.get(v):
                    continue

                if s not in db_ent:
                    # logging.warning('Searching warning: slot %s not in %s db'%(s, domain))
                    match = False
                    break

                # v = 'guesthouse' if v == 'guest house' else v
                # v = 'swimmingpool' if v == 'swimming pool' else v
                v = 'yes' if v == 'free' else v

                if s in ['arrive', 'leave']:
                    try:
                        h,m = v.split(':')   # raise error if time value is not xx:xx format
                        v = int(h)*60+int(m)
                    except:
                        match = False
                        break
                    time = int(db_ent[s].split(':')[0])*60+int(db_ent[s].split(':')[1])
                    if s == 'arrive' and v<time:
                        match = False
                    if s == 'leave' and v>time:
                        match = False
                else:
                    if exactly_match and v != db_ent[s]:
                        match = False
                        break
                    elif v not in db_ent[s]:
                        match = False
                        break

            if match:
                match_result.append(db_ent)

        # sorted entities by hotel type, rank type=hotel first.
        if domain == 'hotel':
            match_result = sorted(match_result, key=lambda x: x['type'], reverse=True)

        if not return_name:
            if 'arrive' in constraints:
                match_result = match_result[::-1]
            return match_result
        else:
            if domain == 'train':
                match_result = [e['id'] for e in match_result]
            else:
                match_result = [e['name'] for e in match_result]
            return match_result

    def _query_JsonsByName(self,
                           domain,
                           constraints,
                           db_entities,
                           exactly_match,
                           return_name):
        match_result = []
        for index, db_ent in enumerate(db_entities):
            if 'name' in db_ent:
                # remove suffix for better match
                cons = constraints['name'].replace(" 's", '').replace("'s", '')
                dbn = db_ent['name'].replace(" 's", '').replace("'s", '')

                # fuzzy match
                fuzzy_match_ratio = 60  # set match ratio to 60
                fuzzy_match_score = fuzz.partial_ratio(cons, dbn)
                if fuzzy_match_score > fuzzy_match_ratio:
                    db_ent = db_ent if not return_name else db_ent['name']
                    # add reference number based on entity index
                    db_ent['reference'] = f'{index:08d}'
                    match_result.append((db_ent, fuzzy_match_score))
        # return empty if no match found
        if not match_result:
            return match_result

        # sort matched result based on match score
        match_result = sorted(match_result, key=lambda x: x[1], reverse=True)
        # for debugging
        # if len(match_result) > 1:
        #     print(constraints['name'])
        #     print('; '.join(['{} {}'.format(r[0]['name'], r[1]) for r in match_result]))
        #     print('')
        # if len(match_result) == 0:
        #     print(constraints['name'])
        #     print('')

        matched_entity = match_result[0][0]

        # check whether the slots of matched entity matched the non-name
        # slots in the belief
        match_result = self._queryJsonsBySlots(
            domain, constraints, [matched_entity], exactly_match, return_name
        )

        # return the top match entity
        return match_result


if __name__ == '__main__':
    dbPATHs = {
        "attraction": "data/multiwoz/db/attraction_db_processed.json",
        "hospital": "data/multiwoz/db/hospital_db_processed.json",
        "hotel": "data/multiwoz/db/hotel_db_processed.json",
        "police": "data/multiwoz/db/police_db_processed.json",
        "restaurant": "data/multiwoz/db/restaurant_db_processed.json",
        "taxi": "data/multiwoz/db/taxi_db_processed.json",
        "train": "data/multiwoz/db/train_db_processed.json"
    }
    db = MultiWozDB(dbPATHs)
    while True:
        constraints = {}
        inp = input('input belief state in fomat: domain-slot1=value1;slot2=value2...\n')
        domain, cons = inp.split('-')
        for sv in cons.split(';'):
            s, v = sv.split('=')
            constraints[s] = v
        # res = db.querySQL(domain, constraints)
        res = db.queryJsons(domain, constraints, return_name=True)
        report = []
        reidx = {
            'hotel': 8,
            'restaurant': 6,
            'attraction':5,
            'train': 1,
        }
        # for ent in res:
        #     if reidx.get(domain):
        #         report.append(ent[reidx[domain]])
        # for ent in res:
        #     if 'name' in ent:
        #         report.append(ent['name'])
        #     if 'trainid' in ent:
        #         report.append(ent['trainid'])
        print(constraints)
        print(res)
        print('count:', len(res), '\nnames:', report)
