
class _Ontology:
    def __init__(self):
        self.all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
        self.db_domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital']


        self.normlize_slot_names = {
            "car type": "car",
            "entrance fee": "price",
            "duration": "time",
            "leaveat": 'leave',
            'arriveby': 'arrive',
            'trainid': 'id'
        }

        self.requestable_slots = {
            "taxi": ["car", "phone"],
            "police": ["postcode", "address", "phone"],
            "hospital": ["address", "phone", "postcode"],
            "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
            "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
            "train": ["time", "leave", "price", "arrive", "id", "reference"],
            "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
        }
        self.all_reqslot = ["car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
                              "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]

        self.informable_slots = {
            "taxi": ["leave", "destination", "departure", "arrive"],
            "police": [],
            "hospital": ["department"],
            "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
            "attraction": ["area", "type", "name"],
            "train": ["destination", "day", "arrive", "departure", "people", "leave"],
            "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
        }
        self.all_infslot = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
                             "leave", "destination", "departure", "arrive", "department", "food", "time"]


        self.booking_slots = {'train': ['people'],
                                   'restaurant': ['time', 'day', 'people'],
                                   'hotel': ['stay', 'day', 'people']}

        self.all_slots = self.all_reqslot + ["stay", "day", "people", "name", "destination", "departure", "department"]
        self.get_slot = {}
        for s in self.all_slots:
            self.get_slot[s] = 1



        # mapping slots in dialogue act to original goal slot names
        self.da_abbr_to_slot_name = {
            'addr': "address",
            'fee': "price",
            'post': "postcode",
            'ref': 'reference',
            'ticket': 'price',
            'depart': "departure",
            'dest': "destination",
        }

        self.dialog_acts = {
            'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
            'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
            'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
            'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
            'taxi': ['inform', 'request'],
            'police': ['inform', 'request'],
            'hospital': ['inform', 'request'],
            # 'booking': ['book', 'inform', 'nobook', 'request'],
            'general': ['bye', 'greet', 'reqmore', 'welcome'],
        }
        self.all_acts = []
        for acts in self.dialog_acts.values():
            for act in acts:
                if act not in self.all_acts:
                    self.all_acts.append(act)


        self.dialog_act_params = {
            'inform': self.all_slots + ['choice', 'open'] ,
            'request': self.all_infslot+['choice', 'price'],
            'nooffer': self.all_slots + ['choice'],
            'recommend': self.all_reqslot + ['choice', 'open'],
            'select': self.all_slots +['choice'],
            # 'book': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
            'nobook': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
            'offerbook':self.all_slots + ['choice'],
            'offerbooked': self.all_slots + ['choice'],
            'reqmore': [],
            'welcome': [],
            'bye': [],
            'greet': [],
        }

        self.dialog_act_all_slots = self.all_slots + ['choice', 'open']


        self.slot_name_to_slot_token = {}


        # special slot tokens in responses
        self.slot_name_to_value_token = {
            # 'entrance fee': '[value_price]',
            # 'pricerange': '[value_price]',
            # 'arriveby': '[value_time]',
            # 'leaveat': '[value_time]',
            # 'departure': '[value_place]',
            # 'destination': '[value_place]',
            # 'stay': 'count',
            # 'people': 'count'
        }

        self.special_tokens = ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>',
                                    '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>','<eos_d>'] # 0,1,2,3,4,5,6,7,8,9,10

        self.eos_tokens = {
            'user': '<eos_u>', 'user_delex': '<eos_u>',
            'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
            'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
            'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
            'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
            'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>'}

global_ontology = _Ontology()
