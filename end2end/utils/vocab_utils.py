import logging
import json
import numpy as np
from collections import OrderedDict

def write_dict(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, indent=2)

class Vocab(object):
    def __init__(self, ontology, vocab_size=0):
        self.ontology = ontology
        self.vocab_size = vocab_size
        self.vocab_size_oov = 0   # get after construction
        self._idx2word = {}   #word + oov
        self._word2idx = {}   # word
        self._freq_dict = {}   #word + oov
        for w in ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>','<eos_u>', '<eos_r>',
                      '<eos_b>', '<eos_a>', '<go_d>','<eos_d>']:
            self._absolute_add_word(w)

    def _absolute_add_word(self, w):
        idx = len(self._idx2word)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: %d' % (len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in self.ontology.all_domains + ['general']:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in self.ontology.all_acts:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in self.ontology.all_slots:
            self._add_to_vocab(word)
        for word in l:
            if word.startswith('[value_') and word.endswith(']'):
                self._add_to_vocab(word)
        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path+'.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_oov = len(self._idx2word)
        print('vocab file loaded from "'+vocab_path+'"')
        print('Vocabulary size including oov: %d' % (self.vocab_size_oov))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict(vocab_path+'.word2idx.json', self._word2idx)
        write_dict(vocab_path+'.freq.json', _freq_dict)


    def encode(self, word, include_oov=True):
        if not include_oov:
            if self._word2idx.get(word, None) is None:
                raise ValueError('Unknown word: %s. Vocabulary should include oovs here.'%word)
            return self._word2idx[word]
        else:
            word = '<unk>' if word not in self._word2idx else word
            return self._word2idx[word]

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def oov_idx_map(self, idx):
        return 2 if idx > self.vocab_size else idx

    def sentence_oov_map(self, index_list):
        return [self.oov_idx_map(_) for _ in index_list]


    def decode(self, idx, indicate_oov=False):
        if not self._idx2word.get(idx):
            raise ValueError('Error idx: %d. Vocabulary should include oovs here.'%idx)
        if not indicate_oov or idx<self.vocab_size:
            return self._idx2word[idx]
        else:
            return self._idx2word[idx]+'(o)'

    def sentence_decode(self, index_list, eos=None, indicate_oov=False):
        l = [self.decode(_, indicate_oov) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]
