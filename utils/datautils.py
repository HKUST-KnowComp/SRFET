import json
import pickle


def load_vocab_file(vocab_file):
    vocab = list()
    term_to_id_dict = dict()
    with open(vocab_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            t = line.strip()
            vocab.append(t)
            term_to_id_dict[t] = i
    return vocab, term_to_id_dict


def read_json_objs(filename):
    objs = list()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            objs.append(json.loads(line))
    return objs


def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
