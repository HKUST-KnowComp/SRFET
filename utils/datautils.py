import json
import pickle
from utils import utils


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


def save_json_objs(objs, output_file):
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    for x in objs:
        fout.write('{}\n'.format(json.dumps(x)))
    fout.close()


def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_indexed_word(w, dec=True):
    p = w.rfind('-')
    s = w[:p]
    idx = int(w[p + 1:])
    if dec:
        idx -= 1
    return s, idx


def next_sent_dependency(fin):
    dep_list = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        dep_tup = line.split(' ')
        wgov, idx_gov = get_indexed_word(dep_tup[0])
        wdep, idx_dep = get_indexed_word(dep_tup[1])
        dep_tup = (dep_tup[2], (idx_gov, wgov), (idx_dep, wdep))
        dep_list.append(dep_tup)
    return dep_list


def read_srl_result_for_sent(fin, to_tag_spans=False):
    try:
        line = next(fin).strip()
    except StopIteration:
        return None

    srl_result = list()
    if not line:
        return srl_result

    srl_result.append(line)
    while True:
        line = next(fin).strip()
        if not line:
            break
        srl_result.append(line)

    if to_tag_spans:
        tag_span_results = list()
        for tag_seq in srl_result:
            tag_seq = tag_seq.split(' ')
            tag_spans = utils.get_srl_tag_spans(tag_seq)
            tag_span_results.append(tag_spans)
        return tag_span_results
    return srl_result


def read_srl_results(filename):
    srl_results_list = list()
    f = open(filename, encoding='utf-8')
    while True:
        srl_results = read_srl_result_for_sent(f, True)
        if srl_results is None:
            break
        srl_results_list.append(srl_results)
    f.close()
    return srl_results_list
