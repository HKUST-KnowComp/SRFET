import numpy as np
from utils import fetutils, datautils


class ResData:
    def __init__(self, type_vocab_file, word_vecs_file):
        self.type_vocab, self.type_id_dict = datautils.load_vocab_file(type_vocab_file)
        self.parent_type_ids_dict = fetutils.get_parent_type_ids_dict(self.type_id_dict)
        self.n_types = len(self.type_vocab)

        if word_vecs_file is not None:
            import config

            print('loading {} ...'.format(word_vecs_file), end=' ', flush=True)
            self.token_vocab, self.token_vecs = datautils.load_pickle_data(word_vecs_file)
            self.token_id_dict = {t: i for i, t in enumerate(self.token_vocab)}
            print('done', flush=True)
            self.zero_pad_token_id = self.token_id_dict[config.TOKEN_ZERO_PAD]
            self.mention_token_id = self.token_id_dict[config.TOKEN_MENTION]
            self.unknown_token_id = self.token_id_dict[config.TOKEN_UNK]


def get_srl_pred_dict(srl_pred_objs):
    mention_preds_dict = dict()
    for x in srl_pred_objs:
        mention_id = x['mention_id']
        pred_objs_tmp = mention_preds_dict.get(mention_id, list())
        if not pred_objs_tmp:
            mention_preds_dict[mention_id] = pred_objs_tmp
        pred_objs_tmp.append(x)

    mention_pred_dict = dict()
    for mention_id, pred_objs in mention_preds_dict.items():
        max_logits = [max(x['logits']) for x in pred_objs]
        max_idx = np.argmax(max_logits)
        if max_logits[max_idx] < 1:
            continue
        mention_pred_dict[mention_id] = pred_objs[max_idx]
    return mention_pred_dict


class PredResultCollect:
    def __init__(self, base_preds_file, srl_preds_file, hypext_file, verif_hypext_file, hypext_logits_file):
        base_pred_objs = datautils.read_json_objs(base_preds_file)
        self.base_preds_dict = {x['mention_id']: x for x in base_pred_objs}
        srl_pred_objs = datautils.read_json_objs(srl_preds_file)
        self.srl_preds_dict = get_srl_pred_dict(srl_pred_objs)

        self.hyp_preds_dict = load_hyp_preds(hypext_file, verif_hypext_file, hypext_logits_file)


def load_hyp_preds(hypext_file, verif_hypext_file, hyp_verif_logits_file, use_verif_logits=True):
    from utils import datautils

    hypext_results = datautils.read_json_objs(hypext_file)

    verif_hypext_results = datautils.read_json_objs(verif_hypext_file)
    with open(hyp_verif_logits_file, encoding='utf-8') as f:
        verif_logits = [float(line.strip()) for line in f]
    assert len(verif_hypext_results) == len(verif_logits)
    if len(hypext_results) != len(verif_hypext_results):
        print('len(hypext_results) != len(verif_hypext_results)')

    verif_hypext_result_dict = {r['mention_id']: (i, r) for i, r in enumerate(verif_hypext_results)}

    hypext_results_dict = dict()
    for r in hypext_results:
        mention_id = r['mention_id']
        tmp = verif_hypext_result_dict.get(mention_id)
        if tmp is None:
            continue
        verif_result_idx, verif_result = tmp
        if not use_verif_logits or verif_logits[verif_result_idx] > 0:
            hypext_results_dict[mention_id] = (r, verif_logits[verif_result_idx])

    return hypext_results_dict


def get_hyp_logits(labels, type_to_id_dict, n_types, child_type_vecs):
    w = 1
    label_ids = [type_to_id_dict[t] for t in labels]
    logits = -np.ones(n_types, np.float32) * w
    for label_id in label_ids:
        logits[label_id] = w
    # if len(label_ids) == 1:
    #     logits += child_type_vecs[label_ids[0]] * w
    return logits


def get_pred_results(preds_collect: PredResultCollect, n_types, type_id_dict, child_type_vecs, mention_id):
    base_result = preds_collect.base_preds_dict.get(mention_id)
    base_logits = base_result['logits'] if base_result is not None else [0.0] * n_types

    srl_result = preds_collect.srl_preds_dict.get(mention_id)
    srl_logits = srl_result['logits'] if srl_result is not None else [0.0] * n_types

    hyp_result = preds_collect.hyp_preds_dict.get(mention_id)
    hyp_verif_logit = 0
    if hyp_result is None:
        hyp_logits = [0.0] * n_types
    else:
        hyp_logits = get_hyp_logits(fetutils.get_full_types(hyp_result[0]['dtype']), type_id_dict, n_types,
                                    child_type_vecs)
        hyp_verif_logit = hyp_result[1]

    return base_logits, srl_logits, hyp_logits, hyp_verif_logit


def load_labeled_samples(type_id_dict, child_type_vecs, labeled_samples_file, pred_file_tup, use_vr, use_hr):
    base_preds_file, srl_preds_file, hyp_file, verif_hyp_file, hypext_logits_file = pred_file_tup
    prc = PredResultCollect(base_preds_file, srl_preds_file, hyp_file, verif_hyp_file, hypext_logits_file)
    n_types = len(type_id_dict)
    samples = list()
    cnt = 0
    f = open(labeled_samples_file, encoding='utf-8')
    for i, line in enumerate(f):
        mention_id = int(line.strip())
        sent_str = next(f).strip()
        # next(f)
        labels_str = next(f).strip()
        if ': ' in labels_str:
            labels_str = next(f).strip()
        next(f)
        if not labels_str or labels_str == '////':
            continue

        srl_pred_obj, hyp_pred_obj = prc.srl_preds_dict.get(mention_id), prc.hyp_preds_dict.get(mention_id)
        if srl_pred_obj is None and hyp_pred_obj is None:
            # print(mention_id)
            cnt += 1
            continue
        if not use_vr and hyp_pred_obj is None:
            continue
        if not use_hr and srl_pred_obj is None:
            continue

        base_logits, srl_logits, hyp_logits, hyp_verif_logit = get_pred_results(
            prc, n_types, type_id_dict, child_type_vecs, mention_id)

        labels = labels_str.split(',')
        for t in labels:
            if not t.startswith('/'):
                print(i, mention_id, t, line)
            assert t.startswith('/')
        labels = fetutils.get_full_types(labels)
        try:
            label_ids = [type_id_dict[t] for t in labels]
        except KeyError:
            print(i, mention_id, labels, line)
            exit()

        sample = (mention_id, base_logits, srl_logits, hyp_logits, hyp_verif_logit, label_ids)
        samples.append(sample)
    f.close()
    print(cnt)
    return samples


def samples_from_test(gres: ResData, child_type_vecs, test_file_tup):
    (mentions_file, sents_file, base_preds_file, srl_preds_file, hypext_file, verif_hypext_file, hypext_logits_file
     ) = test_file_tup
    prc = PredResultCollect(base_preds_file, srl_preds_file, hypext_file, verif_hypext_file, hypext_logits_file)
    mentions = datautils.read_json_objs(mentions_file)
    true_labels_dict = {m['mention_id']: m['labels'] for m in mentions}
    sents = datautils.read_json_objs(sents_file)
    samples = list()
    for i, m in enumerate(mentions):
        mention_id = m['mention_id']
        base_logits, srl_logits, hyp_logits, hyp_verif_logit = get_pred_results(
            prc, gres.n_types, gres.type_id_dict, child_type_vecs, mention_id)

        labels = m['labels']
        label_ids = [gres.type_id_dict[t] for t in labels]

        sample = (mention_id, base_logits, srl_logits, hyp_logits, hyp_verif_logit, label_ids)
        samples.append(sample)
    return samples, true_labels_dict
