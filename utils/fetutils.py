import numpy as np


def __super_types(t):
    types = [t]
    tmpt = t
    while True:
        pos = tmpt.rfind('/')
        if pos == 0:
            break
        tmpt = tmpt[:pos]
        types.append(tmpt)
    return types


def get_full_types(labels):
    types = set()
    for label in labels:
        super_types = __super_types(label)
        for t in super_types:
            types.add(t)
    return list(types)


def get_full_type_ids(type_ids, parent_type_ids_dict):
    full_type_ids = set()
    for type_id in type_ids:
        full_type_ids.add(type_id)
        for tid in parent_type_ids_dict[type_id]:
            full_type_ids.add(tid)
    return list(full_type_ids)


def get_parent_type(t):
    p = t.rfind('/')
    if p <= 1:
        return None
    return t[:p]


def get_parent_types(t):
    parents = list()
    while True:
        p = get_parent_type(t)
        if p is None:
            return parents
        parents.append(p)
        t = p


def get_parent_type_ids_dict(type_id_dict):
    d = dict()
    for t, type_id in type_id_dict.items():
        d[type_id] = [type_id_dict[p] for p in get_parent_types(t)]
    return d


def build_hierarchy_vecs(type_vocab, type_to_id_dict):
    n_types = len(type_vocab)
    l1_type_vec = np.zeros(n_types, np.float32)
    l1_type_indices = list()
    child_type_vecs = np.zeros((n_types, n_types), np.float32)
    for i, t in enumerate(type_vocab):
        p = get_parent_type(t)
        if p is None:
            l1_type_indices.append(i)
            l1_type_vec[type_to_id_dict[t]] = 1
        else:
            child_type_vecs[type_to_id_dict[p]][type_to_id_dict[t]] = 1
    l1_type_indices = np.array(l1_type_indices, np.int32)
    return l1_type_indices, l1_type_vec, child_type_vecs


def inference_labels_full(l1_type_indices, child_type_vecs, scores, extra_label_thres=0.5):
    label_preds_main = inference_labels(l1_type_indices, child_type_vecs, scores)
    label_preds = list()
    for i in range(len(scores)):
        extra_idxs = np.argwhere(scores[i] > extra_label_thres).squeeze(axis=1)
        label_preds.append(list(set(label_preds_main[i] + list(extra_idxs))))
    return label_preds


def inference_labels(l1_type_indices, child_type_vecs, scores):
    l1_type_scores = scores[:, l1_type_indices]
    tmp_indices = np.argmax(l1_type_scores, axis=1)
    max_l1_indices = l1_type_indices[tmp_indices]
    l2_scores = child_type_vecs[max_l1_indices] * scores
    max_l2_indices = np.argmax(l2_scores, axis=1)
    # labels_pred = np.zeros(scores.shape[0], np.int32)
    labels_pred = list()
    for i, (l1_idx, l2_idx) in enumerate(zip(max_l1_indices, max_l2_indices)):
        # labels_pred[i] = l2_idx if l2_scores[i][l2_idx] > 1e-4 else l1_idx
        labels_pred.append([l2_idx] if l2_scores[i][l2_idx] > 1e-4 else [l1_idx])
    return labels_pred


class TypeInfer:
    def __init__(self, type_vocab, type_id_dict, single_type_path, extra_label_thres=0.0):
        self.type_vocab, self.type_id_dict = type_vocab, type_id_dict
        self.l1_type_indices, self.l1_type_vec, self.child_type_vecs = build_hierarchy_vecs(
            self.type_vocab, self.type_id_dict)
        self.single_type_path = single_type_path
        self.extra_label_thres = extra_label_thres

    def inference(self, scores, is_torch_tensor=True):
        if self.single_type_path:
            return self.inference_single(scores, is_torch_tensor)
        return self.inference_full(scores, is_torch_tensor)

    def inference_single(self, scores, is_torch_tensor=True):
        if is_torch_tensor:
            scores = scores.data.cpu().numpy()
        return inference_labels(self.l1_type_indices, self.child_type_vecs, scores)

    def inference_full(self, logits, is_torch_tensor=True):
        if is_torch_tensor:
            logits = logits.data.cpu().numpy()
        return inference_labels_full(self.l1_type_indices, self.child_type_vecs, logits, self.extra_label_thres)


def labels_full_match(labels_true, labels_pred):
    if len(labels_true) != len(labels_pred):
        return False

    for t in labels_true:
        if t not in labels_pred:
            return False
    return True


def strict_acc(true_labels_dict, pred_labels_dict):
    hit_cnt = 0
    for wid, labels_true in true_labels_dict.items():
        labels_pred = pred_labels_dict[wid]
        if labels_full_match(labels_true, labels_pred):
            hit_cnt += 1
    return hit_cnt / len(true_labels_dict)


def count_match(label_true, label_pred):
    return sum(1 if t in label_pred else 0 for t in label_true)


def microf1(true_labels_dict, pred_labels_dict):
    assert len(true_labels_dict) == len(pred_labels_dict)
    l_true_cnt, l_pred_cnt, hit_cnt = 0, 0, 0
    for mention_id, labels_true in true_labels_dict.items():
        labels_pred = pred_labels_dict[mention_id]
        hit_cnt += count_match(labels_true, labels_pred)
        l_true_cnt += len(labels_true)
        l_pred_cnt += len(labels_pred)
    p = hit_cnt / l_pred_cnt
    r = hit_cnt / l_true_cnt
    return 2 * p * r / (p + r + 1e-7)


def macrof1(true_labels_dict, pred_labels_dict):
    assert len(true_labels_dict) == len(pred_labels_dict)
    p_acc, r_acc = 0, 0
    for mention_id, labels_true in true_labels_dict.items():
        labels_pred = pred_labels_dict[mention_id]
        match_cnt = count_match(labels_true, labels_pred)
        p_acc += match_cnt / len(labels_pred)
        r_acc += match_cnt / len(labels_true)
    p, r = p_acc / len(pred_labels_dict), r_acc / len(true_labels_dict)
    f1 = 2 * p * r / (p + r + 1e-7)
    return f1


def eval_fet_performance(true_labels_dict, pred_labels_dict):
    sacc = strict_acc(true_labels_dict, pred_labels_dict)
    maf1 = macrof1(true_labels_dict, pred_labels_dict)
    mif1 = microf1(true_labels_dict, pred_labels_dict)
    return sacc, maf1, mif1
