import logging
import torch
from typing import List
from models.srlfet import SRLFET
from exp import expdata, exputils
from utils import datautils, fetutils, utils


class TrainConfig:
    def __init__(self, learning_rate=0.001, batch_size=64, n_iter=100, loss_name='mm', pos_margin=1.0, neg_margin=1.0,
                 pos_scale=1.0, neg_scale=1.0, schedule_lr=False, n_steps=-1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.loss_name = loss_name
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.schedule_lr = schedule_lr
        self.n_steps = n_steps


def __split_samples_by_arg_idx(samples):
    samples_list = [list(), list(), list()]
    for sample in samples:
        mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids, srl_info = sample
        v_span, arg0_span, arg1_span, arg2_span, mention_arg_idx = srl_info
        samples_list[mention_arg_idx].append(sample)
    return samples_list


def __get_full_type_ids_of_samples(parent_type_ids_dict, samples):
    type_ids_list = list()
    for sample in samples:
        type_ids = fetutils.get_full_type_ids(sample[5], parent_type_ids_dict)
        type_ids_list.append(type_ids)
    return type_ids_list


def samples_from_txt(token_id_dict, unknown_token_id, type_id_dict, mentions_file, sents_file, dep_tags_file,
                     srl_results_file, use_all):
    mentions = datautils.read_json_objs(mentions_file)
    sents = datautils.read_json_objs(sents_file)
    dep_tag_seq_list = None
    if dep_tags_file is not None:
        with open(dep_tags_file, encoding='utf-8') as f:
            dep_tag_seq_list = [datautils.next_sent_dependency(f) for _ in range(len(sents))]
    srl_results_list = datautils.read_srl_results(srl_results_file)
    print(len(sents), len(srl_results_list))

    sent_dict = {sent['sent_id']: (i, sent) for i, sent in enumerate(sents)}
    samples = list()
    for m in mentions:
        mspan = m['span']
        sent_idx, sent = sent_dict[m['sent_id']]
        sent_tokens = sent['text'].split(' ')
        dep_tag_seq = dep_tag_seq_list[sent_idx] if dep_tag_seq_list is not None else None
        srl_results = srl_results_list[sent_idx]
        matched_tag_list, matched_tag_spans_list = utils.match_srl_to_mentions_all(
            sent_tokens, srl_results, mspan, dep_tag_seq)

        if not matched_tag_list:
            continue

        if not use_all:
            matched_tag_list = matched_tag_list[-1:]
            matched_tag_spans_list = matched_tag_spans_list[-1:]

        for matched_tag, matched_tag_spans in zip(matched_tag_list, matched_tag_spans_list):
            type_labels = m.get('labels', ['/PERSON'])
            type_ids = [type_id_dict[t] for t in type_labels]

            matched_tag_pos = int(matched_tag[-1:])
            srl_info = (utils.get_srl_tag_span(matched_tag_spans, 'V'),
                        utils.get_srl_tag_span(matched_tag_spans, 'ARG0'),
                        utils.get_srl_tag_span(matched_tag_spans, 'ARG1'),
                        utils.get_srl_tag_span(matched_tag_spans, 'ARG2'), matched_tag_pos)
            sent_token_ids = [token_id_dict.get(token, unknown_token_id) for token in sent_tokens]

            sample = (m['mention_id'], m['str'], mspan[0], mspan[1], None, type_ids, sent_token_ids, srl_info)
            samples.append(sample)
    return samples


def train_srlfet(device, gres: expdata.ResData, train_pkl, dev_pkl, test_file_tup, lstm_dim,
                 mlp_hidden_dim, type_embed_dim, train_config: TrainConfig, single_type_path,
                 save_model_file_prefix=None):
    train_samples = datautils.load_pickle_data(train_pkl)
    dev_samples = datautils.load_pickle_data(dev_pkl)
    print(len(train_samples))

    # loss_obj = exputils.BinMaxMarginLoss()
    loss_obj = exputils.BinMaxMarginLoss(pos_margin=train_config.pos_margin, neg_margin=train_config.neg_margin,
                                         pos_scale=train_config.pos_scale, neg_scale=train_config.neg_scale)

    batch_size = train_config.batch_size
    learning_rate = train_config.learning_rate
    n_iter = train_config.n_iter

    train_samples_list = __split_samples_by_arg_idx(train_samples)
    print([len(samples) for samples in train_samples_list], 'train samples')
    print([len(samples) // batch_size for samples in train_samples_list], 'batchs per iter')

    dev_samples_list = __split_samples_by_arg_idx(dev_samples)
    dev_sample_type_ids_list = __get_full_type_ids_of_samples(gres.parent_type_ids_dict, dev_samples)
    dev_true_labels_dict = {s[0]: [gres.type_vocab[tid] for tid in type_ids] for type_ids, s in zip(
        dev_sample_type_ids_list, dev_samples)}
    # dev_true_labels_dict = {s.mention_id: [gres.type_vocab[l] for l in s.labels] for s in dev_samples}
    print([len(samples) for samples in dev_samples_list], 'validation samples')
    logging.info(' '.join(['{}={}'.format(k, v) for k, v in vars(train_config).items()]))

    test_samples_list, test_true_labels_dict = None, None
    if test_file_tup is not None:
        mentions_file, sents_file, dep_tags_file, srl_results_file = test_file_tup
        all_test_samples = samples_from_txt(gres.token_id_dict, gres.unknown_token_id, gres.type_id_dict,
                                            mentions_file, sents_file, dep_tags_file, srl_results_file, use_all=True)
        test_samples_list = __split_samples_by_arg_idx(all_test_samples)
        test_sample_type_ids_list = __get_full_type_ids_of_samples(gres.parent_type_ids_dict, all_test_samples)
        test_true_labels_dict = {s[0]: [gres.type_vocab[tid] for tid in type_ids] for type_ids, s in zip(
            test_sample_type_ids_list, all_test_samples)}
        print([len(samples) for samples in test_samples_list], 'test samples')

    word_vec_dim = gres.token_vecs.shape[1]
    models, optimizers = list(), list()
    lr_schedulers = list() if train_config.schedule_lr else None
    for i in range(3):
        model = SRLFET(device, gres.type_vocab, gres.type_id_dict, word_vec_dim, lstm_dim, mlp_hidden_dim,
                       type_embed_dim)
        if device.type == 'cuda':
            model = model.cuda(device.index)
        models.append(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizers.append(optimizer)
        if lr_schedulers is not None:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)
            lr_schedulers.append(lr_scheduler)
    print('start training ...')
    losses = list()
    best_dev_perf = -1
    n_steps_per_iter = len(train_samples) // batch_size
    n_steps = n_iter * n_steps_per_iter
    if train_config.n_steps > -1:
        n_steps = train_config.n_steps
    for i in range(n_steps):
        # print(i)
        for mention_arg_idx, samples in enumerate(train_samples_list):
            model, optimizer = models[mention_arg_idx], optimizers[mention_arg_idx]
            model.train()
            lr_scheduler = None if lr_schedulers is None else lr_schedulers[mention_arg_idx]
            loss_val = __train_step(model, gres.parent_type_ids_dict, gres.token_vecs, samples, mention_arg_idx,
                                    i, batch_size, loss_obj, optimizer, lr_scheduler)
            losses.append(loss_val)

        if (i + 1) % 1000 == 0:
            acc_v, maf1, mif1, _ = __eval(gres, models, dev_samples_list, dev_true_labels_dict)
            # logging.info('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(i + 1, sum(losses), acc_v, maf1, mif1))
            # losses = list()

            if test_samples_list is not None:
                acc_t, maf1_t, mif1_t, _ = __eval(gres, models, test_samples_list, test_true_labels_dict,
                                                  single_type_path=single_type_path)
                # print(i + 1, sum(losses), acc, maf1, mif1)
                logging.info('{} {:.4f} {:.4f} {:.4f} {:.4f} acct={:.4f} maf1t={:.4f} mif1t={:.4f}'.format(
                    i + 1, sum(losses), acc_v, maf1, mif1, acc_t, maf1_t, mif1_t))
            else:
                logging.info('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(i + 1, sum(losses), acc_v, maf1, mif1))
            losses = list()

            if acc_v > best_dev_perf and save_model_file_prefix:
                __save_srl_models(models, save_model_file_prefix)
                best_dev_perf = acc_v


def __get_one_hot_type_vecs(parent_type_ids_dict, samples):
    true_type_vecs = list()
    n_types = len(parent_type_ids_dict)
    type_ids_list = __get_full_type_ids_of_samples(parent_type_ids_dict, samples)
    for type_ids in type_ids_list:
        true_type_vecs.append(exputils.onehot_encode(type_ids, n_types))
    return true_type_vecs


def __train_step(model: SRLFET, parent_type_ids_dict, token_vecs, samples, mention_arg_idx, step, batch_size,
                 loss_obj, optimizer, lr_scheduler):
    device = model.device
    n_batches = len(samples) // batch_size
    step %= n_batches
    samples_batch = samples[step * batch_size:(step + 1) * batch_size]

    mstr_vec_seqs, verb_vec_seqs, arg1_vec_seqs, arg2_vec_seqs = __get_sample_batch_srl_inputs(
        device, model.n_types, token_vecs, samples_batch, mention_arg_idx)
    true_type_vecs = __get_one_hot_type_vecs(parent_type_ids_dict, samples_batch)
    logits = model(mstr_vec_seqs, verb_vec_seqs, arg1_vec_seqs, arg2_vec_seqs)

    true_type_vecs = torch.tensor(true_type_vecs, dtype=torch.float32, device=device)
    # loss = model.get_loss(true_type_vecs, logits, person_loss_vec=None)
    loss = loss_obj.loss(true_type_vecs, logits)
    # loss = model.get_cali_loss(true_type_vecs, logits)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, float('inf'))
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return loss.data.cpu().numpy()


def __eval(gres: expdata.ResData, models: List[SRLFET], samples_list, true_labels_dict, batch_size=16,
           single_type_path=False):
    pred_labels_dict = dict()
    result_objs = list()
    for mention_arg_idx, samples in enumerate(samples_list):
        model = models[mention_arg_idx]
        model.eval()
        device = model.device
        n_batches = (len(samples) + batch_size - 1) // batch_size
        # print('{} batches'.format(n_batches))
        for i in range(n_batches):
            batch_beg, batch_end = i * batch_size, min((i + 1) * batch_size, len(samples))
            samples_batch = samples[batch_beg:batch_end]

            mstr_vec_seqs, verb_vec_seqs, arg1_vec_seqs, arg2_vec_seqs = __get_sample_batch_srl_inputs(
                device, model.n_types, gres.token_vecs, samples_batch, mention_arg_idx)
            with torch.no_grad():
                logits = model(mstr_vec_seqs, verb_vec_seqs, arg1_vec_seqs, arg2_vec_seqs)
                # loss = model.get_loss(true_type_vecs, logits)
            # losses.append(loss)

            if single_type_path:
                preds = model.inference(logits)
            else:
                preds = model.inference_full(logits, extra_label_thres=0.0)
            for j, (sample, type_ids_pred, sample_logits) in enumerate(
                    zip(samples_batch, preds, logits.data.cpu().numpy())):
                labels = fetutils.get_full_types([gres.type_vocab[tid] for tid in type_ids_pred])
                pred_labels_dict[sample[0]] = labels
                result_objs.append({'mention_id': sample[0], 'labels': labels,
                                    'logits': [float(v) for v in sample_logits]})

    strict_acc = fetutils.strict_acc(true_labels_dict, pred_labels_dict)
    # partial_acc = utils.partial_acc(true_labels_dict, pred_labels_dict)
    maf1 = fetutils.macrof1(true_labels_dict, pred_labels_dict)
    mif1 = fetutils.microf1(true_labels_dict, pred_labels_dict)
    return strict_acc, maf1, mif1, result_objs


def __save_srl_models(models, file_prefix):
    for i, model in enumerate(models):
        filepath = '{}-{}.pth'.format(file_prefix, i)
        torch.save(model.state_dict(), filepath)
    logging.info('model saved to {}'.format(file_prefix))


def load_train_models(device, gres, model_file_prefix):
    word_vec_dim = gres.token_vecs.shape[1]
    lstm_dim, mlp_hidden_dim, type_embed_dim = 0, 0, 0
    models = list()
    for i in range(3):
        model_params_file = '{}-{}.pth'.format(model_file_prefix, i)
        logging.info('load model from {}'.format(model_params_file))
        trained_params = torch.load(model_params_file)
        lstm_dim = trained_params['lstm1.weight_hh_l0'].size()[1]
        mlp_hidden_dim = trained_params['mlp.linear_map1.weight'].size()[0]
        type_embed_dim = trained_params['type_embeddings'].size()[0]
        model = SRLFET(device, gres.type_vocab, gres.type_id_dict, word_vec_dim, lstm_dim, mlp_hidden_dim,
                       type_embed_dim)
        model.load_state_dict(trained_params)
        if device.type == 'cuda':
            model = model.cuda(device.index)
        models.append(model)
    print('lstm_dim={} mlp_hidden_dim={} type_embed_dim={}'.format(lstm_dim, mlp_hidden_dim, type_embed_dim))
    return models


def eval_trained(device, gres: expdata.ResData, model_file_prefix, mentions_file, sents_file, srl_results_file,
                 dep_tags_file, single_type_path, output_preds_file):
    models = load_train_models(device, gres, model_file_prefix)

    all_samples = samples_from_txt(gres.token_id_dict, gres.unknown_token_id, gres.type_id_dict,
                                   mentions_file, sents_file, dep_tags_file, srl_results_file, use_all=True)
    samples_list = __split_samples_by_arg_idx(all_samples)
    sample_type_ids_list = __get_full_type_ids_of_samples(gres.parent_type_ids_dict, all_samples)
    true_labels_dict = {s[0]: [gres.type_vocab[tid] for tid in type_ids] for type_ids, s in zip(
        sample_type_ids_list, all_samples)}
    print([len(samples) for samples in samples_list], 'samples')

    acc, maf1, mif1, result_objs = __eval(gres, models, samples_list, true_labels_dict,
                                          single_type_path=single_type_path)
    print(acc, maf1, mif1)

    datautils.save_json_objs(result_objs, output_preds_file)
    print('results saved to {}'.format(output_preds_file))


def __get_sample_batch_srl_inputs(device, n_types, token_vecs, samples, mention_arg_idx):
    token_vec_dim = token_vecs.shape[1]
    # true_type_vecs = list()
    mstr_vec_seqs, verb_vec_seqs, arg1_vec_seqs, arg2_vec_seqs = list(), list(), list(), list()
    arg1_idx = 0 if mention_arg_idx > 0 else 1
    arg2_idx = 1 if mention_arg_idx == 2 else 2
    for i, sample in enumerate(samples):
        mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids, srl_info = sample
        v_span, srl_arg0_span, srl_arg1_span, srl_arg2_span, mention_arg_idx = srl_info

        # true_type_vecs.append(utils.onehot_encode(type_ids, n_types))

        mstr_vec_seqs.append(exputils.get_torch_vec_seq(device, sent_token_ids[pos_beg:pos_end], token_vecs))
        verb_vec_seqs.append(exputils.get_torch_vec_seq(device, sent_token_ids[v_span[0]:v_span[1]], token_vecs))

        srl_arg_spans = (srl_arg0_span, srl_arg1_span, srl_arg2_span)
        arg1_span, arg2_span = srl_arg_spans[arg1_idx], srl_arg_spans[arg2_idx]
        if arg1_span is None:
            arg1_vec_seqs.append(torch.zeros(size=(1, token_vec_dim), dtype=torch.float32, device=device))
        else:
            arg1_vec_seqs.append(exputils.get_torch_vec_seq(
                device, sent_token_ids[arg1_span[0]:arg1_span[1]], token_vecs))
        if arg2_span is None:
            arg2_vec_seqs.append(torch.zeros(size=(1, token_vec_dim), dtype=torch.float32, device=device))
        else:
            arg2_vec_seqs.append(exputils.get_torch_vec_seq(
                device, sent_token_ids[arg2_span[0]:arg2_span[1]], token_vecs))
    return mstr_vec_seqs, verb_vec_seqs, arg1_vec_seqs, arg2_vec_seqs
