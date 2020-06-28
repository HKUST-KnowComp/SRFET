import random
import os
import torch
import numpy as np
import torch.nn.functional as F
from exp import expdata, exputils
from models import mlp
from utils import fetutils


def train_stacking(device, gres: expdata.ResData, use_vr, use_hr, type_infer_train, all_train_samples,
                   gres_test: expdata.ResData, type_infer_test, test_samples, test_true_labels_dict):
    n_runs = 5
    n_iter = 200
    n_dev = 100
    batch_size = 32
    n_mlp_layers = 2
    mlp_hdim = 5
    learning_rate = 0.01
    lr_gamma = 0.9
    margin = 0.5
    dropout = 0.5
    n_labelers = 3
    if not use_vr:
        n_labelers -= 1
    if not use_hr:
        n_labelers -= 1

    print('{} train samples'.format(len(all_train_samples)))

    test_acc_list, test_maf1_list, test_mif1_list = list(), list(), list()
    for i in range(n_runs):
        print('run', i)
        random.shuffle(all_train_samples)
        train_samples = all_train_samples
        dev_samples, dev_true_labels_dict = None, None
        if n_dev > 0:
            dev_samples = all_train_samples[:n_dev]
            dev_true_labels_dict = {s[0]: fetutils.get_full_types(
                gres.type_vocab[tid] for tid in s[-1]) for s in dev_samples}
            train_samples = all_train_samples[n_dev:]

        test_acc, test_maf1, test_mif1 = __do_train(
            device, train_samples, dev_samples, dev_true_labels_dict, test_samples, test_true_labels_dict,
            gres.type_vocab, gres_test.type_vocab, type_infer_train, type_infer_test, use_vr, use_hr, n_labelers,
            n_mlp_layers, mlp_hdim, dropout, margin, learning_rate, batch_size, n_iter, lr_gamma)
        test_acc_list.append(test_acc)
        test_maf1_list.append(test_maf1)
        test_mif1_list.append(test_mif1)
    (avg_acc, avg_maf1, avg_mif1
     ) = sum(test_acc_list) / n_runs, sum(test_maf1_list) / n_runs, sum(test_mif1_list) / n_runs
    print('acc', ' '.join(['{:.4f}'.format(v) for v in test_acc_list]), '*', '{:.4f}'.format(avg_acc))
    print('maf1', ' '.join(['{:.4f}'.format(v) for v in test_maf1_list]), '*', '{:.4f}'.format(avg_maf1))
    print('mif1', ' '.join(['{:.4f}'.format(v) for v in test_mif1_list]), '*', '{:.4f}'.format(avg_mif1))


def __do_train(device, train_samples, dev_samples, dev_true_labels_dict, test_samples,
               test_true_labels_dict, type_vocab_train, type_vocab_test, type_infer_train, type_infer_test,
               use_vr, use_hr, n_labelers, n_mlp_layers, mlp_hdim, dropout, margin, learning_rate, batch_size,
               n_iter, lr_gamma):
    mlp_input_dim = 2
    if use_vr:
        mlp_input_dim += 2
    if use_hr:
        mlp_input_dim += 1

    model = mlp.MLP(n_mlp_layers, mlp_input_dim, n_labelers, mlp_hdim, dropout=dropout)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'lin1_bn', 'lin2_bn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    loss_obj = exputils.BinMaxMarginLoss(pos_margin=margin, neg_margin=margin)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)

    n_batches = (len(train_samples) + batch_size - 1) // batch_size
    if len(train_samples) == batch_size * (n_batches - 1) + 1:
        n_batches -= 1

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_batches * 10, gamma=lr_gamma)

    n_steps = n_batches * n_iter
    losses = list()
    n_types_train = len(type_vocab_train)
    best_dev_loss, best_dev_acc = 1e5, 0
    best_test_acc, best_test_maf1, best_test_mif1 = 0, 0, 0
    for step in range(n_steps):
        bidx = step % n_batches
        bbeg, bend = bidx * batch_size, min((bidx + 1) * batch_size, len(train_samples))
        batch = train_samples[bbeg:bend]
        cur_batch_size = bend - bbeg

        (pred_logits_tensor, max_logits_tensor, true_label_vecs
         ) = __get_batch_input(device, n_types_train, batch, use_vr, use_hr)
        feats = max_logits_tensor

        model.train()
        ens_logits = model(feats)
        final_logits = ens_labeler_logits(ens_logits, pred_logits_tensor, cur_batch_size)
        # print(ens_logits)
        loss = loss_obj.loss(true_label_vecs, final_logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.data.cpu().numpy())

        if (step + 1) % n_batches == 0:
            acc, loss_dev = 0, 0
            if dev_samples is not None:
                acc, maf1, mif1, _, loss_dev, avg_weights = __eval_ens(
                    device, loss_obj, type_vocab_train, model, type_infer_train, use_vr, use_hr,
                    dev_samples, dev_true_labels_dict)
                best_tag = '*' if acc > best_dev_acc or (acc == best_dev_acc and loss_dev < best_dev_loss) else ''
                # print((step + 1) // n_batches, sum(losses), loss_dev, acc, maf1, mif1, avg_weights, best_tag)
                print('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}'.format(
                    (step + 1) // n_batches, sum(losses), loss_dev, acc, maf1, mif1, avg_weights, best_tag))
            if acc > best_dev_acc or (acc == best_dev_acc and loss_dev < best_dev_loss):
                acct, maf1t, mif1t, result_objs, _, _ = __eval_ens(
                    device, None, type_vocab_test, model, type_infer_test, use_vr, use_hr,
                    test_samples, test_true_labels_dict)
                print('TEST {} {} ACC={:.4f} {:.4f} {:.4f}'.format(
                    (step + 1) // n_batches, sum(losses), acct, maf1t, mif1t))

                best_test_acc, best_test_maf1, best_test_mif1 = acct, maf1t, mif1t
                best_dev_acc = acc
                best_dev_loss = loss_dev
            losses = list()
            # random.shuffle(train_samples)
    print('test acc={:.4f} maf1={:.4f} mif1={:.4f}'.format(best_test_acc, best_test_maf1, best_test_mif1))
    return best_test_acc, best_test_maf1, best_test_mif1


def __get_bert_reps(bert_model, bert_token_id_seqs):
    with torch.no_grad():
        bert_hidden, _ = bert_model(bert_token_id_seqs, output_all_encoded_layers=False)
        bert_hidden = bert_hidden[:, 0, :]
    return bert_hidden


def ens_labeler_logits(ens_logits, labeler_logits, batch_size):
    weights = F.softmax(ens_logits, dim=1)
    output_logits = torch.bmm(weights.view(batch_size, 1, -1), labeler_logits)
    return output_logits.view((batch_size, -1))


def __get_top_two_scores(scores):
    idx1, idx2 = -1, -1
    max1, max2 = -1e10, -1e10
    for i, s in enumerate(scores):
        if s > max1:
            idx2 = idx1
            idx1 = i
            max2 = max1
            max1 = s
        elif s > max2:
            max2 = s
            idx2 = i
    return idx1, idx2, max1, max2


def __get_batch_input(device, n_types, samples, use_srl, use_hd):
    pred_logits_list = list()
    max_logits_list = list()
    true_label_vecs = list()
    for i, sample in enumerate(samples):
        (mention_id, base_logits, srl_logits, hyp_logits, hyp_verif_logit, label_ids) = sample

        true_label_vecs.append(exputils.onehot_encode(label_ids, n_types))

        cur_logits = [base_logits]
        if use_srl:
            cur_logits.append(srl_logits)
        if use_hd:
            cur_logits.append(hyp_logits)
        pred_logits_list.append(cur_logits)

        real_logits_list = [base_logits]
        if use_srl:
            real_logits_list.append(srl_logits)
        max_logits = list()
        for logits in real_logits_list:
            idx1, idx2, max1, max2 = __get_top_two_scores(logits)
            max_logits += [max1, max2]
        if use_hd:
            max_logits.append(hyp_verif_logit)
        max_logits_list.append(max_logits)

    max_logits_tensor = torch.tensor(max_logits_list, dtype=torch.float32, device=device)
    pred_logits_tensor = torch.tensor(pred_logits_list, dtype=torch.float32, device=device)
    true_label_vecs = torch.tensor(true_label_vecs, dtype=torch.float32, device=device)
    return pred_logits_tensor, max_logits_tensor, true_label_vecs


def __eval_ens(device, loss_obj, type_vocab, model, type_infer: fetutils.TypeInfer, use_vr, use_hr,
               samples, true_labels_dict):
    batch_size = 16
    n_types = len(type_vocab)
    model.eval()
    pred_labels_dict = dict()
    result_objs = list()
    n_steps = (len(samples) + batch_size - 1) // batch_size
    n_weights = 3
    if not use_vr:
        n_weights -= 1
    if not use_hr:
        n_weights -= 1
    weight_sums = np.zeros(n_weights, np.float32)
    losses = list()
    for step in range(n_steps):
        bbeg, bend = step * batch_size, min((step + 1) * batch_size, len(samples))
        samples_batch = samples[bbeg:bend]
        cur_batch_size = bend - bbeg
        (pred_logits_tensor, max_logits_tensor, true_label_vecs
         ) = __get_batch_input(device, n_types, samples_batch, use_vr, use_hr)
        feats = max_logits_tensor

        with torch.no_grad():
            ens_logits = model(feats)

        weights = torch.nn.functional.softmax(ens_logits, dim=1)
        weights = weights.data.cpu().numpy()
        weight_sums += np.sum(weights, axis=0)
        # ens_logits = torch.tensor([[1, 0.01, 0.01] for _ in range(cur_batch_size)], device=model.device,
        #                           dtype=torch.float32)

        final_logits = ens_labeler_logits(ens_logits, pred_logits_tensor, cur_batch_size)
        if loss_obj is not None:
            loss = loss_obj.loss(true_label_vecs, final_logits)
            losses.append(loss.data.cpu().numpy())

        preds = type_infer.inference(final_logits)
        for j, (sample, type_ids_pred, sample_logits) in enumerate(
                zip(samples_batch, preds, final_logits.data.cpu().numpy())):
            labels = fetutils.get_full_types([type_vocab[tid] for tid in type_ids_pred])
            pred_labels_dict[sample[0]] = labels
            result_objs.append({'mention_id': sample[0], 'labels': labels,
                                'logits': [float(v) for v in sample_logits]})

    strict_acc, maf1, mif1 = fetutils.eval_fet_performance(true_labels_dict, pred_labels_dict)
    return strict_acc, maf1, mif1, result_objs, sum(losses), weight_sums / len(samples)
