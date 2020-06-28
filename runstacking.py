import torch
import datetime
import os
import argparse
from utils.fetutils import TypeInfer
from utils import fetutils
from utils.loggingutils import init_universal_logging
from exp import stackexp, expdata
import config


def __train():
    use_vr = True
    # use_vr = False
    use_hr = True
    # use_hr = False

    train_samples_file = os.path.join(config.DATA_DIR, 'figer/figer-dev-man-labeled.txt')
    train_base_preds_file = os.path.join(config.DATA_DIR, 'figer/ftbfet7069-results-figer-wikival.txt')
    train_srl_preds_file = os.path.join(config.DATA_DIR, 'figer/figer-train-srl1all-preds.txt')
    train_hyp_file = os.path.join(config.DATA_DIR, 'figer/figer-wikival-all-fmm-hypext.txt')
    train_verif_hyp_file = os.path.join(config.DATA_DIR, 'figer/figer-wikival-all-fmm-hypext-bbntyped.txt')
    train_hypext_logits_file = os.path.join(config.DATA_DIR,
                                            'figer/bert-tmhypextbbn-results-all-figer-wikival.txt')
    train_pred_file_tup = (train_base_preds_file, train_srl_preds_file, train_hyp_file, train_verif_hyp_file,
                           train_hypext_logits_file)
    gres = expdata.ResData(config.FIGER_FILES['type-vocab'])
    type_infer_train = TypeInfer(gres.type_vocab, gres.type_id_dict, single_type_path=False)
    _, _, child_type_vecs_train = fetutils.build_hierarchy_vecs(gres.type_vocab, gres.type_id_dict)
    train_samples = expdata.load_labeled_samples(gres.type_id_dict, child_type_vecs_train,
                                                 train_samples_file, train_pred_file_tup, use_vr, use_hr)
    print(len(train_samples), 'labeled samples')

    test_mentions_file = config.FIGER_FILES['test-mentions']
    test_sents_file = config.FIGER_FILES['test-sents']
    test_base_preds_file = os.path.join(config.DATA_DIR, 'figer/ftbfet7069-results-figer.txt')
    test_srl_preds_file = os.path.join(config.DATA_DIR, 'figer/figer-test-srl1-preds.txt')
    test_hyp_preds_file = os.path.join(config.DATA_DIR, 'figer/figer-test-all-fmm-hypext.txt')
    test_verif_hypext_file = os.path.join(config.DATA_DIR, 'figer/figer-test-all-fmm-hypext-bbntyped.txt')
    test_hypext_logits_file = os.path.join(config.DATA_DIR, 'figer/bert-tmhypextbbn-results-all-figer-test.txt')
    test_file_tup = (test_mentions_file, test_sents_file, test_base_preds_file, test_srl_preds_file,
                     test_hyp_preds_file, test_verif_hypext_file, test_hypext_logits_file)
    gres_test = expdata.ResData(config.FIGER_FILES['type-vocab'])

    l1_type_indices_test, l1_type_vec_test, child_type_vecs_test = fetutils.build_hierarchy_vecs(
        gres.type_vocab, gres.type_id_dict)
    type_infer_test = TypeInfer(gres_test.type_vocab, gres_test.type_id_dict, single_type_path=False)
    test_samples, test_true_labels_dict = expdata.samples_from_test(gres_test, child_type_vecs_test, test_file_tup)

    stackexp.train_stacking(device, gres, use_vr, use_hr, type_infer_train, train_samples,
                            gres_test, type_infer_test, test_samples, test_true_labels_dict)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')
    # log_file = os.path.join(config.LOG_DIR, '{}-{}-{}.log'.format(os.path.splitext(
    #     os.path.basename(__file__))[0], str_today, config.MACHINE_NAME))
    log_file = None

    parser = argparse.ArgumentParser(description='dhl')
    parser.add_argument('idx', type=int, default=0, nargs='?')
    parser.add_argument('-d', type=int, default=[], nargs='+')
    args = parser.parse_args()

    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')
    device_ids = args.d

    if args.idx == 0:
        __train()
