import os
import datetime
import torch
import logging
import argparse
from exp import srlfetexp, expdata
from utils.loggingutils import init_universal_logging
import config


def __eval2():
    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE
    model_file_prefix = os.path.join(config.DATA_DIR, 'models/pretrained-srl-{}'.format(dataset))
    # sub_set = 'test'
    sub_set = 'train'

    if sub_set == 'test':
        mentions_file = datafiles['test-mentions']
        sents_file = datafiles['test-sents']
        srl_file = datafiles['test-srl']
        dep_file = datafiles['test-sents-dep']
    else:
        if dataset == 'bbn':
            mentions_file = datafiles['train-mentions']
            sents_file = datafiles['train-sents']
            srl_file = datafiles['train-srl']
            dep_file = datafiles['train-sents-dep']
        else:
            mentions_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-mentions.json')
            sents_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-sents.json')
            srl_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-srl.txt')
            dep_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-tok-dep.txt')

    output_preds_file = os.path.join(
        config.DATA_DIR, '{}/{}-{}-pretrained-srl-preds.txt'.format(dataset, dataset, sub_set))
    single_type_path = True if dataset == 'bbn' else False

    gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
    srlfetexp.eval_trained(device, gres, model_file_prefix, mentions_file, sents_file, srl_file, dep_file,
                           single_type_path, output_preds_file)


def __eval1():
    # dataset = 'figer'
    dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE
    model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl-{}'.format(dataset))
    # sub_set = 'test'
    # sub_set = 'train'
    sub_sets = ['test', 'train']
    for sub_set in sub_sets:
        if sub_set == 'test':
            mentions_file = datafiles['test-mentions']
            sents_file = datafiles['test-sents']
            srl_file = datafiles['test-srl']
            dep_file = datafiles['test-sents-dep']
        else:
            if dataset == 'bbn':
                mentions_file = datafiles['train-mentions']
                sents_file = datafiles['train-sents']
                srl_file = datafiles['train-srl']
                dep_file = datafiles['train-sents-dep']
            else:
                mentions_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-mentions.json')
                sents_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-sents.json')
                srl_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-srl.txt')
                dep_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-tok-dep.txt')

        output_preds_file = os.path.join(
            config.DATA_DIR, '{}/{}-{}-srl-preds.txt'.format(dataset, dataset, sub_set))
        single_type_path = True if dataset == 'bbn' else False

        gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
        srlfetexp.eval_trained(device, gres, model_file_prefix, mentions_file, sents_file, srl_file, dep_file,
                               single_type_path, output_preds_file)


def __eval():
    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE
    model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl-{}'.format(dataset))
    # sub_set = 'test'
    # sub_set = 'train'
    sub_sets = ['test', 'train']
    for sub_set in sub_sets:
        if sub_set == 'test':
            mentions_file = datafiles['test-mentions']
            sents_file = datafiles['test-sents']
            srl_file = datafiles['test-srl']
            dep_file = datafiles['test-sents-dep']
        else:
            if dataset == 'bbn':
                mentions_file = datafiles['train-mentions']
                sents_file = datafiles['train-sents']
                srl_file = datafiles['train-srl']
                dep_file = datafiles['train-sents-dep']
            else:
                mentions_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-mentions.json')
                sents_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-sents.json')
                srl_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-srl.txt')
                dep_file = os.path.join(config.DATA_DIR, 'figer/wiki-valcands-figer-tok-dep.txt')

        output_preds_file = os.path.join(
            config.DATA_DIR, '{}/{}-{}-srl-preds.txt'.format(dataset, dataset, sub_set))
        single_type_path = True if dataset == 'bbn' else False

        gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
        srlfetexp.eval_trained(device, gres, model_file_prefix, mentions_file, sents_file, srl_file, dep_file,
                               single_type_path, output_preds_file)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')
    # log_file = os.path.join(config.LOG_DIR, '{}-{}-{}.log'.format(os.path.splitext(
    #     os.path.basename(__file__))[0], str_today, config.MACHINE_NAME))
    log_file = None
    init_universal_logging(log_file, mode='a', to_stdout=True)

    parser = argparse.ArgumentParser(description='dhl')
    parser.add_argument('idx', type=int, default=0, nargs='?')
    parser.add_argument('-d', type=int, default=[], nargs='+')
    args = parser.parse_args()

    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')

    if args.idx == 0:
        __eval()
    if args.idx == 1:
        __eval1()
    if args.idx == 2:
        __eval2()
