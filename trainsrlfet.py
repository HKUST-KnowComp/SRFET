import os
import datetime
import torch
import logging
import argparse
from exp import srlfetexp, expdata
from utils.loggingutils import init_universal_logging
import config


def __train1():
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME))
    # log_file = None
    init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))

    train_config = srlfetexp.TrainConfig(loss_name='mm', neg_scale=0.1, n_steps=70000)

    lstm_dim = 250
    mlp_hidden_dim = 500
    type_embed_dim = 500
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    # dataset = 'figer'
    dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES

    data_prefix = datafiles['srl-train-data-prefix']
    dev_data_pkl = data_prefix + '-dev.pkl'
    train_data_pkl = data_prefix + '-train.pkl'

    test_file_tup = (datafiles['test-mentions'], datafiles['test-sents'],
                     datafiles['test-sents-dep'], datafiles['test-srl'])
    single_type_path = False if dataset == 'figer' else True

    # output_model_file = None
    save_model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl-{}'.format(dataset))

    gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
    logging.info('dataset={} {}'.format(dataset, data_prefix))
    srlfetexp.train_srlfet(device, gres, train_data_pkl, dev_data_pkl, test_file_tup, lstm_dim, mlp_hidden_dim,
                           type_embed_dim, train_config, single_type_path,
                           save_model_file_prefix=save_model_file_prefix)


def __train():
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], args.idx, str_today, config.MACHINE_NAME))
    # log_file = None
    init_universal_logging(log_file, mode='a', to_stdout=True)
    logging.info('logging to {}'.format(log_file))

    train_config = srlfetexp.TrainConfig(neg_scale=1.0, batch_size=128, schedule_lr=True, n_steps=80000)

    lstm_dim = 250
    mlp_hidden_dim = 500
    type_embed_dim = 500
    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES

    data_prefix = datafiles['srl-train-data-prefix']
    dev_data_pkl = data_prefix + '-dev.pkl'
    train_data_pkl = data_prefix + '-train.pkl'

    test_file_tup = (datafiles['test-mentions'], datafiles['test-sents'],
                     datafiles['test-sents-dep'], datafiles['test-srl'])
    single_type_path = False if dataset == 'figer' else True

    # output_model_file = None
    save_model_file_prefix = os.path.join(config.DATA_DIR, 'models/srl-{}'.format(dataset))

    gres = expdata.ResData(datafiles['type-vocab'], word_vecs_file)
    logging.info('dataset={} {}'.format(dataset, data_prefix))
    srlfetexp.train_srlfet(
        device, gres, train_data_pkl, dev_data_pkl, test_file_tup, lstm_dim, mlp_hidden_dim,
        type_embed_dim, train_config, single_type_path, save_model_file_prefix=save_model_file_prefix)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')

    parser = argparse.ArgumentParser(description='dhl')
    parser.add_argument('idx', type=int, default=0, nargs='?')
    parser.add_argument('-d', type=int, default=[], nargs='+')
    args = parser.parse_args()

    cuda_device_str = 'cuda' if len(args.d) == 0 else 'cuda:{}'.format(args.d[0])
    device = torch.device(cuda_device_str) if torch.cuda.device_count() > 0 else torch.device('cpu')

    if args.idx == 0:
        __train()
    if args.idx == 1:
        __train1()
