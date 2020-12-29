from platform import platform
from os.path import join
import socket

MACHINE_NAME = socket.gethostname()
DATA_DIR = '/data/hldai/data/srfet-data'

LOG_DIR = join(DATA_DIR, 'log')

TOKEN_UNK = '<UNK>'
TOKEN_ZERO_PAD = '<ZPAD>'
TOKEN_EMPTY_PAD = '<EPAD>'
TOKEN_MENTION = '<MEN>'

FIGER_FILES = {
    'test-mentions': join(DATA_DIR, 'figer-afet/figer-dfet-test-mentions.json'),
    'test-sents': join(DATA_DIR, 'figer-afet/figer-dfet-test-sents.json'),
    'test-srl': join(DATA_DIR, 'figer-afet/figer-test-sents-srl.txt'),
    'test-sents-dep': join(DATA_DIR, 'figer-afet/figer-test-sents-tok-texts-dep.txt'),
    'test-pos-tags': join(DATA_DIR, 'figer-afet/figer-test-sents-tok-texts-pos.txt'),
    'presample-train-data-prefix': join(DATA_DIR, 'figer-afet/weakdata/enwiki20151002-anchor-figer-pre0_1'),
    # 'srl-train-data-prefix': join(DATA_DIR, 'figer-afet/weakdata/enwiki20151002-anchor-srl-pre0_1'),
    # 'vfet-train-data-prefix': join(DATA_DIR, 'figer-afet/weakdata/wiki20151002-anchor-srl16'),
    'srl-train-data-prefix': join(DATA_DIR, 'weakdata/wiki20151002-anchor-srl16'),
    'type-vocab': join(DATA_DIR, 'figer-afet/figer-type-vocab.txt'),
}

BBN_FILES = {
    'test-mentions': join(DATA_DIR, 'bbn-afet/bbn-dfet-test-mentions.json'),
    'test-sents': join(DATA_DIR, 'bbn-afet/bbn-dfet-test-sents.json'),
    'test-srl': join(DATA_DIR, 'bbn-afet/bbn-test-sents-srl.txt'),
    'test-sents-dep': join(DATA_DIR, 'bbn-afet/bbn-test-sents-tok-texts-dep.txt'),
    'test-pos-tags': join(DATA_DIR, 'bbn-afet/bbn-test-sents-tok-texts-pos.txt'),
    'train-mentions': join(DATA_DIR, 'bbn-afet/bbn-dfet-train-mentions-aug.json'),
    'train-sents': join(DATA_DIR, 'bbn-afet/bbn-dfet-train-sents.json'),
    'train-pos-tags': join(DATA_DIR, 'bbn-afet/bbn-dfet-train-sents-tok-texts-pos.txt'),
    'train-sents-dep': join(DATA_DIR, 'bbn-afet/bbn-dfet-train-sents-tok-texts-dep.txt'),
    'train-srl': join(DATA_DIR, 'bbn-afet/bbn-dfet-train-sents-srl.txt'),
    'srl-train-data-prefix': join(DATA_DIR, 'weakdata/wiki20151002-anchor-srl16-bbn'),
    'type-vocab': join(DATA_DIR, 'bbn-afet/bbn-type-vocab.txt'),
}
