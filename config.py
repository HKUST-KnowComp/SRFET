from platform import platform
from os.path import join
import socket

MACHINE_NAME = socket.gethostname()
DATA_DIR = '/home/data/hldai/fet/srfet-data'

LOG_DIR = join(DATA_DIR, 'log')

FIGER_FILES = {
    'test-mentions': join(DATA_DIR, 'figer-afet/figer-dfet-test-mentions.json'),
    'test-sents': join(DATA_DIR, 'figer-afet/figer-dfet-test-sents.json'),
    'test-srl': join(DATA_DIR, 'figer-afet/figer-test-sents-srl.txt'),
    'test-sents-dep': join(DATA_DIR, 'figer-afet/figer-test-sents-tok-texts-dep.txt'),
    'test-pos-tags': join(DATA_DIR, 'figer-afet/figer-test-sents-tok-texts-pos.txt'),
    'presample-train-data-prefix': join(DATA_DIR, 'figer-afet/weakdata/enwiki20151002-anchor-figer-pre0_1'),
    'srl-train-data-prefix': join(DATA_DIR, 'figer-afet/weakdata/enwiki20151002-anchor-srl-pre0_1'),
    'vfet-train-data-prefix': join(DATA_DIR, 'figer-afet/weakdata/wiki20151002-anchor-srl16'),
    'type-vocab': join(DATA_DIR, 'figer-afet/figer-type-vocab.txt'),
}
