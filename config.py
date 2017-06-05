#coding=utf-8
FILE_PATH = ''

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3

BUCKETS = [(5, 7) ,(8, 10), (12, 14)]

NUM_LAYERS = 3
UNITS_NUM = 256
BATCH_SIZE = 64

LR = 0.1
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 4

TEV = './files/train_encode_vocabulary'
TDV = './files/train_decode_vocabulary'

TRAIN_ENC_VEC = './files/train_encode.vec'
TRAIN_DEC_VEC = './files/train_decode.vec'
TEST_ENC_VEC = './files/test_encode.vec'
TEST_DEC_VEC = './files/test_decode.vec'

LR_DECAY = 0.97

DEC_VOCAB_SIZE = 10000
ENC_VOCAB_SIZE = 10000

PERSONAL_FILTER = set([u'谁', '?', u'吗', u'什么'])


