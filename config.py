#coding=utf-8
FILE_PATH = ''

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3

PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符

START_VOCABULART = [PAD, UNK, GO, EOS]


BUCKETS = [(5, 7) ,(8, 10), (12, 14)]

NUM_LAYERS = 5 
UNITS_NUM = 256
BATCH_SIZE = 64

LR = 0.1
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512

TRAIN_ENC_FILE = './data/train_encode.dat'
TRAIN_DEC_FILE = './data/train_decode.dat'
TEST_ENC_FILE = './data/test_encode.dat'
TEST_DEC_FILE = './data/test_decode.dat'

TRAIN_ENC_VEC = './data/train_encode.vec'
TRAIN_DEC_VEC = './data/train_decode.vec'
TEST_ENC_VEC = './data/test_encode.vec'
TEST_DEC_VEC = './data/test_decode.vec'

TRAIN_ENC_VOCABULARY = './data/train_encode_vocabulary'
TRAIN_DEC_VOCABULARY = './data/train_decode_vocabulary'

LR_DECAY = 0.97

DEC_VOCAB_SIZE = 66616
ENC_VOCAB_SIZE = 67624

PERSONAL_FILTER = set([u'谁', '?', u'吗', u'什么', u'？'])

YULIAO = '/Users/yixin/sunhao25/myself/nlp/dgk_shooter_min.conv'
CUT_WORDS = './data/words.dat'
VECTOR_BIN = '/data/vector.bin'
VOCABULARY_SIZE = 18185

DROPOUT = 0.8

QUES_DICT = {}


