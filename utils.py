#coding=utf-8
import sys
import random
import codecs
import os
import thulac
import struct
import numpy as np
import config
import collections
import json

#data format origin
#wget https://raw.githubusercontent.com/rustch3n/dgk_lost_conv/master/dgk_shooter_min.conv.zip

"""
E
M 畹/华/吾/侄/
M 你/接/到/这/封/信/的/时/候/
M 不/知/道/大/伯/还/在/不/在/人/世/了/
E
M 您/多/打/我/几/下/不/就/得/了/吗/
M 走/
M 这/是/哪/一/出/啊/…/ / /这/是/
M 撕/破/一/点/就/弄/死/你/
M 唉/
M 记/着/唱/戏/的/再/红/
M 还/是/让/人/瞧/不/起/
M 大/伯/不/想/让/你/挨/了/打/
M 还/得/跟/人/家/说/打/得/好/
M 大/伯/不/想/让/你/再/戴/上/那/纸/枷/锁/
M 畹/华/开/开/门/哪/
E
...
"""

#format data, each E-M pair is a dialog, drop some sentenses to make sure that each dialog sentense is even
#use thulac to cut words
def format_data(file_name, output_file):
    if not os.path.exists(file_name):
        print '数据不存在'
        return False

    out = open(output_file,'w')
    sep_word = thulac.thulac(seg_only=True)

    with codecs.open(file_name, 'r', encoding='utf8') as f:
        seqs = []
        for line in f:
            line = line.strip().replace('/','')

            if line == '':
                continue

            if line[0] == 'E':
                start = len(seqs) % 2
                for word in seqs[start:]:
                    out.write(word + '\n')
                seqs = []

            elif line[0] == 'M':
                ans = line.split(' ')

                if len(ans) == 2:
                    try:
                        j_seq = sep_word.cut(ans[1].encode('utf8'), text=True)
                    except:
                        continue
                    #print j_seq, ans[1].encode('utf8')
                    # j_seq = ''
                    # for v in aseq:
                    #     j_seq += " " + v
                    j_seq = j_seq.strip()
                    if j_seq != '':
                        seqs.append(j_seq)
                
                                            

    for word in seqs[start:]:
        out.write(word.encode('utf8') + '\n')

    out.close()


#load vector bin as a dict
def load_vector_bin(input_file):

    print "begin load vectors"

    input_file = open(input_file, "rb")
    # get words num and dim
    words, dim = input_file.readline().strip().split(' ')
    print "words =", words
    print "dim =", dim

    word_set = {}
    for i in range(0, int(words)):
        word = ''
        # read a word
        while True:
            read_w = input_file.read(1)
            if read_w == False or read_w == ' ':
                break
            word += read_w
        word = word.strip()

        vector = []
        for j in range(int(dim)):
            w_dim = input_file.read(4)
            (weight,) = struct.unpack('f',w_dim)
            vector.append(weight)
        #print word, vector
        word_set[word] = vector

    input_file.close()
    return word_set


# 生成词汇表文件
def gen_vocabulary_file(input_file, output_file, freq = None):
    vocabulary = {}
    with open(input_file) as f:
        counter = 0
        for line in f:
            counter += 1
            tokens = [word for word in line.strip().split(' ')]
            for word in tokens:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1

        words = sorted(vocabulary, key=vocabulary.get, reverse=True)

        vocabulary_size = config.VOCABULARY_SIZE
        #如果设置了freq，则按freq的次数来筛选词表size 否则选配置的size
        if freq is not None:
            for i in range(len(words) - 1, -1, -1):
                if vocabulary[words[i]] < freq:
                    continue
                else:
                    vocabulary_size = max(i + 1, config.VOCABULARY_SIZE)
                    break

        # 取前n个常用汉字, 应该差不多够用了(额, 好多无用字符, 最好整理一下. 我就不整理了)
        vocabulary_list = config.START_VOCABULART + words[:vocabulary_size-4]

        print(input_file + " 词汇表大小:", len(vocabulary_list))
        with open(output_file, "w") as voca_f:
            for word in vocabulary_list:
                voca_f.write(word + "\n")


def make_train_test(input_file, seed=0):
    random.seed(0)
    flag = 0
    test_ques = open(config.TEST_ENC_FILE,'w')
    test_ans = open(config.TEST_DEC_FILE,'w')
    train_ques = open(config.TRAIN_ENC_FILE,'w')
    train_ans = open(config.TRAIN_DEC_FILE,'w')

    f = open(input_file, 'r')
    cnt = 0
    while True:
        line_ques = f.readline().strip()

        cnt += 1
        if not line_ques:
            break
        line_ans = f.readline().strip()
        cnt += 1

        if random.randint(1,10) < 2:
            test_ques.write(line_ques + '\n')
            test_ans.write(line_ans + '\n')
        else:
            train_ques.write(line_ques + '\n')
            train_ans.write(line_ans + '\n')

    f.close()
    test_ques.close()
    test_ans.close()
    train_ques.close()
    train_ans.close()


#load file and convert to vec
def convert_to_vec(input_file, vocabulary_file, output_file):
    tmp_vocab = []
    with open(vocabulary_file, "r") as f:
        tmp_vocab.extend(f.readlines())

    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])

    output_f = open(output_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            line_vec = []
            words = line.strip().split(' ')
            for word in words:
                line_vec.append(vocab.get(word, config.UNK_ID))
            output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
    output_f.close()


def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab

def lcb(target, source):
    #longest common substring
    if len(source) < 1 or len(target) < 1:
        return None

    c = [[0 for i in range(len(target))] for j in range(len(source))]

    for i in range(len(source)):
        c[i][0] = 1 if source[i] == target[0] else 0
    
    for j in range(len(target)):
        c[0][j] = 1 if source[0] == target[j] else 0

    max_ = 0
    index = 0

    #print source, target, len(source), len(target), c

    for i in range(1, len(source)):
        for j in range(1, len(target)):

            c[i][j] = (c[i-1][j-1] + 1) if source[i] == target[j] else 0
            if max_ < c[i][j]:
                max_ = c[i][j]
                index = j

    return max_, target[(index + 1 - max_):]

def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab

def check_pre_ques(ques, id='9527'):
    if config.QUES_DICT.get(id) is None:
        if all([word not in  ques for word in config.PERSONAL_FILTER ]):
            config.QUES_DICT[id] = collections.OrderedDict([('1_','_'), ('2_', '_'), ('3_','_'), ('4_','_'), (ques, 1 )])
        return None
    else:
        if config.QUES_DICT[id].get(ques) > 2 and random.randint(1,4) > 1:
            return random.choice([u'为什么重复这句话？？？', u'我刚刚的回答有问题吗？😈', u'需要我帮你Google一波？'])

        max_ = 0
        ans = ''

        #make sure that we only save 5 personal ques and filter some key words
        if all([word not in  ques for word in config.PERSONAL_FILTER ]):
            if config.QUES_DICT[id].get(ques) is None:
                config.QUES_DICT[id][ques] = 1 
                config.QUES_DICT[id].pop(config.QUES_DICT[id].keys()[0])
            else:
                config.QUES_DICT[id][ques] += 1
            


        for k, v in config.QUES_DICT[id].items():
            if k not in ques and ques not in k:
                lcb_size, target = lcb(k, ques)
                if max_ <= lcb_size:
                    max_ = lcb_size
                    ans = target
        
        if u'我' in ans and max_ >= 2:
            return ans.replace(u'我', u'你')
        elif max_ >= 4:
            return ans.replace(u'我', u'你')
        else:
            return None

def filter_response(ans):
    size = len(ans.decode('utf-8'))
    if ans.startswith('我是'):
        if ans.startswith('我是__UNK__') or size < 6:
            ans = '我是孙小号😁'
    elif ans.startswith('我叫'):
        ans = '我叫克劳德·斯特莱夫'
    else:
        ans = ans.replace('__UNK__', '😜')
    return ans

if __name__ == '__main__':

    #format_data(config.YULIAO, config.CUT_WORDS)
    
    make_train_test(config.CUT_WORDS)

    gen_vocabulary_file(config.TRAIN_ENC_FILE, config.TRAIN_ENC_VOCABULARY, 2)
    gen_vocabulary_file(config.TRAIN_DEC_FILE, config.TRAIN_DEC_VOCABULARY, 2)
    
    train_ques_vec = convert_to_vec(config.TRAIN_ENC_FILE, config.TRAIN_ENC_VOCABULARY, config.TRAIN_ENC_VEC)
    train_ans_vec = convert_to_vec(config.TRAIN_DEC_FILE, config.TRAIN_DEC_VOCABULARY, config.TRAIN_DEC_VEC)

    test_ques_vec = convert_to_vec(config.TEST_ENC_FILE, config.TRAIN_ENC_VOCABULARY, config.TEST_ENC_VEC)
    
    test_ans_vec = convert_to_vec(config.TEST_DEC_FILE, config.TRAIN_DEC_VOCABULARY, config.TEST_DEC_VEC)
    


