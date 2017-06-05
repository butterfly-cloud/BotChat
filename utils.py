#coding=utf-8
import sys
import random
import codecs
import os
import jieba
import struct
import numpy as np
import config

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
#use jieba to cut words
def format_data(file_name, output_file):
    if not os.path.exists(file_name):
        print '数据不存在'
        return False

    out = open(output_file,'w')

    with codecs.open(file_name, 'r', encoding='utf8') as f:
        seqs = []
        for line in f:
            line = line.strip().replace('/','')

            if line == '':
                continue

            if line[0] == 'E':
                start = len(seqs) % 2
                for word in seqs[start:]:
                    out.write(word.encode('utf8') + '\n')
                seqs = []

            elif line[0] == 'M':
                ans = line.split(' ')
                if len(ans) == 2:
                    aseq = jieba.cut(ans[1], cut_all=True)
                    #filter ''
                    j_seq = ''
                    for v in aseq:
                        j_seq += " " + v
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


def make_train_test(input_file, seed=0):
    random.seed(0)
    flag = 0
    test_ques = open('./test_ques.dat','w')
    test_ans = open('./test_ans.dat','w')
    train_ques = open('./train_ques.dat','w')
    train_ans = open('./train_ans.dat','w')

    f = open(input_file, 'r')
    cnt = 0
    while True:
        line_ques = f.readline().strip()
        #print cnt
        cnt += 1
        if not line_ques:
            #print line_ques
            break
        line_ans = f.readline().strip()
        cnt += 1
        #print line_ans
        if random.randint(1,10) < 4:
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
def convert_to_vec(input_file, vector_dict):
    vec_list = []
    with open(input_file,'r') as f:
        for line in f:
            words = line.strip().split(' ')
            vec = []
            for word in words:
                if vector_dict.has_key(word):
                    vec.append(vector_dict[word])
            vec_list.append(vec)

    return vec_list

#generator data for model using
def generator_train_data(ques_all, ans_all, max_words_size=10, word_dim=500):
    train_ques_ans = []
    train_ans = []

    for i in range(len(ques_all)):
        ques = ques_all[i]
        ans = ans_all[i]

        if len(ques) < max_words_size and len(ans) < max_words_size:
            # ques = [np.zeros(word_dim)] * (max_words_size - len(ques)) + list(reversed(ques))
            # ans = [np.ones(word_dim)] + ans + [np.zeros(word_dim)] * (max_words_size - len(ans) - 1)

            # train_ques_ans.append(ques + ans)
            # train_ans.append(ans + [np.zeros(word_dim)])

            ques = [np.zeros(word_dim)] * (max_words_size - len(ques)) + list(reversed(ques))
            ans = ans + [np.zeros(word_dim)] * (max_words_size - len(ans))
            ques += ans
            ans = [np.ones(word_dim)] + ans
            train_ques_ans.append(ques)
            train_ans.append(ans)


    return np.array(train_ques_ans), np.array(train_ans)

def _pad_full(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshape_batch(input_, size, batch_size):
    batch_inputs = []
    for i in xrange(size):
        batch_inputs.append(np.array(
                [input_[batch_id][i] for batch_id in xrange(batch_size)], dtype=np.int32
            ))


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

if __name__ == '__main__':
    #format_data('/Users/yixin/sunhao25/myself/nlp/dgk_shooter_min.conv','./cut_all.words')
    #format_data('/Users/yixin/sunhao25/myself/nlp/test.conv','./test.words')
    #load_vector_bin('./vector.bin')
    make_train_test('./cut_all.words')



