#coding=utf-8
import tensorflow as tf  # 1.1
from seq2seq import Seq2Seq
import os
import numpy as np
import thulac
import config
import utils
import math
import collections
import random
import sys
import logging

class Singleton(object):
    def __new__(cls, *args, **kw):  
        if not hasattr(cls, '_instance'):  
            orig = super(Singleton, cls)  
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

class Predict(object):

    def __init__(self):
        self.logger = logging.getLogger('trainlogger')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
                fmt='%(levelname)s\t%(asctime)s\t%(message)s',
                datefmt='%Y-%m-%dT%H:%M:%S')
        handler = logging.FileHandler('./logs/predict.log','a')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


        self.model = Seq2Seq(batch_size=1, forward_only=True)
        model_path = './models/0612/'
        self.vocab_en, _, = utils.read_vocabulary(config.TRAIN_ENC_VOCABULARY)
        _, self.vocab_de, = utils.read_vocabulary(config.TRAIN_DEC_VOCABULARY)
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt != None:
            self.logger.info('find modal: ' + ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.logger.error("没找到模型")
            raise
        self.sep_word = thulac.thulac(seg_only=True)

    def predict(self, ques, uid='9527'):        
        personal_ans = utils.check_pre_ques(ques.decode('utf-8'), uid)

        if personal_ans is not None:
            return personal_ans

        input_string_vec = []
        aseq = self.sep_word.cut(ques, text=True)
        for words in aseq.split(' '):
            input_string_vec.append(self.vocab_en.get(words, config.UNK_ID))

        bucket_id = min([b for b in range(len(config.BUCKETS)) if config.BUCKETS[b][0] > len(input_string_vec)])
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch({bucket_id: [(input_string_vec, [])]}, bucket_id)
        _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

        if config.EOS_ID in outputs:
            outputs = outputs[:outputs.index(config.EOS_ID)]
        
        response = "".join([tf.compat.as_str(self.vocab_de[output]) for output in outputs])
        response = utils.filter_response(response)

        return response
   

