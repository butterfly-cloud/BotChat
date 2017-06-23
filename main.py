#coding=utf-8
import tensorflow as tf  # 1.1
from seq2seq import Seq2Seq
import os
import numpy as np
import thulac
import config
import utils
import math
import random
import sys
import logging


# 读取*dencode.vec和*decode.vec数据（数据还不算太多, 一次读人到内存）
def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in config.BUCKETS]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(config.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.BUCKETS):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set



def train():
    model = Seq2Seq()

    model_path = './models/0602/'

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allocator_type = 'BFC'  # 防止 out of memory

    with tf.Session(config=tf_config) as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt != None:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('load model success' + ckpt.model_checkpoint_path)
        else:
            sess.run(tf.initialize_all_variables())

        logger.info('read data...')

        train_set = read_data(config.TRAIN_ENC_VEC, config.TRAIN_DEC_VEC)
        test_set = read_data(config.TEST_ENC_VEC, config.TEST_DEC_VEC, 100)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(config.BUCKETS))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]

        loss = 0.0
        total_step = 0
        skip_step = 10000
        previous_losses = []

        logger.info('start to train')
        # 一直训练，每过一段时间保存一次模型
        while True:
            rand = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

            loss += step_loss
            total_step += 1

            #print(total_step)

            if total_step % skip_step == 0:
                logger.info('total step {}: loss {} '.format(total_step, loss / skip_step))

                # 如果模型没有得到提升，减小learning rate
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)
                # 保存模型
                checkpoint_path = model_path + "chatbot_seq2seq.ckpt"
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                loss = 0.0
                # 使用测试数据评估模型
                for bucket_id in range(len(config.BUCKETS)):
                    if len(test_set[bucket_id]) == 0:
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    logger.info('test set: {}, {}'.format(bucket_id, eval_ppx))

def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab



def predict():
    sep_word = thulac.thulac(seg_only=True)

    model = Seq2Seq(batch_size=1, forward_only=True)

    model_path = './models/0612/'

    vocab_en, _, = utils.read_vocabulary(config.TRAIN_ENC_VOCABULARY)
    _, vocab_de, = utils.read_vocabulary(config.TRAIN_DEC_VOCABULARY)

    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt != None:
            print('find modal: ',ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")

        while True:
            input_string = raw_input('me > ')
            # 退出
            if input_string == 'quit':
                exit()
            personal_ans = utils.check_pre_ques(input_string.decode('utf-8'))
            if personal_ans is not None:
                print('AI > ' + personal_ans)
                continue

            input_string_vec = []
            aseq = sep_word.cut(input_string, text=True)
            for words in aseq.split(' '):
                input_string_vec.append(vocab_en.get(words, config.UNK_ID))
            bucket_id = min([b for b in range(len(config.BUCKETS)) if config.BUCKETS[b][0] > len(input_string_vec)])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if config.EOS_ID in outputs:
                outputs = outputs[:outputs.index(config.EOS_ID)]
            
            response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs])
            print('AI > ' + response)


if __name__ == '__main__':

    logger = logging.getLogger('trainlogger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
            fmt='%(levelname)s\t%(asctime)s\t%(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S')
    handler = logging.FileHandler('./logs/train.log','a')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    if sys.argv[1] == 'init':
        logger.info('init data')
        utils.format_data(config.YULIAO, config.CUT_WORDS)
        utils.make_train_test(config.CUT_WORDS)

        utils.gen_vocabulary_file(config.TRAIN_ENC_FILE, config.TRAIN_ENC_VOCABULARY, None)
        utils.gen_vocabulary_file(config.TRAIN_DEC_FILE, config.TRAIN_DEC_VOCABULARY, None)
        
        utils.convert_to_vec(config.TRAIN_ENC_FILE, config.TRAIN_ENC_VOCABULARY, config.TRAIN_ENC_VEC)
        utils.convert_to_vec(config.TRAIN_DEC_FILE, config.TRAIN_DEC_VOCABULARY, config.TRAIN_DEC_VEC)

        utils.convert_to_vec(config.TEST_ENC_FILE, config.TRAIN_ENC_VOCABULARY, config.TEST_ENC_VEC)
        utils.convert_to_vec(config.TEST_DEC_FILE, config.TRAIN_DEC_VOCABULARY, config.TEST_DEC_VEC)
        logger.info('init success')
    elif sys.argv[1] == 'train':
        logger.info('training...')
        train()
    elif sys.argv[1] == 'predict':
        predict()
    else:
        logger.error('command error')

   









