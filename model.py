#coding=utf-8

import time
import numpy as np 
import tensorflow as tf 
import config

class BotChat(object):

    def __init__(self, forward_only, batch_size):
        """
        if set forward_only, it will not use backward pass
        """

        print 'initialize new model'
        self.forward_only = forward_only
        self.batch_size = batch_size

    def _create_placeholders(self):
        print 'create placeholders'
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i)) \
                                for i in xrange(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i)) \
                                for i in xrange(config.BUCKETS[-1][1] + 1)]

        self.decoder_masks = [tf.placeholder(tf.int32, shape=[None], name='mask{}'.format(i)) \
                                for i in xrange(config.BUCKETS[-1][1] + 1)]

        self.targets = self.decoder_inputs[1:]

    def _inference(self):
        print 'inference'
        #whether using sampled softmax
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDEN_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, inputs, labels, config.NUM_SAMPLES, config.DEC_VOCAB)

        self.softmax_loss_func = sampled_loss

        single_cell = tf.nn.rnn_cell.GRUCell(config.HIDEN_SIZE)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell for _ in range(config.NUM_LAYERS)])

    def _create_loss(self):
        print 'creating loss'
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols = config.ENC_VOCAB,
                    num_decoder_symbols = config.DEC_VOCAB,
                    embedding_size = config.HIDEN_SIZE,
                    output_projection = self.output_projection,
                    feed_previous = do_decode
                )

        if self.forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs, self.decoder_inputs,
                                        self.targets, self.decoder_masks,
                                        config.BUCKETS, lambda x, y : _seq2seq_f(x, y, True),
                                        softmax_loss_function = self.softmax_loss_func
                )

            #if feed_previous is True, should decode output so that we can use it as next input
            if self.output_projection:
                for bucket in xrange(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, self.output_projection[0]) + self.output_projection[1] \
                                            for output in self.outputs[bucket]]


        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs, self.decoder_inputs,
                                        self.targets, self.decoder_masks,
                                        config.BUCKETS, lambda x, y : _seq2seq_f(x, y, False),
                                        softmax_loss_function = self.softmax_loss_func
                )
        print 'Time:', time.time() - start


    def _create_optimizer(self):
        print 'create optimizer'
        with tf.variable_scope('traing') as scope:
            self.global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')

            if not self.forward_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in xrange(config.BUCKETS):
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], trainables),\
                                            config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables), global_step = self.global_step))

    def build_graph(self):
        self._create_placeholders()
        self._create_loss()
        self._create_optimizer()




































