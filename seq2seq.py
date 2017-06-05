#coding=utf-8
from __future__ import print_function
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import config
from tensorflow.python.ops import nn_impl

#a sequence to sequence model with attention

class Seq2Seq(object):

    def __init__(self, 
                learning_rate=0.1,
                batch_size = 1,
                use_lstm = False,
                forward_only = False
                ):

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(config.BUCKETS[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(config.BUCKETS[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))
        # Our targets are decoder inputs shifted by one.
        self.targets = self.decoder_inputs[1:]
        self.global_step = tf.Variable(0, trainable=False)

        
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.forward_only = forward_only

        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * config.LR_DECAY)

        self.init_loss_func()
        self.init_rnn_cell()
        self.create_loss()
        self.optimize()




    def init_loss_func(self):
        self.output_projection = None
        self.softmax_loss_function = None

        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB_SIZE:
            w_t = tf.get_variable('proj_w', [ config.DEC_VOCAB_SIZE, config.UNITS_NUM], dtype=tf.float32)
            w = tf.transpose(w_t)
            b = tf.get_variable('proj_b', [config.DEC_VOCAB_SIZE], dtype=tf.float32)
            self.output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)

                # return nn_impl.sampled_softmax_loss(
                #           weights=local_w_t,
                #           biases=local_b,
                #           labels=labels,
                #           inputs=local_inputs,
                #           num_sampled=config.NUM_SAMPLES,
                #           num_classes=config.DEC_VOCAB_SIZE)


                return tf.cast(
                        tf.nn.sampled_softmax_loss(
                            weights=local_w_t,
                            biases=local_b,
                            labels=labels,
                            inputs=local_inputs,
                            num_sampled=config.NUM_SAMPLES,
                            num_classes=config.DEC_VOCAB_SIZE),
                        tf.float32)

            self.softmax_loss_function = sampled_loss

    def init_rnn_cell(self):
        single_cell =lambda : tf.contrib.rnn.BasicLSTMCell(config.UNITS_NUM, reuse = tf.get_variable_scope().reuse) if self.use_lstm \
                         else tf.contrib.rnn.GRUCell(config.UNITS_NUM, reuse = tf.get_variable_scope().reuse)
        if config.NUM_LAYERS > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(config.NUM_LAYERS)])
            self.encoder_cells = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(config.NUM_LAYERS)])
        else:
            self.cell = single_cell()
            self.encoder_cells = single_cell()

    def create_loss(self):
        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                self.encoder_cells,
                self.cell,
                num_encoder_symbols=config.ENC_VOCAB_SIZE,
                num_decoder_symbols=config.DEC_VOCAB_SIZE,
                embedding_size=config.UNITS_NUM,
                output_projection=self.output_projection,
                feed_previous=do_decode,
                dtype=tf.float32)
                   

        # Training outputs and losses.
        if self.forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, self.targets,
                self.target_weights, config.BUCKETS, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function = self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection is not None:
                for b in xrange(len(config.BUCKETS)):
                    self.outputs[b] = [
                        tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, self.targets,
                self.target_weights, config.BUCKETS,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=self.softmax_loss_function)

    def optimize(self):
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not self.forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(config.BUCKETS)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             config.MAX_GRAD_NORM)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())


    def get_batch(self, data_bucket, bucket_id):
        # pad to the max length
        encoder_size, decoder_size = config.BUCKETS[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data_bucket[bucket_id])
            # pad encoder, decoder and reverse encoder because doing this model will be better
            encoder_inputs.append(list(reversed(encoder_input + [config.PAD_ID] * (encoder_size - len(encoder_input)))))
            decoder_inputs.append([config.GO_ID] + decoder_input + [config.PAD_ID] * (decoder_size - len(decoder_input) - 1))

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))


        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == config.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        #Run a step of the model feeding the given inputs.
        encoder_size, decoder_size = config.BUCKETS[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

        input_feed = {}
        for i in xrange(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in xrange(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        # Since our targets are decoder inputs shifted by GO_ID, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for i in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][i])


        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.















