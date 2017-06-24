#!/usr/bin/env python
"""Text generation using a Recurrent Neural Network (LSTM)."""

import os
import random
import time

import gflags
import glog as log
from google.apputils import app
import numpy as np
import tensorflow as tf

gflags.DEFINE_string("test_prefix", "The ",
                     "Prefix to prompt the network in test mode")
gflags.DEFINE_string("training_data", "data/shakespeare.txt",
                     "Input data for training the neural network")
gflags.DEFINE_string("checkpoint_file", "saved/model.ckpt",
                     "Checkpoint file to load")

gflags.DEFINE_integer("num_train_batches", 20000,
                      "Number of training batches")
gflags.DEFINE_integer(
    "test_output_length", 500,
    "Number of test characters of text to generate after training the network")
gflags.DEFINE_boolean(
    "continue_training", False,
    "If true, do more training after loading a checkpoint")


FLAGS = gflags.FLAGS


class ModelNetwork(object):
    """RNN with num_layers LSTM layers and a fully-connected output layer

    The network allows for a dynamic number of iterations, depending on the
    inputs it receives.

       out   (fc layer; out_size)
        ^
       lstm
        ^
       lstm  (lstm size)
        ^
        in   (in_size)
    """
    def __init__(self, in_size, lstm_size, num_layers, out_size, session,
                 learning_rate=0.003, name="rnn"):
        self.scope = name

        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.session = session

        self.learning_rate = tf.constant(learning_rate)

        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))

        with tf.variable_scope(self.scope):
            # (batch_size, timesteps, in_size)
            self.xinput = tf.placeholder(tf.float32,
                                         shape=(None, None, self.in_size),
                                         name="xinput")
            self.lstm_init_value = tf.placeholder(
                tf.float32,
                shape=(None, self.num_layers*2*self.lstm_size),
                name="lstm_init_value")

            # LSTM
            self.lstm_cells = [
                tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0,
                                             state_is_tuple=False)
                for i in range(self.num_layers)
            ]
            self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells,
                                                    state_is_tuple=False)

            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm,
                self.xinput,
                initial_state=self.lstm_init_value,
                dtype=tf.float32)

            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(
                tf.random_normal(
                    (self.lstm_size, self.out_size),
                    stddev=0.01
                ))
            self.rnn_out_B = tf.Variable(
                tf.random_normal(
                    (self.out_size, ),
                    stddev=0.01
                ))

            outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
            network_output = (tf.matmul(outputs_reshaped, self.rnn_out_W)
                              + self.rnn_out_B)

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.out_size)
            )

            # Training: provide target outputs for supervised training.
            self.y_batch = tf.placeholder(tf.float32,
                                          (None, None, self.out_size))
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long)
            )
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9
                                                      ).minimize(self.cost)

    def run_step(self, x, init_zero_state=True):
        """Input: X is a single element, not a list!"""
        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers*2*self.lstm_size,))
        else:
            init_value = self.lstm_last_state

        (out, next_lstm_state) = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.xinput: [x],
                self.lstm_init_value: [init_value],
            }
        )

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

    def train_batch(self, xbatch, ybatch):
        """
        xbatch must be (batch_size, timesteps, input_size)
        ybatch must be (batch_size, timesteps, output_size)
        """
        init_value = np.zeros((xbatch.shape[0],
                               self.num_layers*2*self.lstm_size))

        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.xinput: xbatch,
                self.y_batch: ybatch,
                self.lstm_init_value: init_value
            }
        )

        return cost


def embed_to_vocab(data_, vocab):
    """Embed string to character-arrays

    It generates an array len(data) x len(vocab)
    Vocab is a list of elements
    """
    data = np.zeros((len(data_), len(vocab)))

    cnt = 0
    for s in data_:
        v = [0.0]*len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1

    return data


def decode_embed(array, vocab):
    return vocab[array.index(1)]


def main(args):
    TEST_PREFIX = FLAGS.test_prefix.lower()

    # Load the data
    with open(FLAGS.training_data, 'r') as f:
        data_ = f.read().lower()

    # Convert to 1-hot coding
    vocab = list(set(data_))

    data = embed_to_vocab(data_, vocab)

    in_size = out_size = len(vocab)
    lstm_size = 256   # 128
    num_layers = 2
    batch_size = 64   # 128
    time_steps = 100  # 50

    # Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    net = ModelNetwork(in_size=in_size,
                       lstm_size=lstm_size,
                       num_layers=num_layers,
                       out_size=out_size,
                       session=sess,
                       learning_rate=0.003,
                       name="char_rnn_network")

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())

    # Restore the checkpoint, if any
    if os.path.exists(FLAGS.checkpoint_file):
        saver.restore(sess, FLAGS.checkpoint_file)
        restored = True
    else:
        restored = False

    # 1) TRAIN THE NETWORK
    if FLAGS.continue_training or not restored:
        last_time = time.time()

        batch = np.zeros((batch_size, time_steps, in_size))
        batch_y = np.zeros((batch_size, time_steps, in_size))

        possible_batch_ids = list(range(data.shape[0]-time_steps-1))
        for i in range(FLAGS.num_train_batches):
            # Sample time_steps consecutive samples from the dataset text file
            batch_id = random.sample(possible_batch_ids, batch_size)

            for j in range(time_steps):
                ind1 = [k+j for k in batch_id]
                ind2 = [k+j+1 for k in batch_id]

                batch[:, j, :] = data[ind1, :]
                batch_y[:, j, :] = data[ind2, :]

            cst = net.train_batch(batch, batch_y)

            if (i % 100) == 0:
                new_time = time.time()
                diff = new_time - last_time
                last_time = new_time

                log.info("batch: %s loss: %s speed: %s batches / s",
                         i, cst, (100.0/diff))

            log.info("Saving checkpoint to %s", FLAGS.checkpoint_file)
            saver.save(sess, FLAGS.checkpoint_file)

    # GENERATE FLAGS.test_output_length CHARACTERS USING THE TRAINED NETWORK
    for i in range(len(TEST_PREFIX)):
        out = net.run_step(embed_to_vocab(TEST_PREFIX[i], vocab),
                           i == 0)

    log.info("SENTENCE:")
    gen_str = TEST_PREFIX
    for i in range(FLAGS.test_output_length):
        # Sample character from the network according to the generated
        #  output probabilities
        element = np.random.choice(list(range(len(vocab))), p=out)
        gen_str += vocab[element]

        out = net.run_step(embed_to_vocab(vocab[element], vocab), False)
    print(gen_str)


if __name__ == '__main__':
    app.run()
