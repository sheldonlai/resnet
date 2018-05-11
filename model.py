import math

import numpy as np
import tensorflow as tf

from data_queue import DataQueue
from ops import residual_block


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class ResNetClassifier(object):
    def __init__(self, data_loader, layers=(16, 32, 64), residual_layers=(5, 5, 5), data_augmentation=True,
                 non_core_layers=(1, 1, 1), learning_rate=0.01, batch_size=128, zero_init=False):
        """
        :param layers: tuple that has the depth dimension vector
        """

        assert (len(layers) == len(residual_layers))
        m_data, labels, self.valid_data, self.valid_labels = data_loader.get_data()

        self.q = DataQueue(m_data, labels, batch_size, capacity=200, threads=32, data_aug=data_augmentation)
        self.q.start()

        n, y_dim, x_dim, channel = m_data.shape
        y_dim = x_dim = 32
        self.batch_size = batch_size
        self.batch_len_in_epoch = int(math.ceil(n / self.batch_size)) - 1
        self.layers = layers
        self.residual_layers_between = residual_layers

        self.x = tf.placeholder(tf.float32, shape=(batch_size, y_dim, x_dim, channel))
        self.y = tf.placeholder(tf.int32, shape=(batch_size,))

        self.phase = tf.placeholder(tf.bool, name='phase')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name='learning_rate')
        self.decrease_learning_rate = tf.assign(self.learning_rate, tf.multiply(self.learning_rate, 0.1))

        # Layers
        conv_layer = None
        for i, depth in enumerate(layers):
            if conv_layer is None:
                conv_layer = residual_block(self.x, depth, block_num=str(depth), first_block=True, core=True,
                                            is_training=self.phase)
            else:
                conv_layer = residual_block(conv_layer, depth, block_num=str(depth), first_block=False, core=True,
                                            is_training=self.phase)
            assert (conv_layer.get_shape()[-1] == depth)

            for k in range(residual_layers[i]):
                if k > residual_layers[i] - non_core_layers[i]:
                    # non-core layer
                    conv_layer = residual_block(conv_layer, depth, block_num=str(depth) + '_' + str(k),
                                                zero_init=zero_init, is_training=self.phase)
                else:
                    conv_layer = residual_block(conv_layer, depth, block_num=str(depth) + '_' + str(k), core=True,
                                                zero_init=False, is_training=self.phase)
                assert (conv_layer.get_shape()[-1] == depth)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        pool_layer = tf.reduce_mean(conv_layer, [1, 2])
        self.logits = tf.layers.dense(tf.reshape(pool_layer, [batch_size, -1]), 10)

        self.prediction = tf.argmax(self.logits, 1)
        trainable_vars = tf.trainable_variables()
        core_var_list = [v for v in trainable_vars if 'core' in v.name]

        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bn' not in v.name]) / 2 * 0.0001

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.y)) + loss_l2

        prediction = tf.equal(tf.argmax(self.logits, 1), tf.cast(self.y, tf.int64))

        self.accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        with tf.control_dependencies(self.update_ops):
            with tf.name_scope('train'):
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)

                self.train_op = self.optimizer.minimize(loss=self.loss, global_step=self.global_step)

                # train core only
                self.train_core_op = self.optimizer.minimize(loss=self.loss, var_list=core_var_list,
                                                             global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=10)

    def get_batch(self):
        return self.q.pop()

    def replace_data_queue(self, new_queue):
        """
        Replaces the old data queue
        :param new_queue: a new data queue
        :return: None
        """
        self.q.stop()
        self.q = None
        self.q = new_queue
        self.q.start()

    # def get_training_batch(self, index):
    #     data = self.m_data
    #     index = index % self.batch_len_in_epoch
    #     if (index + 1) * self.batch_size <= len(data):
    #         res = data[index * self.batch_size:(index + 1) * self.batch_size]
    #         labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
    #     else:
    #         res = data[index * self.batch_size:] + data[:len(data) - (index * self.batch_size)]
    #         labels = self.labels[index * self.batch_size:] + self.labels[:len(data) - (index * self.batch_size)]
    #
    #     res = np.array(res).astype(np.float32)
    #     return res, np.array(labels)

    def get_validation_accuracy_op(self, sess):
        return self.get_data_accuracy(sess, self.valid_data, self.valid_labels)

    def get_data_accuracy(self, sess, data, labels):
        batch_num = int(math.ceil(len(data) / self.batch_size))
        length, y_dim, x_dim, channel = data.shape
        prediction_holder = None
        for index in range(batch_num):
            if (index + 1) * self.batch_size <= len(data):
                x = data[index * self.batch_size:(index + 1) * self.batch_size]
            else:
                data_last = data[index * self.batch_size:]
                x = np.vstack((data_last, np.zeros((self.batch_size - len(data_last), y_dim, x_dim, channel))))

            prediction = sess.run(self.prediction, feed_dict={self.x: x, self.phase: 0})
            if prediction_holder is None:
                prediction_holder = prediction
            else:
                prediction_holder = np.concatenate((prediction_holder, prediction))

        return get_accuracy(prediction_holder[:len(data)], labels)


def get_accuracy(prediction, labels):
    eq = np.equal(prediction, labels)
    mean = np.mean(eq)
    return mean
