import unittest

import numpy as np
import tensorflow as tf

class NPTests(unittest.TestCase):

    def test_np_accuracy(self):
        a = [1, 2, 3, 4, 3]
        label = [1, 0, 3, 4, 3]

        eq = np.equal(a, label)
        match = [True, False, True, True, True]
        np.testing.assert_equal(match, eq)

        mean = np.mean(eq)
        self.assertEqual(mean, 0.8)

    def test_tf(self):
        with tf.Session() as sess:
            varo = tf.Variable(0.1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.assign(varo, tf.multiply(varo, 0.1)))

            self.assertEqual(0.01, sess.run(varo))
