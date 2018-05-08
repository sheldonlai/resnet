import unittest

import time
import numpy as np
import tensorflow as tf

from data_loader import CifarDataLoader
from model import ResNetClassifier


class ModelTest(unittest.TestCase):

    def test_valid_accuracy(self):
        data_loader = CifarDataLoader(augmentation=True)

        model = ResNetClassifier(data_loader=data_loader, zero_init=True,
                                 layers=[16, 32, 64],
                                 data_augmentation=True,
                                 residual_layers=[5, 5, 5])
        start_time = time.time()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(500):
                model.get_validation_accuracy_op(sess)
                print("time used %4.4f" % (time.time() - start_time))
                start_time = time.time()

