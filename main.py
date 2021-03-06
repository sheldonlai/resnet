import tensorflow as tf
import time
import os

import numpy as np

from data_loader import CifarDataLoader
from data_queue import DataQueue
from model import ResNetClassifier, get_accuracy

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_list('layers', [16, 32, 64], """layers and their depths""")
tf.app.flags.DEFINE_list('residual_layers', [5, 5, 5], """number of residual layers in between""")
tf.app.flags.DEFINE_list('residual_not_core_layers', [1, 1, 1], """# of residual layers that are not core""")
tf.app.flags.DEFINE_list('steps', [21000, 36000, 50000], "steps to decrease learning rate, the last one is ending step")

tf.app.flags.DEFINE_boolean('train', True, """layers and their depths""")
tf.app.flags.DEFINE_boolean('recover', False, """recover previous model or not""")
tf.app.flags.DEFINE_boolean('progressive', False, """train only the core layers""")
tf.app.flags.DEFINE_boolean('data_augmentation', True, """to apply data_augmentation""")

tf.app.flags.DEFINE_string('model_dir', './checkpoints', """recover previous model or not""")
tf.app.flags.DEFINE_string('sub_dir', '', """the sub directory in model_dir and summaries_dir""")
tf.app.flags.DEFINE_string('summaries_dir', './log', """recover previous model or not""")

tf.app.flags.DEFINE_float('learning_rate', 0.1, """the learning rate""")
tf.app.flags.DEFINE_integer('batch_size', 128, """batch size""")
tf.app.flags.DEFINE_integer('save', 500, """save every x iteration""")
tf.app.flags.DEFINE_integer('progressive_step', 5000, """train only the core layers util step x""")
tf.app.flags.DEFINE_integer('validate', 500, """validate every""")

tf.app.flags.DEFINE_integer('partition', 1, """ divided the training data into portions""")
tf.app.flags.DEFINE_integer('selection', 0, """ if partition is enabled, select the x batch to train""")
tf.app.flags.DEFINE_integer('change_data_set_step', 26000, """step to switch the data set""")

data_loader = CifarDataLoader(augmentation=FLAGS.data_augmentation, partition=FLAGS.partition, selection=FLAGS.selection)
model = ResNetClassifier(data_loader=data_loader, zero_init=FLAGS.progressive,
                         layers=[int(float(e)) for e in FLAGS.layers],
                         data_augmentation=FLAGS.data_augmentation,
                         residual_layers=[int(float(e)) for e in FLAGS.residual_layers],
                         learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size)
sess = tf.Session()

valid_acc_results = []

assert(len(FLAGS.layers) == len(FLAGS.residual_layers))
if FLAGS.progressive:
    assert(len(FLAGS.residual_not_core_layers) == len(FLAGS.residual_layers))


def recover():
    if FLAGS.recover:
        try:
            model.saver.restore(sess, tf.train.latest_checkpoint(save_path))
            step = tf.train.global_step(sess, model.global_step)
            print("Recover session at step: %d." % step)
        except:
            print("Unable to recover session starting anew.")
            step = tf.train.global_step(sess, model.global_step)
    else:
        step = tf.train.global_step(sess, model.global_step)
    return step


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


DECREASE_IF_SIMILAR = 3


def should_decrease_learning_rate():
    if len(valid_acc_results) < 3:
        return False
    l = np.array(valid_acc_results[-DECREASE_IF_SIMILAR:])
    mean = np.mean(l[:-1])
    stdd = np.std(l[:-1])
    return np.abs(mean - l[-1]) < stdd


save_path = os.path.join(FLAGS.model_dir, FLAGS.sub_dir)
log_path = os.path.join(FLAGS.summaries_dir, FLAGS.sub_dir)

make_dir_if_not_exists(save_path)
make_dir_if_not_exists(log_path)

if FLAGS.train:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, FLAGS.sub_dir), sess.graph)
    sess.run(tf.global_variables_initializer())
    step = recover()
    start_time = time.time()
    last_valid_time = time.time()
    learning_rate = sess.run(model.learning_rate)
    print('learning rate is %f' % learning_rate)

    valid_accs = []
    train_accs = []

    if FLAGS.progressive and step < FLAGS.progressive_step:
        train_op = model.train_core_op
        acc_op = model.accuracy
    else:
        train_op = model.train_op
        acc_op = model.accuracy

    while step <= int(float(FLAGS.steps[-1])):

        # print('in training loop')
        x, y = model.get_batch()

        _, t_acc = sess.run([train_op, model.accuracy], feed_dict={model.x: x, model.y: y, model.phase: 1})
        # train_writer.add_summary(summary, step)
        step = tf.train.global_step(sess, model.global_step)
        train_accs.append(t_acc)

        if step == FLAGS.change_data_set_step and FLAGS.partition > 1:
            # change dataset
            new_loader = CifarDataLoader(augmentation=FLAGS.data_augmentation)
            m_data, labels, _, _ = new_loader.get_data()
            new_q = DataQueue(m_data, labels, FLAGS.batch_size, capacity=200,
                              threads=32, data_aug=FLAGS.data_augmentation)
            model.replace_data_queue(new_q)

        if step == FLAGS.progressive_step and FLAGS.progressive:
            print("Now training core")
            train_op = model.train_op
            acc_op = model.accuracy

        if step in [int(float(e)) for e in FLAGS.steps[:-1]]:
            print("decreasing learning rate")
            learning_rate = sess.run(model.learning_rate)
            sess.run(tf.assign(model.learning_rate, learning_rate / 10))
            learning_rate = sess.run(model.learning_rate)
            print('learning rate is %f' % learning_rate)

        if step % FLAGS.save == 0:
            model.saver.save(sess, save_path + '/ckpt', step)
            if FLAGS.save != FLAGS.validate:
                print('step: %d, time elaspsed: %3.5f' % (step, time.time() - start_time))
                start_time = time.time()

        if step % FLAGS.validate == 0:
            acc = model.get_validation_accuracy_op(sess) * 100
            t_acc = float(np.mean(train_accs) * 100)
            loss = sess.run(model.loss, feed_dict={model.x: x, model.y: y, model.phase: 0})
            print('step %d, valid accuracy %f%%, training_acc %f%%, training_loss %f, time elasped %s, '
                  'processed batches in queue: %d' %
                  (step, acc, t_acc, loss, time.time() - last_valid_time, model.q.size()))
            train_accs = []
            last_valid_time = time.time()

else:
    sess.run(tf.global_variables_initializer())
    recover()

    data, labels = data_loader.get_test_data()
    valid_acc = model.get_validation_accuracy_op(sess) * 100
    acc = model.get_data_accuracy(sess, data, labels) * 100
    print('valid_acc %f%%, test accuracy %f%%' % (valid_acc, acc))
