import os

import tensorflow as tf
import numpy as np

glove = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'glove_ops.so'))


def main():
    with tf.Session() as sess:
        vocab_words, indices, values = glove.glove_model('testfile')
        size = sess.run(tf.shape(vocab_words))
        coocurrence_matrix = tf.SparseTensor(
            indices, values, dense_shape=[size[0], size[0]])
        print(sess.run([vocab_words]))
        print(sess.run([indices]))
        print(sess.run([values]))

if __name__ == '__main__':
    main()
