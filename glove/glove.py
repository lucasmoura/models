import os

import tensorflow as tf

glove = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'glove_ops.so'))


def main():
    with tf.Session() as sess:
        vocab_words = glove.glove_model('testfile')
        print(sess.run([vocab_words]))

if __name__ == '__main__':
    main()
