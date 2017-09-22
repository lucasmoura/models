import os

import tensorflow as tf

glove = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'glove_ops.so'))


"""
The flags and Config class are based on the official word2vec model on
tensorflow

https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py
"""

flags = tf.app.flags

flags.DEFINE_string('save_path', None, 'Directory to write the model and '
                    'training summaries.')
flags.DEFINE_string('train_data', None, 'Training text file. '
                    'E.g., unzipped file http://mattmahoney.net/dc/text8.zip.')
flags.DEFINE_string(
    'eval_data', None, 'File consisting of analogies of four tokens.'
    'embedding 2 - embedding 1 + embedding 3 should be close to embedding 4.')
flags.DEFINE_integer('embedding_size', 200, 'The embedding dimension size.')
flags.DEFINE_integer(
    'epochs_to_train', 15,
    'Number of epochs to train. Each epoch processes the training data once '
    'completely.')
flags.DEFINE_float('learning_rate', 0.2, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 16,
                     'Number of training examples processed per step '
                     '(size of a minibatch).')
flags.DEFINE_integer('concurrent_steps', 12,
                     'The number of concurrent training steps.')
flags.DEFINE_integer('window_size', 5,
                     'The number of words to predict to the left and right '
                     'of the target word.')
flags.DEFINE_integer('min_count', 5,
                     'The minimum number of word occurrences for it to be '
                     'included in the vocabulary.')
flags.DEFINE_integer('statistics_interval', 5,
                     'Print statistics every n seconds.')
flags.DEFINE_integer('summary_interval', 5,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval).')
flags.DEFINE_integer('checkpoint_interval', 600,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds (rounded up to statistics interval).')

FLAGS = flags.FLAGS


class Config():
    """Configuration used by our GloVe model."""

    def __init__(self):
        # Embedding dimension.
        self.emb_dim = FLAGS.embedding_size

        # Training options.
        # The training text file.
        self.train_data = FLAGS.train_data

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = FLAGS.epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # Number of words to predict to the left and right of the target word.
        self.window_size = FLAGS.window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = FLAGS.min_count

        # How often to print statistics.
        self.statistics_interval = FLAGS.statistics_interval

        # How often to write to the summary file (rounds up to the nearest
        # statistics_interval).
        self.summary_interval = FLAGS.summary_interval

        # How often to write checkpoints (rounds up to the nearest statistics
        # interval).
        self.checkpoint_interval = FLAGS.checkpoint_interval

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Eval options.
        # The text file for eval.
        self.eval_data = FLAGS.eval_data




def main():
    pass


if __name__ == '__main__':
    main()
