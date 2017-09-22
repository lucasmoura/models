import os
import sys

import tensorflow as tf
from word_embedding import WordEmbedding, FLAGS, Options

from six.moves import xrange  # pylint: disable=redefined-builtin

glove = tf.load_op_library(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'glove_ops.so'))


class GloVe(WordEmbedding):

    def forward(self, inputs, labels, **kwargs):
        opts = self._options
        init_width = 1.0

        self.input_embeddings = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="input_embeddings")

        # Transposed context embeddings
        self.context_embeddings = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="context_embeddings")

        input_biases = tf.Variable(
            tf.random_uniform([opts.vocab_size], -init_width, init_width),
            name="input_biases")

        context_biases = tf.Variable(
            tf.random_uniform([opts.vocab_size], -init_width, init_width),
            name="context_biases")

        # Embeddings for examples: [batch_size, emb_dim]
        input_embedings = tf.nn.embedding_lookup(
            self.input_embeddings, inputs)

        # Embeddings for labels: [batch_size, vocab_size]
        labels_embeddings = tf.nn.embedding_lookup(
            self.context_embeddings, labels)

        # biases for examples: [batch_size]
        input_biases = tf.nn.embedding_lookup(
            input_biases, inputs)

        # biases for labels: [batch_size]
        labels_biases = tf.nn.embedding_lookup(
            context_biases, labels)

        self.global_step = tf.Variable(0, name="global_step")

        return (input_embedings, input_biases,
                labels_embeddings, labels_biases)

    def loss(self, **kwargs):
        ccounts = kwargs['ccounts']
        inputs_embeddings = kwargs['inputs_embeddings']
        inputs_biases = kwargs['inputs_biases']
        labels_embeddings = kwargs['labels_embeddings']
        labels_biases = kwargs['labels_biases']
        alpha_value = 0.75
        x_max = 100

        # Co-ocurrences log
        log_coocurrences = tf.log(tf.to_float(ccounts))

        embedding_product = tf.reduce_sum(
            tf.multiply(inputs_embeddings, labels_embeddings), 1)

        distance_score = tf.square(
                tf.add_n([embedding_product,
                          inputs_biases,
                          labels_biases,
                          -log_coocurrences]))

        weighting_factor = tf.minimum(
            1.0,
            tf.pow(tf.div(ccounts, x_max), alpha_value))

        loss = tf.reduce_sum(
            tf.multiply(weighting_factor, distance_score))

        return loss

    def optimize(self, loss):
        opts = self._options
        lr = opts.learning_rate

        optimizer = tf.train.AdagradOptimizer(lr)
        self._lr = lr
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train

    def build_graph(self):
        opts = self._options

        print('Calculating Co-ocurrence matrix...')
        (words, _, _, words_per_epoch,
         self._epoch, self._words, examples, labels,
         ccounts) = glove.glove_model(filename=opts.train_data,
                                      batch_size=opts.batch_size,
                                      window_size=opts.window_size,
                                      min_count=opts.min_count)
        (opts.vocab_words, opts.words_per_epoch) = self._session.run(
                 [words, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)

        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)

        self._examples = examples
        self._labels = labels
        self._ccounts = ccounts
        self._id2word = opts.vocab_words

        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        (inputs_embeddings, inputs_biases,
         labels_embeddings, labels_biases) = self.forward(examples, labels)

        loss_variables = {"ccounts": ccounts,
                          "inputs_embeddings": inputs_embeddings,
                          "inputs_biases": inputs_biases,
                          "labels_embeddings": labels_embeddings,
                          "labels_biases": labels_biases}
        loss = self.loss(**loss_variables)

        self._loss = loss
        tf.summary.scalar("GloVe loss", loss)
        self.optimize(loss)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def set_embeddings(self):
        self._embeddings = tf.add(self.input_embeddings,
                                  self.context_embeddings)


def main():
    """Train a GloVe model."""
    if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
        print("--train_data --eval_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = GloVe(opts, session)
            model.read_analogies()  # Read analogy questions
        for _ in xrange(opts.epochs_to_train):
            model.train()  # Process one epoch
            model.eval()  # Eval analogies.
            # Perform a final save.
        model.saver.save(session,
                         os.path.join(opts.save_path, "model.ckpt"),
                         global_step=model.global_step)


if __name__ == '__main__':
    main()
