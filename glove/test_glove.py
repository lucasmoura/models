import tensorflow as tf
import threading

glove = tf.load_op_library('./glove_ops.so')


class GloveTest(tf.test.TestCase):

    def testCoocurrenceMatrix(self):
        filename = 'testfile'
        window_size = 5
        min_count = 0
        batch_size = 6

        with self.test_session():
            (vocab_word, indices, values, inputs, labels,
             ccounts, current_epoch) = glove.glove_model(filename,
                                                         batch_size,
                                                         window_size,
                                                         min_count)
            vocab_size = tf.shape(vocab_word)[0]
            indices_size = tf.shape(indices)[0]
            values_size = tf.shape(values)[0]

            self.assertEqual(vocab_size.eval(), 6)
            self.assertEqual(indices_size.eval(), 21)
            self.assertEqual(values_size.eval(), 21)

            expected_indices = [(4, 5), (4, 3), (4, 2), (4, 1),
                                (5, 4), (5, 3), (5, 2), (5, 1),
                                (3, 4), (3, 5), (3, 2), (3, 1),
                                (2, 4), (2, 5), (2, 3), (2, 1),
                                (1, 4), (1, 5), (1, 3), (1, 2), (1, 1)]

            self.assertAllClose(indices.eval(), expected_indices)

            expected_values = [1.0, 0.5, 0.333333, 0.45,
                               1.0, 1.0, 0.5, 0.583333,
                               0.5, 1.0, 1.0, 0.833333,
                               0.333333, 0.5, 1.0, 1.5,
                               0.45, 0.583333, 0.833333, 1.5, 2.0]

            self.assertAllClose(values.eval(), expected_values)

    def testBatchExamples(self):
        filename = 'testfile'
        window_size = 5
        min_count = 0
        batch_size = 5
        concurrent_steps = 5

        (vocab_word, indices, values, inputs, labels,
         ccounts, current_epoch) = glove.glove_model(filename,
                                                     batch_size,
                                                     window_size,
                                                     min_count)

        sess = tf.Session()
        t_indices = sess.run(indices).tolist()
        t_values = sess.run(values).tolist()
        expected_epoch = 1

        def test_body():
            inputs_, labels_, ccounts_, epoch = sess.run(
                [inputs, labels, ccounts, current_epoch])

            for word, label, ccount in zip(inputs_, labels_, ccounts_):
                pos = t_indices.index([word, label])
                self.assertEqual(t_values[pos], ccount)

        workers = []

        for _ in range(concurrent_steps):
            t = threading.Thread(target=test_body)
            t.start()
            workers.append(t)

        for t in workers:
            t.join()

        curr_epoch = sess.run(current_epoch)
        self.assertEqual(expected_epoch, curr_epoch)


if __name__ == '__main__':
    tf.test.main()
