import tensorflow as tf

glove = tf.load_op_library('./glove_ops.so')


class GloveTest(tf.test.TestCase):

    def testCoocurrenceMatrix(self):
        filename = 'testfile'
        window_size = 5
        min_count = 0

        with self.test_session():
            vocab_word, indices, values = glove.glove_model(
                filename, window_size, min_count)
            vocab_size = tf.shape(vocab_word)[0]
            indices_size = tf.shape(indices)[0]
            values_size = tf.shape(values)[0]

            self.assertEqual(vocab_size.eval(), 6)
            self.assertEqual(indices_size.eval(), 21)
            self.assertEqual(values_size.eval(), 21)

            expected_indices = [(4, 5), (4, 3), (4, 2), (4, 1), (5, 4), (5, 3),
                                (5, 2), (5, 1), (3, 4), (3, 5), (3, 2), (3, 1),
                                (2, 4), (2, 5), (2, 3), (2, 1), (1, 4), (1, 5),
                                (1, 3), (1, 2), (1, 1)]

            for i, index in enumerate(indices.eval()):
                self.assertEqual(expected_indices[i][0], index[0])
                self.assertEqual(expected_indices[i][1], index[1])

            expected_values = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2,
                               2, 2, 2, 2, 2]

            for i, value in enumerate(values.eval()):
                self.assertEqual(expected_values[i], value)


if __name__ == '__main__':
    tf.test.main()
