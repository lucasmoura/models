import tensorflow as tf

glove = tf.load_op_library('./glove_ops.so')


class GloveTest(tf.test.TestCase):

    def testCoocurrenceMatrix(self):
        filename = 'testfile'
        window_size = 5
        min_count = 0
        batch_size = 6;

        with self.test_session():
            vocab_word, indices, values, inputs, labels, ccounts, current_epoch = glove.glove_model(
                filename, batch_size, window_size, min_count)
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


    def testBatchExamples(self):
        filename = 'testfile'
        window_size = 5
        min_count = 0
        batch_size = 5;

        with self.test_session():
            vocab_word, indices, values, inputs, labels, ccounts, current_epoch = glove.glove_model(
                filename, batch_size, window_size, min_count)

            t_indices = indices.eval().tolist()
            t_values = values.eval().tolist()
            expected_epoch = 1;

            for i in range(6):
                for input_w, label, ccount in zip(inputs.eval(), labels.eval(), ccounts.eval()):
                    pos = t_indices.index([input_w, label])
                    self.assertEqual(t_values[pos], ccount)

                current_epoch.eval()

            self.assertEqual(current_epoch.eval(), expected_epoch)


if __name__ == '__main__':
    tf.test.main()
