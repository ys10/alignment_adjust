# coding=utf-8
import tensorflow as tf


def get_data_set_from_generator(generator, epochs=1, batch_size=3):
    data_set = tf.data.Dataset.from_generator(generator,
                                              output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
                                              output_shapes=(tf.TensorShape([65]), tf.TensorShape([]),
                                                             tf.TensorShape([]), tf.TensorShape([1])))
    data_set = data_set.repeat(epochs)
    data_set = data_set.batch(batch_size)
    return data_set


class DataLoader(object):
    def __init__(self, train_config, predict_config, generator):
        with tf.variable_scope("data"):
            self.train_set = get_data_set_from_generator(generator, epochs=train_config.epochs_per_loop,
                                                         batch_size=train_config.batch_size)
            self.predict_set = get_data_set_from_generator(generator, epochs=predict_config.epochs_per_loop,
                                                           batch_size=predict_config.batch_size)
            self.iterator = self.train_set.make_one_shot_iterator()
            samples, last_phones, next_phones, labels = self.iterator.get_next()
            self.next_data = {'samples': samples, 'last_phones': last_phones, 'next_phones': next_phones, 'labels': labels}
            self.training_init_op = self.iterator.make_initializer(self.train_set)
            self.predict_init_op = self.iterator.make_initializer(self.predict_set)
