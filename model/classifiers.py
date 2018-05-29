# coding=utf-8
import tensorflow as tf
from base.base_model import BaseModel


class FCNetModel(BaseModel):
    def __init__(self, config, data):
        super(FCNetModel, self).__init__(config)
        self.data = data
        self.build_model()
        # init the saver
        self.init_saver()

    def build_model(self):
        with tf.variable_scope(self.config.name, reuse=tf.AUTO_REUSE):
            samples = tf.cast(self.data['samples'], tf.float32)
            last_phones = tf.one_hot(self.data['last_phones'], depth=70,)
            next_phones = tf.one_hot(self.data['next_phones'], depth=70,)
            features = tf.concat([samples, last_phones, next_phones], axis=1)
            labels = tf.cast(self.data["labels"], tf.int32)

            output = tf.layers.batch_normalization(features, training=self.config.training, name="bn_layer_0")
            output = tf.layers.dense(inputs=output, units=64, activation=tf.nn.relu, name="hidden_layer_1")
            output = tf.layers.batch_normalization(output, training=self.config.training, name="bn_layer_1")
            output = tf.layers.dense(inputs=output, units=64, activation=tf.nn.relu, name="hidden_layer_2")
            output = tf.layers.batch_normalization(output, training=self.config.training, name="bn_layer_2")
            output = tf.layers.dense(inputs=output, units=16, activation=tf.nn.relu, name="hidden_layer_3")
            output = tf.layers.batch_normalization(output, training=self.config.training, name="bn_layer_3")
            logits = tf.layers.dense(inputs=output, units=3, activation=tf.nn.softmax, name="output_layer")

            with tf.name_scope("loss"):
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)
                self.predictions = tf.argmax(logits, axis=-1)
                acc, self.acc_op = tf.metrics.accuracy(labels, self.predictions)
                rec, self.rec_op = tf.metrics.recall(labels, self.predictions)
                pre, self.pre_op = tf.metrics.precision(labels, self.predictions)
                self.f1_score = tf.divide(2 * tf.multiply(self.rec_op, self.pre_op), tf.add(self.rec_op, self.pre_op))
