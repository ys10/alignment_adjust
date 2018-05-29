#  coding=utf-8
import tensorflow as tf


class BasePredict:
    def __init__(self, sess, model, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def predict(self):
        tf.logging.info('Inference...')
        return self.predict_epoch()

    def predict_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def predict_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
