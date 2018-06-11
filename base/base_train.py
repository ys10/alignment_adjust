#  coding=utf-8
import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        # self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.sess.run(self.init)

    def train(self):
        # self.model.load(self.sess)
        tf.logging.info('Training...')
        start_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for cur_epoch in range(start_epoch, start_epoch + self.config.epochs_per_loop, 1):
            tf.logging.info('   epoch:%d' % cur_epoch)
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
        self.model.save(self.sess)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
