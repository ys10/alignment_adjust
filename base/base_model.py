#  coding=utf-8
import os
import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    def set_data(self, data):
        self.data = data

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        tf.logging.info("Saving model to {}...".format(self.config.checkpoint_dir))
        global_step = self.global_step_tensor.eval(sess)
        tf.logging.info("  Global step was: {}".format(global_step))
        self.saver.save(sess, self.config.checkpoint_dir, global_step)
        tf.logging.info("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        tf.logging.info("Trying to restore saved checkpoints from {} ...".format(self.config.checkpoint_dir))
        ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        if ckpt:
            tf.logging.info("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path
                              .split('/')[-1]
                              .split('-')[-1])
            tf.logging.info("  Global step was: {}".format(global_step))
            tf.logging.info("  Restoring...")
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            tf.logging.debug('global_step_tensor_value: {}.'.format(self.global_step_tensor.eval(sess)))
            tf.logging.debug('cur_epoch_tensor_value: {}.'.format(self.cur_epoch_tensor.eval(sess)))
            tf.logging.info(" Done.")
            return global_step
        else:
            tf.logging.warning(" No checkpoint found.")
            return None

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign_add(self.cur_epoch_tensor,1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        raise NotImplementedError
