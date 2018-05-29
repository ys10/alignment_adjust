# coding=utf-8
import tensorflow as tf
import os
import sys


def load_model(saver, sess, logdir):
    tf.logging.info("Trying to restore saved checkpoints from {} ...".format(logdir))
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        tf.logging.info("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        tf.logging.info("  Global step was: {}".format(global_step))
        tf.logging.info("  Restoring...")
        saver.restore(sess, ckpt.model_checkpoint_path)
        tf.logging.info(" Done.")
        return global_step
    else:
        tf.logging.warning(" No checkpoint found.")
        return None


def save_model(saver, sess, logdir, step):
    model_name = "model.ckpt"
    checkpoint_path = os.path.join(logdir, model_name)
    tf.logging.info("Storing checkpoint to {} ...".format(logdir))
    sys.stdout.flush()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    tf.logging.info(" Done.")
