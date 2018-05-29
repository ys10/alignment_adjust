# coding=utf-8
import tensorflow as tf
from model.classifiers import FCNetModel
from model.em import EMFramework
from features.feature_extraction import FeatureGenerator
from features.data_loader import DataLoader
from align.align_mender import AlignMender
from utils.configs import process_config
from utils.logger import Logger


def is_converged(bound_count, bound_moved, move_dist_mean):
    converged = False
    if bound_moved/bound_count < 1e-3 or move_dist_mean < 0.001:
        converged = True
    return converged


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    train_config = process_config("configs/train.json")
    predict_config = process_config("configs/predict.json")
    feature_generator = FeatureGenerator(train_config.phones_path, train_config.wav_dir_path, train_config.aligns_path)
    align_mender = AlignMender()
    data_loader = DataLoader(train_config, predict_config, feature_generator.gen_feats)

    with tf.Session() as sess:
        '''init model'''
        sess.run([data_loader.training_init_op])
        next_data = data_loader.next_data
        classifier = FCNetModel(train_config, next_data)
        '''run'''
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        logger = Logger(sess, train_config)
        em = EMFramework(train_config, predict_config, feature_generator,
                         data_loader, align_mender, classifier, logger, sess)
        for i in range(train_config.em_loops):
            em.e_step()
            if is_converged(*em.m_step()):
                break
    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
