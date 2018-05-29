# coding=utf-8
from align.align_tools import save_aligns
from operators.predictor import Predictor
from operators.trainer import Trainer
import tensorflow as tf


class EMFramework(object):
    def __init__(self, train_config, predict_config, feature_generator, data_loader,
                 align_mender, classifier, logger, sess):
        self.feature_generator = feature_generator
        self.data_loader = data_loader
        self.align_mender = align_mender
        self.classifier = classifier
        self.train_config = train_config
        self.predict_config = predict_config
        self.logger = logger
        self.sess = sess

    def e_step(self):
        """
        :return:
        """
        '''Prepare data'''
        self.sess.run([self.data_loader.training_init_op])
        next_data = self.data_loader.next_data
        '''reset model parameters'''
        self.classifier.data = next_data
        self.classifier.config = self.train_config
        '''load saved model'''
        self.classifier.load(self.sess)
        '''train'''
        trainer = Trainer(self.sess, self.classifier, self.train_config, self.logger)
        trainer.train()

    def m_step(self):
        """
        :return:
        """
        '''Prepare data'''
        self.sess.run([self.data_loader.predict_init_op])
        next_data = self.data_loader.next_data
        bound_info = self.feature_generator.gen_bound_info()
        tf.logging.debug('bound info length: {}'.format(len(bound_info)))
        '''reset model parameters'''
        self.classifier.config = self.predict_config
        self.classifier.data = next_data
        '''load saved model'''
        self.classifier.load(self.sess)
        '''predict'''
        predictor = Predictor(self.sess, self.classifier, self.predict_config, self.logger)
        predictions = predictor.predict()
        '''amend alignment'''
        new_aligns_dict, new_bound_dict, bound_count, bound_moved, move_dist_mean =\
            self.align_mender.mend(self.feature_generator.aligns_dict, predictions, bound_info)
        self.feature_generator.aligns_dict = new_aligns_dict
        self.feature_generator.bound_dict = new_bound_dict
        '''save new aligns dict'''
        save_aligns(new_aligns_dict, self.predict_config.aligns_path)
        return bound_count, bound_moved, move_dist_mean
