#  coding=utf-8
from base.base_predict import BasePredict


class Predictor(BasePredict):
    def __init__(self, sess, model, config, logger):
        super(Predictor, self).__init__(sess, model, config, logger)

    def predict_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        predictions = []
        for _ in range(self.config.iter_per_epoch):
            pred = self.predict_step()
            predictions.extend(pred[0])
        return predictions

    def predict_step(self):
        predictions = self.sess.run([self.model.predictions])
        return predictions
