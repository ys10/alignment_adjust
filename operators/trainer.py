#  coding=utf-8
from base.base_train import BaseTrain
from tqdm import tqdm


class Trainer(BaseTrain):
    def __init__(self, sess, model, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.iter_per_epoch))
        for _ in loop:
            loss, acc, rec, pre, f1 = self.train_step()
            summaries_dict = {
                'loss': loss,
                'acc': acc,
                'rec': rec,
                'pre': pre,
                'f1': f1,
            }
            cur_it = self.model.global_step_tensor.eval(self.sess)
            self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        _, loss, acc, rec, pre, f1 = self.sess.run([
            self.model.train_op, self.model.loss, self.model.acc_op,
            self.model.rec_op, self.model.pre_op, self.model.f1_score])
        return loss, acc, rec, pre, f1
