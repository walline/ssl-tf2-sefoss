import tensorflow as tf
from absl import flags, app
from utils import dataset_cardinality
from doublematch import DoubleMatch
from fixmatch_ossl import FixMatchOSSL
import os
from data import OpenSSLDataSets
from tensorflow.keras import mixed_precision

FLAGS = flags.FLAGS

class DoubleMatchOSSL(DoubleMatch, FixMatchOSSL):
    
    def setup_model(self, *args, **kwargs):

        super().setup_model(*args, **kwargs)
        
        with tf.device("CPU:0"):

            self.testset_id_size = dataset_cardinality(self.datasets.test)
            self.testset_ood_size = dataset_cardinality(self.datasets.test_ood)
            self.testset_unseen_size = dataset_cardinality(self.datasets.test_unseen)
            
            self.labelset_size = dataset_cardinality(self.datasets.train_labeled)
            self.ulset_size = (dataset_cardinality(self.datasets.train_unlabeled_id)
                               + dataset_cardinality(self.datasets.train_unlabeled_ood))

            self.scores = ["energy", "conf", "maxlogit"]

            self.scores_testset_id = {}
            self.scores_testset_ood = {}
            self.scores_testset_unseen = {}

            for key in self.scores:
                self.scores_testset_id[key] = tf.Variable(tf.zeros(self.testset_id_size, dtype=tf.float32),
                                                          trainable=False)
                self.scores_testset_ood[key] = tf.Variable(tf.zeros(self.testset_ood_size, dtype=tf.float32),
                                                           trainable=False)
                self.scores_testset_unseen[key] = tf.Variable(tf.zeros(self.testset_unseen_size, dtype=tf.float32),
                                                           trainable=False)

    @tf.function
    def test_step(self, inputs):

        return FixMatchOSSL.test_step(self, inputs)

    def evaluate_and_save_checkpoint(self, *args, **kwargs):

        FixMatchOSSL.evaluate_and_save_checkpoint(self, *args, **kwargs)

    def train(self, train_steps, eval_steps):

        FixMatchOSSL.train(self, train_steps, eval_steps)
        
        
def main(argv):

    datasets = OpenSSLDataSets(FLAGS.dataset,
                               FLAGS.datasetood,
                               FLAGS.nlabeled,
                               FLAGS.seed,
                               FLAGS.datasetunseen)
    
    
    model = DoubleMatchOSSL(
        os.path.join(FLAGS.traindir, datasets.name),
        datasets,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        ws=FLAGS.ws,
        wp=FLAGS.wp,
        nclass=datasets.nclass,
        momentum=FLAGS.momentum,
        batch=FLAGS.batch,
        arch=FLAGS.arch,
        confidence=FLAGS.confidence
    )

    model.train(FLAGS.trainsteps, FLAGS.evalsteps)
    
if __name__ == '__main__':
    flags.DEFINE_string("datadir", None, "Directory for data")
    flags.DEFINE_string("traindir", "./experiments", "Directory for results and checkpoints")        
    flags.DEFINE_string("arch", "WRN-28-2", "Network architecture")
    flags.DEFINE_integer("trainsteps", int(1e5), "Number of training steps")
    flags.DEFINE_integer("evalsteps", int(3e3), "Number of steps between model evaluations")
    flags.DEFINE_integer("batch", 64, "Batch size")
    flags.DEFINE_string("dataset", "cifar10", "Name of dataset")
    flags.DEFINE_string("datasetood", "svhn", "Name of ood dataset")
    flags.DEFINE_string("datasetunseen", "cifar100", "Name of unseen OOD set")    
    flags.DEFINE_float("confidence", 0.95, "Confidence threshold for pseudo-labels")
    flags.DEFINE_integer("uratio", 7, "Unlabeled batch size ratio")
    flags.DEFINE_integer("seed", 1, "Seed for labeled data")
    flags.DEFINE_integer("nlabeled", 40, "Number of labeled data")
    flags.DEFINE_integer("keepckpt", 1, "Number of checkpoints to keep")
    flags.DEFINE_float("wd", 0.0005, "Weight decay regularization")
    flags.DEFINE_float("ws", 5.0, "Weight for self-supervised loss")
    flags.DEFINE_float("wp", 1.0, "Weight for pseudo-labeling loss")
    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("decayfactor", 7/8, "Decay factor for cosine learning rate")
    flags.DEFINE_float("ema_decay", 0.999, "Momentum parameter for exponential moving average for model parameters")
    flags.DEFINE_float("momentum", 0.9, "SGD momentum")
    flags.DEFINE_string("rerun", "", "Additional experiment specification")
    flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)    

    app.run(main)
