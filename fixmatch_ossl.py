import tensorflow as tf
from absl import flags, app
from tqdm import trange
from models import EmaPredictor
from utils import auroc_rank_calculation, dataset_cardinality, log_partition
import os
import math
from data import OpenSSLDataSets, PARSEDICT
from augment import weak_augmentation_pair, weak_augmentation, rand_augment_cutout_batch
from fixmatch import FixMatch
from tensorflow.keras import mixed_precision


FLAGS = flags.FLAGS

class FixMatchOSSL(FixMatch):
    
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

        images = inputs["image"]
        labels = inputs["label"]
        dataset_labels = inputs.get("dataset", None)
        
        logits, _ = self.models["classifier"](images, training=False)
        logits = tf.cast(logits, tf.float32)

        energies = log_partition(logits)
        maxlogit = tf.reduce_max(logits, axis=-1)        
        conf = tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1)
        
        l2_loss = sum(tf.nn.l2_loss(v) for v in self.models["classifier"].trainable_variables
                          if "kernel" in v.name)

        return {
            "logits": logits,
            "energy": -1.0*energies,
            "conf": conf,
            "maxlogit": maxlogit,
            "setlabels": dataset_labels,
            "labels": labels,
            "l2_loss": l2_loss,
        }

    def evaluate_and_save_checkpoint(self,
                                     data_test,
                                     data_test_ood,
                                     data_test_unseen,
                                     **kwargs):

        batch = FLAGS.batch
                
        # predictions on (ID) test set using non-ema model
        for inputs in data_test:
            results = self.strategy.run(self.test_step, args=(inputs,))
            logits = self.strategy.gather(results["logits"], axis=0)
            labels = self.strategy.gather(results["labels"], axis=0)
            l2_loss = self.strategy.reduce("MEAN", results["l2_loss"], axis=None)
            xe_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))            

            self.metrics["test/accuracy"].update_state(labels, tf.nn.softmax(logits))
            self.metrics["test/xe_loss"].update_state(xe_loss)

        self.metrics["monitors/wd_loss"].update_state(l2_loss)

        current_lr = self.cosine_learning_rate(self.ckpt.step)
        self.metrics["monitors/lr"].update_state(current_lr)

        # predictions on test sets using ema model
        with EmaPredictor(self.models["classifier"], self.strategy):
            for b, inputs in enumerate(data_test):
                results = self.strategy.run(self.test_step, args=(inputs,))
                for key in self.scores:
                    scores = self.strategy.gather(results[key], axis=0)
                    self.scores_testset_id[key][b*batch:(b+1)*batch].assign(scores)
                
                logits = self.strategy.gather(results["logits"], axis=0)
                labels = self.strategy.gather(results["labels"], axis=0)
                self.metrics["test/accuracy_ema"].update_state(labels, tf.nn.softmax(logits))

            for b, inputs in enumerate(data_test_ood):
                results = self.strategy.run(self.test_step, args=(inputs,))
                for key in self.scores:
                    scores = self.strategy.gather(results[key], axis=0)
                    self.scores_testset_ood[key][b*batch:(b+1)*batch].assign(scores)

            for b, inputs in enumerate(data_test_unseen):
                results = self.strategy.run(self.test_step, args=(inputs,))
                for key in self.scores:
                    scores = self.strategy.gather(results[key], axis=0)
                    self.scores_testset_unseen[key][b*batch:(b+1)*batch].assign(scores)        
        
        total_results = {name: metric.result() for name, metric in self.metrics.items()}

        with self.summary_writer.as_default():
            for name, result in total_results.items():
                tf.summary.scalar(name, result, step=self.ckpt.step)

            # log images
            tf.summary.image("Weak aug",
                             tf.expand_dims(self.log_weak_augmentation, 0),
                             step=self.ckpt.step)
            tf.summary.image("Strong aug",
                             tf.expand_dims(self.log_strong_augmentation, 0),
                             step=self.ckpt.step)

            # pairwise test auroc
            for key in self.scores:
                id_scores = self.scores_testset_id[key]
                ood_scores = self.scores_testset_ood[key]
                unseen_scores = self.scores_testset_unseen[key]
                auroc = auroc_rank_calculation(id_scores, ood_scores)
                auroc_unseen = auroc_rank_calculation(id_scores, unseen_scores)
                tf.summary.scalar("ood/auroc_{}".format(key), auroc, step=self.ckpt.step)
                tf.summary.scalar("ood/auroc_{}_{}".format(FLAGS.datasetunseen, key),
                                  auroc_unseen,
                                  step=self.ckpt.step)
            
                
        for metric in self.metrics.values():
            metric.reset_states()

        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}".format(self.ckpt.step.numpy()))
        print("Current test accuracy (ema): {}".format(total_results["test/accuracy_ema"]))

    def train(self, train_steps, eval_steps):

        batch = FLAGS.batch
        uratio = FLAGS.uratio

        shift = int(self.datasets.shape[0]*0.125)
        weakaug = lambda x: weak_augmentation(x, shift)
        weakaugpair = lambda x: weak_augmentation_pair(x, shift)

        # get parse function matching your dataset
        parse = lambda x: PARSEDICT[FLAGS.dataset](x, self.datasets.shape)        
        
        dl = self.datasets.train_labeled.shuffle(FLAGS.shuffle).map(parse, tf.data.AUTOTUNE)
        dl = dl.repeat().batch(batch).map(weakaug, tf.data.AUTOTUNE).prefetch(16)

        dul = self.datasets.train_unlabeled_id.concatenate(self.datasets.train_unlabeled_ood)
        dul = dul.shuffle(self.ulset_size).repeat()
        dul = dul.map(parse, tf.data.AUTOTUNE).batch(batch*uratio).map(weakaugpair, tf.data.AUTOTUNE)
        dul = dul.map(rand_augment_cutout_batch, tf.data.AUTOTUNE).prefetch(16)

        dt = self.datasets.test.map(parse, tf.data.AUTOTUNE).batch(batch).prefetch(16)

        dtood = self.datasets.test_ood.map(parse, tf.data.AUTOTUNE).batch(batch).prefetch(16)

        dtunseen = self.datasets.test_unseen.map(parse, tf.data.AUTOTUNE).batch(batch).prefetch(16)
                
        data_train_labeled = self.strategy.experimental_distribute_dataset(dl)
        data_train_unlabeled = self.strategy.experimental_distribute_dataset(dul)
        data_test = self.strategy.experimental_distribute_dataset(dt)
        data_test_ood = self.strategy.experimental_distribute_dataset(dtood)
        data_test_unseen = self.strategy.experimental_distribute_dataset(dtunseen)        

        labeled_iterator = iter(data_train_labeled)
        unlabeled_iterator = iter(data_train_unlabeled)

        nr_evals = math.ceil(FLAGS.trainsteps/FLAGS.evalsteps)        
        
        # training loop
        while self.ckpt.step < FLAGS.trainsteps:
            
            self.evaluate_and_save_checkpoint(data_test,
                                              data_test_ood,
                                              data_test_unseen)
            
            desc = "Evaluation {}/{}".format(1+self.ckpt.step//FLAGS.evalsteps, nr_evals)
            loopstart = self.ckpt.step%FLAGS.evalsteps
            loopend = min(FLAGS.evalsteps,
                          loopstart + FLAGS.trainsteps - self.ckpt.step)
            loop = trange(loopstart,
                          loopend,
                          leave=False,
                          unit="step",
                          desc=desc,
                          mininterval=10)

            # evaluation loop
            for _ in loop:
                labeled_item = next(labeled_iterator)
                unlabeled_item = next(unlabeled_iterator)
                self.strategy.run(self.train_step, args=(labeled_item, unlabeled_item))
                self.ckpt.step.assign_add(1)


            # place augmented images in logging variables
            if self.strategy.num_replicas_in_sync > 1:
                unlabeled_images = unlabeled_item["image"].values[0]
            else:
                unlabeled_images = unlabeled_item["image"]

            self.log_weak_augmentation.assign((unlabeled_images[0,0,:,:,:]+1.0)/2.0)
            self.log_strong_augmentation.assign((unlabeled_images[0,1,:,:,:]+1.0)/2.0)

        assert self.ckpt.step == FLAGS.trainsteps
        assert self.optimizer.iterations == FLAGS.trainsteps
            
        # final evaluation
        self.evaluate_and_save_checkpoint(data_test,
                                          data_test_ood,
                                          data_test_unseen)

        
        
def main(argv):

    datasets = OpenSSLDataSets(FLAGS.dataset,
                               FLAGS.datasetood,
                               FLAGS.nlabeled,
                               FLAGS.seed,
                               FLAGS.datasetunseen)
    
    model = FixMatchOSSL(
        os.path.join(FLAGS.traindir, datasets.name),
        datasets,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
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
    flags.DEFINE_boolean("savestats", False, "Save stats as csv files")    
    flags.DEFINE_integer("seed", 1, "Seed for labeled data")
    flags.DEFINE_integer("nlabeled", 40, "Number of labeled data")
    flags.DEFINE_integer("keepckpt", 1, "Number of checkpoints to keep")
    flags.DEFINE_float("wd", 0.0005, "Weight decay regularization")
    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("ema_decay", 0.999, "Momentum parameter for exponential moving average for model parameters")
    flags.DEFINE_float("momentum", 0.9, "SGD momentum")
    flags.DEFINE_string("rerun", "", "Additional experiment specification")
    flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    app.run(main)
