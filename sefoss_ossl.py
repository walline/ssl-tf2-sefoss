import tensorflow as tf
from absl import flags, app
from tqdm import trange
import math
from models import get_model, EmaPredictor
from utils import CosineDecay, auroc_rank_calculation, dataset_cardinality, log_partition
from base import BaseModel
import os
from data import OpenSSLDataSets, PARSEDICT
from augment import weak_augmentation_pair, weak_augmentation, rand_augment_cutout_batch
from tensorflow.keras import mixed_precision


FLAGS = flags.FLAGS

class SefossOSSL(BaseModel):
    
    def setup_model(self, arch, nclass, wd, ws, we, lr, momentum, **kwargs):

        # metrics that are not updated at training time
        # these are placed outside of strategy scope
        self.metrics = {
            "test/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
            "test/accuracy_ema": tf.keras.metrics.SparseCategoricalAccuracy(),
            "test/xe_loss": tf.keras.metrics.Mean(),
            "valid/accuracy_ema": tf.keras.metrics.SparseCategoricalAccuracy(),
            "monitors/lr": tf.keras.metrics.Mean(),
        }
        
        with self.strategy.scope():

            self.models["classifier"] = get_model(arch, nclass)

            self.models["classifier"].build(ema_decay=FLAGS.ema_decay,
                                            input_shape=[None] + self.datasets.shape)

            output_shape = self.models["classifier"].compute_output_shape(
                input_shape=[None] + self.datasets.shape)

            feature_dimension = output_shape[-1][-1]

            self.models["projection_head"] = tf.keras.layers.Dense(
                feature_dimension,
                kernel_initializer=tf.keras.initializers.GlorotNormal())
            self.models["projection_head"].build(input_shape=[None, feature_dimension])

            self.wd = tf.constant(wd, tf.float32)
            self.ws = tf.constant(ws, tf.float32)
            self.we = tf.constant(we, tf.float32)

            self.id_threshold = tf.Variable(0.0, dtype=tf.float32, trainable=False)
            self.ood_threshold = tf.Variable(0.0, dtype=tf.float32, trainable=False)
            self.ood_target = tf.Variable(0.0, dtype=tf.float32, trainable=False)

            self.thresholds_set = tf.Variable(False, dtype=tf.bool, trainable=False)
            
            self.cosine_learning_rate = CosineDecay(lr,
                                                    decay_steps=FLAGS.trainsteps,
                                                    decay_factor=FLAGS.decayfactor,
                                                    pretrain_steps=FLAGS.pretrainsteps)
            
            self.optimizer = tf.keras.optimizers.SGD(self.cosine_learning_rate,
                                                     momentum=tf.Variable(momentum),
                                                     nesterov=True)

            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)            

            # need this line because of some tf bug when restoring checkpoints
            self.optimizer.decay=tf.Variable(0.0)            
            
            self.metrics.update({
                "train/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
                "train/xe_loss": tf.keras.metrics.Mean(),
                "train/us_loss": tf.keras.metrics.Mean(),
                "train/energy_loss": tf.keras.metrics.Mean(),
                "monitors/wd_loss": tf.keras.metrics.Mean(),
                "monitors/mask": tf.keras.metrics.Mean(),
                "monitors/we": tf.keras.metrics.Mean(),
                "monitors/wpl": tf.keras.metrics.Mean(),
                "monitors/ood_mask": tf.keras.metrics.Mean()
            })

            # initialize checkpoint
            ckpt_dir = os.path.join(self.train_dir, "checkpoints")
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                            optimizer=self.optimizer,
                                            models=self.models,
                                            id_threshold=self.id_threshold,
                                            ood_threshold=self.ood_threshold,
                                            ood_target=self.ood_target,
                                            thresholds_set=self.thresholds_set)
            
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=FLAGS.keepckpt)

            # restore from previous checkpoint if exists
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if self.ckpt_manager.latest_checkpoint:
                print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
                assert self.optimizer.iterations == self.ckpt.step
            else:
                print("Initializing from scratch")

        with tf.device('CPU:0'):
            self.log_weak_augmentation = tf.Variable(
                initial_value=tf.zeros(self.datasets.shape, dtype=tf.float32),
                trainable=False)
            
            self.log_strong_augmentation = tf.Variable(
                initial_value=tf.zeros(self.datasets.shape, dtype=tf.float32),
                trainable=False)

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

            self.train_oodpreds = tf.Variable(tf.zeros(self.labelset_size, dtype=tf.float32),
                                                       trainable=False)



    @tf.function
    def train_step(self, labeled_inputs, unlabeled_inputs):


        labeled_images = labeled_inputs["image"]
        labels = labeled_inputs["label"]

        unlabeled_images = unlabeled_inputs["image"]

        local_batch = int(FLAGS.batch / self.strategy.num_replicas_in_sync)
        uratio = int(FLAGS.uratio)

        step_fn = tf.maximum(0.0, tf.cast(tf.sign(self.ckpt.step-FLAGS.pretrainsteps), tf.float32))
        wpl = step_fn
        we = step_fn*self.we

        with tf.GradientTape() as tape:

            x = tf.concat([labeled_images, unlabeled_images[:,0], unlabeled_images[:,1]], 0)
            
            logits, embeds = self.models["classifier"](x, training=True)
            logits, embeds = tf.cast(logits, tf.float32), tf.cast(embeds, tf.float32)

            energies_weak = log_partition(logits[local_batch:-local_batch*uratio])
            
            projected_embeds = self.models["projection_head"](embeds[-local_batch*uratio:])
            projected_embeds = tf.cast(projected_embeds, tf.float32)
            embeds_weak = embeds[local_batch:-local_batch*uratio]
            
            logits_labeled = logits[:local_batch]
            logits_weak, logits_strong = tf.split(logits[local_batch:], 2)
            
            xe_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                logits_labeled,
                                                                from_logits=True))

            pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))

            xep_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(pseudo_labels, axis=1),
                                                                       logits_strong,
                                                                       from_logits=True)

            pseudo_mask = tf.cast(energies_weak <= self.id_threshold, tf.float32)

            ood_mask = tf.cast(energies_weak > self.ood_threshold, tf.float32)

            xep_loss = tf.reduce_mean(xep_loss * pseudo_mask)                                    

            # unsupervised representation loss
            normalized_proj_embeds = tf.linalg.l2_normalize(projected_embeds, axis=-1)
            normalized_embeds = tf.linalg.l2_normalize(tf.stop_gradient(embeds_weak), axis=-1)
            cosine_similarity = tf.reduce_sum(tf.multiply(normalized_proj_embeds, normalized_embeds), axis=-1)
            us_loss = tf.reduce_mean(-cosine_similarity+1)

            # energy regularization
            energy_loss_out = tf.pow(tf.maximum(tf.cast(0, tf.float32), self.ood_target - energies_weak),2)
            
            energy_loss = tf.reduce_sum(ood_mask * energy_loss_out)/tf.maximum(tf.reduce_sum(ood_mask), 1.0)

            variables = self.models["classifier"].trainable_variables + \
                self.models["projection_head"].trainable_variables
            
            l2_loss = sum(tf.nn.l2_loss(v) for v in variables
                          if "kernel" in v.name)

            full_loss =  xe_loss + wpl*xep_loss + self.ws*us_loss + self.wd*l2_loss + we*energy_loss

            # scale loss for the current strategy b.c. apply_gradients sums over all replicas
            full_loss = full_loss / self.strategy.num_replicas_in_sync
            full_loss = self.optimizer.get_scaled_loss(full_loss)


        grads = tape.gradient(full_loss, variables)
        grads = self.optimizer.get_unscaled_gradients(grads)        
        
        self.optimizer.apply_gradients(zip(grads, variables))
        self.models["classifier"].ema.apply(self.models["classifier"].trainable_variables)

        self.metrics["train/us_loss"].update_state(us_loss)
        self.metrics["train/xe_loss"].update_state(xe_loss)
        self.metrics["train/accuracy"].update_state(labels, tf.nn.softmax(logits_labeled))
        self.metrics["train/energy_loss"].update_state(energy_loss)        
        self.metrics["monitors/mask"].update_state(tf.reduce_mean(pseudo_mask))
        self.metrics["monitors/ood_mask"].update_state(tf.reduce_mean(ood_mask))
        self.metrics["monitors/we"].update_state(we)
        self.metrics["monitors/wpl"].update_state(wpl)
        self.metrics["monitors/wd_loss"].update_state(l2_loss)
        
        

    def set_thresholds(self, data_train):

        batch = FLAGS.batch
                                                                                  
        with EmaPredictor(self.models["classifier"], self.strategy):
            for b, inputs in enumerate(data_train):
                results = self.strategy.run(self.test_step, args=(inputs,))
                ood_preds = self.strategy.gather(results["energy"], axis=0)
                self.train_oodpreds[b*batch:(b+1)*batch].assign(-1.0*ood_preds)

        
        sorted_scores = tf.sort(self.train_oodpreds)
        
        lq_idx = tf.cast(tf.round(self.labelset_size*0.25), tf.int32)
        uq_idx = tf.cast(tf.round(self.labelset_size*0.75), tf.int32)
        uq = sorted_scores[uq_idx]
        lq = sorted_scores[lq_idx]
        
        iqr = uq - lq

        if self.labelset_size % 2 == 1:
            idx = tf.cast(tf.floor(self.labelset_size/2), tf.int32)
            median = sorted_scores[idx]
        elif self.labelset_size % 2 == 0:
            idx1 = tf.cast(self.labelset_size/2, tf.int32)
            idx2 = idx1 - 1
            median = 0.5 * (sorted_scores[idx1] + sorted_scores[idx2])

        self.id_threshold.assign(median-iqr*FLAGS.inlierrange)
        self.ood_threshold.assign(median+iqr*FLAGS.outlierrange)

        self.ood_target.assign(median+iqr*FLAGS.outliertarget)        

        self.thresholds_set.assign(True)
        

        
    @tf.function
    def test_step(self, inputs):

        images = inputs["image"]
        labels = inputs["label"]
        
        logits, _ = self.models["classifier"](images, training=False)
        logits = tf.cast(logits, tf.float32)

        energies = log_partition(logits)
        maxlogit = tf.reduce_max(logits, axis=-1)
        conf = tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1)        
        
        return {
            "logits": logits,
            "energy": -1.0*energies,
            "conf": conf,
            "maxlogit": maxlogit,
            "labels": labels,
        }      


    def evaluate_and_save_checkpoint(self,
                                     data_test,
                                     data_test_ood,
                                     data_test_unseen,
                                     data_valid,
                                     **kwargs):

        batch = FLAGS.batch
        
        # predictions on (ID) test set using non-ema model
        for inputs in data_test:
            results = self.strategy.run(self.test_step, args=(inputs,))
            logits = self.strategy.gather(results["logits"], axis=0)
            labels = self.strategy.gather(results["labels"], axis=0)
            xe_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))            

            self.metrics["test/accuracy"].update_state(labels, tf.nn.softmax(logits))
            self.metrics["test/xe_loss"].update_state(xe_loss)

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

        # predictions on validation set using ema-model
        with EmaPredictor(self.models["classifier"], self.strategy):
            for b, inputs in enumerate(data_valid):
                results = self.strategy.run(self.test_step, args=(inputs,))                
                logits = self.strategy.gather(results["logits"], axis=0)
                labels = self.strategy.gather(results["labels"], axis=0)
                self.metrics["valid/accuracy_ema"].update_state(labels, tf.nn.softmax(logits))
            
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
            
            # thresholds
            tf.summary.scalar("ood/id_threshold", self.id_threshold, step=self.ckpt.step)
            tf.summary.scalar("ood/ood_threshold", self.ood_threshold, step=self.ckpt.step)
            tf.summary.scalar("ood/ood_target", self.ood_target, step=self.ckpt.step)
                
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

        dl = self.datasets.train_labeled.shuffle(self.labelset_size)

        dtr_raw = dl.skip(FLAGS.nvalid).map(parse, tf.data.AUTOTUNE).batch(batch)
        
        dvalid = dl.take(FLAGS.nvalid)
        dvalid = dvalid.map(parse, tf.data.AUTOTUNE).batch(batch)

        
        dl = dl.skip(FLAGS.nvalid).map(parse, tf.data.AUTOTUNE)
        dl = dl.shuffle(FLAGS.shuffle).repeat().batch(batch).map(weakaug, tf.data.AUTOTUNE)
        dl = dl.prefetch(16)
        
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
        data_valid = self.strategy.experimental_distribute_dataset(dvalid)
        
        data_train_raw = self.strategy.experimental_distribute_dataset(dtr_raw)

        labeled_iterator = iter(data_train_labeled)
        unlabeled_iterator = iter(data_train_unlabeled)        

        assert FLAGS.trainsteps >= FLAGS.pretrainsteps        
        nr_evals = math.ceil(FLAGS.trainsteps/FLAGS.evalsteps)
        
        # training loop
        while self.ckpt.step < FLAGS.trainsteps:

            self.evaluate_and_save_checkpoint(data_test,
                                              data_test_ood,
                                              data_test_unseen,
                                              data_valid)

            desc = "Evaluation {}/{}".format(
                1+self.ckpt.step//FLAGS.evalsteps, nr_evals)
            loopstart = self.ckpt.step%FLAGS.evalsteps
            loopend = min(FLAGS.evalsteps,
                          loopstart + FLAGS.trainsteps - self.ckpt.step)

            loop = trange(loopstart,
                          loopend,
                          leave=False,
                          unit="step",
                          desc=desc,
                          mininterval=10)
            
            for _ in loop:
                labeled_item = next(labeled_iterator)
                unlabeled_item = next(unlabeled_iterator)
                self.strategy.run(self.train_step, args=(labeled_item, unlabeled_item))
                self.ckpt.step.assign_add(1)

                if not self.thresholds_set and self.ckpt.step >= FLAGS.pretrainsteps:
                    self.set_thresholds(data_train_raw)

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
                                          data_test_unseen,
                                          data_valid)
        

        
        
def main(argv):

    datasets = OpenSSLDataSets(FLAGS.dataset,
                               FLAGS.datasetood,
                               FLAGS.nlabeled,
                               FLAGS.seed,
                               FLAGS.datasetunseen)    
    
    model = SefossOSSL(
        os.path.join(FLAGS.traindir, datasets.name),
        datasets,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        ws=FLAGS.ws,
        we=FLAGS.we,
        nclass=datasets.nclass,
        momentum=FLAGS.momentum,
        batch=FLAGS.batch,
        arch=FLAGS.arch,
        ilr=FLAGS.inlierrange,
        olr=FLAGS.outlierrange,
        tro=FLAGS.outliertarget
    )

    model.train(FLAGS.trainsteps, FLAGS.evalsteps)
    
if __name__ == '__main__':
    flags.DEFINE_string("datadir", None, "Directory for data")
    flags.DEFINE_string("traindir", "./experiments", "Directory for results and checkpoints")        
    flags.DEFINE_string("arch", "WRN-28-2", "Network architecture")
    flags.DEFINE_integer("trainsteps", int(1e5), "Number of training steps")
    flags.DEFINE_integer("evalsteps", int(3e3), "Number of steps between model evaluations")
    flags.DEFINE_integer("pretrainsteps", int(5e3), "Number of pretraining steps")
    flags.DEFINE_integer("batch", 64, "Batch size")
    flags.DEFINE_string("dataset", "cifar10", "Name of dataset")
    flags.DEFINE_string("datasetood", "svhn", "Name of ood dataset")
    flags.DEFINE_string("datasetunseen", "cifar100", "Name of unseen OOD set")
    flags.DEFINE_integer("uratio", 7, "Unlabeled batch size ratio")
    flags.DEFINE_integer("nvalid", 0, "Number of validation data")    
    flags.DEFINE_integer("seed", 1, "Seed for labeled data")
    flags.DEFINE_integer("nlabeled", 40, "Number of labeled data")
    flags.DEFINE_integer("keepckpt", 1, "Number of checkpoints to keep")
    flags.DEFINE_float("wd", 0.0005, "Weight decay regularization")
    flags.DEFINE_float("ws", 5.0, "Weight for self-supervised loss")
    flags.DEFINE_float("we", 0.0001, "Weight for energy loss")
    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("decayfactor", 7/8, "Decay factor for cosine decay of learning rate")
    flags.DEFINE_float("outlierrange", 1.3, "IQR multiplier for outlier detection")
    flags.DEFINE_float("inlierrange", 0.2, "IQR multiplier for inlier detection")
    flags.DEFINE_float("outliertarget", 1.9, "IQR multiplier for outlier target")
    flags.DEFINE_float("ema_decay", 0.999, "Momentum parameter for exponential moving average for model parameters")
    flags.DEFINE_float("momentum", 0.9, "SGD momentum")
    flags.DEFINE_string("rerun", "", "Additional experiment specification")
    flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    app.run(main)
