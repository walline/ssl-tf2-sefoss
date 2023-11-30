import tensorflow as tf
from absl import flags, app
from models import get_model
from utils import CosineDecay
from fixmatch import FixMatch
import os
from tensorflow.keras import mixed_precision
from data import SemiSupervisedDataSets


FLAGS = flags.FLAGS

class DoubleMatch(FixMatch):
    
    def setup_model(self, arch, nclass, wd, ws, wp, lr, momentum, confidence, **kwargs):

        # metrics that are not updated at training time
        # these are placed outside of strategy scope
        self.metrics = {
            "test/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
            "test/accuracy_ema": tf.keras.metrics.SparseCategoricalAccuracy(),
            "test/xe_loss": tf.keras.metrics.Mean(),
            "monitors/wd_loss": tf.keras.metrics.Mean(),
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
            
            self.confidence = tf.constant(confidence, tf.float32)
            self.wd = tf.constant(wd, tf.float32)
            self.ws = tf.constant(ws, tf.float32)
            self.wp = tf.constant(wp, tf.float32)
            
            self.cosine_learning_rate = CosineDecay(lr,
                                                    decay_steps=FLAGS.trainsteps,
                                                    decay_factor=FLAGS.decayfactor)
            
            self.optimizer = tf.keras.optimizers.SGD(self.cosine_learning_rate, momentum=momentum,
                                                     nesterov=True)

            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)            

            # need this line because of some tf bug when restoring checkpoints
            self.optimizer.decay=tf.Variable(0.0)
            
            self.metrics.update({
                "train/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
                "train/xe_loss": tf.keras.metrics.Mean(),
                "train/us_loss": tf.keras.metrics.Mean(),                
                "monitors/mask": tf.keras.metrics.Mean(),
            })

            # initialize checkpoint
            ckpt_dir = os.path.join(self.train_dir, "checkpoints")
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                            optimizer=self.optimizer, models=self.models)
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

    @tf.function
    def train_step(self, labeled_inputs, unlabeled_inputs):

        labeled_images = labeled_inputs["image"]
        labels = labeled_inputs["label"]

        unlabeled_images = unlabeled_inputs["image"]

        local_batch = int(FLAGS.batch / self.strategy.num_replicas_in_sync)
        uratio = int(FLAGS.uratio)

        with tf.GradientTape() as tape:

            x = tf.concat([labeled_images, unlabeled_images[:,0], unlabeled_images[:,1]], 0)
            
            logits, embeds = self.models["classifier"](x, training=True)
            logits, embeds = tf.cast(logits, tf.float32), tf.cast(embeds, tf.float32)

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
            pseudo_mask = tf.cast(tf.reduce_max(pseudo_labels, axis=1) >= self.confidence, tf.float32)

            xep_loss = tf.reduce_mean(xep_loss * pseudo_mask)                                    

            # unsupervised representation loss
            normalized_proj_embeds = tf.linalg.l2_normalize(projected_embeds, axis=-1)
            normalized_embeds = tf.linalg.l2_normalize(tf.stop_gradient(embeds_weak), axis=-1)
            cosine_similarity = tf.reduce_sum(tf.multiply(normalized_proj_embeds, normalized_embeds), axis=-1)
            us_loss = tf.reduce_mean(-cosine_similarity+1)

            variables = self.models["classifier"].trainable_variables + \
                self.models["projection_head"].trainable_variables
            
            l2_loss = sum(tf.nn.l2_loss(v) for v in variables if "kernel" in v.name)

            full_loss =  xe_loss + self.wp*xep_loss + self.ws*us_loss + self.wd*l2_loss

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
        self.metrics["monitors/mask"].update_state(tf.reduce_mean(pseudo_mask))


    @tf.function
    def test_step(self, inputs):

        images = inputs["image"]
        labels = inputs["label"]
        
        logits, _ = self.models["classifier"](images, training=False)
        logits = tf.cast(logits, tf.float32)

        variables = self.models["classifier"].trainable_variables + \
                self.models["projection_head"].trainable_variables
        
        l2_loss = sum(tf.nn.l2_loss(v) for v in variables
                          if "kernel" in v.name)

        return {
            "logits": logits,
            "labels": labels,
            "l2_loss": l2_loss,
        }        
        
def main(argv):

    datasets = SemiSupervisedDataSets(FLAGS.dataset, FLAGS.nlabeled, FLAGS.seed)
    
    model = DoubleMatch(
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
    flags.DEFINE_float("confidence", 0.95, "Confidence threshold for pseudo-labels")
    flags.DEFINE_integer("uratio", 7, "Unlabeled batch size ratio")
    flags.DEFINE_integer("seed", 1, "Seed for labeled data")
    flags.DEFINE_integer("nlabeled", 40, "Number of labeled data")
    flags.DEFINE_integer("keepckpt", 1, "Number of checkpoints to keep")
    flags.DEFINE_float("wd", 0.0005, "Weight decay regularization")
    flags.DEFINE_float("ws", 5.0, "Weight for self-supervised loss")
    flags.DEFINE_float("wp", 1.0, "Weight for pseudo-labeling loss")
    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("decayfactor", 7/8, "Decay for cosine learning rate")
    flags.DEFINE_float("ema_decay", 0.999, "Momentum parameter for exponential moving average for model parameters")
    flags.DEFINE_float("momentum", 0.9, "SGD momentum")
    flags.DEFINE_string("rerun", "", "Additional experiment specification")
    flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)    

    app.run(main)
