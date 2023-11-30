import tensorflow as tf
from absl import flags, app
from tqdm import trange
from models import get_model, EmaPredictor
from utils import CosineDecay
from base import BaseModel
import os
import math
from data import DataSets, PARSEDICT
from augment import weak_augmentation
from tensorflow.keras import mixed_precision

FLAGS = flags.FLAGS

class VanillaSupervised(BaseModel):
    
    def setup_model(self, arch, nclass, wd, lr, momentum, **kwargs):

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
            

            self.wd = tf.constant(wd, tf.float32)
            
            self.cosine_learning_rate = CosineDecay(lr, decay_steps=FLAGS.trainsteps)
            self.optimizer = tf.keras.optimizers.SGD(self.cosine_learning_rate, momentum=momentum,
                                                     nesterov=True)

            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)            

            # need this line because of some tf bug when restoring checkpoints
            self.optimizer.decay=tf.Variable(0.0)            


            self.metrics.update({
                "train/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
                "train/xe_loss": tf.keras.metrics.Mean(),
            })

            # initialize checkpoint
            ckpt_dir = os.path.join(self.train_dir, "checkpoints")
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                            optimizer=self.optimizer,
                                            models=self.models)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=FLAGS.keepckpt)

            # restore from previous checkpoint if exists
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if self.ckpt_manager.latest_checkpoint:
                print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
                assert self.optimizer.iterations == self.ckpt.step
            else:
                print("Initializing from scratch")

        with tf.device('CPU:0'):
            # for logging images in tensorboard
            self.log_image = tf.Variable(
                initial_value=tf.zeros(self.datasets.shape, dtype=tf.float32),
                trainable=False)


    @tf.function
    def train_step(self, inputs):
        images = inputs["image"]
        labels = inputs["label"]

        with tf.GradientTape() as tape:
            logits, _ = self.models["classifier"](images, training=True)
            logits = tf.cast(logits, tf.float32)
            xe_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                logits,
                                                                from_logits=True))

            
            l2_loss = sum(tf.nn.l2_loss(v) for v in self.models["classifier"].trainable_variables
                          if "kernel" in v.name)

            full_loss =  xe_loss + self.wd*l2_loss

            # scale loss for the current strategy b.c. apply_gradients sums over all replicas
            full_loss = full_loss / self.strategy.num_replicas_in_sync
            full_loss = self.optimizer.get_scaled_loss(full_loss)            

        grads = tape.gradient(full_loss, self.models["classifier"].trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(grads)
        
        self.optimizer.apply_gradients(zip(grads, self.models["classifier"].trainable_variables))
        self.models["classifier"].ema.apply(self.models["classifier"].trainable_variables)
        
        self.metrics["train/xe_loss"].update_state(xe_loss)
        self.metrics["train/accuracy"].update_state(labels, tf.nn.softmax(logits))


    @tf.function
    def test_step(self, inputs):

        images = inputs["image"]
        labels = inputs["label"]
        
        logits, _ = self.models["classifier"](images, training=False)
        logits = tf.cast(logits, tf.float32)

        l2_loss = sum(tf.nn.l2_loss(v) for v in self.models["classifier"].trainable_variables
                      if "kernel" in v.name)        

        return {
            "logits": logits,
            "labels": labels,
            "l2_loss": l2_loss,
        }

    def evaluate_and_save_checkpoint(self, data_test, **kwargs):

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
        
        with EmaPredictor(self.models["classifier"], self.strategy):
            for inputs in data_test:
                results = self.strategy.run(self.test_step, args=(inputs,))
                logits = self.strategy.gather(results["logits"], axis=0)
                labels = self.strategy.gather(results["labels"], axis=0)
                self.metrics["test/accuracy_ema"].update_state(labels, tf.nn.softmax(logits))        
            
        total_results = {name: metric.result() for name, metric in self.metrics.items()}

        with self.summary_writer.as_default():
            for name, result in total_results.items():
                tf.summary.scalar(name, result, step=self.ckpt.step)

            # log images
            tf.summary.image("Weak aug",
                             tf.expand_dims(self.log_image, 0),
                             step=self.ckpt.step)

        for metric in self.metrics.values():
            metric.reset_states()

        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}".format(self.ckpt.step.numpy()))
        print("Current test accuracy (ema): {}".format(total_results["test/accuracy_ema"]))

    def train(self, train_steps, eval_steps):

        batch = FLAGS.batch

        shift = int(self.datasets.shape[0]*0.125)
        weakaug = lambda x: weak_augmentation(x, shift)

        # get parse function matching your dataset
        parse = lambda x: PARSEDICT[FLAGS.dataset](x, self.datasets.shape)        
        
        dtr = self.datasets.train.shuffle(FLAGS.shuffle).map(parse, tf.data.AUTOTUNE)
        dtr = dtr.repeat().batch(batch)
        dtr = dtr.map(weakaug, tf.data.AUTOTUNE).prefetch(16)

        dte = self.datasets.test.map(parse, tf.data.AUTOTUNE).batch(batch)
        
        data_train = self.strategy.experimental_distribute_dataset(dtr)
        data_test = self.strategy.experimental_distribute_dataset(dte)

        training_iterator = iter(data_train)

        nr_evals = math.ceil(FLAGS.trainsteps/FLAGS.evalsteps)        
        
        # training loop
        while self.ckpt.step < FLAGS.trainsteps:
            
            self.evaluate_and_save_checkpoint(data_test)
            
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

            # evaluation loop
            for _ in loop:
                item = next(training_iterator)
                self.strategy.run(self.train_step, args=(item,))
                self.ckpt.step.assign_add(1)

            # place augmented images in logging variables
            if self.strategy.num_replicas_in_sync > 1:
                images = item["image"].values[0]
            else:
                images = item["image"]

            self.log_image.assign((images[0,:,:,:]+1.0)/2.0)

        assert self.ckpt.step == FLAGS.trainsteps
        assert self.optimizer.iterations == FLAGS.trainsteps
            
        # final evaluation
        self.evaluate_and_save_checkpoint(data_test)

        
        
def main(argv):

    datasets = DataSets(FLAGS.dataset)
    
    model = VanillaSupervised(
        os.path.join(FLAGS.traindir, datasets.name),
        datasets,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        nclass=datasets.nclass,
        momentum=FLAGS.momentum,
        batch=FLAGS.batch,
        arch=FLAGS.arch,
    )

    model.train(FLAGS.trainsteps, FLAGS.evalsteps)
    
if __name__ == '__main__':
    flags.DEFINE_string("datadir", None, "Directory for data")
    flags.DEFINE_string("traindir", "./experiments", "Directory for results and checkpoints")
    flags.DEFINE_string("arch", "WRN-28-2", "Network architecture")
    flags.DEFINE_string("dataset", "cifar10", "Name of dataset")
    flags.DEFINE_integer("trainsteps", int(1e5), "Number of training steps")
    flags.DEFINE_integer("evalsteps", int(1e3), "Number of steps between model evaluations")
    flags.DEFINE_integer("batch", 64, "Batch size")
    flags.DEFINE_integer("keepckpt", 3, "Number of checkpoints to keep")
    flags.DEFINE_float("wd", 0.0005, "Weight decay regularization")
    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("ema_decay", 0.999, "Momentum parameter for exponential moving average for model parameters")
    flags.DEFINE_float("momentum", 0.9, "SGD momentum")
    flags.DEFINE_string("rerun", "", "Additional experiment specification")
    flags.DEFINE_integer("shuffle", 8192, "Size of dataset shuffling")

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)    

    app.run(main)
