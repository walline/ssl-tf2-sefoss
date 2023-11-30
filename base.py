import os
import tensorflow as tf
from absl import flags
from utils import print_model_summary

FLAGS = flags.FLAGS

class BaseModel:

    def __init__(self, train_dir, datasets, **kwargs):

        self.train_dir = os.path.join(train_dir, FLAGS.rerun, self.experiment_name(**kwargs))
        self.datasets = datasets

        if tf.config.list_physical_devices("GPU"):
            self.strategy = tf.distribute.MirroredStrategy()
            assert FLAGS.batch % self.strategy.num_replicas_in_sync == 0
            print("Nr of GPUs in use: {}".format(self.strategy.num_replicas_in_sync))
        else:
            self.strategy = tf.distribute.get_strategy()
            print("No GPU in use")

        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.train_dir, "summaries"))

        self.models = {}
        
        self.setup_model(**kwargs)

        print(" Config ".center(80, "-"))
        print("Train dir", self.train_dir)
        print("{:<32} {}".format("Model", self.__class__.__name__))
        print("{:<32} {}".format("Dataset", datasets.name))
        for k, v in sorted(kwargs.items()):
            print("{:<32} {}".format(k, v))

        for name, model in self.models.items():
            print_model_summary(model, name)

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    def setup_model(self, *args, **kwargs):
        raise NotImplementedError

