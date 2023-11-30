import tensorflow as tf
from keras import backend

class BatchNorm(tf.keras.layers.BatchNormalization):
    """
    Subclassing layers.BatchNormalization to disable
    behaviour that self.trainable = False sets entire
    layer to inference mode. Now instead, setting
    self.trainable = False only disables updates
    of moving mean and variances.
    """

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if self._USE_V2_BEHAVIOR:
            if isinstance(training, int):
                training = bool(training)
        return training

from .wideresnet import WideResNet
from .resnet import ResNet

def get_model(arch, num_classes):

    if arch == "WRN-28-2":
        depth = 28
        width = 2
        model = WideResNet(depth, width, num_classes)
    elif arch == "WRN-28-8":
        depth = 28
        width = 8
        model = WideResNet(depth, width, num_classes)
    elif arch == "WRN-28-4":
        depth = 28
        width = 4
        model = WideResNet(depth, width, num_classes)
    elif arch == "ResNet18":
        blocks_per_layer = [2, 2, 2, 2]
        model = ResNet(blocks_per_layer, num_classes)
    else:
        raise NotImplementedError(
            "Arch: {} not implemented".format(arch))
    
    return model

class EmaPredictor(object):
    def __init__(self, ema_model, strategy):
        self.ema_model = ema_model
        self.strategy = strategy

    def __enter__(self):
        self.ema_model.swap_weights(self.strategy)
        return self.ema_model

    def __exit__(self, type, value, traceback):
        self.ema_model.swap_weights(self.strategy)
