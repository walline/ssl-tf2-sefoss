import math
import tensorflow as tf
from tensorflow import keras
from models import BatchNorm


kaiming_normal = lambda: keras.initializers.VarianceScaling(scale=2.0,
                                                            mode='fan_out',
                                                            distribution='untruncated_normal')

class Conv2D(tf.keras.layers.Layer):

    def __init__(self, filters, stride=1):

        super(Conv2D, self).__init__()
        
        self.pad = tf.keras.layers.ZeroPadding2D(
            padding=1,
        )
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            use_bias=False,
            kernel_initializer=kaiming_normal(),
        )

    def call(self, inputs, **kwargs):
        x = self.pad(inputs)
        x = self.conv(x)
        return x

def BatchNormalization():

    kwargs = {
        "momentum": 0.9,
        "epsilon": 1e-5,
    }

    return BatchNorm(**kwargs)
        

class ResNetBlock(tf.keras.layers.Layer):

    def __init__(self, filters, stride=1, downsample=None):

        super(ResNetBlock, self).__init__()
        self.conv1 = Conv2D(filters, stride=stride)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters)
        self.bn2 = BatchNormalization()

        self.downsample = downsample

    def call(self, inputs, training=True):

        x = inputs
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.downsample is not None:
            inputs = self.downsample[0](inputs)
            inputs = self.downsample[1](inputs, training=training)

        out = inputs + x
        out = tf.nn.relu(out)

        return out

class ResNetLayer(tf.keras.layers.Layer):

    def __init__(self, inplanes, planes, blocks, stride=1):

        super(ResNetLayer, self).__init__()        

        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = [
                tf.keras.layers.Conv2D(
                    filters=planes,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False,
                    kernel_initializer=kaiming_normal(),
                ),
                BatchNormalization()
                ]

        self.blocks = [None]*blocks
        self.blocks[0] = ResNetBlock(planes, stride, downsample)
        for i in range(1, blocks):
            self.blocks[i] = ResNetBlock(planes)

        
    def call(self, inputs, training=True):

        x = inputs

        for block in self.blocks:
            x = block(x, training=training)

        return x        

class ResNet(tf.keras.Model):

    def __init__(self, blocks_per_layer, num_classes):

        super(ResNet, self).__init__()

        self.zeropad1 = tf.keras.layers.ZeroPadding2D(padding=3)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            use_bias=False,
            kernel_initializer=kaiming_normal(),
        )
        self.bn = BatchNormalization()
        self.zeropad2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=3,
            strides=2)

        self.layer1 = ResNetLayer(64, 64, blocks_per_layer[0])
        self.layer2 = ResNetLayer(64, 128, blocks_per_layer[1], stride=2)
        self.layer3 = ResNetLayer(128, 256, blocks_per_layer[2], stride=2)
        self.layer4 = ResNetLayer(256, 512, blocks_per_layer[3], stride=2)

        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        initializer = lambda: keras.initializers.RandomUniform(
            -1.0 / math.sqrt(512),
            1.0 / math.sqrt(512),
            )

        self.dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_initializer=initializer(),
            bias_initializer=initializer())

    def call(self, inputs, training=True):
        
        x = inputs
        x = self.zeropad1(x)
        x = self.conv1(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.zeropad2(x)
        x = self.maxpool(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.gap(x)
        embeds = x

        x = self.dense(x)

        return x, embeds

    def build(self, ema_decay, *args, **kwargs):

        super(ResNet, self).build(*args, **kwargs)
        self.initiate_ema(ema_decay=ema_decay)

    def initiate_ema(self, ema_decay=0.999):

        self.ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        self.ema.apply(self.trainable_variables)
        self.ema_variables = [self.ema.average(var) for var in self.trainable_variables]


    def swap_weights(self, strategy):
        """Swap the average and moving weights.
        This is a convenience method to allow one to evaluate the averaged weights
        at test time. Loads the weights stored in `self._average_weights` into the model,
        keeping a copy of the original model weights. Swapping twice will return
        the original weights.
        """
        #if tf.distribute.in_cross_replica_context():
        #    strategy = tf.distribute.get_strategy()
        #    return strategy.run(self._swap_weights, args=())
        #else:
        #    raise ValueError(
        #        "Swapping weights must occur under a " "tf.distribute.Strategy"
        #    )
        return strategy.run(self._swap_weights, args=())

    @tf.function
    def _swap_weights(self):
        def fn_0(a, b):
            return a.assign_add(b)

        def fn_1(b, a):
            return b.assign(a - b)

        def fn_2(a, b):
            return a.assign_sub(b)

        def swap(strategy, a, b):
            """Swap `a` and `b` and mirror to all devices."""
            for a_element, b_element in zip(a, b):
                strategy.extended.update(
                    a_element, fn_0, args=(b_element,)
                )  # a = a + b
                strategy.extended.update(
                    b_element, fn_1, args=(a_element,)
                )  # b = a - b
                strategy.extended.update(
                    a_element, fn_2, args=(b_element,)
                )  # a = a - b

        ctx = tf.distribute.get_replica_context()
        return ctx.merge_call(swap, args=(self.ema_variables, self.trainable_variables))        

