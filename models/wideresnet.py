import functools
from models import BatchNorm

import tensorflow as tf

leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)

def Conv2D(filters, kernel_size, **kwargs):

    stddev = tf.math.rsqrt(0.5*kernel_size*kernel_size*filters)
    default_kwargs = {
        "padding": "same",
        "use_bias": True,
        "kernel_initializer": tf.keras.initializers.RandomNormal(stddev=stddev)
    }
    default_kwargs.update(kwargs)
    
    return tf.keras.layers.Conv2D(filters, kernel_size, **default_kwargs)

def BatchNormalization():

    kwargs = {
        "epsilon": 0.001,
        "momentum": 0.999
    }
    
    return BatchNorm(**kwargs)                
    
    
class ResNetBlock(tf.keras.layers.Layer):

    def __init__(self, filters, strides, shape_correction=False):


        super(ResNetBlock, self).__init__()
        
        self.shape_correction = shape_correction

        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(filters, 3, strides=strides)
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(filters, 3, strides=1)

        if shape_correction:
            self.conv_correction = Conv2D(filters, 1, strides=strides)



    def call(self, inputs, training=True):

        x = inputs
        y = inputs

        y = self.bn1(y, training=training)
        y = leaky_relu(y)
        y = self.conv1(y)
        y = self.bn2(y, training=training)
        y = leaky_relu(y)
        y = self.conv2(y)

        if self.shape_correction:
            x = self.conv_correction(x)

        x = tf.math.add(x, y)

        return x

class ResNetGroup(tf.keras.layers.Layer):

    def __init__(self, filters, strides, num_blocks, shape_correction=True):

        super(ResNetGroup, self).__init__()
        
        self.blocks = [None]*num_blocks

        self.blocks[0] = ResNetBlock(filters, strides, shape_correction=shape_correction)

        for i in range(1,num_blocks):
            self.blocks[i] = ResNetBlock(filters, 1, shape_correction=False)

    def call(self, inputs, training=True):

        x = inputs
        for block in self.blocks:
            x = block(x, training=training)

        return x

class WideResNet(tf.keras.Model):

    def __init__(self, depth, width_multiplier, num_classes):

        super(WideResNet, self).__init__()
        
        if (depth - 4) % 6 != 0:
            raise ValueError("depth should be 6n+4 (e.g., 16, 22, 28, 40)")

        num_blocks = (depth - 4) // 6


        self.conv = Conv2D(16, 3, strides=1)

        self.group1 = ResNetGroup(filters=16*width_multiplier,
                                  strides=1,
                                  num_blocks=num_blocks,
                                  shape_correction=width_multiplier!=1)

        self.group2 = ResNetGroup(filters=32*width_multiplier,
                                  strides=2,
                                  num_blocks=num_blocks,
                                  shape_correction=True)

        self.group3 = ResNetGroup(filters=64*width_multiplier,
                                  strides=2,
                                  num_blocks=num_blocks,
                                  shape_correction=True)

        self.bn = BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(num_classes,
                                           kernel_initializer=tf.keras.initializers.GlorotNormal())


    def call(self, inputs, training=True):


        x = inputs
        x = self.conv(x)

        x = self.group1(x, training=training)
        x = self.group2(x, training=training)
        x = self.group3(x, training=training)
        x = self.bn(x, training=training)
        x = leaky_relu(x)
        # x = tf.nn.avg_pool(x, ksize=8, strides=1, padding="VALID")
        x = tf.reduce_mean(x, axis=[1,2])
        x = self.flatten(x)
        embeds = x

        x = self.dense(x)

        return x, embeds

    def build(self, ema_decay, *args, **kwargs):


        super(WideResNet, self).build(*args, **kwargs)

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
