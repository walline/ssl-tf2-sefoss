import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags
#import tensorflow_probability as tfp
import math

FLAGS = flags.FLAGS

def augment_mirror(x):
    return tf.image.random_flip_left_right(x)

def augment_shift(x, w):
    """
    Takes x in [B,H,W,C]
    Shifts entire batch randomly
    """

    parallel_iters = FLAGS.uratio*FLAGS.batch if hasattr(FLAGS, "uratio") else FLAGS.batch    
    
    y = tf.pad(x, [[0] * 2, [w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
    return tf.map_fn(lambda z: tf.image.random_crop(z, tf.shape(x)[1:]),
                     y,
                     parallel_iterations=parallel_iters)

def augment_shift_single(x, w):
    """
    Takes x in [H,W,C]
    Shift single image randomly.
    """
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
    return tf.image.random_crop(y, tf.shape(x))
    

def weak_augmentation_pair(data, shift=4):
    """
    expects data["image"] of shape [B,H,W,C]
    returns data["image"] of shape [B,A,H,W,C] where dimension A represents 2
    different weak augmentations of the same image
    """

    augmentations = [augment_shift(augment_mirror(data["image"]), shift) for _ in range(2)]
    
    data["image"] = tf.stack(augmentations, axis=1)
    return data

def weak_augmentation(data, shift=4):
    data["image"] = augment_shift(augment_mirror(data["image"]), shift)
    return data


def rotations(data, batch):

    rot_labels = [0, 1, 2, 3]

    rotated_images = []
    
    for rot in rot_labels:
        rotated_images.append(tf.image.rot90(data["image"], rot))

    data["image"] = tf.concat(rotated_images, 0)
    data["rotlabel"] = tf.constant([x for x in rot_labels for _ in range(batch)], tf.int64)

    return data


def _blend(image1, image2, factor):
    """
    factor = 0.0 means image1 is returned
    factor = 1.0 means image2 is returned
    """

    difference = image2 - image1
    blended_image = image1 + difference*factor
    return blended_image


def autocontrast(image, *args):

    n_ch = 3
    
    def scale_channel(img_channel):

        high = tf.reduce_max(img_channel)
        low = tf.reduce_max(img_channel)

        def scale_values(vals):
            vals = 2.0 * (vals-low) / (high-low) - 1.0
            vals = tf.clip_by_value(vals, -1.0, 1.0)
            return vals

        scaled_channel = tf.cond(high > low,
                                 lambda: scale_values(img_channel),
                                 lambda: img_channel)
        return scaled_channel

    return tf.stack([scale_channel(image[:,:,ch]) for ch in range(n_ch)], axis=2)

def brightness(image, B):
    # B in (0.05, 0.95)

    black_image = tf.ones_like(image)*-1.0
    return _blend(black_image, image, B)    


def color(image, C):
    # C in (0.05, 0.95)

    gray_image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return _blend(gray_image, image, C)


def contrast(image, C):
    # C in (0.05, 0.95)

    gray_image = tf.image.rgb_to_grayscale(image)
    mean = tf.reduce_mean(gray_image)
    degenerate = tf.ones_like(gray_image, dtype=tf.float32) * mean
    degenerate = tf.image.grayscale_to_rgb(degenerate)

    return _blend(degenerate, image, C)

def equalize(image, *args):
    int_image = tf.image.convert_image_dtype((image+1.0)/2.0, dtype=tf.uint8)
    eq_image = tfa.image.equalize(int_image)
    return tf.cast(eq_image, tf.float32) * 2.0 / 255.0 - 1.0

def posterize(image, B):
    # reduces each pixed to B bits
    # B in [4,...,8]
    shift = tf.cast(8-B, tf.uint8)
    int_image = tf.image.convert_image_dtype((image+1.0)/2.0, dtype=tf.uint8)
    shifted_image = tf.bitwise.left_shift(tf.bitwise.right_shift(int_image, shift), shift)
    return tf.cast(shifted_image, tf.float32) * 2.0 / 255.0 - 1.0    
    

def rotate(image, theta):
    # thega [deg] in (-30, 30)
    return tfa.image.rotate(image, angles=theta*math.pi/180.0)
    
def sharpness(image, S):
    # S in (0.05, 0.95)

    image = tf.expand_dims(image, 0)
    n_ch = 3

    kernel = tf.constant([[1,1,1], [1,5,1], [1,1,1]],
                         dtype=tf.float32,
                         shape=[3, 3, 1, 1])
    kernel = kernel / 13.0
    kernel = tf.tile(kernel, [1, 1, n_ch, 1])

    degenerate = tf.nn.depthwise_conv2d(image,
                                        kernel,
                                        strides=[1,1,1,1],
                                        padding="VALID",
                                        dilations=[1,1])

    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[0,0], [1,1], [1,1], [0,0]])
    padded_degenerate = tf.pad(degenerate, [[0,0], [1,1], [1,1], [0,0]])

    blurred_image = tf.where(tf.equal(padded_mask, 1), padded_degenerate, image)

    image = tf.squeeze(image)
    blurred_image = tf.squeeze(blurred_image)

    return _blend(blurred_image, image, S)    
    

def shear_x(image, R):
    # R in (-0.3, 0.3)
    return tfa.image.shear_x(image, R, replace=0)

def shear_y(image, R):
    # R in (-0.3, 0.3)
    return tfa.image.shear_y(image, R, replace=0)


def solarize(image, T):
    # T in (0,1)
    threshold = -1.0 + 2*T
    return tf.where(image < threshold, image, -image)

def translate_x(image, lambda_):
    # lambda in (-0.3, 0.3)
    width = tf.shape(image, out_type=tf.int32)[1]
    width = tf.cast(width, tf.float32)
    return tfa.image.translate_xy(image, [0, width*lambda_], replace=0)


def translate_y(image, lambda_):
    # lambda in (-0.3, 0.3)
    height = tf.shape(image, out_type=tf.int32)[0]
    height = tf.cast(height, tf.float32)
    return tfa.image.translate_xy(image, [height*lambda_, 0], replace=0)


def cutout(image):
    # TODO: can this be vectorized to operate on the full batch of images?
    
    size = tf.reduce_min(image.shape[:-1])//2

    img_height, img_width = image.shape[:-1]

    height_loc = tf.random.uniform([], 0, img_height, dtype=tf.int32)
    width_loc = tf.random.uniform([], 0, img_width, dtype=tf.int32)

    upper_coord = tf.maximum(0, height_loc - size//2), tf.maximum(0, width_loc - size//2)
    lower_coord = tf.minimum(img_height, height_loc + size//2), tf.minimum(img_width, width_loc + size//2)

    cutout_shape = (lower_coord[0]-upper_coord[0], lower_coord[1]-upper_coord[1])
    padding_dimensions = [[upper_coord[0], img_height-lower_coord[0]],
                          [upper_coord[1], img_width-lower_coord[1]]]

    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=tf.float32),
        padding_dimensions,
        constant_values=1.0)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    cutout_image = tf.where(
        tf.equal(mask, 0),
        tf.zeros(image.shape, dtype=tf.float32),
        image)

    return cutout_image


def rand_augment_single_image(image, nops=2):

    operations = (("Autocontrast", autocontrast, lambda: None),
                  ("Brightness", brightness, lambda: tf.random.uniform([], 0.05, 0.95)),
                  ( "Color", color, lambda: tf.random.uniform([], 0.05, 0.95)),
                  ("Contrast", contrast, lambda: tf.random.uniform([], 0.05, 0.95)),
                  ("Equalize", equalize, lambda: None),
                  ("Identity", lambda x, *args: x, lambda: None),
                  ("Posterize", posterize, lambda: tf.random.uniform([], 4, 9, dtype=tf.int32)),
                  ("Rotate", rotate, lambda: tf.random.uniform([], -30, 30)),
                  ("Sharpness", sharpness, lambda: tf.random.uniform([], 0.05, 0.95)),
                  ("Shear_x", shear_x, lambda: tf.random.uniform([], -0.3, 0.3)),
                  ("Shear_y", shear_y, lambda: tf.random.uniform([], -0.3, 0.3)),
                  ("Solarize", solarize, lambda: tf.random.uniform([], 0, 1)),
                  ("Translate_x", translate_x, lambda: tf.random.uniform([], -0.3, 0.3)),
                  ("Translate_y", translate_y, lambda: tf.random.uniform([], -0.3, 0.3)))

    
    # we sample operations with replacement because this is what other implementations do
    sampled_operations = tf.random.uniform([nops], maxval=len(operations), dtype=tf.int32)
    
    for selected_idx in sampled_operations:
        for idx, (_, fn, arg_fn) in enumerate(operations):
            image = tf.cond(tf.equal(selected_idx, idx),
                            lambda: fn(image, arg_fn()),
                            lambda: image)

    return image

def rand_augment_cutout_batch(data):

    weak_images = data["image"][:,0,:,:,:]
    strong_images = data["image"][:,1,:,:,:]

    parallel_iters = FLAGS.uratio*FLAGS.batch if hasattr(FLAGS, "uratio") else FLAGS.batch    

    strong_images = tf.map_fn(rand_augment_single_image,
                              strong_images,
                              parallel_iterations=parallel_iters)
    strong_images = tf.map_fn(cutout,
                              strong_images,
                              parallel_iterations=parallel_iters)

    data["image"] = tf.stack([weak_images, strong_images], axis=1)
    return data

def mixup_labeled(data, nclass, alpha):
    # takes batch of labeled data

    images = data["image"]
    labels = data["label"]

    labels_onehot = tf.one_hot(labels, nclass)
    batch_size = tf.shape(images)[0]


    # tensorflow probability does not seem to work, need another way to sample from beta
    #dist = tfp.distributions.Beta(alpha, alpha)
    #lambda_ = dist.sample()

    batch_range = tf.range(batch_size)
    random_idxs = tf.random.uniform([batch_size],
                                    minval=1,
                                    maxval=batch_size,
                                    dtype=tf.int32)

    sum_idxs = batch_range + random_idxs
    pick_idxs = tf.math.floormod(sum_idxs, batch_size)

    new_images = tf.gather(images, pick_idxs)
    new_labels = tf.gather(labels_onehot, pick_idxs)

    mixed_images = lambda_*images + (1.0 - lambda_)*new_images
    mixed_labels = lambda_*labels_onehot + (1.0 - lambda_)*new_labels

    data["image"] = tf.concat([images, mixed_images], axis=0)
    data["label"] = tf.concat([labels_onehot, mixed_labels], axis=0)

    return data

    
    



    
    

    
    
    
    
    

                  

    
