import tensorflow as tf
from absl import flags
import csv
import os

FLAGS = flags.FLAGS

class smart_dict(dict):
    @staticmethod
    def __missing__(key):
        return key

SET_MAPPING = smart_dict()
for x in ["10pc", "20pc", "30pc", "40pc",
          "50pc", "60pc", "70pc", "80pc",
          "90pc"]:
    SET_MAPPING["cifar10_{}".format(x)] = "cifar10"
    SET_MAPPING["cifar100_{}".format(x)] = "cifar100"
SET_MAPPING["cifar10_onlylabeled"] = "cifar10"
SET_MAPPING["cifar100_onlytest"] = "cifar100"
SET_MAPPING["noise10000"] = "uniform_noise"
SET_MAPPING["tin10000"] = "tinyimagenet_resized"

MIX_MAPPING = smart_dict()
MIX_MAPPING["cifar10animals+cifar10notanimals"] = "cifar10"
MIX_MAPPING["tin_firsthalf+tin_secondhalf"] = "tinyimagenet"
MIX_MAPPING["cifar10_onlylabeled+cifar100"] = "cifar100"
MIX_MAPPING["cifar10+cifar100_onlytest"] = "cifar10"
MIX_MAPPING["cifar100fiftyfive+cifar100fortyfive"] = "cifar100"
MIX_MAPPING["imagenet30_id+imagenet30_ood"] = "imagenet30"

TEST_MAPPING = smart_dict()
TEST_MAPPING["imagenet30"] = ["imagenet30_id", "imagenet30_ood"]

SHAPES = {"cifar10": [32, 32, 3],
          "cifar100": [32, 32, 3],
          "svhn": [32, 32, 3],
          "uniform_noise": [32, 32, 3],
          "lsun_church_outdoor": [32, 32, 3],
          "cifar10animals": [32, 32, 3],
          "cifar10notanimals": [32, 32, 3],
          "cifar100fiftyfive": [32, 32, 3],
          "cifar100fortyfive": [32, 32, 3],
          "tinyimagenet": [64, 64, 3],
          "tin_id70": [64, 64, 3],
          "tin_ood70": [64, 64, 3],
          "tin_unseen60": [64, 64, 3],
          "tin_firsthalf": [64, 64, 3],
          "tin_secondhalf": [64, 64, 3],
          "tinyimagenet_resized": [32, 32, 3],
          "imagenet30_id": [224, 224, 3],
          "imagenet30_ood": [224, 224, 3],
          "imagenet30": [224, 224, 3],
          "imagenet100_id": [224, 224, 3],
          "imagenet100_ood": [224, 224, 3],
          }

NCLASS = {"cifar10": 10,
          "cifar100": 100,
          "cifar10animals": 6,
          "cifar100fiftyfive": 55,
          "tin_id70": 70,
          "tin_firsthalf": 100,
          "imagenet30_id": 20,
          "imagenet30": 30,
          "imagenet100_id": 50,
          }

@tf.function
def _smart_resize(img_arr, target_size=256):

    shape = tf.shape(img_arr)
    height, width = shape[0], shape[1]
    target_size = tf.convert_to_tensor(target_size, tf.int32)

    target_height, target_width = tf.cond(height >= width,
                                          lambda: (tf.cast(tf.math.round(target_size*height/width), tf.int32),
                                                           target_size),
                                          lambda: (target_size,
                                                   tf.cast(tf.math.round(target_size*width/height), tf.int32)))

    img_arr = tf.image.resize(img_arr, (target_height, target_width))

    return img_arr

@tf.function
def _center_crop(img_arr, target_size=224):

    shape = tf.shape(img_arr)
    height, width = shape[0], shape[1]

    height_diff = height - target_size
    width_diff = width - target_size

    h_start = tf.cast(height_diff / 2, tf.int32)
    w_start = tf.cast(width_diff / 2, tf.int32)

    img_arr = tf.image.crop_to_bounding_box(
        img_arr, h_start, w_start, target_size, target_size)

    return img_arr

def image_normalization(data):
    data["image"] = tf.cast(data["image"], tf.float32) * 2.0 / 255.0 - 1.0
    return data

def record_parse_batch(serialized_example, image_shape=None):
    features = tf.io.parse_example(
        serialized_example,
        features={'image': tf.io.FixedLenFeature([], tf.string),
                  'label': tf.io.FixedLenFeature([], tf.int64)})
    
    image = tf.map_fn(tf.image.decode_image, features["image"],
                      fn_output_signature=tf.uint8)
    
    if image_shape:
        image = tf.ensure_shape(image, [None] + image_shape)
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(image=image, label=features['label'])

def record_parse_imagenet_batch(item, image_shape=None):
    filenames, labels = item[:,0], item[:,1]

    def full_pipeline(path):
        full_path = tf.strings.join([FLAGS.datadir, path])        
        imgfile = tf.io.read_file(full_path)
        img_arr = tf.io.decode_jpeg(imgfile, channels=3)
        img_arr = _smart_resize(img_arr, 256)
        img_arr = _center_crop(img_arr, 224)
        return img_arr

    image = tf.map_fn(full_pipeline,
                      filenames,
                      fn_output_signature=tf.float32)

    if image_shape:
        image = tf.ensure_shape(image, [None] + image_shape)    

    image = image * 2.0 / 255.0 - 1.0
    return dict(image=image, label=tf.strings.to_number(labels, tf.int64))

def record_parse_imagenet(item, image_shape=None):
    filename, label = item[0], item[1]

    full_path = tf.strings.join([FLAGS.datadir, filename])
    imgfile = tf.io.read_file(full_path)
    img_arr = tf.io.decode_jpeg(imgfile, channels=3)
    img_arr = _smart_resize(img_arr, 256)
    image = _center_crop(img_arr, 224)

    if image_shape:
        image = tf.ensure_shape(image, image_shape)

    image = image * 2.0 / 255.0 - 1.0
    return dict(image=image, label=tf.strings.to_number(label, tf.int64))

def record_parse(serialized_example, image_shape=None):
    features = tf.io.parse_single_example(
        serialized_example,
        features={'image': tf.io.FixedLenFeature([], tf.string),
                  'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features["image"])
    if image_shape:
        image.set_shape(image_shape)
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    return dict(image=image, label=features["label"])

def add_index(idx, data):
    data["index"] = idx
    return data

def ds_from_txt(path):

    items = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            items.append([row[0], row[1]])

    ds = tf.data.Dataset.from_tensor_slices(items)
    return ds

def ds_from_records_or_txt(path):

    txt_path = os.path.splitext(path)[0] + '.txt'
    
    if tf.io.gfile.exists(txt_path):
        return ds_from_txt(txt_path)
    else:
        return tf.data.TFRecordDataset(path)
    

class DataSets:

    def __init__(self, name):

        self.name = name
        self.shape = SHAPES[name]
        self.nclass = NCLASS[name]

        self.train = tf.data.TFRecordDataset(
            os.path.join(FLAGS.datadir, "{}-train.tfrecord".format(name)))

        test_names = TEST_MAPPING[name]
        test_names = [test_names] if isinstance(test_names, str) else test_names
        
        self.test = tf.data.TFRecordDataset(
            [os.path.join(FLAGS.datadir, "{}-test.tfrecord".format(name))
             for name in test_names])
        

class SemiSupervisedDataSets:

    def __init__(self, name, nlabeled, seed):
                        
        self.name = "{}.{}@{}".format(name, seed, nlabeled)
        self.shape = SHAPES[name]
        self.nclass = NCLASS[name]

        self.train_unlabeled = tf.data.TFRecordDataset(
                os.path.join(FLAGS.datadir, "{}-train.tfrecord".format(name)))
        self.train_labeled = tf.data.TFRecordDataset(
            os.path.join(FLAGS.datadir, "SSL2", "{}-label.tfrecord".format(self.name)))
        self.test = tf.data.TFRecordDataset(
            os.path.join(FLAGS.datadir, "{}-test.tfrecord".format(name)))            

        

class OpenSSLDataSets:

    def __init__(self, id_name, ood_name, nlabeled, seed, unseen):

        self.name = "{}.{}@{}+{}".format(id_name, seed, nlabeled, ood_name)
        
        # load id data        
        id_set_name = SET_MAPPING[id_name]
        labeled_filename = "{}.{}@{}".format(id_set_name, seed, nlabeled)

        self.shape = SHAPES[id_set_name]
        self.nclass = NCLASS[id_set_name]

        self.train_labeled = ds_from_records_or_txt(
            os.path.join(FLAGS.datadir, "SSL2", "{}-label.tfrecord".format(labeled_filename)))
        
        self.test = ds_from_records_or_txt(
            os.path.join(FLAGS.datadir, "{}-test.tfrecord".format(id_set_name)))

        # load ood data
        ood_set_name = SET_MAPPING[ood_name]

        id_train_path = os.path.join(FLAGS.datadir, "{}-train.tfrecord".format(id_name))
        ood_train_path = os.path.join(FLAGS.datadir, "{}-train.tfrecord".format(ood_name))
        
        test_ood_path = os.path.join(FLAGS.datadir, "{}-test.tfrecord".format(ood_set_name))

        test_unseen_path = os.path.join(FLAGS.datadir, "{}-test.tfrecord".format(unseen))

        assert self.shape == SHAPES[ood_set_name]
        assert self.shape == SHAPES[unseen]
        
        self.train_unlabeled_id = ds_from_records_or_txt(id_train_path)
        self.train_unlabeled_ood = ds_from_records_or_txt(ood_train_path)
        
        self.test_unseen = ds_from_records_or_txt(test_unseen_path)
        
        self.test_ood = ds_from_records_or_txt(test_ood_path)


class TestSets:


    def __init__(self):


        self.datasets = {}

        self.datasets["svhn"] = tf.data.TFRecordDataset(
            os.path.join(FLAGS.datadir, "svhn-test.tfrecord"))
        self.datasets["cifar10"] = tf.data.TFRecordDataset(
            os.path.join(FLAGS.datadir, "cifar10-test.tfrecord"))
        self.datasets["cifar100"] = tf.data.TFRecordDataset(
            os.path.join(FLAGS.datadir, "cifar100-test.tfrecord"))


class parse_dict(dict):
    @staticmethod
    def __missing__(key):
        return record_parse
        
PARSEDICT = parse_dict()
PARSEDICT["imagenet100_id"] = record_parse_imagenet
PARSEDICT["imagenet100_ood"] = record_parse_imagenet
