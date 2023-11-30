import collections
import os
import tarfile
import tempfile
from urllib import request

import numpy as np
import scipy.io
import tensorflow as tf
from absl import app, flags
from tqdm import trange

FLAGS = flags.FLAGS

URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz',
}


def _encode_png(images):
    raw = []
    for x in trange(images.shape[0], desc="PNG Encoding", leave=False):
        raw.append(tf.image.encode_png(images[x]))
    return raw


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] -= 1
        splits[split] = dataset
    return splits

def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_cifar100():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar100'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/train.mat'))
        train_set = {'images': data_dict['data'],
                     'labels': data_dict['fine_labels'].flatten()}
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/test.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['fine_labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _load_noise():
    train_size = 50000
    test_size = 10000
    image_size = 32

    train_data = tf.random.uniform([train_size, image_size, image_size, 3],
                                  minval=0,
                                  maxval=255,
                                  dtype=tf.int32)
    train_data = tf.cast(train_data, tf.uint8)

    test_data = tf.random.uniform([test_size, image_size, image_size, 3],
                                  minval=0,
                                  maxval=255,
                                  dtype=tf.int32)
    test_data = tf.cast(test_data, tf.uint8)

    train_set = {"images": _encode_png(train_data),
                 "labels": tf.zeros([train_size], tf.int64)}
    test_set = {"images": _encode_png(test_data),
                 "labels": tf.zeros([test_size], tf.int64)}

    return dict(train=train_set, test=test_set)    
    
    

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(FLAGS.datadir, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x].numpy()),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(FLAGS.datadir, '%s-%s.tfrecord' % (name, subset))
        if not tf.io.gfile.exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        tf.io.gfile.makedirs(os.path.join(FLAGS.datadir, folder))
    for filename, contents in files.items():
        with tf.io.gfile.GFile(os.path.join(FLAGS.datadir, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return tf.io.gfile.exists(os.path.join(FLAGS.datadir, name, folder))


CONFIGS = dict(
    cifar10=dict(loader=_load_cifar10, checksums=dict(train=None, test=None)),
    cifar100=dict(loader=_load_cifar100, checksums=dict(train=None, test=None)),
    svhn=dict(loader=_load_svhn, checksums=dict(train=None, test=None, extra=None)),
    uniform_noise=dict(loader=_load_noise, checksums=dict(train=None, test=None)),
)


def main(argv):
    if len(argv[1:]):
        subset = set(argv[1:])
    else:
        subset = set(CONFIGS.keys())
    tf.io.gfile.makedirs(FLAGS.datadir)
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(FLAGS.datadir, '%s-%s.txt' % (name, sub_name))
                with tf.io.gfile.GFile(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(FLAGS.datadir, file_and_data.filename)
                    with tf.io.gfile.GFile(path, "wb") as f:
                        f.write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))


if __name__ == '__main__':
    flags.DEFINE_string("datadir", ".", "Directory for data")
    app.run(main)
