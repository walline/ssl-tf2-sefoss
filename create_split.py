import json
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange, tqdm

FLAGS = flags.FLAGS


def get_class(serialized_example):
    return tf.io.parse_single_example(serialized_example,
                                      features={'label': tf.io.FixedLenFeature([], tf.int64)})['label']

def main(argv):
    assert FLAGS.size
    argv.pop(0)
    if any(not tf.io.gfile.exists(f) for f in argv[1:]):
        raise FileNotFoundError(argv[1:])
    target = '%s.%d@%d' % (argv[0], FLAGS.seed, FLAGS.size)
    if tf.io.gfile.exists(target):
        raise FileExistsError('For safety overwriting is not allowed', target)
    input_files = argv[1:]
    count = 0
    id_class = []
    class_id = defaultdict(list)
    print('Computing class distribution')
    dataset = tf.data.TFRecordDataset(input_files).map(get_class, 4)

    for i in tqdm(dataset, leave=False):
        i = int(i)
        id_class.append(i)
        class_id[i].append(count)
        count += 1
    
    print('%d records found' % count)
    nclass = len(class_id)
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
    train_stats /= train_stats.max()

    print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]
    if FLAGS.seed:
        np.random.seed(FLAGS.seed)
        for i in range(nclass):
            np.random.shuffle(class_id[i])

    # Distribute labels to match the input distribution.
    npos = np.zeros(nclass, np.int64)
    label = []
    for i in range(FLAGS.size):
        c = np.argmax(train_stats - npos / max(npos.max(), 1))
        label.append(class_id[c][npos[c]])
        npos[c] += 1

    del npos, class_id
    label = frozenset([int(x) for x in label])

    print('Creating split in %s' % target)
    tf.io.gfile.makedirs(os.path.dirname(target))
    with tf.io.TFRecordWriter(target + '-label.tfrecord') as writer_label:
        pos, loop = 0, trange(count, desc='Writing records')
        for input_file in input_files:
            for record in tf.data.TFRecordDataset(input_file):
                if pos in label:
                    writer_label.write(record.numpy())
                pos += 1
                loop.update()
        loop.close()
    with tf.io.gfile.GFile(target + '-label.json', 'w') as writer:
        writer.write(json.dumps(dict(distribution=train_stats.tolist(), label=sorted(label)), indent=2, sort_keys=True))

if __name__ == '__main__':
    flags.DEFINE_integer('seed', 0, 'Random seed to use, 0 for no shuffling.')
    flags.DEFINE_integer('size', 0, 'Size of labelled set.')
    app.run(main)
