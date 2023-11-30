import tensorflow as tf
import math
import io
import csv
import os
import json
import tqdm

class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_steps, decay_factor=7/8, pretrain_steps=0):

        super(CosineDecay, self).__init__()

        # TODO: maybe rename attributes to decay_end and decay_start

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps # where to end decay
        self.decay_factor = decay_factor
        self.pretrain_steps = pretrain_steps # where to start decay

    def __call__(self, step):

        with tf.name_scope("CosineDecay"):

            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_factor = tf.cast(self.decay_factor, dtype)

            pretrain_steps = tf.cast(self.pretrain_steps, dtype)
            
            global_step_recomp = tf.cast(step, dtype)
            complete_fraction = tf.clip_by_value(
                (global_step_recomp-pretrain_steps)/(decay_steps-pretrain_steps),
                0.0,
                1.0)

            decay_factor = tf.cos(decay_factor*tf.constant(math.pi, dtype=dtype)*complete_fraction/2.0)

            return tf.multiply(initial_learning_rate, decay_factor)

class ConstantLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate,):
        super(ConstantLR, self).__init__()
        self.learning_rate = tf.cast(learning_rate, tf.float32)

    def __call__(self, step):
        return self.learning_rate

class T2TSchedule(CosineDecay):

    def __init__(self, initial_piecewise_lr, gamma, boundaries,
                 initial_cosine_lr, decay_steps, decay_factor=7/8, transition=0):

        super(T2TSchedule, self).__init__(
            initial_cosine_lr, decay_steps, decay_factor, pretrain_steps=0)

        self.transition = transition
        self.initial_piecewise_lr = initial_piecewise_lr
        self.gamma = gamma
        self.boundaries = boundaries

    def __call__(self, step):
        step_recomp = tf.convert_to_tensor(step)
        transition = tf.convert_to_tensor(self.transition, dtype=step.dtype)

        return tf.cond(step_recomp >= transition,
                       lambda: super(T2TSchedule, self).__call__(step_recomp),
                       lambda: self.piecewise_warmup(step_recomp))

    def piecewise_warmup(self, step):

        lr = tf.convert_to_tensor(self.initial_piecewise_lr)
        gamma = tf.convert_to_tensor(self.gamma)

        boundaries = tf.nest.map_structure(
            tf.convert_to_tensor, tf.nest.flatten(self.boundaries))

        nr_steps = tf.convert_to_tensor(len(self.boundaries))
        step_recomp = tf.convert_to_tensor(step)

        for i, b in enumerate(boundaries):
            b = tf.cast(b, step_recomp.dtype.base_dtype)
            boundaries[i] = b

        pred_fn_pairs = []
        pred_fn_pairs.append((step_recomp <= boundaries[0], lambda: lr))
        pred_fn_pairs.append(
            (step_recomp > boundaries[-1], lambda: lr * tf.pow(gamma, tf.cast(nr_steps, tf.float32))))

        #for low, high, i in zip(boundaries[:-1], boundaries[1:], range(1, nr_steps)):

        i = 1
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            pred = (step_recomp > low) & (step_recomp <= high)
            pred_fn_pairs.append((pred, lambda i=i: lr * tf.pow(gamma, i)))
            i += 1

        return tf.case(pred_fn_pairs, exclusive=True)


            

            
        

        
        






def log_partition(logits):
    return -tf.reduce_logsumexp(logits, axis=1)

def dataset_cardinality(dataset):
    # TODO: should implement something to avoid infinite loops here
    
    count = 0
    for item in dataset:
        count += 1

    return count
    
def plot_to_image(fig):
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def energy_histograms(id_vals, ood_vals, train_vals, ax, title="ema"):
    ax.hist(id_vals, color="red", density=True, bins=20, alpha=0.3, label="ID")
    ax.hist(ood_vals, color="blue", density=True, bins=20, alpha=0.3, label="OOD")
    ax.hist(train_vals, color="green", density=True, bins=20, alpha=0.3, label="TRAIN")
    ax.set_ylabel("density")
    ax.set_xlabel("E(x)")
    ax.set_title(title)
    ax.legend()
    return ax

def maxpreds_histograms(id_vals, ood_vals, ax):

    ax.hist(id_vals, color="red", density=True, bins=20, alpha=0.3, label="ID")
    ax.hist(ood_vals, color="blue", density=True, bins=20, alpha=0.3, label="OOD")
    ax.set_ylabel("density")
    ax.set_xlabel("maxpred")
    ax.legend()
    return ax

def auroc_rank_calculation(id_vals, ood_vals):

    pairwise_tests = tf.greater(id_vals[:, tf.newaxis], ood_vals[tf.newaxis, :])
    num_pairs = tf.size(id_vals)*tf.size(ood_vals)
    num_pairs = tf.cast(num_pairs, tf.float32)
    auroc = tf.divide(tf.reduce_sum(tf.cast(pairwise_tests, tf.float32)), num_pairs)
    return auroc


def _extrema(arr, func, include_endpoints=True):
    bool1 = func(arr[1:], arr[:-1])
    bool2 = func(arr[:-1], arr[1:])
    bool1 = tf.concat(([include_endpoints], bool1), axis=0)    
    bool2 = tf.concat((bool2, [include_endpoints]), axis=0)
    return tf.logical_and(bool1, bool2)
    
def first_local_minima(arr):

    strict_minima = _extrema(arr, tf.less, False)
    nonstrict_minima = _extrema(arr, tf.less_equal, True)

    has_strict_minima = tf.reduce_any(strict_minima)

    idxs = tf.cond(
        has_strict_minima,
        true_fn=lambda: tf.where(strict_minima),
        false_fn=lambda: tf.where(nonstrict_minima)
    )
    
    return int(idxs[0])

def last_local_maxima(arr):

    strict_maxima = _extrema(arr, tf.greater, False)
    nonstrict_maxima = _extrema(arr, tf.greater_equal, True)

    has_strict_maxima = tf.reduce_any(strict_maxima)

    idxs = tf.cond(
        has_strict_maxima,
        true_fn=lambda: tf.where(strict_maxima),
        false_fn=lambda: tf.where(nonstrict_maxima)
    )
    
    return int(idxs[-1])

def print_model_summary(model, name):

    WIDTH=70

    def shorten_variable_name(name_string):
        splitted = name_string.split("/")
        if len(splitted)>1:
            return "{}/{}".format(splitted[-2], splitted[-1])
        else:
            return splitted[-1]

    format_string = "{:<35} {:<20} {}"

    print("="*WIDTH)
    print("Model: {}".format(name))
    print("-"*WIDTH)
    print(format_string.format("Variable", "Shape", "Nr. params"))
    print("-"*WIDTH)

    for var in model.trainable_variables:
        print(format_string.format(
            shorten_variable_name(var.name),
            str(var.shape.as_list()),
            tf.reduce_prod(var.shape).numpy()))

    print("-"*WIDTH)
    print("Nr trainable params: {:,}".format(
        sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])))
    print("Nr non-trainable params: {:,}".format(
        sum([tf.reduce_prod(var.shape) for var in model.non_trainable_variables])))
    print("="*WIDTH)

def save_energies_csv(energies,
                      energies_ema,
                      similarities,
                      similarities_ema,
                      datasets,
                      labels,
                      save_dir,
                      step):

    fname = os.path.join(save_dir, "energies_{}.csv".format(step.read_value()))

    def set_precision(list_, decimals=4):
        return ["{:.{p}f}".format(x, p=decimals) for x in list_]
    
    zipdata = list(zip(set_precision(energies.numpy().tolist()),
                       set_precision(energies_ema.numpy().tolist()),
                       set_precision(similarities.numpy().tolist()),
                       set_precision(similarities_ema.numpy().tolist()),
                       datasets.numpy().astype(str).tolist(),
                       labels.numpy().tolist()))
    
    with open(fname, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["energies", "energies_ema", "similarities", "similarities_ema", "dataset", "labels"])
        writer.writerows(zipdata)

def pairwise_distances(a, b):
    """
    Computes pairwise euclidian distances between the rows of a and the rows of b
    """
    asquare = tf.reduce_sum(tf.pow(a, 2), axis=1, keepdims=True)
    bsquare = tf.reduce_sum(tf.pow(b, 2), axis=1, keepdims=True)
    d = asquare + tf.transpose(bsquare) - 2*tf.matmul(a, b, transpose_b=True)

    return tf.sqrt(d)
    
        
def save_stats_csv(data, labels, save_dir, step, floatdecimals=4):

    fname = os.path.join(save_dir, "stats_{}.csv".format(step.read_value()))
    assert len(data) == len(labels)

    def set_precision(list_, decimals=floatdecimals):
        return ["{:.{p}f}".format(x, p=decimals) for x in list_]

    data = [set_precision(x) if isinstance(x[0], float) else x for x in data]
    
    zipdata = list(zip(*data))

    with open(fname, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        writer.writerows(zipdata)


def create_labeled_mask(data_dir, id_name, nlabeled, seed, ds, ulset_size):

    def parse(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={"dataset": tf.io.FixedLenFeature([], tf.string, default_value="missing"),
                      "index": tf.io.FixedLenFeature([], tf.int64)})
        return dict(dataset=features["dataset"], index=features["index"])
    
    ds = ds.map(parse)
    mask = [False]*int(ulset_size)
    idxs_dict = {}

    name = "{}.{}@{}-label.json".format(id_name, seed, nlabeled)
    path = os.path.join(data_dir, "SSL2", name)
    
    with open(path) as f:
        data = json.load(f)

    label_idxs = data["label"]

    progress = tqdm.tqdm(iterable=None, desc="Fetching indices from unlabeled data", unit="Img")
    for i, item in enumerate(ds):
        dataset = item["dataset"]
        idx = int(item["index"])
        
        if dataset == id_name or dataset == "missing":
            idxs_dict[idx] = i
            
        progress.update()

    progress = tqdm.tqdm(iterable=None, desc="Creating labeled mask", unit="Labeled img")
    for label_idx in label_idxs:
        i = idxs_dict[label_idx]
        mask[i] = True
        progress.update()

    del idxs_dict
        
    mask = tf.constant(mask, tf.bool)

    assert nlabeled == tf.reduce_sum(tf.cast(mask, tf.int64))

    return mask
        
    
