import argparse
import glob
import os
import sys
import random
import shutil

import numpy as np
import tensorflow as tf

from utils import get_module_logger, get_dataset

def split_files(filenames, source, dest, train_share=0.8, val_share=0.1):
    """Given a list of filenames shuffle them and return
    three lists train, val and test
    """
    np.random.shuffle(filenames)

    ntrain = int(train_share*len(filenames))
    nval = int((train_share+val_share)*len(filenames))

    train, val, test = np.split(filenames, [ntrain, nval])

    train_dir = os.path.join(dest, "train")
    val_dir = os.path.join(dest, "val")
    test_dir = os.path.join(dest, "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    for filename in train:
        shutil.move(os.path.join(source, filename), train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    for filename in val:
        shutil.move(os.path.join(source, filename), val_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for filename in test:
        shutil.move(os.path.join(source, filename), test_dir)

def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """

    # Given perfect shuffling we could perform split randomly. Perfect shuffling could guarantee
    # that images of each type end up in each of the train, val and test sets. However perfect
    # shuffling is imposible due to RAM limitations of one local machine.

    # Since the number of images with cyclists is very low we are going to treat them separately.
    # This is to ensure they are well represented in all three sets

    tfrecord_path = os.path.join(source, "*.tfrecord")
    dataset = get_dataset(tfrecord_path)

    DEVELOP = False
    if DEVELOP:
        dataset = dataset.take(500)

    def cyclist_present(ex):
        return tf.reduce_any(tf.equal(ex["groundtruth_classes"], tf.constant(4, dtype=tf.int64)))

    dataset_cyclists = dataset.filter(cyclist_present)
    cyclists_count = dataset_cyclists.reduce(0, lambda count, _: count + 1)
    logger.info(f"Found {cyclists_count} examples with cyclists")

    def preproces_filenames(ds):
        filenames = ds.map(lambda ex: ex['source_id'])
        filenames = list(filenames.as_numpy_iterator())
        filenames = [name.decode('utf-8') for name in filenames]
        filenames = list(map(lambda s: s[:s.rfind('_')] + '.tfrecord', filenames))
        filenames = list(set(filenames))

        return filenames

    filenames_cyclist = preproces_filenames(dataset_cyclists)
    logger.info(f"Found {len(filenames_cyclist)} tfrecords with cyclists")

    dataset_other = dataset.filter(lambda ex: not cyclist_present(ex))
    count_other = dataset_other.reduce(0, lambda count, _: count + 1)
    logger.info(f"Found {count_other} images without cyclists")

    filenames_other = preproces_filenames(dataset_other)
    filenames_other = list(set(filenames_other) - set(filenames_cyclist))
    logger.info(f"Found {len(filenames_other)} tfrecords without cyclists")

    split_files(filenames_cyclist, source, destination)
    split_files(filenames_other, source, destination)

# run like this: 
# python create_splits.py --source /app/project/data --destination /app/project/split
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)