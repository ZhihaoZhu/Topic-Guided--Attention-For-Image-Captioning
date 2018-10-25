# -*- coding: utf-8 -*-

import h5py
import numpy as np


def main():
    with h5py.File('./f8_topic/f8_topic.h5', 'r') as f:
        topic_train = f['all_topic'][0:5999]
        topic_val = f['all_topic'][5999 : 5999+1000]
        topic_test = f['all_topic'][5999+1000 : 5999+2000]

        train = h5py.File("../data/f8_data/train/train.topics.h5", "w")
        train.create_dataset("topics", data=topic_train)

        val = h5py.File("../data/f8_data/val/val.topics.h5", "w")
        val.create_dataset("topics", data=topic_val)

        test = h5py.File("../data/f8_data/test/test.topics.h5", "w")
        test.create_dataset("topics", data=topic_test)

if __name__ == '__main__':
    main()