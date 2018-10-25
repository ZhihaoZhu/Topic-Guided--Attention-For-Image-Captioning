from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
import h5py


def _build_vocab1000(annotations, word_to_idx, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']): 
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:  
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    vocab1000 = counter.most_common(150)
    vocab1000_idx = np.ndarray(100).astype(np.int32)

    for i in range(100):
        vocab1000_idx[i] = word_to_idx[vocab1000[i+50][0]]
    return vocab1000_idx

def _get_attribute(annotations, word_to_idx, split, vocab1000):
    image_path = list(annotations['file_name'].unique())
    n_examples = len(image_path)
    length = n_examples
    image_ids = {}



    image_attribute = np.zeros((length, 100)).astype(np.int32)
    i = -1

    for caption, image_id in zip(annotations['caption'], annotations['image_id']):
        if not image_id in image_ids:
            image_ids[image_id] = 0
            i += 1

            words = caption.split(" ")

            for word in words:
                if word in word_to_idx:
                    if word_to_idx[word] in vocab1000:
                        pos = np.argwhere(vocab1000 == word_to_idx[word])
                        image_attribute[i][pos] = 1
    return image_attribute




def main():
    max_length = 15
    word_count_threshold = 1

    for split in ['train','test','val']:
        annotations = load_pickle('../data/f8_data/%s/%s.annotations.pkl' % (split, split))

        word_to_idx = load_pickle('../data/f8_data/%s/word_to_idx.pkl' % ('train'))

        if split == 'train':  
            word_to_idx = load_pickle('../data/f8_data/%s/word_to_idx.pkl' % ('train'))
            vocab1000 = _build_vocab1000(annotations=annotations, word_to_idx=word_to_idx, threshold=word_count_threshold)

            save_pickle(vocab1000, '../data/f8_data/%s/%s.pkl' % ('train', 'vocab1000'))

        attributes = _get_attribute(annotations, word_to_idx, split, vocab1000)

        save_path = h5py.File('../data/f8_data/%s/%s.attributes.h5' %(split, split),'w')
        save_path.create_dataset("attributes", data=attributes)



if __name__ == '__main__':

    main()