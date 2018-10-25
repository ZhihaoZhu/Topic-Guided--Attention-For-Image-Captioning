#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:06:42 2018

@author: xz
"""
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
import shutil

def split_dataset():
    
    train=[]
    val=[]
    test=[]
    for split in ['train']:  
        fd = file( '/home/Fdisk/flcikr8/drive-download-20171213T072617Z-001/Flickr8k_text/Flickr_8k.%s.txt'% (split), "r" )  
      
        for line in fd.readlines():  
            train.append(line.strip())
    
    for split in ['test']:  
        fd = file( '/home/Fdisk/flcikr8/drive-download-20171213T072617Z-001/Flickr8k_text/Flickr_8k.%s.txt'% (split), "r" )  
      
        for line in fd.readlines():  
            test.append(line.strip())
    for split in ['val']:  
        fd = file( '/home/Fdisk/flcikr8/drive-download-20171213T072617Z-001/Flickr8k_text/Flickr_8k.%s.txt'% (split), "r" )  
      
        for line in fd.readlines():  
            val.append(line.strip())
    return train,val,test
#    fileDir='/home/Fdisk/flcikr8/drive-download-20171213T072617Z-001/Flickr8k_Dataset/Flicker8k_Dataset'
#    for root, dirs, files in os.walk(fileDir):
#        for xx in files:
#            old_path=os.path.join(root, xx)
#
#            if xx in train:
#                new_path=os.path.join('/home/Fdisk/flcikr8/f8_train',xx)
#                shutil.copyfile(old_path,new_path)
#            if xx in val:
#                new_path=os.path.join('/home/Fdisk/flcikr8/f8_val',xx)
#                shutil.copyfile(old_path,new_path)  
#            if xx in test:
#                new_path=os.path.join('/home/Fdisk/flcikr8/f8_test',xx)
#                shutil.copyfile(old_path,new_path)  



def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
                
def load_data(doc):
    id_=0
    data_train = []
    data_val = []
    data_test = []
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        file_name = image_id+'.jpg'
        image_desc = ' '.join(image_desc)
        train = []
        val = []
        test = []
        train,val,test = split_dataset()
        if file_name in train:
            file_name = os.path.join('image/trainf8_resized/',file_name)
            mapping['image_id']=image_id
            mapping['caption']=image_desc
            mapping['file_name']=file_name
            mapping['id']=id_
            id_ +=1
            data_train += [mapping]
            mapping={}
        if file_name in val:
            file_name = os.path.join('image/valf8_resized/',file_name)
            mapping['image_id']=image_id
            mapping['caption']=image_desc
            mapping['file_name']=file_name
            mapping['id']=id_
            id_ +=1
            data_val += [mapping]
            mapping={} 
        if file_name in test:
            file_name = os.path.join('image/testf8_resized/',file_name)
            mapping['image_id']=image_id
            mapping['caption']=image_desc
            mapping['file_name']=file_name
            mapping['id']=id_
            id_ +=1
            data_test += [mapping]
            mapping={} 
    return data_train,data_val,data_test

def _process_caption_data(data,max_length):

    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)

    
    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" %len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" %len(caption_data)
    return caption_data
def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file        
def main():


    

    xx = load_pickle('/home/Fdisk/imagecaption/data/f8_data/train/train.file.names.pkl')
    train = []
    val = []
    test = []
    train,val,test = split_dataset()
    
    train_1= []
    for aa in train:
        train_1.append(os.path.join('image/trainf8_resized/',aa))

    for i in train_1:
        if i not in xx:
            print i
            
        
        
    
#    batch_size = 100
#    max_length = 15
#    word_count_threshold = 1
#    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
#    filename = '/home/Fdisk/flcikr8/drive-download-20171213T072617Z-001/Flickr8k_text/Flickr8k.token.txt'
#    doc = load_doc(filename)
#    train_data=[]
#    val_data= []
#    test_data=[]
#    train_data,val_data,test_data= load_data(doc)
#    print len(train_data)
#    train_dataset =_process_caption_data(train_data,max_length=max_length)
#    val_dataset =_process_caption_data(val_data,max_length=max_length)
#    test_dataset =_process_caption_data(test_data,max_length=max_length)  
#    print 'Finished processing caption data'
#    save_pickle(train_dataset, 'data/f8_data/train/train.annotations.pkl')
#    save_pickle(val_dataset, 'data/f8_data/val/val.annotations.pkl')
#    save_pickle(test_dataset, 'data/f8_data/test/test.annotations.pkl')
#    
#    for split in ['train', 'val', 'test']:
#        annotations = load_pickle('./data/f8_data/%s/%s.annotations.pkl' % (split, split))
#
#        if split == 'train':
#            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
#            save_pickle(word_to_idx, './data/f8_data/%s/word_to_idx.pkl' % split)
#        
#        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
#        save_pickle(captions, './data/f8_data/%s/%s.captions.pkl' % (split, split))
#
#        file_names, id_to_idx = _build_file_names(annotations)
#        save_pickle(file_names, './data/f8_data/%s/%s.file.names.pkl' % (split, split))
#
#        image_idxs = _build_image_idxs(annotations, id_to_idx)
#        save_pickle(image_idxs, './data/f8_data/%s/%s.image.idxs.pkl' % (split, split))
#
#        # prepare reference captions to compute bleu scores later
#        image_ids = {}
#        feature_to_captions = {}
#        i = -1
#        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
#            if not image_id in image_ids:
#                image_ids[image_id] = 0
#                i += 1
#                feature_to_captions[i] = []
#            feature_to_captions[i].append(caption.lower() + ' .')
#        save_pickle(feature_to_captions, './data/f8_data/%s/%s.references.pkl' % (split, split))
#        print "Finished building %s caption dataset" %split
#
#    # extract conv5_3 feature vectors
#    vggnet = Vgg19(vgg_model_path)
#    vggnet.build()
#    with tf.Session() as sess:
#        tf.initialize_all_variables().run()
#        for split in [ 'train','val', 'test']:
#            anno_path = './data/f8_data/%s/%s.annotations.pkl' % (split, split)
#            #save_path = './data/%s/%s.features.hkl' % (split, split)
#            save_path = h5py.File('./data/f8_data/%s/%s.h5' %(split, split),'w')
#            #save_path1 = './data/%s/%s.features1.hkl' % (split, split)#
#            annotations = load_pickle(anno_path)
#            image_path = list(annotations['file_name'].unique())
#            n_examples = len(image_path)
#
#            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)
#            #all_feats1 = np.ndarray([n_examples-45000, 196, 512], dtype=np.float32)#
#
#            for start, end in zip(range(0, n_examples, batch_size),
#                                  range(batch_size, n_examples + batch_size, batch_size)):
#                image_batch_file = image_path[start:end]
#                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
#                    np.float32)
#                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
#
#                all_feats[start:end, :] = feats
#
#                print ("Processed %d %s features.." % (end, split))
#
#            # use hickle to save huge feature vectors
#            #hickle.dump(all_feats, save_path)
#            save_path.create_dataset('features', data=all_feats)
#
#            print ("Saved %s.." % (save_path))
    
    


if __name__ == "__main__":
    main()