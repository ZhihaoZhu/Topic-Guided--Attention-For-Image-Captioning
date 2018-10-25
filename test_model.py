#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:23:36 2017

@author: xz
"""

import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.bleu import evaluate


plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


data = load_coco_data(data_path='./data/coco_data/', split='test')
with open('./data/coco_data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

#print '~~~~~~~~~~~~~~~~~~~~~~~'
#
#for i in range(data['features'].shape[0]):
#    
#    if data['file_names'][i] =='image/train2014_resized/COCO_train2014_000000013140.jpg':
#        print i
#        print data['file_names'][i]
#print data['file_names'][1813]


model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                   dim_hidden=1024, n_time_step=16, prev2out=True, 
                                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

solver = CaptioningSolver(model, data, data, n_epochs=20, batch_size=128, update_rule='adam',
                                      learning_rate=0.0025, print_every=2000, save_every=1, image_path='./image/val2014_resized',
                                pretrained_model=None, model_path='./model/preview_model/', test_model='./model/preview_model/model-20',
                                 print_bleu=False, log_path='./log/')


#solver.test(data, split='val')
#test = load_coco_data(data_path='./data/coco_data', split='test')
#tf.get_variable_scope().reuse_variables()
solver.test(data, split='test')
#evaluate(data_path='./data/coco_data', split='val')
evaluate(data_path='./data/coco_data', split='test')

#solver.test(data, split='test')
#
#evaluate(data_path='./data', split='test')
