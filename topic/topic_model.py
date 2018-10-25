#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:31:21 2018

@author: xz
"""


# Imports
import numpy as np
import tensorflow as tf
import pickle
import os
import hickle
import h5py

def sample_coco_minibatch(topic_data, feature, batch_size):
    data_size = feature.shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = feature[mask]
    file_names = topic_data[mask]
    return features, file_names


#def convLayer(x, kHeight, kWidth, strideX, strideY,  
#              featureNum, name, padding = "SAME"):  
#    """convlutional"""  
#    channel = int(x.get_shape()[-1]) #获取channel数  
#    with tf.variable_scope(name) as scope:  
#        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])  
#        b = tf.get_variable("b", shape = [featureNum])  
#
##        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
##        x = tf.nn.bias_add(x, b, name='bias_add')
##        x = tf.nn.relu(x, name='relu')
#
#        
#        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)  
#        out = tf.nn.bias_add(featureMap, b)
#        return tf.nn.relu(out, name = scope.name) 
#        #return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)  
#
#
#def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):  
#    """max-pooling"""  
#    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],  
#                          strides = [1, strideX, strideY, 1], padding = padding, name = name) 
#    
def dropout(x, keepPro, name = None):  
    """dropout"""  
    return tf.nn.dropout(x, keepPro, name)  
#  
#def fcLayer(x, inputD, outputD, reluFlag, name):  
#    """fully-connect"""  
#    with tf.variable_scope(name) as scope:  
#        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")  
#        b = tf.get_variable("b", [outputD], dtype = "float")  
#        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)  
#        if reluFlag:  
#            return tf.nn.relu(out)  
#        else:  
#            return out  
    

def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


def FC_layer(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
    return x
        
KEEPPRO = 0.5
is_pretrain = True

def inference(input):
    #input = tf.convert_to_tensor(input)
    flat = tf.reshape(input, [-1,196,512,3])
    x = conv('conv5_4', flat, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True) 

    x = FC_layer('fc6', x, out_nodes=4096)
    x = dropout(x,KEEPPRO)
    x = batch_norm(x)
    x = FC_layer('fc7', x, out_nodes=1000)
    x = dropout(x,KEEPPRO)
    x = batch_norm(x)
    x = FC_layer('fc8', x, out_nodes=80)        
    
    
#    fcIn = tf.reshape(pool5, [-1, 7*7*512])
#    fc6 = fcLayer(fcIn, 7*7*512, 4096, True, "fc6")  
#    dropout1 = dropout(fc6, KEEPPRO)
#
#    fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")  
#    dropout2 = dropout(fc7, KEEPPRO) 
#
#
#    fc8 = fcLayer(dropout2, 4096, 1000, True, "fc8") 
#    dropout3 = dropout(fc8, KEEPPRO)  
#    
#    fc9 = fcLayer(dropout3,1000,80,True,"fcout")    
    
    
    #logits = tf.layers.dense(inputs=pool5, units=80)
    #logits = tf.nn.dropout(logits, 0.5)
    #logits = tf.layers.dense(inputs=flat, units=80,activation=tf.nn.sigmoid)
    #logits=tf.nn.softmax(logits)
    return x

def train():

    #TODO
#    image_topic = []
#    topic_path = './val.topics.h5'
#    with h5py.File(topic_path, 'r') as f:
#        image_topic = np.asarray(f['topics'])
#    print ('image_topic ok!')
#    # TODO
#    features = []
#    feature_path = '../data/coco_data/val/val.h5'
#    with h5py.File(feature_path, 'r') as f:
#        features = np.asarray(f['features'])
#    #features = hickle.load(feature_path)
#    print ('features ok!')
    
    features = np.random.rand(5000, 196, 512)
    image_topic = np.random.rand(5000, 80)
    

    log_path = './log/'
    model_path = './model/'

    n_examples = len(features)
    print(n_examples)
    batch_size = 64
    n_epoch = 10
    save_every = 1

    x = tf.placeholder(tf.float32, [batch_size, 196, 512,3], name='x-input')
    _y = tf.placeholder(tf.float32, [batch_size, 80], name='y-input')
    y = inference(x)

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)) / batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#    grads = tf.gradients(loss, tf.trainable_variables())
#    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    train_op = optimizer.minimize(loss=loss)
    n_iters_per_epoch = int(np.ceil(float(n_examples) / batch_size))
    print (n_iters_per_epoch)

    tf.summary.scalar('loss', loss)  
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
#    for grad, var in grads_and_vars:
#        tf.summary.histogram(var.op.name+'/gradient', grad)    
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    
    
#    config = tf.ConfigProto(allow_soft_placement = True)
#    #config.gpu_options.per_process_gpu_memory_fraction=0.9
#    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        print '-.-'
        for e in range(n_epoch):
            rand_idxs = np.random.permutation(n_examples)

            for i in range(n_iters_per_epoch):
                xs = features[rand_idxs[i * batch_size:(i + 1) * batch_size]]
                ys = image_topic[rand_idxs[i * batch_size:(i + 1) * batch_size]]
                feed_dict={x: xs, _y: ys}
                _, l = sess.run([train_op, loss], feed_dict)

                if i % 40 == 0:
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, e * n_iters_per_epoch + i)
                    #print ("Processed %d features.." % (e * n_iters_per_epoch + i*batch_size))

            if (e + 1) % save_every == 0:
                saver.save(sess, model_path+'model.ckpt', global_step=e + 1)
                print("model-%s saved." % (e + 1))

def test():
    x = tf.placeholder(tf.float32, [None, 196,512], name='x-input')
    # _y = tf.placeholder(tf.float32, [None, 80], name='y-input')
    y = inference(x)
    #y = tf.sigmoid(y)
    #ys = tf.nn.softmax(y)
    
    features = []
    feature_path = '../data/coco_data/val/val.h5'
    with h5py.File(feature_path, 'r') as f:
        features = np.asarray(f['features'])

    image_topic = []
    topic_path = './val.topics.h5'
    with h5py.File(topic_path, 'r') as f:
        image_topic = np.asarray(f['topics'])


    logs_train_dir='./model/'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')

        feed_dict = {x: features}
        y = sess.run(y,feed_dict)
        #print(sess.run(op_to_restore, feed_dict))
        print(y[10])
        print(image_topic[10])
       

def main():
    train()

if __name__ == "__main__":
    main()
