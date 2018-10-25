#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:36:21 2018

@author: xz
"""
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


def inference(input):
    flat = tf.reshape(input, [-1, 14 * 14 * 512])
    
    logits = tf.layers.dense(inputs=flat, units=80)
    logits = tf.nn.dropout(logits, 0.5)
    #logits = tf.layers.dense(inputs=flat, units=80,activation=tf.nn.sigmoid)
    logits=tf.nn.softmax(logits)
    return logits

def train():

    #TODO
    image_topic = []
    topic_path = './val.topics.h5'
    with h5py.File(topic_path, 'r') as f:
        image_topic = np.asarray(f['topics'])
    print ('image_topic ok!')
    # TODO
    features = []
    feature_path = '../data/coco_data/val/val.h5'
    with h5py.File(feature_path, 'r') as f:
        features = np.asarray(f['features'])
    #features = hickle.load(feature_path)
    print ('features ok!')
    
#    features = np.random.rand(5000, 196, 512)
#    image_topic = np.random.rand(5000, 80)
    

    log_path = './log/'
    model_path = './model/'

    n_examples = len(features)
    print(n_examples)
    batch_size = 100
    n_epoch = 20
    save_every = 1

    x = tf.placeholder(tf.float32, [None, 196, 512], name='x-input')
    _y = tf.placeholder(tf.float32, [None, 80], name='y-input')
    y = inference(x)

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=y)) / batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
#    grads = tf.gradients(loss, tf.trainable_variables())
#    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    train_op = optimizer.minimize(loss=loss)
    n_iters_per_epoch = int(np.ceil(float(n_examples) / batch_size))

    tf.summary.scalar('loss', loss)  
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
#    for grad, var in grads_and_vars:
#        tf.summary.histogram(var.op.name+'/gradient', grad)    
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

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
    test()

if __name__ == "__main__":
    main()

