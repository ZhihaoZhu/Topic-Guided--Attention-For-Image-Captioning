import numpy as np
import cPickle as pickle
import hickle
import time
import os
import h5py

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  


data_path='./data/coco_data' 
split='test'
data_path = os.path.join(data_path, split)

data = {}
  
   
with open(os.path.join(data_path, '%s.file.names.pkl' %split), 'rb') as f:
    data['file_names'] = pickle.load(f)   
with open(os.path.join(data_path, '%s.captions.pkl' %split), 'rb') as f:
    data['captions'] = pickle.load(f)

with open(os.path.join(data_path, '%s.image.idxs.pkl' %split), 'rb') as f:
    data['image_idxs'] = pickle.load(f)
    
with open(os.path.join(data_path, '%s.candidate.captions_seq.pkl' %split), 'rb') as f:
    data['captions_seq'] = pickle.load(f)

attributes_file = os.path.join(data_path, '%s.attributes.h5' %split)
with h5py.File(attributes_file, 'r') as f:
    data['attributes'] = np.asarray(f['attributes']) 
    

    
print len(data['captions_seq'])
for i in range(10):
    for j in range(20):
        if data['captions_seq'][i][j]//1 != 0:
            print 'false'
            print data['captions_seq'][i][j]







    