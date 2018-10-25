import matplotlib.pyplot as plt
import numpy as np
import lda
import pickle
from collections import defaultdict
import h5py

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' % path)
        return file


# word_to_idx: Mapping dictionary from word to index
def _build_caption_vector(annotations, word_to_idx, max_length=15):

    captions = {}
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ")
        cap_vec = []
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        captions[i] = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions


# file_names: Image file names of shape (82783, )
def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    print len(annotations['file_name'])
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1
        
    print idx

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations['image_id']), dtype=np.int32)

    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' % path)


def main():
    annotations_train = load_pickle('./f8_topic/%s.annotations.pkl' % ('train'))
    annotations_val = load_pickle('./f8_topic/%s.annotations.pkl' % ('val'))
    annotations_test = load_pickle('./f8_topic/%s.annotations.pkl' % ('test'))
    word_to_idx = load_pickle('./f8_topic/word_to_idx.pkl')
    
#    xx = load_pickle('/home/Fdisk/imagecaption/data/f8_data/test/test.file.names.pkl')
#    xx1 = load_pickle('/home/Fdisk/imagecaption/data/f8_data/val/val.file.names.pkl')

    



    annotations = {}
    for split in ['image_id', 'file_name', 'caption']:
        x = annotations_train[split]
        x = x.tolist()
        y = annotations_val[split]
        y = y.tolist()
        z = annotations_test[split]
        z = z.tolist()
        for i in range(len(y)):
            x.append(y[i])
        for j in range(len(z)):
            x.append(z[j])
        annotations[split] = x
            


    len_vocab = len(word_to_idx)
    print len_vocab
    captions = _build_caption_vector(annotations, word_to_idx, 15)
    
    file_names, id_to_idx = _build_file_names(annotations)
    image_idxs = _build_image_idxs(annotations, id_to_idx)


    #ldac = np.zeros((82783 + 4052 + 4047, len_vocab)).astype(np.int32)
    ldac = np.zeros((5999 + 1000 + 1000, len_vocab)).astype(np.int32)
    for i in range(len(captions)):
        for j in range(len(captions[i])):
            if captions[i][j] < len_vocab:
                ldac[image_idxs[i]][captions[i][j]] += 1

    X = ldac
    model = lda.LDA(n_topics=80, n_iter=2000, random_state=1)
    model.fit(X)
    plt.plot(model.loglikelihoods_[5:])
    doc_topic = model.doc_topic_

    save_path = h5py.File('./f8_topic/f8_topic.h5','w')
    save_path.create_dataset('all_topic', data=doc_topic)
    print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()