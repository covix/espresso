#!/usr/bin/env/python3

import numpy as np
import pickle
import sys
import os
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin-1')
    return d


if __name__ == '__main__':
    dirname = sys.argv[1]
    batches = (x for x in os.listdir(dirname) if '_batch' in x)
    output_folder = 'cifar10-extracted'
    caffe_data_folder = '/opt/caffe/data/'
    txtfiles = {}

    for i in ['train', 'val']:
        os.makedirs(os.path.join(output_folder, i), exist_ok=True)
        txtfiles[i] = open(os.path.join(output_folder, i + '.txt'), 'w')

    for batch in batches:
        datadict = unpickle(os.path.join(dirname, batch))
        print("processing", datadict['batch_label'])

        ds = 'val' if 'test' in datadict['batch_label'] else 'train'

        X = datadict["data"]
        Y = datadict['labels']
        filenames = datadict['filenames']

        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        Y = np.array(Y)

        for i in range(X.shape[0]):
            im = Image.fromarray(X[i])
            im.save(os.path.join(output_folder, ds, filenames[i]))
            image_line = "{} {}\n".format(filenames[i], Y[i])
            txtfiles[ds].write(image_line)

            if i % 1000 == 0:
                print("\tSaved {} images".format(i))
