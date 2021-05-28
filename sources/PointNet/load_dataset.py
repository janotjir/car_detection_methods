'''
    Dataset for wheel recognition
'''

import os
import os.path
import numpy as np
import sys
import h5py


def pc_normalize(pc, norm_type):
    # zero-mean
    # centroid = np.mean(pc[:,0:2], axis=0)
    # pc[:,0:2] = pc[:,0:2] - centroid

    # normalization - scaling (fixed or dynamic)
    if norm_type == 0:
        m = np.max(np.sqrt(pc[:,0] **2 + pc[:,1]**2))
        pc[:, 0:2] = pc[:, 0:2] / m
    else:
        pc[:, 0:2] = pc[:, 0:2] / 40

    return pc


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def loadDataFile(filename):
    return load_h5(filename)


class CarDataset():
    def __init__(self, root, num_classes=3, npoints=1200, classification=False, normalize=-1):
        self.root = root
        self.npoints = npoints
        self.classification = classification
        self.normalize = normalize
        self.num_classes = num_classes

        self.data = []  # array of arrays with points
        self.labels = []  # array of arrays with labels
        self.data, self.labels = loadDataFile(os.path.join(self.root, "data.h5"))

        print(self.data.shape)
        print(self.labels.shape)

    def __getitem__(self, index):
        data = self.data[index]
        point_set = data[:, 0:3].astype(np.float32)
        #redundant_data = data[:, 3:6]
        label = self.labels[index].astype(np.int32)

        # normalization
        centroid = np.mean(point_set[:, 0:2], axis=0)
        point_set[:, 0:2] = point_set[:, 0:2] - centroid
        point_set[point_set[:, 2] == 1, 2] = 0.4
        point_set[point_set[:, 2] == 2, 2] = 0.7
        point_set[point_set[:, 2] == 3, 2] = 1
        if self.normalize != -1:
            point_set = pc_normalize(point_set, self.normalize)

        # resample and choose accurate quantity of point set
        choice = np.random.choice(len(label), self.npoints, replace=True)
        point_set = point_set[choice, :]
        label = label[choice]

        return point_set, label

    def __len__(self):
        return len(self.labels)
