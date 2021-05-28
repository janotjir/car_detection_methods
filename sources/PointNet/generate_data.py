#!/usr/bin/env python
import os
import h5py
import numpy as np
import math
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='trn', help="choose which type of data to generate")
opt = parser.parse_args()


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def euclidean_distance(x_1, x_2, y_1, y_2):
    return math.sqrt((x_1-x_2)*(x_1-x_2)+(y_1-y_2)*(y_1-y_2))


if opt.type == 'tst':
    data, labels = load_h5('../../datasets/tst_data.h5')
    out_dir = "tst" + "_data_PN"
else:
    data, labels = load_h5('../../datasets/trn_data.h5')
    out_dir = "trn" + "_data_PN"


npoints = 1200
data_out = np.empty((0, npoints, 3), dtype = np.float32)
labels_out = np.empty((0, npoints))



for i in range(0, len(data), 1):
    print("Processing {}/{}".format(i, len(data)))

    frame = data[i][:, :3]
    frame_labels = labels[i]

    origin_x = frame[0][0]
    origin_y = frame[0][1]

    frame = frame[1:][:]  # discard robot position
    frame_labels = frame_labels[1:][:]  # discard robot position

    distances = []
    centroid = np.mean(frame[:, 0:2], axis=0)

    id = 0
    # delete all 0 positions that were added and calculate distances
    while id != len(frame_labels):
        if frame[id][0] == 0.0 and frame[id][1] == 0.0 and frame[id][2] == 0.0 and frame_labels[id] == 0:
            frame = np.delete(frame, id, 0)
            frame_labels = np.delete(frame_labels, id)
        else:
            dist = euclidean_distance(origin_x, frame[id][0], origin_y, frame[id][1])
            frame[id][0] -= centroid[0]
            frame[id][1] -= centroid[1]
            if frame_labels[id] == 1:
                distances.append(-dist)
            else:
                distances.append(dist)
            id += 1

    # according to length of point set duplicate or delete some furthest points
    act_npoints = len(frame)
    if act_npoints < npoints:
        # duplicate
        for i in range(npoints-act_npoints):
            act_id = random.randint(0, act_npoints-1)
            frame = np.vstack((frame, [frame[act_id]]))
            frame_labels = np.hstack((frame_labels, frame_labels[act_id]))
    elif act_npoints > npoints:
        # delete
        indexes = np.argsort(distances)
        frame_labels = frame_labels[indexes]
        frame = frame[indexes]

        frame_labels = frame_labels[:npoints]
        frame = frame[:npoints]

    data_out = np.vstack((data_out, [frame]))
    labels_out = np.vstack((labels_out, [frame_labels]))



data_out = np.asarray(data_out)
labels_out = np.asarray(labels_out)

try:
    os.makedirs(out_dir)
except OSError:
    pass

dataFile = out_dir + "/" + "data.h5"
f = h5py.File(dataFile, 'w')
lab = f.create_dataset("label", data=labels_out)
dat = f.create_dataset("data", data=data_out)
f.close()
print("Generated data with labels of size {}".format(data_out.shape))



