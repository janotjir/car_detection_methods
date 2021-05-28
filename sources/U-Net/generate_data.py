#!/usr/bin/env python
import os
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='trn', help="choose which type of data to generate")
opt = parser.parse_args()


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


H = 256
W = 256
H_center = H//2 - 1
W_center = W//2 -1
resolution = 0.075


if opt.type == 'tst':
    data, labels = load_h5('../../datasets/tst_data.h5')
    out_dir = "tst"
else:
    data, labels = load_h5('../../datasets/trn_data.h5')
    out_dir = "trn"


out_labels = torch.zeros([len(data), H, W], dtype=torch.int8)
out_data = torch.zeros([len(data), 3, H, W], dtype=torch.int8)


for i in range(0, len(data), 1):
    print("Loaded {}/{}".format(i, len(data)))
    frame = data[i]
    frame = frame[1:][:] # discard robot position
    frame_labels = labels[i]
    frame_labels = frame_labels[1:][:] # discard robot position

    # count zero positions that were added for fixed size
    l = 0
    for j in range(len(frame)):
        if frame[j][0] == 0 and frame[j][1] == 0 and frame[j][2] == 0:
            l += 1
    x = np.sum(frame, axis=0)
    center_x = x[0]/(len(frame)-l)
    center_y = x[1]/(len(frame)-l)

    grid = np.zeros((3, H, W))
    grid_n = np.zeros((H, W, 3))
    grid_labels = np.zeros((H, W))

    for j in range(len(frame)):
        if frame[j][0] == 0 and frame[j][1] == 0 and frame[j][2] == 0:
            continue
        col = math.ceil(W_center+((frame[j][0]-center_x)/resolution))
        row = math.ceil(H_center-((frame[j][1]-center_y)/resolution))
        if not (0<=row<=H-1 and 0<=col<=W-1):
            continue

        grid[int(frame[j][2]-1)][row][col] = 1
        grid_n[row][col][int(frame[j][2]-1)] = 1
        out_data[i][int(frame[j][2])-1][row][col] = 1
        if frame_labels[j] == 1:
            grid_labels[row][col] = 1
            out_labels[i][row][col] = 1

    #UNCOMMENT TO VISUALIZE ACTUAL INPUT FRAME
    '''f = plt.figure()
    plt.ion()
    plt.imshow(grid_n)
    plt.show()
    input()
    plt.imshow(out_labels[i])
    plt.show()
    input()'''

'-------------------SAVE DATA-------------------'
torch.save(out_labels, out_dir + '_labels_UNet.pt')
torch.save(out_data, out_dir + '_data _UNet.pt')
print("Generated data with labels of size: {}".format(out_data.shape))

