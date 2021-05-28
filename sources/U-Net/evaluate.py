#!/usr/bin/env python
import os
import h5py
import numpy as np
import math
import random
import torch
import timeit
import argparse

from unet_model import SmallerUnet
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--criteria', type=str, default='acc', help="choose criteria to evaluate")
opt = parser.parse_args()


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def get_device(gpu=0):  # Manually specify gpu
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device


def load_model(device='cpu'):
    model = SmallerUnet().to(device)
    model.load_state_dict(torch.load("epoch7.pth", map_location=device))
    return model


def euclidean_distance(x_1, x_2, y_1, y_2):
    return math.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2))


'-------------------LOAD DATA and MODEL----------------------'
if opt.criteria == 'wheels':
    data_in, labels_in = load_h5('../../datasets/tst_data_wheels.h5')
elif opt.criteria == 'snow':
    data_in, labels_in = load_h5('../../datasets/tst_data_snow.h5')
else:
    data_in, labels_in = load_h5('../../datasets/tst_data.h5')
print("Loaded dataset of size {} loaded".format(data_in.shape))

device = get_device()
print("Device: {}".format(device))
model = load_model(device=device)
print("Model restored, ready to evaluate")
'--------------------------------------------------------------'

H = 256
W = 256
H_center = H//2 - 1
W_center = W//2 -1
resolution = 0.075

WIDTH = 1.5
LENGTH = 2.5
DIAGONAL = 3.1
DIVISION_FACTOR = 5
THRESHOLD = 0.35

time_processing = 0
time_method = 0
time_wheels = 0
wheels_extractions_skipped = 0
TP = FP = TN = FN = 0

wheels_dropoff =np.array([[-3.827, -4.196], [-2.893, -5.588], [-5.839, -5.680], [-4.840, -7.118]])
FC = 0
F2W = 0
F3W = 0
F4W = 0
TOTAL = 0

'-----------------------EVALUATION---------------------------'
for j in range(0, len(data_in), 1):
    print("Processing frame {}/{}".format(j, len(data_in)))

    '''----------OBTAIN ACT FRAME-----------'''
    curr_frame = data_in[j, :, :3]
    curr_frame_labels = labels_in[j]

    curr_frame = curr_frame[1:][:]
    curr_frame_labels = curr_frame_labels[1:]
    id = 0
    # delete all 0 positions that were added
    while id != len(curr_frame_labels):
        if curr_frame[id][0] == 0.0 and curr_frame[id][1] == 0.0 and curr_frame[id][2] == 0.0 and curr_frame_labels[
            id] == 0:
            curr_frame = np.delete(curr_frame, id, 0)
            curr_frame_labels = np.delete(curr_frame_labels, id)
            continue
        id += 1

    curr_frame[curr_frame[:, 2] == 1, 2] = 0.4
    curr_frame[curr_frame[:, 2] == 2, 2] = 0.7
    curr_frame[curr_frame[:, 2] == 3, 2] = 1

    '''---------------INPUT PREPROCESSING---------------'''
    start = timeit.default_timer()

    indices = np.empty((0, 2), dtype=np.int8)
    grid = torch.zeros([1, 3, H, W], dtype=torch.float32).to(device)
    points = np.empty((0, 3), dtype=np.float32)

    centroid = np.mean(curr_frame[:, 0:2], axis=0)

    idx1 = np.nonzero(curr_frame[:, 2] == 0.4)
    cloud_right = np.squeeze(curr_frame[idx1, :])
    rows = np.expand_dims(np.ceil(H_center - ((cloud_right[:, 1] - centroid[1]) / resolution)), axis=1)
    cols = np.expand_dims(np.ceil(W_center + ((cloud_right[:, 0] - centroid[0]) / resolution)), axis=1)
    ids1 = np.nonzero(np.logical_and.reduce((rows[:, 0] >= 0, rows[:, 0] <= H - 1, cols[:, 0] >= 0, cols[:, 0] <= W - 1)))
    rows = rows[ids1].astype(int)
    cols = cols[ids1].astype(int)
    grid[0][0][rows[:, 0], cols[:, 0]] = 1
    indices = np.vstack((indices, np.hstack((rows, cols))))
    points = np.vstack((points, cloud_right[ids1]))

    idx2 = np.nonzero(curr_frame[:, 2] == 0.7)
    cloud_left = np.squeeze(curr_frame[idx2, :])
    rows = np.expand_dims(np.ceil(H_center - ((cloud_left[:, 1] - centroid[1]) / resolution)), axis=1)
    cols = np.expand_dims(np.ceil(W_center + ((cloud_left[:, 0] - centroid[0]) / resolution)), axis=1)
    ids2 = np.nonzero(np.logical_and.reduce((rows[:, 0] >= 0, rows[:, 0] <= H - 1, cols[:, 0] >= 0, cols[:, 0] <= W - 1)))
    rows = rows[ids2].astype(int)
    cols = cols[ids2].astype(int)
    grid[0][1][rows[:, 0], cols[:, 0]] = 1
    indices = np.vstack((indices, np.hstack((rows, cols))))
    points = np.vstack((points, cloud_left[ids2]))

    idx3 = np.nonzero(curr_frame[:, 2] == 1)
    cloud_middle = np.squeeze(curr_frame[idx3, :])
    rows = np.expand_dims(np.ceil(H_center - ((cloud_middle[:, 1] - centroid[1]) / resolution)), axis=1)
    cols = np.expand_dims(np.ceil(W_center + ((cloud_middle[:, 0] - centroid[0]) / resolution)), axis=1)
    ids3 = np.nonzero(np.logical_and.reduce((rows[:, 0] >= 0, rows[:, 0] <= H - 1, cols[:, 0] >= 0, cols[:, 0] <= W - 1)))
    rows = rows[ids3].astype(int)
    cols = cols[ids3].astype(int)
    grid[0][2][rows[:, 0], cols[:, 0]] = 1
    indices = np.vstack((indices, np.hstack((rows, cols))))
    points = np.vstack((points, cloud_middle[ids3]))

    end = timeit.default_timer()
    time_processing += end - start

    # UNCOMMENT FOR PREPROCESSED POINT CLOUD VISUALIZATION
    '''plt.figure()
    plt.imshow(grid[0].permute(1, 2, 0).cpu())
    plt.show()
    input()'''

    '''------------------LABELS PREPARATION----------------'''
    labels = np.empty(0)
    labels_discarded = np.empty(0)

    r_labels = curr_frame_labels[idx1]
    labels = np.hstack((labels, r_labels[ids1]))
    r_labels_discarded = np.delete(r_labels, ids1)
    labels_discarded = np.hstack((labels_discarded, r_labels_discarded))

    l_labels = curr_frame_labels[idx2]
    labels = np.hstack((labels, l_labels[ids2]))
    l_labels_discarded = np.delete(l_labels, ids2)
    labels_discarded = np.hstack((labels_discarded, l_labels_discarded))

    m_labels = curr_frame_labels[idx3]
    labels = np.hstack((labels, m_labels[ids3]))
    m_labels_discarded = np.delete(m_labels, ids3)
    labels_discarded = np.hstack((labels_discarded, m_labels_discarded))

    '''----------------------------UNET----------------------------'''
    start = timeit.default_timer()

    with torch.no_grad():
        model.eval()
        output = model(grid)
        output = torch.argmax(output, dim=1)
        output = output[0].cpu()

    pred = output[indices[:,0], indices[:,1]]

    end = timeit.default_timer()
    time_method += end - start

    # UNCOMMENT FOR CLASSIFICATIONS VISUALIZATION
    '''plt.figure()
    plt.imshow(output.cpu())
    plt.show()
    input()'''

    '''----------------------PRECISION CALC----------------------'''
    conf_vector = pred / labels

    TP += torch.sum(conf_vector == 1).item()  # pred=1 and label=1
    FP += torch.sum(conf_vector == float('inf')).item()  # pred=1 and label=0
    TN += torch.sum(torch.isnan(conf_vector)).item()  # pred=0 and label=0
    FN += torch.sum(conf_vector == 0).item()  # pred=0 and label=1

    if len(labels_discarded) > 0:
        conf_vector = np.zeros(len(labels_discarded)) / labels_discarded
        TP += np.sum(conf_vector == 1).item()  # pred=1 and label=1
        FP += np.sum(conf_vector == float('inf')).item()  # pred=1 and label=0
        TN += np.sum(np.isnan(conf_vector)).item()  # pred=0 and label=0
        FN += np.sum(conf_vector == 0).item()  # pred=0 and label=1'''

    #print(TP, FP, TN, FN)

    '''----------------------------WHEEL EXTRACTION----------------------------'''
    start = timeit.default_timer()
    pred = torch.squeeze(torch.nonzero(pred)).to('cpu')
    points = points[pred]
    if points.shape[0] < 3:
        print("SKIPPED WHEEL EXTRACTION")
        wheels_extractions_skipped += 1
        continue
    '''----apply DBSCAN----'''
    dbscan = DBSCAN(eps=0.35, min_samples=3)
    clustering = dbscan.fit_predict(points)

    num_clusters = np.unique(clustering)
    num_clusters = len(num_clusters[num_clusters != -1])
    cluster_centers = np.empty((0, 3))
    parent_clusters = np.empty(0)
    cluster_centers_middle = np.empty((0, 3))

    '''----calculate cluster centers----'''
    for i in range(num_clusters):
        # calculate centers of clusters from middle scanner
        cluster = points[np.nonzero(clustering == i)]
        indices = cluster[:, 2] == 1  # indices of points from middle scanner
        if indices.shape[0] > 2:
            center = np.mean(cluster[indices], axis=0)
            cluster_centers_middle = np.vstack((cluster_centers_middle, [center[0], center[1], i]))

        # calculate centers of clusters from right/left scanners
        cluster = cluster[np.logical_not(indices)]  # indices of pts from other scanners
        if cluster.shape[0] < 3: continue
        diffs = (cluster - cluster[:, np.newaxis]) ** 2
        dists = np.sqrt(diffs[:, :, 0] + diffs[:, :, 1])  # dist matrix = sqrt((x_1 - y_1)**2 + (x_2- y_2)**2)
        max_dist = np.max(dists)

        if (1 < max_dist < DIAGONAL) and (len(cluster) > 25):
            sub_cluster_size = int(math.ceil(len(cluster) / DIVISION_FACTOR))
            decision_dists = dists[0]  # array of distances from the first point
            indexes = np.argsort(decision_dists)
            cluster = cluster[indexes]  # sort array of points according to distances
            for j in range(DIVISION_FACTOR - 1):
                tmp = j * sub_cluster_size
                center = np.mean(cluster[tmp:tmp + sub_cluster_size, :], axis=0)
                cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
                parent_clusters = np.append(parent_clusters, i)
            tmp = (DIVISION_FACTOR - 1) * sub_cluster_size
            center = np.mean(cluster[tmp:, :], axis=0)
            cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
            parent_clusters = np.append(parent_clusters, i)
        else:
            center = np.mean(cluster, axis=0)
            cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
            parent_clusters = np.append(parent_clusters, i)

    '''----wheels fitting----'''
    fragment_ids = np.negative(np.ones(len(cluster_centers), dtype=int))
    fragments = []
    for j in range(len(cluster_centers)):
        for k in range(j + 1, len(cluster_centers)):
            d = euclidean_distance(cluster_centers[j][0], cluster_centers[k][0], cluster_centers[j][1],
                                   cluster_centers[k][1])
            if abs(d - WIDTH) < THRESHOLD:
                w1 = cluster_centers[j]
                w2 = cluster_centers[k]

                # CALC APPROX POSITIONS OF REMAINING WHEELS ON BOTH SIDES
                n = [(w1[1] - w2[1]) / d, (w2[0] - w1[0]) / d]  # n=(u2, -u1)/norm - normalized normal vector
                w3_1 = [w1[0] + LENGTH * n[0], w1[1] + LENGTH * n[1]]
                w4_1 = [w2[0] + LENGTH * n[0], w2[1] + LENGTH * n[1]]
                w3_2 = [w1[0] - LENGTH * n[0], w1[1] - LENGTH * n[1]]
                w4_2 = [w2[0] - LENGTH * n[0], w2[1] - LENGTH * n[1]]

                # FIND POTENTIAL WHEELS
                d_values = [float('inf')] * 4  # d3_1_best, d3_2_best, d4_1_best, d4_2_best
                d_values = np.array(d_values)
                w_ids = [-1] * 4  # w3_1, w3_2, w4_1, w4_2
                w_ids = np.array(w_ids)

                for l in range(len(cluster_centers)):
                    if l == j or l == k: continue
                    dw3_1 = euclidean_distance(cluster_centers[l][0], w3_1[0], cluster_centers[l][1], w3_1[1])
                    dw4_1 = euclidean_distance(cluster_centers[l][0], w4_1[0], cluster_centers[l][1], w4_1[1])
                    dw3_2 = euclidean_distance(cluster_centers[l][0], w3_2[0], cluster_centers[l][1], w3_2[1])
                    dw4_2 = euclidean_distance(cluster_centers[l][0], w4_2[0], cluster_centers[l][1], w4_2[1])
                    # one side check
                    if (dw3_1 < THRESHOLD) and (dw3_1 < d_values[0]):
                        d_values[0] = dw3_1
                        w_ids[0] = l
                    elif (dw4_1 < THRESHOLD) and (dw4_1 < d_values[2]):
                        d_values[2] = dw4_1
                        w_ids[2] = l
                    # opposite side check
                    elif (dw3_2 < THRESHOLD) and (dw3_2 < d_values[1]):
                        d_values[1] = dw3_2
                        w_ids[1] = l
                    elif (dw4_2 < THRESHOLD) and (dw4_2 < d_values[3]):
                        d_values[3] = dw4_2
                        w_ids[3] = l

                # decide which side gives better results if any
                tmp = np.array(w_ids[:] != -1, dtype=int)
                d_values[d_values[:] == float('inf')] = 0
                numw = np.array([tmp[0] + tmp[2] + 2, tmp[1] + tmp[3] + 2])
                idx = -1

                if not (numw[0] > 3 or numw[1] > 3):
                    for l in range(len(cluster_centers_middle)):
                        d1 = euclidean_distance(cluster_centers[j][0], cluster_centers_middle[l][0],
                                                cluster_centers[j][1], cluster_centers_middle[l][1])
                        d2 = euclidean_distance(cluster_centers[k][0], cluster_centers_middle[l][0],
                                                cluster_centers[k][1], cluster_centers_middle[l][1])
                        if 0.3 < d1 < 1.35 and 0.3 < d2 < 1.35:
                            if numw[0] == numw[1]:
                                idx = np.argmin(np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))
                            else:
                                idx = np.argmin(np.array(numw))
                            break
                else:
                    # decide which is better - have more wheels/better distances
                    if numw[0] > numw[1]:
                        idx = 0
                    elif numw[1] == numw[0]:
                        idx = np.argmin(np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))
                    else:
                        idx = 1

                # determine the best global fit
                if idx != -1:
                    the_best = True
                    act_ids = np.array([j, k, w_ids[idx], w_ids[idx + 2]])
                    act_ids = act_ids[act_ids != -1]
                    act_d = d_values[idx] + d_values[idx + 2]
                    ids = np.unique(fragment_ids[act_ids])
                    ids = ids[ids != -1]
                    # check if the clusters aren't part of other car
                    for l in range(len(ids)):
                        if numw[idx] < fragments[ids[l]][0]:
                            the_best = False
                            break
                        elif numw[idx] == fragments[ids[l]][0]:
                            if act_d < fragments[ids[l]][1]:
                                continue
                            else:
                                the_best = False
                                break
                    if not the_best:
                        continue
                    # check if clusters from the same parent cluster aren't part of other car
                    related_parents = np.unique(parent_clusters[act_ids])
                    parents_ids = np.unique(fragment_ids[np.nonzero(np.in1d(parent_clusters, related_parents))])
                    parents_ids = parents_ids[parents_ids != -1]
                    for l in range(len(parents_ids)):
                        if numw[idx] < fragments[parents_ids[l]][0]:
                            the_best = False
                            break
                        elif numw[idx] == fragments[parents_ids[l]][0]:
                            if act_d < fragments[parents_ids[l]][1]:
                                continue
                            else:
                                the_best = False
                                break
                    # if all check went gut, discard eventually other cars and add this
                    if the_best:
                        indices = np.nonzero(np.in1d(fragment_ids, ids))
                        fragment_ids[indices] = -1
                        indices = np.nonzero(np.in1d(fragment_ids, parents_ids))
                        fragment_ids[indices] = -1
                        fragments.append([numw[idx], act_d, act_ids])
                        fragment_ids[act_ids] = len(fragments) - 1
    end = timeit.default_timer()
    time_wheels += end - start

    if opt.criteria == 'wheels':
        TOTAL += 1
        car_idx = np.unique(fragment_ids)
        car_idx = car_idx[car_idx != -1]
        foundw = np.zeros(4)
        for i in range(len(car_idx)):
            for j in range(len(fragments[car_idx[i]][2])):
                act_wheel = cluster_centers[fragments[car_idx[i]][2][j]]
                for k in range(len(wheels_dropoff)):
                    d = euclidean_distance(act_wheel[0], wheels_dropoff[k][0], act_wheel[1], wheels_dropoff[k][1])
                    if d < 0.4:
                        foundw[k] = 1
        if sum(foundw) == 4:
            F4W += 1
            FC += 1
        elif sum(foundw) == 3:
            F3W += 1
            FC += 1
        elif sum(foundw) == 2:
            F2W += 1
            FC += 1

    # UNCOMMENT FOR WHEELS LOCALIZATION VISUALIZATION
    # red/green/blue points correspond to the data from the right/left/middle LIDAR
    # yellow points represent the found wheels
    '''plt.figure()
    plt.ion()
    plt.scatter(curr_frame[np.nonzero(curr_frame[:, 2] == 0.4), 0], curr_frame[np.nonzero(curr_frame[:, 2] == 0.4), 1], color='red', s=2)
    plt.scatter(curr_frame[np.nonzero(curr_frame[:, 2] == 0.7), 0], curr_frame[np.nonzero(curr_frame[:, 2] == 0.7), 1], color='green', s=2)
    plt.scatter(curr_frame[np.nonzero(curr_frame[:, 2] == 1), 0], curr_frame[np.nonzero(curr_frame[:, 2] == 1), 1],color='blue', s=2)
    car_idx = np.unique(fragment_ids)
    car_idx = car_idx[car_idx != -1]
    for i in range(len(car_idx)):
        for j in range(len(fragments[car_idx[i]][2])):
            plt.scatter(cluster_centers[fragments[car_idx[i]][2][j]][0], cluster_centers[fragments[car_idx[i]][2][j]][1], color='yellow')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    input()'''

'''-------------------FINAL RESULTS--------------------'''
print("Average time for preprocessing: {}".format(time_processing / len(data_in)))
print("Average time for UNet application: {}".format(time_method / len(data_in)))
print("Average time for wheel extraction: {}".format(time_wheels / (len(data_in) - wheels_extractions_skipped)))
print("Average time in total: {}".format((time_processing + time_method) / len(data_in) + time_wheels / (len(data_in) - wheels_extractions_skipped)))
print("TP: {}, FP: {}, TN: {}, FN: {}".format(TP, FP, TN, FN))
print("Precision on the testing set: {}".format(TP / (TP + FP)))
print("Recall on the testing set: {}".format(TP / (TP + FN)))
print("Accuracy on the testing set: {}".format((TP + TN) / (TP + TN + FP + FN)))
if opt.criteria == 'wheels':
    print("Found {}/{} wheels".format(FC, TOTAL))
    print("Found {} four wheels, {} three wheels, {} two wheels". format(F4W, F3W, F2W))
