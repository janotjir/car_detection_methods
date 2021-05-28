#!/usr/bin/env python
import os
import h5py
import numpy as np
import math
import timeit

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def euclidean_distance(x_1, x_2, y_1, y_2):
    return math.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2))

'-------------------LOAD DATA----------------------'
data_in, labels_in = load_h5('../../datasets/tst_data_wheels.h5')
print("Loaded dataset of size {} loaded".format(data_in.shape))

'--------------------------------------------------------------'

WIDTH = 1.5
LENGTH = 2.5
DIAGONAL = 3.1
DIVISION_FACTOR = 5
THRESHOLD = 0.35

wheels_dropoff = [[-3.827, -4.196], [-2.893, -5.588], [-5.839, -5.680], [-4.840, -7.118]]
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

    origin_x = curr_frame[0][0]
    origin_y = curr_frame[0][1]
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

    dbscan = DBSCAN(eps=0.35, min_samples=3)
    clustering = dbscan.fit_predict(curr_frame)

    num_clusters = np.unique(clustering)
    num_clusters = len(num_clusters[num_clusters != -1])

    # PROCESSING: CALCULATE CENTERS AND SPLIT POTENTIAL BUMPERS
    cluster_centers = np.empty((0, 3))
    for i in range(num_clusters):
        cluster_indices = np.nonzero(clustering == i)
        cluster = curr_frame[cluster_indices]

        diffs = (cluster - cluster[:, np.newaxis]) ** 2
        dists = np.sqrt(diffs[:, :, 0] + diffs[:, :, 1])  # dist matrix = sqrt((x_1 - y_1)**2 + (x_2- y_2)**2)
        max_dist = np.max(dists)
        # arg = np.unravel_index(dists.argmax(), dists.shape)    # dists[arg] = np.max(dists)

        if (1 < max_dist < DIAGONAL) and (len(cluster) > 25):
            # arg = np.unravel_index(dists.argmax(), dists.shape)  # dists[arg] = np.max(dists)
            # cluster_centroids = np.vstack((cluster_centroids, [cluster[arg[0]][0], cluster[arg[0]][1], i]))
            # cluster_centroids = np.vstack((cluster_centroids, [cluster[arg[1]][0], cluster[arg[1]][1], i]))

            sub_cluster_size = int(math.ceil(len(cluster) / DIVISION_FACTOR))
            decision_dists = dists[0]  # array of distances from the first point
            indexes = np.argsort(decision_dists)
            cluster = cluster[indexes]  # sort array of points according to distances

            for j in range(DIVISION_FACTOR - 1):
                tmp = j * sub_cluster_size
                center = np.mean(cluster[tmp:tmp + sub_cluster_size, :], axis=0)
                cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
            tmp = (DIVISION_FACTOR - 1) * sub_cluster_size
            center = np.mean(cluster[tmp:, :], axis=0)
            cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
        else:
            center = np.mean(cluster, axis=0)
            cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))

    '''----------------------------WHEELS-FITTING----------------------------'''
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
                w_ids = [-1] * 4  # w3_1, w3_2, w4_1, w4_2

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

                # DECIDE WHICH SIDE GIVES BETTER RESULTS
                idx = -1
                if (w_ids[0] != -1) and (w_ids[2] != -1):
                    if (w_ids[1] != -1) and (w_ids[3] != -1):
                        idx = np.argmin(np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))
                    else:
                        idx = 0
                elif (w_ids[1] != -1) and (w_ids[3] != -1):
                    idx = 1
                elif (w_ids[0] != -1) or (w_ids[2] != -1):
                    if (w_ids[1] != -1) or (w_ids[3] != -1):
                        idx = np.argmin(np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))

                # DETERMINE THE GLOBAL BEST FIT (each cluster can be in max. 1 car)
                if idx != -1:
                    the_best = True
                    act_ids = [j, k, w_ids[idx], w_ids[idx + 2]]
                    act_d = d_values[idx] + d_values[idx + 2]
                    ids = np.unique(fragment_ids[act_ids])
                    ids = ids[ids != -1]
                    for l in range(len(ids)):
                        if act_d < fragments[ids[l]][0]:
                            continue
                        else:
                            the_best = False
                            break
                    # if better option found, discard all associated
                    if the_best:
                        indices = np.nonzero(np.in1d(fragment_ids, ids))
                        fragment_ids[indices] = -1
                        fragments.append([act_d, act_ids])
                        fragment_ids[act_ids] = len(fragments) - 1

    TOTAL += 1
    car_idx = np.unique(fragment_ids)
    car_idx = car_idx[car_idx != -1]
    foundw = np.zeros(4)
    for i in range(len(car_idx)):
        for j in range(len(fragments[car_idx[i]][1])):
            act_wheel = cluster_centers[fragments[car_idx[i]][1][j]]
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
        for j in range(len(fragments[car_idx[i]][1])):
            plt.scatter(cluster_centers[fragments[car_idx[i]][1][j]][0], cluster_centers[fragments[car_idx[i]][1][j]][1], color='yellow')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    input()'''

'''-------------------FINAL RESULTS--------------------'''
print("Found {}/{} wheels".format(FC, TOTAL))
print("Found {} four wheels, {} three wheels, {} two wheels". format(F4W, F3W, F2W))
