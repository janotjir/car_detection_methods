#!/usr/bin/env python
import os
import os.path
import sys
import h5py
import numpy as np
import math
import random
import timeit
import pickle
import argparse

from sklearn.cluster import DBSCAN
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--criteria', type=str, default='acc', help="choose criteria to evaluate")
opt = parser.parse_args()


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label

'-------------------LOAD DATA and MODEL----------------------'
if opt.criteria == 'wheels':
    data_in, labels_in = load_h5('../../datasets/tst_data_wheels.h5')
elif opt.criteria == 'snow':
    data_in, labels_in = load_h5('../../datasets/tst_data_snow.h5')
else:
    data_in, labels_in = load_h5('../../datasets/tst_data.h5')
print("Loaded dataset of size {} loaded".format(data_in.shape))

with open('svm_model.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Model and scaler restored, ready to evaluate")

'--------------------------------------------------------------'
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

wheels_dropoff = [[-3.827, -4.196], [-2.893, -5.588], [-5.839, -5.680], [-4.840, -7.118]]
FC = 0
F2W = 0
F3W = 0
F4W = 0
TOTAL = 0


def euclidean_distance(x_1, x_2, y_1, y_2):
    return math.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2))


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
    # delete all 0 positions that were added for
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

    # run dbscan
    dbscan = DBSCAN(eps=0.35, min_samples=3)
    clustering = dbscan.fit_predict(curr_frame)

    num_clusters = np.unique(clustering)
    num_clusters = num_clusters[num_clusters != -1]
    num_clusters = len(num_clusters)

    cluster_centroids = np.empty((0, 2))
    tmp_data = np.empty((0, 10))
    tmp_data_id = np.empty(0)

    for i in range(0, num_clusters):
        cluster_indices = np.nonzero(clustering == i)
        cluster = curr_frame[cluster_indices]

        # get maximum distance (= length) inside cluster
        diffs = (cluster - cluster[:, np.newaxis]) ** 2
        dists = np.sqrt(diffs[:, :, 0] + diffs[:, :, 1])  # dist matrix = sqrt((x_1 - y_1)**2 + (x_2- y_2)**2)
        max_dist = np.max(dists)
        if max_dist > 4.2:
            continue

        centroid = np.mean(cluster, axis=0)
        cluster_centroids = np.vstack((cluster_centroids, [centroid[0], centroid[1]]))

        # get depth (= width) of the cluster
        arg = np.unravel_index(dists.argmax(), dists.shape)  # dists[arg] = np.max(dists)
        a1 = cluster[arg[0]][1] - cluster[arg[1]][1]
        b1 = cluster[arg[1]][0] - cluster[arg[0]][0]
        c1 = cluster[arg[1]][1] * cluster[arg[0]][0] - cluster[arg[1]][0] * cluster[arg[0]][1]
        n1 = (a1 ** 2 + b1 ** 2) ** 0.5
        dists2 = abs(a1 / n1 * cluster[:, 0] + b1 / n1 * cluster[:, 1] + c1 / n1)
        max_depth = np.max(dists2)
        # arg2 = np.unravel_index(dists2.argmax(), dists2.shape)

        # PCA analysis
        A = cluster[:, 0:2] - centroid[0:2]
        pca = PCA()
        pca.fit(A)

        # all features
        f1_r = np.count_nonzero(cluster[:, 2] == 0.4)
        f1_l = np.count_nonzero(cluster[:, 2] == 0.7)
        f1_m = np.count_nonzero(cluster[:, 2] == 1)
        f2 = euclidean_distance(origin_x, centroid[0], origin_y, centroid[1])
        f4 = max_dist
        f5 = max_depth
        f6 = np.sqrt(np.sum(A[:, 0] ** 2 + A[:, 1] ** 2) / (len(cluster) - 1))  # standard deviation
        f7 = pca.explained_variance_[0] / pca.explained_variance_[1]  # PCA ellipse
        f8 = pca.explained_variance_[1]  # PCA error of fitting with line

        tmp_data = np.vstack((tmp_data, [f1_r, f1_l, f1_m, f2, 0, f4, f5, f6, f7, f8]))
        tmp_data_id = np.append(tmp_data_id, i)

    # compute dists between clusters and add it as f3
    diffs_c = (cluster_centroids - cluster_centroids[:, np.newaxis]) ** 2
    dists_c = np.sqrt(diffs_c[:, :, 0] + diffs_c[:, :, 1])
    tmp_data[:, 4] = (dists_c[:, :] <= 2).sum(1) - 1

    # normalize - using scaler
    tmp_data = scaler.transform(tmp_data)

    end = timeit.default_timer()
    time_processing += end - start

    '''----------------------------SVM----------------------------'''
    start = timeit.default_timer()

    y_pred = svm_classifier.predict(tmp_data)
    pred = np.in1d(clustering, tmp_data_id[np.nonzero(y_pred)]).astype(int)

    end = timeit.default_timer()
    time_method += end - start

    # UNCOMMENT FOR CLASSIFICATIONS VISUALIZATION
    # - orange points the classified cars, blue are the background
    '''plt.figure()
    plt.scatter(curr_frame[np.nonzero(pred), 0], curr_frame[np.nonzero(pred), 1], color='orange', s=2)
    plt.scatter(curr_frame[np.nonzero(pred == 0), 0], curr_frame[np.nonzero(pred == 0), 1], color='blue', s=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ion()
    plt.show()
    input()'''

    '''----------------------PRECISION CALC----------------------'''

    conf_vector = pred / curr_frame_labels

    TP += np.sum(conf_vector == 1).item()  # pred=1 and label=1
    FP += np.sum(conf_vector == float('inf')).item()  # pred=1 and label=0
    TN += np.sum(np.isnan(conf_vector)).item()  # pred=0 and label=0
    FN += np.sum(conf_vector == 0).item()  # pred=0 and label=1

    '''----------------------------WHEEL EXTRACTION----------------------------'''
    curr_frame = np.squeeze(curr_frame[np.nonzero(pred), :])
    if curr_frame.shape[0] < 3:
        print("SKIPPED WHEEL EXTRACTION")
        wheels_extractions_skipped += 1
        continue

    '''----modify DBSCAN from preprocessing----'''
    clustering = clustering[np.nonzero(pred)]

    num_clusters = np.unique(clustering)
    num_clusters = len(num_clusters[num_clusters != -1])
    cluster_centers = np.empty((0, 3))
    parent_clusters = np.empty(0)
    cluster_centers_middle = np.empty((0, 3))

    '''----calculate cluster centers----'''
    for i in np.unique(clustering):
        # calculate centers of clusters from middle scanner
        cluster = curr_frame[np.nonzero(clustering == i)]
        indices = cluster[:, 2] == 1  # indices of points from middle scanner
        if np.count_nonzero(indices) > 2:
            center = np.mean(cluster[np.nonzero(indices)], axis=0)
            cluster_centers_middle = np.vstack((cluster_centers_middle, [center[0], center[1], i]))

        # calculate centers of clusters from right/left scanners
        cluster = cluster[np.nonzero(np.logical_not(indices))]  # indices of pts from other scanners
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
            d = euclidean_distance(cluster_centers[j][0], cluster_centers[k][0], cluster_centers[j][1],cluster_centers[k][1])
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
                        d1 = euclidean_distance(cluster_centers[j][0], cluster_centers_middle[l][0],cluster_centers[j][1], cluster_centers_middle[l][1])
                        d2 = euclidean_distance(cluster_centers[k][0], cluster_centers_middle[l][0],cluster_centers[k][1], cluster_centers_middle[l][1])
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
print("Average time for PN application: {}".format(time_method / len(data_in)))
print("Average time for wheel extraction: {}".format(time_wheels / (len(data_in) - wheels_extractions_skipped)))
print("Average time in total: {}".format((time_processing + time_method) / len(data_in) + time_wheels / (len(data_in) - wheels_extractions_skipped)))
print("TP: {}, FP: {}, TN: {}, FN: {}".format(TP, FP, TN, FN))
print("Precision on the testing set: {:.15f}".format(float(TP) / (TP + FP)))
print("Recall on the testing set: {:.15f}".format(float(TP) / (TP + FN)))
print("Accuracy on the testing set: {:.15f}".format(float(TP + TN) / (TP + TN + FP + FN)))
if opt.criteria == 'wheels':
    print("Found {}/{} wheels".format(FC, TOTAL))
    print("Found {} four wheels, {} three wheels, {} two wheels". format(F4W, F3W, F2W))
