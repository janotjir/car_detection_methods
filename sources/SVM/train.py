import os
import numpy as np
import h5py
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label

'''-------------LOAD DATASETS---------------'''
data_path = 'trn_data_SVM/data.h5'
x_train, y_train = load_h5(data_path)
print("Loaded training data of shape {}".format(x_train.shape))
data_path = 'tst_data_SVM/data.h5'
x_test, y_test = load_h5(data_path)
print("Loaded testing data of shape {}".format(x_test.shape))

'''--------------NORMALIZE DATASETS (Z-SCORE)--------------'''
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Save scaler? Type 'y', any key otherwise")
i = input()
if i == "y":
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler")

'''--------------SVM CLASSIFIER----------------'''
svm_classifier = svm.SVC(kernel='rbf',  C = 100, gamma=0.1)
svm_classifier.fit(x_train, y_train)

print("Score on testing dataset {}".format(svm_classifier.score(x_test, y_test)))
print("Support vectors: {}".format(svm_classifier.n_support_))

print("Save SVM model? Type 'y', any key otherwise")
i = input()
if i == "y":
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_classifier, f)
    print("Saved SVM model")

