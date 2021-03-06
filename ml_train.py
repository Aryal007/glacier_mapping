#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:48:53 2020

@author: mibook
"""
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import pickle, time, os, math
from matplotlib import pyplot
import yaml, json, pathlib, os
from addict import Dict
import warnings
warnings.filterwarnings("ignore")

class Dataset():
    def __init__(self, trainX, testX, trainY, testY, 
                 classes = ["land", "water"]):
        self.raw = None
        self.trainX = np.load(trainX)
        self.trainY = np.load(trainY)
        self.testX = np.load(testX)
        self.testY = np.load(testY)
        self.classes = classes

    def get_train_data(self):
        return [self.trainX, self.trainY]

    def get_test_data(self):
        return [self.testX, self.testY]

    def num_classes(self):
        return len(self.classes)

    def num_data(self):
        return len(self.trainY) + len(self.testY)

    def get_histogram(self, X, y, channel=0):
        """
        This function takes X, y, channel and plots the histogram for that 
        channel in the X for all classes in y
        Parameters
        ----------
        X : Numpy array
            DESCRIPTION.
        y : Numpy array
            DESCRIPTION.
        channel : TYPE, optional
            DESCRIPTION. The default is 0.
        Returns
        -------
        None.
        """
        classes = self.classes.copy()
        if self.background:
            classes.append("Background")
        X = X[:,channel].astype('int16')
        try:
            bins = np.linspace(np.amin(X), np.amax(X), np.amax(X)-np.amin(X))
        except:
            bins = np.linspace(0, 100, 1)
        pyplot.title("Channel "+str(channel))
        for key, value in enumerate(classes):
            _x = X[y[:,key] == 1]
            pyplot.hist(_x, bins, alpha=0.5, density = True, label=value, log=True)
        pyplot.legend(loc='upper right')
        pyplot.ylabel('Probability')
        pyplot.xlabel('Intensity')
        pyplot.show()


    def info(self):
        print("No. of classes: {}".format(self.num_classes()))
        print ("Class labels: {}".format(self.classes))
        print ("Total data samples: {}".format(self.num_data()))

        if self.trainY is not None:
            print("Train samples: {}".format(len(self.trainY)))
            
            for k in range(len(self.classes)):
                print("\t {}:{} = {}".format(k, self.classes[k], np.sum(self.trainY == k)))

        if self.testY is not None:
            print ("Test stats: {}".format(len(self.testY)))

            for k in range(len(self.classes)):
                print("\t {}:{} = {}".format(k, self.classes[k], np.sum(self.testY == k)))

class Classifier():
    def __init__(self, savepath,
                 bands = ["Red","Green","Blue","NIR", "NDVI", "NDWI", "NDSWI"]):
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.bands = bands

    def train_and_evaluate(self, estimator, trainX, trainY, testX, testY):
        start = time.time()        
        estimator.fit(trainX, trainY)
        elapsed_time = time.time()-start
        print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
        outputs = estimator.predict(testX)
        return estimator, outputs

    def random_forest(self, trainX, trainY, testX, testY, 
                        tune_estimators = False, tune_depth = False, train=True, 
                        n_estimators = 10, max_depth = 3, 
                         min_samples_split = 100, feature_importance = False):
        print('\nRandom Forest')
        n_estimators_range = np.asarray([2, 5, 10, 15, 20, 25, 50])
        max_depth_range = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        if tune_estimators:
            scores = np.zeros(len(n_estimators_range))
            for i, n_estimators in enumerate(n_estimators_range):
                estimator = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, random_state=42) 
                clf = Pipeline([
                    ('clf', estimator)
                ])
                print(f"Estimators: {n_estimators}, Depth: {max_depth}")
                estimator, outputs = self.train_and_evaluate(clf, trainX, trainY, testX, testY)
                scores[i] = estimator.score(testX, testY)
            score_matrix = np.vstack((n_estimators_range, scores))
            print(score_matrix)
            np.save(self.savepath / 'tune_estimators', score_matrix)

        if tune_depth:
            scores = np.zeros(len(max_depth_range))
            for i, max_depth in enumerate(max_depth_range):
                estimator = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, random_state=42) 
                clf = Pipeline([
                    ('clf', estimator)
                ])
                print(f"Estimators: {n_estimators}, Depth: {max_depth}")
                estimator, outputs = self.train_and_evaluate(clf, trainX, trainY, testX, testY)
                scores[i] = estimator.score(testX, testY)
            score_matrix = np.vstack((n_estimators_range, scores))
            print(score_matrix)
            np.save(self.savepath / 'tune_estimators', score_matrix)

        if train:
            estimator = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, random_state=42) 
            clf = Pipeline([
                ('clf', estimator)
            ])
            estimator, outputs = self.train_and_evaluate(clf, trainX, trainY, testX, testY)
            print("Accuracy on train Set: ")
            print(estimator.score(trainX, trainY))
            print("Accuracy on Test Set: ")
            print(estimator.score(testX, testY))
            print("Classification Report: ")
            print(metrics.classification_report(testY, outputs))
            print("Confusion Matrix: ")
            print(metrics.confusion_matrix(testY, outputs))
            pickle.dump(estimator, open(self.savepath / 'estimator.sav', 'wb'))
            
            if feature_importance:
                self.get_feature_importance(estimator)

    def xgboost(self, trainX, trainY, testX, testY, 
                tune_estimators = False, tune_depth = False, train=True, 
                n_estimators = 10, max_depth = 3, 
                min_samples_split = 100, feature_importance = False):
        print('\nXGBoost')
        n_estimators_range = np.asarray([2, 5, 10, 15, 20, 25, 50])
        max_depth_range = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        if tune_estimators:
            scores = np.zeros(len(n_estimators_range))
            for i, n_estimators in enumerate(n_estimators_range):
                estimator = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, random_state=42) 
                clf = Pipeline([
                    ('clf', estimator)
                ])
                print(f"Estimators: {n_estimators}, Depth: {max_depth}")
                estimator, outputs = self.train_and_evaluate(clf, trainX, trainY, testX, testY)
                scores[i] = estimator.score(testX, testY)
            score_matrix = np.vstack((n_estimators_range, scores))
            print(score_matrix)
            np.save(self.savepath / 'tune_estimators', score_matrix)

        if tune_depth:
            scores = np.zeros(len(max_depth_range))
            for i, max_depth in enumerate(max_depth_range):
                estimator = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, random_state=42) 
                clf = Pipeline([
                    ('clf', estimator)
                ])
                print(f"Estimators: {n_estimators}, Depth: {max_depth}")
                estimator, outputs = self.train_and_evaluate(clf, trainX, trainY, testX, testY)
                scores[i] = estimator.score(testX, testY)
            score_matrix = np.vstack((n_estimators_range, scores))
            print(score_matrix)
            np.save(self.savepath / 'tune_estimators', score_matrix)

        if train:
            estimator = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, random_state=42) 
            clf = Pipeline([
                ('clf', estimator)
            ])
            estimator, outputs = self.train_and_evaluate(clf, trainX, trainY, testX, testY)
            print("Accuracy on train Set: ")
            print(estimator.score(trainX, trainY))
            print("Accuracy on Test Set: ")
            print(estimator.score(testX, testY))
            outputs = estimator.predict(testX)
            print("Classification Report: ")
            print(metrics.classification_report(testY, outputs))
            print("Confusion Matrix: ")
            print(metrics.confusion_matrix(testY, outputs))
            pickle.dump(estimator, open(self.savepath / 'estimator.sav', 'wb'))
            
            if feature_importance:
                self.get_feature_importance(estimator)
    
    def get_feature_importance(self, estimator):
        feat_importances = pd.Series(estimator._final_estimator.feature_importances_, index=self.bands)
        feat_importances = feat_importances.sort_values(ascending=True).tail(10)
        feat_importances.plot.barh()
        pyplot.tight_layout()
        pyplot.show()

if __name__ == "__main__":

    conf = Dict(yaml.safe_load(open('./conf/ml_train.yaml')))
    data_dir = pathlib.Path(conf.data_dir)
    estimator_name = conf.estimator_name
    
    dataset = Dataset(trainX = data_dir / "X_train.npy", 
                    testX = data_dir / "X_val.npy",
                    trainY = data_dir / "y_train.npy",
                    testY = data_dir / "y_val.npy")

    dataset.info()

    classifier = Classifier(savepath=data_dir / estimator_name)

    if estimator_name == "random_forest":
        classifier.random_forest(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                  tune_estimators = conf.tune_estimators, tune_depth = conf.tune_depth, 
                  train = conf.train, n_estimators = conf.n_estimator, 
                  max_depth = conf.max_depth,  min_samples_split = conf.min_samples_split, 
                  feature_importance = conf.feature_importance)

    elif estimator_name == "xgboost":
        classifier.xgboost(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                  tune_estimators = conf.tune_estimators, tune_depth = conf.tune_depth, 
                  train = conf.train, n_estimators = conf.n_estimator, 
                  max_depth = conf.max_depth,  min_samples_split = conf.min_samples_split, 
                  feature_importance = conf.feature_importance)

    else:
        print("Classifier not clear. Enter 'random_forest' or 'xgboost'.")
