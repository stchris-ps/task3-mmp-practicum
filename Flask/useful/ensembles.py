#!/usr/bin/env python
# coding: utf-8

# In[6]:


def modif_mse(c, pred_m_1, pred_m, y_true):
    tmp = pred_m_1 - c*pred_m
    return np.sqrt(np.sum((tmp - y_true)**2))


# In[1]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
import time


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.trees = n_estimators
        self.dep = max_depth
        self.size = feature_subsample_size
        self.other_par = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
            
        y_val : numpy ndarray
            Array of size n_val_objects           
        """
        self.fitted = []
        history = []
        history_about_train = []
        times_his = []
        pred = np.zeros_like(y_val)
        pred_train = np.zeros_like(y)
        ind1 = range(X.shape[1])
        self.feats = []
        t = time.time()
        
        
        if not X_val is None:
            val_len = len(y_val)
        for i in range(self.trees):
            ind = np.random.choice(len(X),len(X))
            if not self.size is None:
                ind1 = np.random.choice(X.shape[1],self.size, replace=False)
                self.feats.append(ind1)
            alg = DecisionTreeRegressor(max_depth=self.dep, **self.other_par)
            X1 = X[ind]
            alg.fit(X1[:,ind1], y[ind])
            self.fitted.append(alg)
            
            pred_train += alg.predict(X[:,ind1])
            tmp = pred_train/len(self.fitted)
            history_about_train.append(np.sqrt(np.sum((y-tmp)**2/len(y))))

            times_his.append(time.time()-t)
            if not X_val is None:
                pred += alg.predict(X_val[:,ind1])
                tmp = pred/len(self.fitted)
                history.append(np.sqrt(np.sum((y_val-tmp)**2/val_len)))
            t = time.time()
        if not X_val is None:
            return history_about_train, history, times_his
        else:
            return history_about_train, None, times_his
        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(len(X))
        if self.size is None:
            for i in self.fitted:
                pred += i.predict(X)
        else:
            for j,i in enumerate(self.fitted):
                pred += i.predict(X[:,self.feats[j]])
        return pred/self.trees


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.other_par = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        beg_weig = np.zeros(len(y))
        self.fitted = []
        history = []
        history_about_train = []
        pred_val = np.zeros_like(y_val)
        pred_train = np.zeros_like(y)
        self.coefs = []
        ind = range(X.shape[1])
        times_his = []
        self.feats = []
        t = time.time()
        
        
        if not X_val is None:
            val_len = len(y_val)
        for i in range(self.n_estimators):
            if not self.feature_subsample_size is None:
                ind = np.random.choice(X.shape[1], self.feature_subsample_size, replace=False)
                self.feats.append(ind)
            grad = y-beg_weig
            alg = DecisionTreeRegressor(max_depth=self.max_depth, **self.other_par)
            alg.fit(X[:,ind], grad)
            self.fitted.append(alg)
            pred = alg.predict(X[:,ind])
            c = minimize_scalar(modif_mse, args=(beg_weig, pred, y), bounds=(0, 1000))
            self.coefs.append(c['x'])
            beg_weig -= self.learning_rate*c['x']*pred
            
            pred_train -= self.learning_rate*c['x']*alg.predict(X[:,ind])
            history_about_train.append(np.sqrt(((y-pred_train)**2/len(y)).sum()))
            
            times_his.append(time.time()-t)
            if not X_val is None:
                pred_val -= self.learning_rate*c['x']*alg.predict(X_val[:,ind])
                history.append(np.sqrt(((y_val-pred_val)**2/val_len).sum()))
            t = time.time()
        if not X_val is None:
            return history_about_train, history, times_his
        else:
            return history_about_train, None, times_his

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(len(X))
        if self.feature_subsample_size is None:
            for i in range(len(self.fitted)):
                pred -= self.learning_rate*self.coefs[i]*self.fitted[i].predict(X)
        else:
            for i in range(len(self.fitted)):
                pred -= self.learning_rate*self.coefs[i]*self.fitted[i].predict(X[:,self.feats[i]])
        return pred
