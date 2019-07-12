# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import ToolBox
import datetime
from sklearn.decomposition import PCA
class DropRowsWithNoTarget(BaseEstimator, TransformerMixin):
    #Ne sera pas utilisé dans la pipeline
    def __init__(self,target,dataframe):
        self.dataframe = dataframe
        self.target = target
    def fit(self,X,y):
        return self
    def transform(self,X):
        return X
class FillNaNCategoricalWithNone(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y):     #create fit method
        return self

    def transform(self,X):    #create transform method
        X = X.copy()
        catCols = X.select_dtypes(include='object').columns
        print(type(X))
        X[catCols] = X[catCols].fillna("None")
        print(X[catCols])
        return X
class CategoricalToDiscrete(BaseEstimator, TransformerMixin):
    def __init__(self,dictionnaryCat):
        self.dictionnaryCat = dictionnaryCat
    def fit(self,X,y):     #create fit method
        return self

    def transform(self,X):    #create transform method
        X = X.copy()
        for key, value in self.dictionnaryCat.items():
            print(key)
            X[key] = X[key].replace(value).astype(float)
        return X
class CategoricalTo0and1(BaseEstimator,TransformerMixin):
    def __init__(self,dictionnary):
        self.dictionnary = dictionnary
    def fit(self,X,y):
        return self
    def transform(self,X):    #create transform method
        X = X.copy()
        for key, value in self.dictionnary.items():
            X[key] = X[key].apply(lambda x : 1 if (x == value) else 0).astype(float)
        return X
class ReplaceValueBy0(BaseEstimator,TransformerMixin):
    def __init__(self,features):
        self.features = features
    def fit(self,X,y):
        return self
    def transform(self,X):
        X = X.copy()
        X[self.features] = X[self.features].fillna(0)
        return X
class ReplaceValueByMedian(BaseEstimator,TransformerMixin):
    def __init__(self,features):
        self.features = features
        self.medianByFeatures = {}
    def fit(self,X,y):
        for feat in self.features:
            self.medianByFeatures[feat] = X[feat].median()
        return self
    def transform(self,X):
        X = X.copy()
        for feat in self.features:
            X[feat] = X[feat].fillna(self.medianByFeatures[feat])
        return X
class CategoryToDummies(BaseEstimator,TransformerMixin):
    def __init__(self,dictionnary):
        self.dictionnary = dictionnary
    def fit(self,X,y):
        return self
    def transform(self,X):
        X = X.copy()
        for key, value in self.dictionnary.items():
            X[key]=pd.Categorical(X[key], categories=value)
        X = pd.get_dummies(X,prefix_sep="_-_")
        return X
class YearToAbsolute(BaseEstimator,TransformerMixin):
    def __init__(self,yearFeatures):
        self.yearFeatures = yearFeatures
    def fit(self,X,y):
        return self
    def transform(self,X):
        X = X.copy()
        now = datetime.datetime.now()
        for feature in self.yearFeatures:
            X[feature] = now.year - X[feature]
        return X
class StandardizeFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self,X):
        X = X.copy()
        colNotDummies = []
        for col in X.columns.values:
            if("_-_" not in col):
                colNotDummies.append(col)
        ss = StandardScaler()
        X[colNotDummies] = ss.fit_transform(X[colNotDummies])
        return X
class PCAReduction(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self,X):
        X= X.copy()
        pca = PCA(n_components=10)
        X = pca.fit_transform(X)
        return X
class ModelPrediction(BaseEstimator,TransformerMixin):
    def __init__(self,estimator):
        self.estimator = estimator
    def fit(self,X,y):
        self.estimator.fit(X,y)
        return self
    def transform(self,X):
        y = self.estimator.predict(X)
        return y