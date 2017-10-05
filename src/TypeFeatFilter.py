from sklearn.base import BaseEstimator, TransformerMixin
from math import modf
import numpy as np
import pandas as pd



class DiscreteFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, dataCatCols = None):
        self.dataCatCols = dataCatCols
        
    def fit(self, X, y = None):    
        return self

    def transform(self, X):      
        return X[:,self.dataCatCols == 1]
    
class ContinuousFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, dataCatCols = None):
        self.dataCatCols = dataCatCols
        
    def fit(self, X, y = None):    
        return self

    def transform(self, X):      
        return X[:,self.dataCatCols == 0]   
