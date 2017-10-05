from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import chi2
from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd


class ContinuousFS(BaseEstimator, TransformerMixin):
    
    def __init__(self, dataCatCols=None, perc=None, oneMinFeat = True):  
        self.dataCatCols = dataCatCols        
        self.percent = perc
        self.oneMinFeat = oneMinFeat
        self.support = []

    def config(self, dataCatCols, perc):
        self.dataCatCols = dataCatCols       
        self.percent = perc
        
    def fit(self, X, y = None):        
        XCat = X[:,self.dataCatCols == 0]

        p_val = []
        for i in range(XCat.shape[1]):
            s, p = mannwhitneyu(XCat[:,i],y)
            p_val.append(p)
        
        #Create support
        i = 0
        for c in self.dataCatCols:
            
            #Numeric columns
            if c == 0:
                if p_val[i] < 0.05:
                    self.support.append(1)
                else:
                    self.support.append(0)
                i += 1
            else:
                self.support.append(0)                        
        
        self.support = np.array(self.support)
        
        if self.oneMinFeat and np.sum(self.support==1) == 0:
            ix = np.argmin(p_val)
            self.support[ix] = 1
            
        return self

    def transform(self, X):
        return X[:,self.support == 1]
    
class DiscreteFS(BaseEstimator, TransformerMixin):
    
    def __init__(self, dataCatCols=None, perc=None, oneMinFeat = True):  
        self.dataCatCols = dataCatCols        
        self.percent = perc
        self.oneMinFeat = oneMinFeat
        self.support = []

    def config(self, dataCatCols, perc):
        self.dataCatCols = dataCatCols
        self.percent = perc
        
    def fit(self, X, y = None): 

        XCat = X[:,self.dataCatCols == 1]
        
        p_val = []
        for i in range(XCat.shape[1]):
            s, p = chi2(XCat[:,i].reshape(-1,1),y)
            p_val.append(p)
        
        #Create support
        i = 0
        for c in self.dataCatCols:
            
            #Categoric columns
            if c == 1:
                if p_val[i] < 0.05:
                    self.support.append(1)
                else:
                    self.support.append(0)
                i += 1
            else:
                self.support.append(0)                
                
        self.support = np.array(self.support)
        if self.oneMinFeat and np.sum(self.support==1) == 0:
            ix = np.argmin(p_val)
            self.support[ix] = 1
        
        return self

    def transform(self, X):
        return X[:,self.support == 1]
