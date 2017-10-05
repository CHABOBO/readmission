from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

__all__ = [
    'WithinClassMeanImputer',
]

class TypeFeatImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, dataCatCols = None, allNameCols = None):
        self.fill = None
        self.dataCatCols = dataCatCols
        self.allNameCols = allNameCols
        
    def fit(self, X, y = None):

        X = pd.DataFrame(X, columns=self.allNameCols)
        
        self.fill = pd.Series([X[c].value_counts().idxmax()
            if self.dataCatCols[i] == 1 else X[c].mean() for i,c in enumerate(X.columns)],
            index=X.columns)
        
        return self

    def transform(self, X):
        
        X = pd.DataFrame(X, columns=self.allNameCols)    
        X.fillna(self.fill, inplace=True)
              
        return X.values
 


