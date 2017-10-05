from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import numpy as np
import pandas as pd

class UnivCombineFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, catCols = None, columns = None, percentile = None):
        self.ixCols = []
        self.catCols = catCols
        self.scorers = {0: [f_classif], 1: [chi2]}
        self.percentile = percentile
        self.columns = columns
        self.colsSelected = []
        self.percPvalue = None
        
    def fit(self, X, y = None):

        dfAll = []
        for typeCol in self.scorers.keys():
            for scorer in self.scorers[typeCol]:

                X_selected = X[:,self.catCols == typeCol]
                selector = SelectPercentile(scorer, percentile=self.percentile)
                selector.fit(X_selected, y)

                # Order features by scorer
                res = pd.DataFrame(np.array([(self.columns[self.catCols == typeCol][i], float(selector.pvalues_[i])) 
                            for i in np.argsort(selector.scores_)[::-1] ]),columns=["feature", "pvalue"])
                
                dfAll.append(res)
        
        dfAll = pd.concat(dfAll, axis=0)
        
        nanVals = np.where(dfAll.pvalue.values == 'nan')[0]        
        if len(nanVals > 0):
            dfAll.pvalue.iloc[nanVals] = 0.0
            
        dfAll.pvalue = pd.to_numeric(dfAll.pvalue)
        dfAll.sort_values("pvalue", ascending=True, inplace=True)
                
        self.percPvalue = dfAll.pvalue.quantile(self.percentile/100.0)
        
        #Get feats by percentile
        self.colsSelected = dfAll[dfAll.pvalue <= self.percPvalue].feature.values
        #print dfAll[dfAll.pvalue <= self.percPvalue]

        for c in self.colsSelected:
            self.ixCols.extend(np.where(self.columns == c)[0])
        
        return self

    def transform(self, X):      
        return X[:,self.ixCols]
    
    def _colsSelected(self):        
        return self.colsSelected
    
    def _pvaluePercentile(self):
        return self.percPvalue

