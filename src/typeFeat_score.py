import numpy as np
from math import modf
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import chi2

def typeFeat_score(X, y = None, catCols=None):
    scores = np.zeros(X.shape[1])
    
    #Sign. test for categorical
    sc, pval = chi2(X,y)
    scores[catCols == 1] = pval[catCols == 1]
    
    #Sign. test for numerical
    pval = []
    for i in range(X.shape[1]):
        s, p = mannwhitneyu(X[:,i],y)
        pval.append(p)
    pval = np.array(pval)
    scores[catCols == 0] = pval[catCols == 0]
    return scores
