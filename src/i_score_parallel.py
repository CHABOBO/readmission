import numpy as np

def i_score(X_param, y_param):

    i_score = []
    
    y = y_param.astype(float)
    X = X_param.astype(float)
    
    #Compute denominator
    y_mean = np.mean(y.astype(float))
    den = np.sum((y.astype(float) - y_mean)**2)

    #Stratify data in quartiles (by column)
    m = []
    m.append(X <= np.percentile(X,q=25, axis=0))
    m.append(np.logical_and(
                                X > np.percentile(X,q=25, axis=0),
                                X <= np.percentile(X,q=50, axis=0)
                  ))
    m.append(np.logical_and(
                X > np.percentile(X,q=50, axis=0),
                X <= np.percentile(X,q=75, axis=0)
            ))
    m.append(np.logical_and(
                X > np.percentile(X,q=75, axis=0),
                X <= np.percentile(X,q=100, axis=0)
            ))

    #Compute numerator column-wise
    num = []
    y_cp = np.tile(y, (X.shape[1], 1)).T    
    for j in range(len(m)):
        mk = np.ma.masked_array(y_cp, m[j] == 0)
        y_ij_mean = np.ma.filled(np.mean(mk,axis=0),0.0)
        n = np.sum(m[j],axis=0)
        num.append((n**2) * (y_ij_mean-y_mean)**2)


    num = np.sum(num, axis=0)
    i_score = (num / float(den))
    
    return i_score


