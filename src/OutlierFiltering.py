from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope
from sklearn.preprocessing import StandardScaler,OneHotEncoder,Imputer,Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierFiltering(BaseEstimator, TransformerMixin):
    
    def __init__(self, colNames, codes, norm="l2"):
        self.colNames = colNames
        self.codes = codes
        self.codesToDel = []
	self.norm=norm
        
    def fit(self, X, y = None):
        
        scaler_norm = Normalizer(norm=self.norm).fit(X)
        df_all_norm = scaler_norm.transform(X).astype(float)

        df_all_norm = pd.DataFrame(
            data = np.hstack((df_all_norm, y.reshape((-1,1)))), 
            columns = self.colNames)

        X_SCALED = df_all_norm.iloc[:,:-1].values

        robust_cov_all = EmpiricalCovariance().fit(X_SCALED[:,:])
        robust_mahal_all = robust_cov_all.mahalanobis(X_SCALED[:,:] - robust_cov_all.location_) ** (0.33)

        #ALT:2
        rm = pd.DataFrame(robust_mahal_all, columns=["value"])

        iqr = float(rm["value"].quantile(0.75)) - float(rm["value"].quantile(0.25))
        outlierRatioRob_all_1 = rm["value"].quantile(0.75) + (1.5 * iqr)
        outlierRatioRob_all_2 = rm["value"].quantile(0.25) - (1.5 * iqr)

        print iqr, rm["value"].quantile(0.25), rm["value"].quantile(0.75)
        print "Ouliers min ratio:", outlierRatioRob_all_1, outlierRatioRob_all_2
        print "Num outliers detected:", len(X_SCALED[robust_mahal_all > outlierRatioRob_all_1, 0])
        print "Num outliers detected:", len(X_SCALED[robust_mahal_all < outlierRatioRob_all_2, 0])
        print [(self.codes[r],robust_mahal_all[r]) for r in range(len(robust_mahal_all))]

        patients_out = self.codes[robust_mahal_all > outlierRatioRob_all_1]
        self.codesToDel.append(patients_out)
        print "Patients outliers above: {}".format(patients_out)

        patients_out = self.codes[robust_mahal_all < outlierRatioRob_all_2]
        self.codesToDel.append(patients_out)
        print "Patients outliers below: {}".format(patients_out)
        
        return self
    
    def transform(self, X):
        return X[np.in1d(self.codes, np.array(self.codesToDel)) == False]
