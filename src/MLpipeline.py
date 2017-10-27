from sklearn import preprocessing
from time import time
import numpy as np
import csv
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE,ADASYN, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline

from operator import truediv
from datetime import datetime
import pandas as pd
import time
import os

from pylab import *
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:,.2f}'.format

import sys
from FeatFilter import FeatFilter
from TypeFeatImputer import TypeFeatImputer
from UnivCombineFilter import UnivCombineFilter


#Generic helper methods
def train_test_partition(df_all, ts_thr=0.30):
    y = df_all.readmitted
    y = y.values

    X = df_all.iloc[:,:-1].values
    sss = StratifiedShuffleSplit(y, 1, test_size=ts_thr, random_state=32) #random_state=42
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    return X_train, X_test, y_train, y_test

def train_partition(X_train, y_train, tr_thr=0.10):
    X_train_aux = []
    y_train_aux = []
    if tr_thr >= 1.0:
            X_train_aux = X_train
            y_train_aux = y_train
    else:
        sss = StratifiedShuffleSplit(y_train, 1, test_size=1-tr_thr, random_state=32) #random_state=42
        for train_index, test_index in sss:
            X_train_aux = X_train[train_index]
            y_train_aux = y_train[train_index]

    return X_train_aux, y_train_aux

def create_pipelines(catCols,reducedCols, hyperparams, fs_methods, sm_method, sm_types, cls_methods, lms, featsToFilter=None):
    
    basePipeline = Pipeline([
            ("FeatFilter", FeatFilter(featsToFilter)),
            ("Imputer", TypeFeatImputer(catCols, reducedCols)),
            ("Scaler", StandardScaler()),
            ("Variance", VarianceThreshold(threshold=0.0))
        ])

    pipeline = [] 

    for fs_method in fs_methods:
        for sm_type in sm_types:
            for cls_method in cls_methods:
                for lm in lms:
                    if not (fs_method == "rfe_rf_fs" and cls_method == "rf") and not(fs_method == "lasso_fs" and cls_method == "logReg"):
                        params = {}   
                        pipe = Pipeline(list(basePipeline.steps))

                        if fs_method == "combine_fs":
                            pipe.steps.insert(1,(fs_method, UnivCombineFilter(catCols,np.array(reducedCols))))
                            pm = hyperparams[hyperparams[:,1] == fs_method,2][0]
                            params.update(pm)                            


                        if fs_method == "rfe_rf_fs":
                            pipe.steps.append((fs_method, RFE(estimator=RandomForestClassifier(class_weight='balanced',
                                                                                               n_estimators=100,
                                                                                               random_state=33))))
                            pm = hyperparams[hyperparams[:,1] == fs_method,2][0]
                            params.update(pm) 
                            
                        if fs_method == 'lasso_fs':
                            pipe.steps.append((fs_method, SelectFromModel(
                                        LogisticRegression(n_jobs=-1, penalty="l1", dual=False, random_state=42))))
                            pm = hyperparams[hyperparams[:,1] == fs_method,2][0]
                            params.update(pm) 
                            
                        #Add classifiers
                        if cls_method == "knn":
                            pipe.steps.append((cls_method, KNeighborsClassifier()))
                            pm = hyperparams[hyperparams[:,1] == cls_method,2][0]
                            params.update(pm) 
                            
                        if cls_method == "logReg":
                            pipe.steps.append((cls_method, LogisticRegression(random_state=42)))
                            pm = hyperparams[hyperparams[:,1] == cls_method,2][0]
                            params.update(pm) 
                            
                        if cls_method == "svmRBF":
                            pipe.steps.append((cls_method, SVC(random_state=42,probability=True)))
                            pm = hyperparams[hyperparams[:,1] == cls_method,2][0]
                            params.update(pm) 

                        if cls_method == "rf":
                            pipe.steps.append((cls_method, RandomForestClassifier(n_jobs=-1,class_weight='balanced',random_state=42)))
                            pm = hyperparams[hyperparams[:,1] == cls_method,2][0]
                            params.update(pm) 
                            
                        if cls_method == "gbt":
                            pipe.steps.append((cls_method, GradientBoostingClassifier(random_state=42,subsample=0.1,loss="deviance")))
                            pm = hyperparams[hyperparams[:,1] == cls_method,2][0]
                            params.update(pm) 

                        if cls_method == "nb":
                            pipe.steps.append((cls_method, GaussianNB()))
                            params.update({}) 
                            
                        if cls_method == "nn":
                            pipe.steps.append((cls_method, MLPClassifier(
                                        activation='logistic',
                                        solver='lbfgs', 
                                        hidden_layer_sizes=(5, 2), 
                                        random_state=13)))
                            pm = hyperparams[hyperparams[:,1] == cls_method,2][0]
                            params.update(pm) 
                            

                        #Add sampling
                        pipe_imb = make_pipeline(*[p[1] for p in pipe.steps])
                        stps = len(pipe_imb.steps)        
                        for s in range(stps):
                            pipe_imb.steps.remove(pipe_imb.steps[0])
                        for s in range(stps):
                            pipe_imb.steps.append(pipe.steps[s])

                        if sm_type == "after":                    
                            pipe_imb.steps.insert(stps - 1, 
                                                  (sm_method, SMOTE(ratio='auto', kind='regular', random_state=32)))
                            pm = hyperparams[hyperparams[:,1] == cls_method,2][0]
                            params.update(pm) 

                        pipeline.append([fs_method,sm_type,cls_method,lm,pipe_imb,params])

    pipelines = pd.DataFrame(pipeline, columns=["fs","sm","cls","metric","pipe","pipe_params"])
    pipelines.sort_values("fs", inplace=True)

    return pipelines

def precision_0(ground_truth, predictions):
    prec = metrics.precision_score(ground_truth, predictions,pos_label=0)
    return prec

def recall_0(ground_truth, predictions):
    rec = metrics.recall_score(ground_truth, predictions,pos_label=0)
    return rec

def specificity(ground_truth, predictions):
    cm_train = metrics.confusion_matrix(ground_truth, predictions)
    tn = cm_train[0,0]
    fp = cm_train[0,1]
    fn = cm_train[1,0]
    tp = cm_train[1,1]
    train_spec = tn / float(tn+fp)
    return train_spec


# One Experiment One file
def run_pipeline(name, pipeline, X_train, X_test, y_train, y_test, tr_thr=1.0, ts_thr=0.3, cv_folds=5, cv_thr=0.3, verbose=True, save=True):
    
    results = []

    print "\nDataSet:"
    print "**********"
    print "**********"
    print "SIZE:", tr_thr
    print "NAME:", name

    print "ALL TRAIN:", X_train.shape
    print "TRAIN:", "[0's:", np.sum(y_train==0), "1's:", np.sum(y_train==1), "]"
    print "ALL TEST:", X_test.shape
    print "TEST:", "[0's:", np.sum(y_test==0), "1's:", np.sum(y_test==1), "]"

    for num_exp in range(pipeline.shape[0]):

        # Run experiment
        start = time.time()

        #Prepare pipe_cls      
        pipeline_cls = pipeline["pipe"].iloc[num_exp]
        pipeline_params = pipeline["pipe_params"].iloc[num_exp]
        fs = pipeline["fs"].iloc[num_exp]
        sm = pipeline["sm"].iloc[num_exp]
        cls = pipeline["cls"].iloc[num_exp]
        lm = pipeline["metric"].iloc[num_exp]

        print "\nNum experiment:", str(num_exp), "/", str(pipeline.shape[0] - 1)
        print "****************"

        print "FS:",fs
        print "SM:",sm
        print "CLS:",cls
        print "METRIC:",lm

        #Prepare cv
        cv_inner = StratifiedShuffleSplit(y_train, n_iter=cv_folds, test_size=cv_thr, random_state=24)
        cv_outer = StratifiedShuffleSplit(y_train, n_iter=cv_folds, test_size=cv_thr, random_state=42)

        #Fit pipeline with CV                        
        grid_pipeline = GridSearchCV(pipeline_cls, param_grid=pipeline_params, verbose=verbose, 
                                     n_jobs=-1, cv=cv_inner, scoring= lm, error_score = 0) 
        grid_pipeline.fit(X_train, y_train)


        # Compute Train scores (with best CV params)
        y_pred = grid_pipeline.best_estimator_.predict(X_train)  
        train_f1_w = metrics.f1_score(y_train, y_pred, average='weighted', pos_label=None)
        train_p, train_r, train_f1, train_s = metrics.precision_recall_fscore_support(y_train, y_pred,labels=None,average=None, sample_weight=None)
        fpr, tpr, _ = metrics.roc_curve(y_train, y_pred)
        train_auc = metrics.auc(fpr, tpr)                
        cm_train = metrics.confusion_matrix(y_train, y_pred)
        tn = cm_train[0,0]
        fp = cm_train[0,1]
        fn = cm_train[1,0]
        tp = cm_train[1,1]
        train_sens = train_r[1]
        train_spec = tn / float(tn+fp)                
        print "\nTRAIN f1 (weighted): %0.3f" % (train_f1_w)
        print "TRAIN Precision [c=0,1]:", train_p
        print "TRAIN Recall [c=0,1]:", train_r
        print "TRAIN AUC: %0.3f" % (train_auc)                 
        print "TRAIN sensitivity:", train_sens
        print "TRAIN Specificity: ", train_spec

        # Compute evaluation scores
        print "\nCV INNER metric: {}".format(lm)
        print "CV INNER selected params {}".format(grid_pipeline.best_params_.values())
        print "CV INNER score: {}".format(grid_pipeline.best_score_)

        scorings = {'roc_auc': 'roc_auc',
                    'f1_weighted':'f1_weighted',
                    'precision_1':'precision',
                    'recall_1':'recall',
                    'precision_0' : metrics.make_scorer(precision_0),
                    'recall_0' : metrics.make_scorer(recall_0),
                    'spec': metrics.make_scorer(specificity)
                   } 

        cv_scores = cross_validate(grid_pipeline.best_estimator_, X_train, y_train, 
                                   cv=cv_outer, scoring=scorings, n_jobs=-1, 
                                   return_train_score = False)

        cv_f1_w_mean = np.mean(cv_scores["test_f1_weighted"])
        cv_f1_w_std = np.std(cv_scores["test_f1_weighted"])
        cv_p1_mean = np.mean(cv_scores["test_precision_1"])
        cv_p1_std = np.std(cv_scores["test_precision_1"])
        cv_r1_mean = np.mean(cv_scores["test_recall_1"])
        cv_r1_std = np.std(cv_scores["test_recall_1"])                
        cv_p0_mean = np.mean(cv_scores["test_precision_0"])
        cv_p0_std = np.std(cv_scores["test_precision_0"])
        cv_r0_mean = np.mean(cv_scores["test_recall_0"])
        cv_r0_std = np.std(cv_scores["test_recall_0"])

        cv_auc_mean = np.mean(cv_scores["test_roc_auc"])
        cv_auc_std = np.std(cv_scores["test_roc_auc"])                
        cv_spec_mean = np.mean(cv_scores["test_spec"])
        cv_spec_std = np.std(cv_scores["test_spec"])
        cv_sens_mean = cv_r1_mean
        cv_sens_std = cv_r1_std

        print "\nCV OUTER f1-weighted score: %0.3f  (+/-%0.03f)" % (cv_f1_w_mean,cv_f1_w_std)               
        print "CV OUTER prec score [c=0,1]: {:.3f} (+/- {:.3f}), {:.3f}  (+/- {:.3f})".format(cv_p0_mean,cv_p0_std,cv_p1_mean,cv_p1_std)                
        print "CV OUTER rec  score [c=0,1]: {:.3f} (+/- {:.3f}), {:.3f}  (+/- {:.3f})".format(cv_r0_mean,cv_r0_std,cv_r1_mean,cv_r1_std)
        print "CV OUTER AUC score: %0.3f  (+/-%0.03f)" % (cv_auc_mean,cv_auc_std) 
        print "CV OUTER sensitivity score: %0.3f  (+/-%0.03f)" % (cv_sens_mean,cv_sens_std) 
        print "CV OUTER Specificity score: %0.3f  (+/-%0.03f)" % (cv_spec_mean,cv_spec_std)
        print "Selected params (bests from CV) {}".format(grid_pipeline.best_params_.values())


        #Compute test scores
        y_pred = grid_pipeline.best_estimator_.predict(X_test)
        test_f1_w = metrics.f1_score(y_test, y_pred, average='weighted', pos_label=None)
        test_p, test_r, test_f1, test_s = metrics.precision_recall_fscore_support(y_test, y_pred,labels=None,average=None, sample_weight=None)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        test_auc = metrics.auc(fpr, tpr)                
        cm_test = metrics.confusion_matrix(y_test, y_pred)
        tn = cm_test[0,0]
        fp = cm_test[0,1]
        fn = cm_test[1,0]
        tp = cm_test[1,1]
        test_sens = test_r[1]
        test_spec = tn / float(tn+fp)

        print "\nTEST f1 (weighted): %0.3f" % (test_f1_w)
        print "TEST Precision [c=0,1]:", test_p
        print "TEST Recall [c=0,1]:", test_r                
        print "TEST AUC: %0.3f" % (test_auc)                
        print "TEST sensitivity:", test_sens
        print "TEST Specificity:", test_spec
        print "Confussion matrix:"
        print "         | PRED"
        print "REAL-->  v "
        print cm_test

        end = time.time()
        print "\nTotal time:", end - start

        res = [num_exp,
               name,
               tr_thr,
               fs,
               sm,
               cls,
               lm,
               grid_pipeline.best_params_.values(),
               train_sens,
               train_spec,
               train_auc,
               train_r,
               train_p,
               train_f1_w,
               cv_sens_mean,
               cv_sens_std,
               cv_spec_mean,
               cv_spec_std,
               cv_auc_mean,
               cv_auc_std,
               [cv_p0_mean,cv_p1_mean],
               [cv_p0_std,cv_p1_std],
               [cv_r0_mean,cv_r1_mean],
               [cv_r1_std,cv_r0_std],
               cv_f1_w_mean,
               cv_f1_w_std,
               test_sens,
               test_spec,
               test_auc,
               test_r,
               test_p,
               test_f1_w,
               cm_test,              
               end - start,
               grid_pipeline.best_estimator_
              ]
        results.append(res)

        #Save results
        if save:
            df = pd.DataFrame(np.array(res).reshape(1,35), columns=
                              ["exp", "name",
                               "size_tr","fs","sm","cls","metric","params",
                               "tr_sens","tr_spec","tr_auc",
                               "tr_prec","tr_rec","tr_f1",
                               "cv_sens_mean","cv_sens_std","cv_spec_mean","cv_spec_std","cv_auc_mean","cv_auc_std",
                               "cv_prec_mean","cv_prec_std","cv_rec_mean","cv_rec_std",
                               "cv_f1_mean","cv_f1_std",
                               "test_sens","test_spec","test_auc",
                               "test_rec","test_prec","test_f1",
                               "cm_test",
                               "time","pipeline"])

            df.to_pickle(os.path.join("resources", "results",
                                      'results_pipe_' + 
                                      "test_" + str(ts_thr) + "_" +
                                      "train_" + str(tr_thr) + "_" +
                                      str(name) + '_' +
                                      str(fs) + '_' +
                                      str(sm) + '_' +
                                      str(lm) + '_' +
                                      str(cls) + '_' +
                                      time.strftime("%Y%m%d-%H%M%S") +
                                      '.pkl'))
    return results

# Execute pipelines method
def run(name,df_all, catCols, reducedCols, hyperparams, ts_thr, tr_thrs, fs_methods, sm_method, 
        sm_types, cls_methods, lms, cv_folds, cv_thr, verbose=True, save=False):

    results = []
    for tr_thr in tr_thrs:

        #Create pipelines
        pipeline = create_pipelines(catCols, reducedCols, hyperparams, fs_methods, sm_method, sm_types, cls_methods, lms)                    
            
        #Partition data
        X_train, X_test, y_train, y_test = train_test_partition(df_all, ts_thr)
        X_train, y_train = train_partition(X_train, y_train, tr_thr)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    
        #Run pipelines
        results.extend(run_pipeline(name, pipeline, X_train, X_test, y_train, y_test, ts_thr, verbose, save))

    return results
