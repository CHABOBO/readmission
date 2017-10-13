from sklearn import preprocessing
from time import time
import numpy as np
import csv
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from TypeFeatImputer import TypeFeatImputer
from UnivCombineFilter import UnivCombineFilter

def load_data(typeEncounter, typeDiagnosis, typeDataFeatures):
    if typeDataFeatures == "minimum":
        df_all=pd.read_pickle(os.path.join('resources','clean_data_' + typeEncounter + "_" +  typeDiagnosis+ '_hyp_1.pkl'))
    if typeDataFeatures == "extended":
        df_all=pd.read_pickle(os.path.join('resources','clean_data_' + typeEncounter + "_" +  typeDiagnosis+ '_hyp_1_' + typeDataFeatures + '.pkl'))

    return df_all

def get_columns(df_all, typeDiagnosis):
    if typeDiagnosis == "diag_1":
        colsDiseases = [u'Diabetis_1', u'Circulatory_1', u'Digestive_1', u'Genitourinary_1', u'Poisoning_1', u'Muscoskeletal_1',
               u'Neoplasms_1', u'Respiratory_1']

    if typeDiagnosis == "diag_3":
        colsDiseases = [u'Diabetis_3', u'Circulatory_3', u'Digestive_3', u'Genitourinary_3', u'Poisoning_3', u'Muscoskeletal_3',
               u'Neoplasms_3', u'Respiratory_3']
    
    colsNonDiseases = [c for c in df_all.columns if c not in colsDiseases]
    
    return colsDiseases, colsNonDiseases

def filter_data_by_class(df_all, typeHypothesis):
    
    # Readmitted none vs readmitted
    if typeHypothesis == "all_readmisssion_vs_none":
        df_all["readmitted"][df_all["readmitted"].values > 0] = 1

    # Readmitted none vs early readmitted            
    if typeHypothesis == "early_readmission_vs_none":
        df_all= df_all[df_all["readmitted"].isin([0,1])]
        
    return df_all

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

def create_pipelines(catCols,reducedCols, fs_methods, sm_method, sm_types, cls_methods, lms):
    basePipeline = Pipeline([
            ("Imputer", TypeFeatImputer(catCols, reducedCols)),
            ("Scaler", StandardScaler()),
            ("Variance", VarianceThreshold(threshold=0.0))
        ])

    pipeline = [] 

    for fs_method in fs_methods:
        for sm_type in sm_types:
            for cls_method in cls_methods:
                for lm in lms:
                    if not (fs_method == "rfe_rf_fs" and cls_method == "rf"):
                        params = {}   
                        pipe = Pipeline(list(basePipeline.steps))

                        if fs_method == "combine_fs":
                            pipe.steps.insert(1,(fs_method, UnivCombineFilter(catCols,np.array(reducedCols))))
                            params.update({fs_method + '__percentile':[5,10,20,30,40,50]})

                        if fs_method == "rfe_rf_fs":
                            pipe.steps.append((fs_method, RFE(estimator=RandomForestClassifier(class_weight='balanced',
                                                                                               n_estimators=100,
                                                                                               random_state=33))))
                            params.update({fs_method + '__step':[0.1]})
                            params.update({fs_method + '__n_features_to_select':[
                                                            int(len(reducedCols)*0.4), 
                                                            int(len(reducedCols)*0.6), 
                                                            int(len(reducedCols)*0.8)]})

                        if fs_method == 'lasso_fs':
                            pipe.steps.append((fs_method, SelectFromModel(
                                        LogisticRegression(n_jobs=-1, penalty="l1", dual=False, random_state=42))))
                            params.update({fs_method + '__estimator__C': [0.001,0.01,0.1,1]})

                        #Add classifiers
                        if cls_method == "knn":
                            pipe.steps.append((cls_method, KNeighborsClassifier()))
                            params.update({'knn__n_neighbors':[1,3,5,7,9,11], 'knn__weights':['uniform', 'distance']})

                        if cls_method == "logReg":
                            pipe.steps.append((cls_method, LogisticRegression(random_state=42)))
                            params.update({'logReg__C': [0.00001,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,15,30]})
                            params.update({'logReg__class_weight': [None, 'balanced']})
                            params.update({'logReg__penalty': ['l1', 'l2']})

                        if cls_method == "svmRBF":
                            pipe.steps.append((cls_method, SVC(random_state=42,probability=True)))
                            params.update({'svmRBF__C': [0.01,0.1,0.5,1,5,10,30,50,100], 
                             'svmRBF__gamma' : [0.0001,0.001,0.01, 0.1,1,5]})
                            params.update({'svmRBF__class_weight': [None, 'balanced']})

                        if cls_method == "rf":
                            pipe.steps.append((cls_method, RandomForestClassifier(n_jobs=-1,random_state=42)))
                            params.update({'rf__n_estimators': [100,150,200,250,500], 
                                           'rf__criterion': ['entropy','gini'],
                                           'rf__max_depth' : [None,4,6]})
                            params.update({'rf__class_weight': [None, 'balanced']})

                        if cls_method == "nb":
                             pipe.steps.append((cls_method, GaussianNB()))
                             params.update({})
                                               
                        if cls_method == "nn":
                            pipe.steps.append((cls_method, MLPClassifier(
                                        activation='logistic',
                                        solver='lbfgs', 
                                        hidden_layer_sizes=(5, 2), 
                                        random_state=13)))
                            params.update({
                                    'nn__alpha': [1e-5,0.00001,0.0001,0.001,0.01,0.1,1,3,5,10],
                                    'nn__hidden_layer_sizes':[(30,),(50,),(70,),(100,),(150,),
                                                              (30,30),(50,50),(70,70),(100,100),
                                                              (30,30,30),(50,50,50),(70,70,70)
                                                             ]
                                          })

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
                            params.update({sm_method + "__k_neighbors":[3,4,5]})


                        pipeline.append([fs_method,sm_type,cls_method,lm,pipe_imb,params])
    pipelines = pd.DataFrame(pipeline, columns=["fs","sm","cls","metric","pipe","pipe_params"])
    pipelines.sort_values("fs", inplace=True)
    print pipelines.shape
    return pipelines

def compute_type_features(df_all, typeDataFeatures):
    if typeDataFeatures == "minimum":
        numCols = ['time_in_hospital']

    if typeDataFeatures == "extended":
        numCols = ['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 
                'number_emergency', 'number_inpatient', 'number_diagnoses']

    catCols = []
    cols = df_all.columns
    reducedCols = cols[:-1]

    for i in range(len(cols)-1):
        if cols[i] not in numCols:
            catCols.append(1)
        else:
            catCols.append(0)
    catCols = np.array(catCols)
    
    return catCols, reducedCols

def get_diseases(colsDiseases, typeDisease):
    if typeDisease == "subset":
        return ["subset"]
    else:
        if typeDisease in colsDiseases:
            return [typeDisease]
        else:
            return colsDiseases

def filter_data_by_diseases(df_all, disease, typeDataExperiment, colsNonDiseases):
    if disease == "subset":
        df_all_filtered = df_all.copy()
    else:
        cols_filtered = colsNonDiseases[:]
        cols_filtered.insert(-1, disease)
        df_all_filtered = df_all[cols_filtered].copy()    
    
    if typeDataExperiment == "disease" and disease != "subset":
        df_all_filtered = df_all_filtered[df_all_filtered[disease] == 1]
        df_all_filtered = df_all_filtered[[c for c in df_all_filtered.columns if c != disease]]
    
    return df_all_filtered

def run(df_all, colsNonDiseases, diseases, typeDataFeatures, typeDataExperiment, typeEncounter, typeHypothesis, typeDiagnosis, ts_thr, tr_thrs, fs_methods, sm_method, sm_types, cls_methods, lms, cv_folds, cv_thr, verbose=True, save=False):

    results = []
    for tr_thr in tr_thrs:
        for disease in diseases:

            df_all_filtered = filter_data_by_diseases(df_all, disease, typeDataExperiment, colsNonDiseases)               
            X_train, X_test, y_train, y_test = train_test_partition(df_all_filtered, ts_thr)
            X_train, y_train = train_partition(X_train, y_train, tr_thr)

            catCols, reducedCols = compute_type_features(df_all_filtered, typeDataFeatures)
            pipeline = create_pipelines(catCols,reducedCols, fs_methods, sm_method, sm_types, cls_methods, lms)

            print "\nDataSet:"
            print "**********"
            print "**********"
            print "SIZE:", tr_thr
            print "DISEASE:", disease

            print df_all_filtered.shape
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

                # Compute pipeline evaluation with CVmetric
                print "\nCV INNER metric: {}".format(lm)
                print "CV INNER selected params {}".format(grid_pipeline.best_params_.values())
                print "CV INNER score: {}".format(grid_pipeline.best_score_)

                cv_f1 = cross_val_score(grid_pipeline.best_estimator_, X_train, y_train, 
                                                         cv=cv_outer, scoring='f1_weighted', n_jobs=-1)

                cv_prec = cross_val_score(grid_pipeline.best_estimator_, X_train, y_train, 
                                                         cv=cv_outer, scoring='precision_weighted', n_jobs=-1)

                cv_rec = cross_val_score(grid_pipeline.best_estimator_, X_train, y_train, 
                                                         cv=cv_outer, scoring='recall_weighted', n_jobs=-1)

                print "\nCV OUTER f1 score: %0.3f  (+/-%0.03f)" % (np.mean(cv_f1), np.std(cv_f1))
                print "CV OUTER prec score: %0.3f  (+/-%0.03f)" % (np.mean(cv_prec), np.std(cv_prec))
                print "CV OUTER rec score: %0.3f  (+/-%0.03f)" % (np.mean(cv_rec), np.std(cv_rec))
                print "Selected params (bests from CV) {}".format(grid_pipeline.best_params_.values())

                # Computel Train score (with best CV params)
                y_pred = grid_pipeline.best_estimator_.predict(X_train)
                train_prec_scores = metrics.precision_score(y_train, y_pred, average='weighted', pos_label=None)
                train_rec_scores = metrics.recall_score(y_train, y_pred, average='weighted', pos_label=None)    
                train_f1_scores = metrics.f1_score(y_train, y_pred, average='weighted', pos_label=None)

                print "\nTR F1 score:", train_f1_scores
                print "TR Prec score:", train_prec_scores
                print "TR Rec score:", train_rec_scores

                #Compute test score
                y_pred = grid_pipeline.best_estimator_.predict(X_test)
                test_f1 = metrics.f1_score(y_test, y_pred, average='weighted', pos_label=None)
                test_prec = metrics.recall_score(y_test, y_pred, average='weighted', pos_label=None)
                test_rec = metrics.precision_score(y_test, y_pred, average='weighted', pos_label=None)
                test_auc = metrics.roc_auc_score(y_test, y_pred, average='weighted')
                cm = metrics.confusion_matrix(y_test, y_pred)
                tn = cm[0,0]
                fp = cm[0,1]
                fn = cm[1,0]
                tp = cm[1,1]

                print "\nTest f1: %0.3f" % (test_f1)
                print "Test Precision: %0.3f" % (test_prec)
                print "Test Recall (Sensitivity): %0.3f" % (test_rec)
                print "Test Specificity: %0.3f" % (tn / float(tn + fp))
                print "Test AUC: %0.3f" % (test_auc)
                print 
                print "with following performance in test:"
                print metrics.classification_report(y_test, y_pred)
                print cm

                end = time.time()
                print "\nTotal time:", end - start
                results = [num_exp,
                               disease,
                               typeEncounter,
                               typeHypothesis,
                               typeDataFeatures,
                               typeDiagnosis,
                               tr_thr,
                               fs,
                               sm,
                               cls,
                               lm,
                               grid_pipeline.best_params_.values(),
                               train_f1_scores,
                               train_prec_scores,
                               train_rec_scores,
                               np.mean(cv_f1), 
                               np.std(cv_f1),
                               np.mean(cv_prec), 
                               np.std(cv_prec),
                               np.mean(cv_rec), 
                               np.std(cv_rec),                    
                               test_f1,
                               test_prec,
                               test_rec,
                               test_auc,                    
                               end - start,
                               grid_pipeline.best_estimator_
                              ]

                #Save results
                if save:
                    df = pd.DataFrame(np.array(results).reshape(1,27), columns=
                          ["exp", "typeDisease","typeEncounter","typeHypothesis","typeDataFeatures","typeDiagnosis",
                           "size_tr","fs","sm","cls","metric","params",
                           "tr_f1","tr_prec","tr_rec",
                           "cv_f1_mean","cv_f1_std","cv_prec_mean","cv_prec_std","cv_rec_mean","cv_rec_std",
                           "test_f1","test_prec","test_rec","test_auc",
                           "time","pipeline"])
                
                    df.to_pickle(os.path.join("resources", "results",
                                          'results_pipe_' + 
                                          "test_" + str(ts_thr) + "_" +
                                          "train_" + str(tr_thr) + "_" +
                                          str(disease) + '_' +
                                          str(typeEncounter) + '_' +
                                          str(typeHypothesis) + '_' +
                                          str(typeDataFeatures) + '_' +
                                          str(typeDataExperiment) + '_' +
                                          str(typeDiagnosis) + '_' +
                                          str(fs) + '_' +
                                          str(sm) + '_' +
                                          str(lm) + '_' +
                                          str(cls) + '_' +
                                          '.pkl'))

