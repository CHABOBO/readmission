from datetime import datetime
from time import time
import numpy as np
import pandas as pd
import time
import os

#Local methods

def load_data(typeEncounter, typeDataFeatures):
        
    df_all=pd.read_pickle(os.path.join('resources','prepared_clean_data_' + typeEncounter  + "_" + typeDataFeatures + '.pkl'))

    return df_all

def filter_data_by_class(df_all, typeHypothesis):
    
    # Readmitted none vs readmitted
    if typeHypothesis == "all_readmisssion_vs_none":
        df_all["readmitted"][df_all["readmitted"].values > 0] = 1

    # Readmitted none vs early readmitted            
    if typeHypothesis == "early_readmission_vs_none":
        df_all= df_all[df_all["readmitted"].isin([0,1])]
        
    return df_all

def compute_type_features(df_all):

    numCols = ['add_in_out', 'add_procs_meds', 'div_visits_time', 'div_em_time', 'div_visit_med', 
               'div_em_med',"sum_ch_med",
               'time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 
               'number_emergency', 'number_inpatient', 'number_diagnoses',            
               'number_treatment','number_treatment_0','number_treatment_1','number_treatment_2','number_treatment_3']
    
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


def get_experiments(typeDataExperiment):
    
    if typeDataExperiment == "disease":
        return ['Diabetis', 'Circulatory', 'Digestive', 'Genitourinary', 'Poisoning', 'Muscoskeletal',
                   'Neoplasms', 'Respiratory']

    if typeDataExperiment == "all":
        return ["all"]

def filter_data_by_experiment(df_all, experiment):
        
    #Copy original dataset
    df_all_filtered = df_all.copy()
        
    #Experiment diff 
    if experiment != "all":        
        
        for sufx in ["_1", "_3"]:
            
            #Current disease
            curr_disease = experiment + sufx            
            
            if curr_disease in df_all_filtered.columns:

                # Filter by disease
                df_all_filtered = df_all_filtered[df_all_filtered[curr_disease] == 1]

                # Remove disease column
                df_all_filtered = df_all_filtered[[c for c in df_all_filtered.columns if c != curr_disease]]
    
    return df_all_filtered