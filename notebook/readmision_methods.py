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

def compute_type_features(cols):

    numCols = ['add_in_out', 'add_procs_meds', 'div_visits_time', 'div_em_time', 'div_visit_med', 
               'div_em_med',"sum_ch_med",
               'time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 
               'number_emergency', 'number_inpatient', 'number_diagnoses',            
               'number_treatment','number_treatment_0','number_treatment_1','number_treatment_2','number_treatment_3']
    
    catCols = []
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

def create_filters(df):
    featFilters = []
    flt_0 = []
    flt_1 = []
    flt_2 = []
    flt_3 = []
    flt_4 = []
    flt_5 = []
    flt_6 = []
    flt_7 = []
    flt_8 = []
    flt_9 = []
    
    patientCols = [u'gender', u'age', u'race_AfricanAmerican', u'race_Caucasian',u'race_Other']
    
    admDisCols = patientCols[:]
    admDisCols.extend([u'adm_src_1',
       u'adm_src_2', u'adm_src_3', u'adm_src_4', u'adm_src_5', u'adm_src_6',
       u'adm_src_7', u'adm_src_8', u'adm_src_10', u'adm_src_11', u'adm_src_13',
       u'adm_src_14', u'adm_src_22', u'adm_src_25'])
    
    admDisCols_1 = patientCols[:]
    admDisCols_1.extend([u'adm_1', u'adm_2',
       u'adm_3', u'adm_4', u'adm_7'])
    
    admDisCols_2 = patientCols[:]
    admDisCols_2.extend([u'medSpec_cardio', u'medSpec_Family/GeneralPractice',
       u'medSpec_InternalMedicine', u'medSpec_surgery',"diss_home"])    
    
    admDisCols_3 = patientCols[:]
    admDisCols_3.extend([u'adm_src_1',
       u'adm_src_2', u'adm_src_3', u'adm_src_4', u'adm_src_5', u'adm_src_6',
       u'adm_src_7', u'adm_src_8', u'adm_src_10', u'adm_src_11', u'adm_src_13',
       u'adm_src_14', u'adm_src_22', u'adm_src_25', u'adm_1', u'adm_2',
       u'adm_3', u'adm_4', u'adm_7',u'medSpec_cardio', u'medSpec_Family/GeneralPractice',
       u'medSpec_InternalMedicine', u'medSpec_surgery',"diss_home"])
    
    hospitalCols = patientCols[:]
    hospitalCols.extend([u'num_lab_procedures', u'num_procedures',u'time_in_hospital',u'HbA1c'])
    
    visitCols = patientCols[:]
    visitCols.extend([u'number_outpatient', u'number_emergency', u'number_inpatient'])
    
    diagCols = patientCols[:]
    diagCols.extend([ u'number_diagnoses',u'Diabetis_3', u'Circulatory_3', u'Digestive_3',
       u'Genitourinary_3', u'Poisoning_3', u'Muscoskeletal_3', u'Neoplasms_3',
       u'Respiratory_3'])
    
    medCols = patientCols[:]
    medCols.extend([u'insulin', u'metformin', u'pioglitazone',
       u'glimepiride', u'glipizide', u'repaglinide', u'nateglinide', 
       u'Change', u'num_medications', u'diabetesMed'])
    
    extraCols = patientCols[:]
    extraCols.extend(['ComplexHbA1c', 'add_in_out', 'add_procs_meds', 'div_visits_time', 'div_em_time', 'div_visit_med', 'div_em_med', 'sum_ch_med', 'number_treatment_0', 'number_treatment_1', 'number_treatment_2', 'number_treatment_3'])
    
    selectCols = [u'age', u'race_AfricanAmerican', u'race_Caucasian',u'number_inpatient',u'HbA1c', u'diabetesMed',
                  "diss_home",u'adm_src_7',u'adm_1',u'adm_3',u'adm_7',u'number_diagnoses','div_em_time', 'div_visit_med',
                  u'Circulatory_3','sum_ch_med','add_in_out']
    
    for i in range(len(df.columns[:-1])):
        
        #Patient filter
        if df.columns[i] in patientCols:
            flt_0.append(1)
        else:
            flt_0.append(0)
    
        #Patient admission-discharge
        if df.columns[i] in admDisCols_3:
            flt_1.append(1)
        else:
            flt_1.append(0)
          
        #Patient hospital
        if df.columns[i] in hospitalCols:
            flt_2.append(1)
        else:
            flt_2.append(0)
            
        #Patient visits
        if df.columns[i] in visitCols:
            flt_3.append(1)
        else:
            flt_3.append(0)
            
        #Patient diagnosis
        if df.columns[i] in diagCols:
            flt_4.append(1)
        else:
            flt_4.append(0)
            
        #Patient admission
        if df.columns[i] in medCols:
            flt_5.append(1)
        else:
            flt_5.append(0)
            
        #None filter
        flt_6.append(1)
        
        #Patient extra
        if df.columns[i] in extraCols:
            flt_7.append(1)
        else:
            flt_7.append(0)
            
        #Patient none diagnosis
        if df.columns[i] not in [u'Diabetis_3', u'Circulatory_3', u'Digestive_3',
       u'Genitourinary_3', u'Poisoning_3', u'Muscoskeletal_3', u'Neoplasms_3',
       u'Respiratory_3']:
            flt_8.append(1)
        else:
            flt_8.append(0)

        #Patient selected
        if df.columns[i] in selectCols:
            flt_9.append(1)
        else:
            flt_9.append(0)            
        
    featFilters = [["patient_filter",np.array(flt_0)],["admision_discharge_filter", np.array(flt_1)],
                   ["hospital_filter",np.array(flt_2)],["Visits_filter",np.array(flt_3)],
                   ["diagnosis_filter",np.array(flt_4)],["medicines_filter",np.array(flt_5)],
                   ["extra_filter",np.array(flt_6)],["none_filter",np.array(flt_7)], 
                   ["no_diagnosis_filter",np.array(flt_8)],["selected_filter",np.array(flt_9)]]
    
    return featFilters