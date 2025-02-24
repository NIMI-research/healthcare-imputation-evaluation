import os
import argparse
import pandas as pd

from utils.load_config import get_config, get_yaml
from utils.load_dataset import load_obs_data, load_eval_columns, load_full_data
from utils.create_missing_data import create_missing_dataset
from utils.plot_predictions import plot_line

from imputers.knn import KNN as knn
from imputers.mice import MiceImputer as mice
from imputers.nnimputer import NeuralNetworkImputer
from imputers.ssl_imputer import SSLImputer

from evals.eval import Evaluation as eval
from dataset_config import MimicConfig, TudConfig, HopperConfig


import warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--impute', action='store_true', help='Use the imputation mode.')
    group.add_argument('--eval', action='store_true', help='Evaluate the imputations based in config file.')
    group.add_argument('--create_missing',action='store_true', help='Create Missing data.')


    args = parser.parse_args()

    if args.impute:
        imputer, dataset_name = get_config("impute")

        if dataset_name == "mimiciv":
            dataset_cfg = MimicConfig()
            #columns = ["vent_etco2", "vent_fio2", "vital_spo2", "vital_hr","vent_rrtot", "blood_paco2", "blood_pao2"]
        elif dataset_name == "tud":
            dataset_cfg = TudConfig()
            #columns = [i for i in range(observation_size)]
        elif dataset_name == "hopper":
            dataset_cfg = HopperConfig()

        columns = dataset_cfg.state_vector
        print("Dataset: ", dataset_name.capitalize(), " Imputer: ", imputer)
        dataframe = load_obs_data(dataset_cfg)

        imputers = {
        "knn" : knn.impute_data,
        "mice" : mice.mice_impute,
        "miceforest" : mice.mice_forest,
        "nn" : None,
        "ssl" : None
        }
        
        if imputer in imputers.keys():
            if imputer == "nn":
                nni = NeuralNetworkImputer(dataset_cfg)

                train_run, test_run, train_test, model_path = get_yaml()

                missing_data = pd.read_csv(dataset_cfg.missing_data_path, usecols=dataset_cfg.state_vector)
                
                for col in dataset_cfg.missing_state_vector:
                    
                    if train_run:
                        nni.fit(col)
                    elif test_run:
                        col_imputations = nni.transform(col, saved_model_path=model_path)
                        missing_data = nni.fill_imputation(missing_data, col, col_imputations)
                    elif train_test:
                        col_imputations = nni.fit_transform(col)
                        missing_data = nni.fill_imputation(missing_data, col, col_imputations)
                
                if train_run:
                    pass
                else:
                    nni.save_imputed_data(dataset_cfg.imputed_data_path, missing_data)

            elif imputer == "ssl":
                ssl = SSLImputer(dataset_cfg)
                ssl.build_model()
                ssl.fit(dataframe)

                # TODO: Add Intermediate step to fill the missing value for prediction
                ssl.transform(dataframe)

                               
            else:
                imputers[imputer](dataframe, dataset_cfg)

    elif args.create_missing:

        dataset_name, _, _, _, _, missing_type, p_missing, steps = get_config("create_missing")
        print("Dataset: ", dataset_name.capitalize())

        if dataset_name == "mimiciv":
            dataset_cfg = MimicConfig()
        elif dataset_name == "tud":
            dataset_cfg = TudConfig()
        elif dataset_name == "hopper":
            dataset_cfg = HopperConfig()
            
        columns = dataset_cfg.state_vector

        dataframe = load_full_data(dataset_cfg.full_data_path, columns)

        create_missing_dataset(dataset_cfg.data_name, dataframe, dataset_cfg.missing_state_vector, steps=None, random=True, percent = p_missing)

    elif args.eval:
        dataset_name, imputer = get_config("eval")
        #og_col_1, og_col_2, imputed_col_1, imputed_col_2 = load_eval_columns(dataset_name, observation_size, imputed_data_path)
        
        if dataset_name == "MIMICIV":
            dataset_cfg = MimicConfig()
        elif dataset_name == "MINIC":
            dataset_cfg = TudConfig()
        elif dataset_name == "hopper":
            dataset_cfg = HopperConfig()

        eval_cols_dict = load_eval_columns(dataset_cfg, imputer)
        for i, col in enumerate(dataset_cfg.missing_state_vector, start=1):
            og_col = eval_cols_dict[f"og_{col}_{i}"]
            imputed_col = eval_cols_dict[f"imputed_{col}_{i}"]
            #eval.scores(og_col, imputed_col, dataset_cfg.imputed_data_path, imputer, col)
            
            mimic_map = {'stay_id':'stay_id', 'mv_id':'mv_id', 'timepoints':'timepoints', 'age':'age',
            'blood_be':'BE', 'blood_hco3':'HCO3', 'blood_ph':'pH', 'drugs_vaso4h':'Vasopressor', 'vital_map':'MAP', 'vital_mpap':'MPAP', 'vital_DBP':'DBP', 'ecmo_bloodflow':'ecmo_bloodflow', 'ecmo_rpm':'ecmo_rpm', 'ecmo_sweep':'ecmo_sweep', 'ecmo_active':'ecmo_active',
            'vital_SBP':'SBP', 'vital_SVRI': 'SVR', 'blood_INR': 'INR', 'blood_PTT': 'PTT', 'daemo_sex':'Sex',
            'daemo_weight':'Weight', 'daemo_height':'Height', 'daemo_discharge':'Discharge', 'blood_calcium':'Calcium', 'blood_chlorid':'Chloride', 'blood_caion':'Ionized Calcium', 'blood_magnes':'Magnesium', 'blood_potas':'Potasium', 'blood_sodium':'Sodium', 'vital_cvp':'CVP', 'cum_fluid_balance':'Fluid Balance', 'state_ivfluid4h':'IV Fluid', 'vent_etco2':'EtCO2', 'blood_paco2':'PaCO2', 'blood_pao2':'PaO2', 'vent_fio2':'FiO2', 'vital_spo2':'SpO2', 'blood_sao2':'SaO2', 'blood_svo2':'SvO2', 'blood_sco2':'SCO2', 'blood_smvo2':'SmvO2', 'blood_plat':'Platelets', 'blood_hb':'Hb', 'blood_hct':'Hct', 'blood_wbc':'WBC', 'vital_co':'CO', 'vital_hr':'HR', 'vital_rr':'RR', 'vital_temp':'Temp', 'vital_urine':'Urine', 'state_bun':'BUN', 'blood_crea':'Creatinine', 'state_urin4h':'Urine Output', 'blood_album':'Albumin', 'blood_ast':'AST', 'blood_alt':'ALT', 'blood_billi':'Bilirubin', 'blood_lac':'Lactate', 'blood_gluco':'Glucose', 'state_temp':'Temperature', 'vent_inspexp':'Insp/Exp', 'vent_pinsp':'Pinsp', 'vent_mairpress':'Mairpress', 'vent_mv':'MV', 'vent_peep':'PEEP', 'vent_rsbi':'RSBI', 'vent_rrtot':'RRTot', 'vent_rrcontrolled':'RRControlled', 'vent_rrspont':'RRSpont', 'vent_suppress':'press_sup', 'vent_vt':'VT', 'vent_vtnorm':'VTNorm', 'vent_mode':'Mode', 'state_airtype':'Airtype', 'blood_ffp':'FFP', 'blood_prbc':'PRBC', 'daemo_morta':'Mortality', 'episode_id':'episode_id'}

            tud_map = {
                    'caseid':'caseid', 'min duration':'min duration', 'max duration':'max duration', 'BM':'BM', 'FiO2':'FiO2', 'Hilfsdruck':'Hilfsdruck',
                    'IE':'IE', 'MV':'MV', 'PEEP':'PEEP', 'Pinsp':'Pinsp', 'RR':'RR', 'VT':'VT', 'etCO2':'EtCO2', 'Abnahme':'Abnahme', 'BEa':'BE',
                    'HaCO3':'HCO3', 'Hba':'Hb', 'Lactat':'Lactat', 'PaCO2':'PaCO2', 'PaO2':'PaO2', 'PaO2/FiO2':'FiO2', 'SaO2':'SaO2', 'pHa':'pH',
                    'ART':'ART', 'HR':'HR', 'SpO2':'SpO2', 'BG':'BG', 'cont_vent':'cont_vent', 'invasive':'invasive', 'purespont':'purespont', 'Temp':'Temp',
                    'Totraum':'Totraum', 'CVP':'CVP', 'Hbv':'Hbv', 'CO':'CO', 'BEv':'BEv', 'HvCO3':'HvCO3', 'PvCO2':'PvCO2', 'PvO2':'PvO2', 'SvO2':'SvO2',
                    'pHv':'pHv', 'SV':'SV', 'SVR':'SVR', 'PAP':'PAP', 'PWP':'PWP', 'PVR':'PVR', 'AMV_spont':'AMV_spont', 'Compliance':'Compliance',
                    'Resistance':'Resistance', 'Pmean':'Pmean', 'BEmv':'BEmv', 'Hbmv':'Hbmv', 'HmvCO3':'HmvCO3', 'PmvCO2':'PmvCO2', 'PmvO2':'PmvO2',
                    'SmvO2':'SmvO2', 'pHmv':'pHmv', 'BEc':'BEc', 'Hbc':'Hbc', 'HcCO3':'HcCO3', 'PcCO2':'PcCO2', 'PcO2':'PcO2', 'ScO2':'ScO2', 'pHc':'pHc',
                    'Pplat':'Pplat', 'BE':'BE', 'SO2':'SO2', 'pH':'pH', 'RR_spont':'RR_spont', 'Hb':'Hb', 'SVer':'SVer', 'VO2':'VO2', 'AZV_spont':'AZV_spont'
                    }  
            
            col = mimic_map[col] if dataset_name == "MIMICIV" else tud_map[col] 
            plot_line(imputed_col[:200], og_col[:200], dataset_name, imputer, col)