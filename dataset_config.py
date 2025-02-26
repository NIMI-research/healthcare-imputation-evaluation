class HopperConfig:
    def __init__(self):
        self.data_name = "Hopper-v4"
        self.state_vector = ["obs_1", "obs_2", "obs_3", "obs_4", "obs_5", 
                             "obs_6", "obs_7", "obs_8", "obs_9", "obs_10", 
                             "obs_11"]
        self.missing_state_vector = ["obs_6", "obs_7"]
        self.action_vector = []
        self.full_data_path = "datasets/Hopper-v4.csv"
        self.missing_data_path = "datasets/Hopper-v4_w_missing_values_random.csv"
        self.imputed_data_path = "Imputation_results/hopper_imputed"

    
class MimicConfig:
    def __init__(self):
        self.data_name = "mimiciv"

        self.full_data_path = "datasets\\mimiciv_full_data.csv"
        self.state_vector = ["vent_etco2", "blood_paco2", "blood_pao2", "vent_fio2", 
                             "vital_spo2", "vital_hr", "vent_rrtot"]
        self.missing_state_vector = ["blood_paco2", "blood_pao2"]
        self.missing_data_path = "datasets/mimiciv_w_missing_values_random.csv"

        
        '''self.full_data_path = "datasets\\mimiciv_state_vectors_v3.csv"
        self.state_vector = ['blood_be','blood_hco3', 'blood_ph', 'vital_map','vital_DBP','vital_SBP','blood_paco2', 'vent_pinsp', 'vent_mairpress','vent_peep','vent_suppress']
        self.missing_state_vector = ["blood_be", "vital_map", "vent_pinsp"]
        self.action_vector = []
        self.missing_data_path = "datasets/mimiciv_3_w_missing_values_random.csv"'''
        
        self.imputed_data_path = f"Imputation_results/{self.data_name}_imputed"

    
class MinicConfig:
    def __init__(self):
        self.data_name = "minic_3"

        self.state_vector = ["BEa","HaCO3","Lactat","PaCO2", "SaO2", "pHa", "SpO2"]
        self.missing_state_vector = ["BEa","pHa"]
        self.action_vector = []
        self.full_data_path = "datasets\\minic_state_vector_v3.csv"
        self.missing_data_path = "datasets/minic_3_w_missing_values_random.csv"
        self.imputed_data_path = "Imputation_results/minic_imputed"


        '''self.state_vector = ["BEa","HaCO3","Lactat","PaCO2","PaO2", "SaO2", "pHa", "HR", "SpO2"]
        self.missing_state_vector = ["PaCO2","PaO2"]
        self.action_vector = []
        self.full_data_path = "datasets\\minic_state_vector_v4.csv"
        self.missing_data_path = "datasets/minic_4_w_missing_values_random.csv"
        self.imputed_data_path = "Imputation_results/minic_4_imputed_8"'''
