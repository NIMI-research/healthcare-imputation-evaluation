import json
import yaml
#from loguru import logger
import sys

def get_config(mode):
    try:
        config_file = open("configs/config.json")
        config = json.load(config_file)
    except Exception as exc:
        #logger.exception(exc)
        print(exc)
        sys.exit(1)

    if mode == "create_missing":
        
        create_missing_config = config["create_missing"]
        dataset_name = create_missing_config["dataset"]
        path = create_missing_config["path"]
        obs_size = create_missing_config["observation_size"]
        act_size = create_missing_config["action_size"]
        missing_cols = create_missing_config["missing_columns"]
        missing_type = create_missing_config["missing_type"]
        p_missing = create_missing_config["percent_missing"]
        steps = create_missing_config["steps"]

        return dataset_name, path, obs_size, act_size, missing_cols, missing_type, p_missing, steps

    elif mode == "impute":
        imputation_config = config["imputation_config"]

        imputer = imputation_config["imputer"]
        dataset_name = imputation_config["dataset"]

        return imputer, dataset_name
    
    elif mode == "eval":
        eval_config = config["eval_config"]
        
        dataset_name = eval_config["dataset"]
        imputer = eval_config["imputer"]
        return dataset_name, imputer
    
def get_yaml():

    try:
        config_file = open("configs/nn_config.yml")
        nn_config = yaml.safe_load(config_file)
    except Exception as exc:
        #logger.exception(exc)
        print(exc)
        sys.exit(1)

    train, test, train_test = nn_config["TRAIN"], nn_config["TEST"], nn_config["TRAIN_TEST"]
    model_path = nn_config["MODEL_PATH"]

    return train, test, train_test, model_path