from sklearn.model_selection import ParameterGrid
from model.main.hybrid_elm import ABfoLSElm
from utils.IOUtil import read_dataset_file
from utils.SettingPaper import abfols_elm_paras_final as param_grid
from utils.SettingPaper import traffic_eu, traffic_uk, worldcup

rv_data = [traffic_eu, traffic_uk, worldcup]
data_file = ["it_eu_5m", "it_uk_5m", "worldcup98_5m"]
test_type = "normal"                ### normal: for normal test, stability: for n_times test
run_times = None

if test_type == "normal":           ### For normal test
    run_times = 1
    pathsave = "paper/results/final/"
    all_model_file_name = "elm_log_models"
elif test_type == "stability":      ### For stability test (n times run with the same parameters)
    run_times = 15
    pathsave = "paper/results/stability/"
    all_model_file_name = "stability_abfols_elm"
else:
    pass

def train_model(item):
    root_base_paras = {
        "dataset": dataset,
        "feature_size": feature_size,
        "data_idx": (0.7, 0.15, 0.15),
        "sliding": item["sliding"],
        "multi_output": requirement_variables[2],
        "output_idx": requirement_variables[3],
        "method_statistic": 0,  # 0: sliding window, 1: mean, 2: min-mean-max, 3: min-median-max
        "log_filename": all_model_file_name,
        "path_save_result": pathsave + requirement_variables[4],
        "test_type": test_type,
        "draw": True,
        "print_train": 0  # 0: nothing, 1 : full detail, 2: short version
    }
    root_hybrid_paras = {
        "hidden_size": item["hidden_size"], "activation": item["activation"], "epoch": item["epoch"],
        "train_valid_rate": item["train_valid_rate"], "domain_range": item["domain_range"]
    }
    abfols_paras = {
        "epoch": item["epoch"], "pop_size": item["pop_size"], "Ci": item["Ci"], "Ped": item["Ped"], "Ns": item["Ns"],
        "N_minmax": item["N_minmax"]
    }
    md = ABfoLSElm(root_base_paras=root_base_paras, root_hybrid_paras=root_hybrid_paras, abfols_paras=abfols_paras)
    md._running__()

for _ in range(run_times):
    for loop in range(len(rv_data)):
        requirement_variables = rv_data[loop]
        filename = requirement_variables[0] + data_file[loop] + ".csv"
        dataset = read_dataset_file(filename, requirement_variables[1])
        feature_size = len(requirement_variables[1])
        # Create combination of params.
        for item in list(ParameterGrid(param_grid)):
            train_model(item)






