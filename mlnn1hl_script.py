#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 00:51, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from os.path import splitext, basename, realpath
from sklearn.model_selection import ParameterGrid
from model.main.traditional_ffnn import Mlnn1HL
from utils.SettingPaper import *
from utils.IOUtil import load_dataset
from utils.SettingPaper import mlnn1hl_paras_final as param_grid

if SP_RUN_TIMES == 1:
    all_model_file_name = SP_LOG_FILENAME
else:  # If runs with more than 1, like stability test --> name of the models ==> such as: rnn1hl.csv
    all_model_file_name = str(splitext(basename(realpath(__file__)))[0])


def train_model(item):
    root_base_paras = {
        "dataset": dataset,
        "feature_size": feature_size,
        "data_idx": SP_DATA_SPLIT_INDEX,
        "sliding": item["sliding"],
        "multi_output": multi_output,
        "output_idx": output_index,
        "method_statistic": SP_PREPROCESSING_METHOD,
        "log_filename": all_model_file_name,
        "n_runs": SP_RUN_TIMES,                         # 1 or others
        "path_save_result": SP_PATH_SAVE_BASE + SP_DATA_FILENAME[loop] + "/",
        "draw": SP_DRAW,
        "print_train": SP_PRINT_TRAIN,  # 0: nothing, 1 : full detail, 2: short version
    }
    paras_name = "hs_{}-ep_{}-bs_{}-lr_{}-ac_{}-op_{}-lo_{}".format(item["hidden_sizes"], item["epoch"], item["batch_size"], item["learning_rate"],
                                                                    item["activations"], item["optimizer"], item["loss"])

    root_mlnn_paras = {
        "hidden_sizes": item["hidden_sizes"], "epoch": item["epoch"], "batch_size": item["batch_size"], "learning_rate": item["learning_rate"],
        "activations": item["activations"], "optimizer": item["optimizer"], "loss": item["loss"], "paras_name": paras_name
    }
    md = Mlnn1HL(root_base_paras=root_base_paras, root_mlnn_paras=root_mlnn_paras)
    md._running__()


for _ in range(SP_RUN_TIMES):
    for loop in range(len(SP_DATA_FILENAME)):
        filename = SP_LOAD_DATA_FROM + SP_DATA_FILENAME[loop]
        dataset = load_dataset(filename, cols=SP_DATA_COLS[loop])
        feature_size = len(SP_DATA_COLS[loop])
        multi_output = SP_DATA_MULTI_OUTPUT[loop]
        output_index = SP_OUTPUT_INDEX[loop]
        # Create combination of params.
        for item in list(ParameterGrid(param_grid)):
            train_model(item)

