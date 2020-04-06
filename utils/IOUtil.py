#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 00:51, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

import numpy as np
import pandas as pd
from csv import DictWriter
from os import getcwd, path, makedirs


def save_all_models_to_csv(item=None, log_filename=None, pathsave=None):
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)
    with open(pathsave + log_filename + ".csv", 'a') as file:
        w = DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=item.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(item)


def save_prediction_to_csv(y_test=None, y_pred=None, filename=None, pathsave=None):
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)

    temp = np.concatenate((y_test, y_pred), axis=1)
    np.savetxt(pathsave + filename + ".csv", temp, delimiter=",")
    return None


def save_loss_train_to_csv(error=None, filename=None, pathsave=None):
    np.savetxt(pathsave + filename + ".csv", np.array(error), delimiter=",")
    return None


def load_dataset(path_to_data=None, cols=None):
    df = pd.read_csv(path_to_data + ".csv", usecols=cols)
    return df.values





def save_run_test(num_run_test=None, data=None, filepath=None):
    t0 = np.reshape(data, (num_run_test, -1))
    np.savetxt(filepath, t0, delimiter=",")

def load_prediction_results(pathfile=None, delimiter=",", header=None):
    df = pd.read_csv(pathfile, sep=delimiter, header=header)
    return df.values[:, 0:1], df.values[:, 1:2]

def save_number_of_vms(data=None, pathfile=None):
    t0 = np.reshape(data, (-1, 1))
    np.savetxt(pathfile, t0, delimiter=",")

def load_number_of_vms(pathfile=None, delimiter=",", header=None):
    df = pd.read_csv(pathfile, sep=delimiter, header=header)
    return df.values[:, 0:1]



def save_scaling_results_to_csv(data=None, path_file=None):
    np.savetxt(path_file + ".csv", np.array(data), delimiter=",")
    return None

def read_dataset_file(filepath=None, usecols=None, header=0, index_col=False, inplace=True):
    df = pd.read_csv(filepath, usecols=usecols, header=header, index_col=index_col)
    df.dropna(inplace=inplace)
    return df.values

def save_formatted_data_csv(dataset=None, filename=None, pathsave=None):
    np.savetxt(pathsave + filename + ".csv", dataset, delimiter=",")
    return None