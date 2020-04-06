#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:21, 06/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

import time
from model.root.root_base import RootBase
import utils.MathUtil as my_math


class RootElm(RootBase):
    def __init__(self, root_base_paras=None, root_elm_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.activation = root_elm_paras["activation"]
        if root_elm_paras["hidden_size"][1]:
            self.hidden_size = root_elm_paras["hidden_size"][0]
        else:
            self.hidden_size = 2*root_base_paras["sliding"]*root_base_paras["feature_size"] + 1
        ## New discovery
        self._activation__ = getattr(my_math, self.activation)

    def _running__(self):
        self.time_system = time.time()
        self._preprocessing_2d__()
        self.time_total_train = time.time()
        self._training__()
        self.time_total_train = round(time.time() - self.time_total_train, 4)
        self.time_epoch = None
        self.time_predict = time.time()
        y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled = self._forecasting__()
        self.time_predict = round(time.time() - self.time_predict, 8)
        self.time_system = round(time.time() - self.time_system, 4)
        self._save_results__(y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled, self.loss_train, self.n_runs)
