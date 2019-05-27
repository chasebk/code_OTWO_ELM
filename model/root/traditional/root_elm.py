from model.root.root_base import RootBase
from utils.MathUtil import elu, relu, tanh, sigmoid
import time

class RootElm(RootBase):
    def __init__(self, root_base_paras=None, root_elm_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.activation = root_elm_paras["activation"]
        if root_elm_paras["hidden_size"][1]:
            self.hidden_size = root_elm_paras["hidden_size"][0]
        else:
            self.hidden_size = 2*root_base_paras["sliding"]*root_base_paras["feature_size"] + 1

        if self.activation == 0:
            self._activation__ = elu
        elif self.activation == 1:
            self._activation__ = relu
        elif self.activation == 2:
            self._activation__ = tanh
        else:
            self._activation__ = sigmoid

    def _running__(self):
        self.time_system = time.time()
        self._preprocessing_2d__()
        self.time_total_train = time.time()
        self._training__()
        self.time_total_train = round(time.time() - self.time_total_train, 4)
        self.time_epoch = None
        self.time_predict = time.time()
        y_actual, y_predict, y_actual_normalized, y_predict_normalized = self._forecasting__()
        self.time_predict = round(time.time() - self.time_predict, 8)
        self.time_system = round(time.time() - self.time_system, 4)
        if self.test_type == "normal":
            self._save_results__(y_actual, y_predict, y_actual_normalized, y_predict_normalized, self.loss_train)
        elif self.test_type == "stability":
            self._save_results_ntimes_run__(y_actual, y_predict, y_actual_normalized, y_predict_normalized)




