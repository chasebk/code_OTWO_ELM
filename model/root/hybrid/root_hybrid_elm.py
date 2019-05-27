import numpy as np
import time
from model.root.root_base import RootBase
from utils.MathUtil import elu, relu, tanh, sigmoid
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RootHybridElm(RootBase):
    """
        This is root of all hybrid models which include Extreme Learning Machine and Optimization Algorithms.
    """
    def __init__(self, root_base_paras=None, root_hybrid_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.epoch = root_hybrid_paras["epoch"]
        self.activation = root_hybrid_paras["activation"]
        self.train_valid_rate = root_hybrid_paras["train_valid_rate"]
        self.domain_range = root_hybrid_paras["domain_range"]
        if root_hybrid_paras["hidden_size"][1]:
            self.hidden_size = root_hybrid_paras["hidden_size"][0]
        else:
            self.hidden_size = 2*(root_base_paras["sliding"]*root_base_paras["feature_size"])**2 + 1

        if self.activation == 0:
            self._activation__ = elu
        elif self.activation == 1:
            self._activation__ = relu
        elif self.activation == 2:
            self._activation__ = tanh
        else:
            self._activation__ = sigmoid

    def _setting__(self):
        self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]
        self.w1_size = self.input_size * self.hidden_size
        self.b_size = self.hidden_size
        self.w2_size = self.hidden_size * self.output_size
        self.problem_size = self.w1_size + self.b_size
        self.root_algo_paras = {
            "X_train": self.X_train, "y_train": self.y_train, "X_valid": self.X_valid, "y_valid": self.y_valid,
            "problem_size": self.problem_size, "train_valid_rate": self.train_valid_rate,
            "domain_range": self.domain_range, "print_train": self.print_train,
            "_get_average_error__": self._get_average_error__
        }

    ## Helper functions
    def _get_model__(self, individual=None):
        X_train = np.concatenate( (self.X_train, self.X_valid), axis=0 )
        y_train = np.concatenate( (self.y_train, self.y_valid), axis=0 )
        w1 = np.reshape(individual[:self.w1_size], (self.input_size, self.hidden_size))
        b = np.reshape(individual[self.w1_size:self.w1_size + self.b_size], (-1, self.hidden_size))
        H = self._activation__(np.add(np.matmul(X_train, w1), b))
        w2 = np.dot(np.linalg.pinv(H), y_train)  # calculate weights between hidden and output layer
        self.model = {"w1": w1, "b": b, "w2": w2}

    def _get_average_error__(self, individual=None, X_data=None, y_data=None):
        w1 = np.reshape(individual[:self.w1_size], (self.input_size, self.hidden_size))
        b = np.reshape(individual[self.w1_size:self.w1_size + self.b_size], (-1, self.hidden_size))

        H = self._activation__(np.add(np.matmul(X_data, w1), b))
        H_pinv = np.linalg.pinv(H)              # compute a pseudoinverse of H
        w2 = np.dot(H_pinv, y_data)             # calculate weights between hidden and output layer

        y_pred = np.matmul(H, w2)
        return [mean_squared_error(y_pred, y_data), mean_absolute_error(y_pred, y_data)]

    def _forecasting__(self):
        hidd = self._activation__(np.add(np.matmul(self.X_test, self.model["w1"]), self.model["b"]))
        y_pred = np.matmul(hidd, self.model["w2"])
        real_inverse = self.scaler.inverse_transform(self.y_test)
        pred_inverse = self.scaler.inverse_transform(np.reshape(y_pred, self.y_test.shape))
        return real_inverse, pred_inverse, self.y_test, y_pred

    def _running__(self):
        self.time_system = time.time()
        self._preprocessing_2d__()
        self._setting__()
        self.time_total_train = time.time()
        self._training__()
        self._get_model__(self.solution)
        self.time_total_train = round(time.time() - self.time_total_train, 4)
        self.time_epoch = round(self.time_total_train / self.epoch, 4)
        self.time_predict = time.time()
        y_actual, y_predict, y_actual_normalized, y_predict_normalized = self._forecasting__()
        self.time_predict = round(time.time() - self.time_predict, 6)
        self.time_system = round(time.time() - self.time_system, 4)
        if self.test_type == "normal":
            self._save_results__(y_actual, y_predict, y_actual_normalized, y_predict_normalized, self.loss_train)
        elif self.test_type == "stability":
            self._save_results_ntimes_run__(y_actual, y_predict, y_actual_normalized, y_predict_normalized)


