from model.root.root_base import RootBase
import time

class RootRnn(RootBase):
    def __init__(self, root_base_paras=None, root_rnn_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.epoch = root_rnn_paras["epoch"]
        self.batch_size = root_rnn_paras["batch_size"]
        self.learning_rate = root_rnn_paras["learning_rate"]
        self.activations = root_rnn_paras["activations"]
        self.optimizer = root_rnn_paras["optimizer"]
        self.loss = root_rnn_paras["loss"]
        self.dropouts = root_rnn_paras["dropouts"]
        if root_rnn_paras["hidden_sizes"][-1]:
            self.hidden_sizes = root_rnn_paras["hidden_size"][:-1]
        else:
            num_hid = len(root_rnn_paras["hidden_size"]) - 1
            self.hidden_sizes = [(num_hid - i) * root_base_paras["sliding"] * root_base_paras["feature_size"] + 1 for i in range(num_hid)]

    def _forecasting__(self):
        y_pred = self.model.predict(self.X_test)
        pred_inverse = self.scaler.inverse_transform(y_pred)
        real_inverse = self.scaler.inverse_transform(self.y_test)
        return real_inverse, pred_inverse, self.y_test, y_pred

    def _running__(self):
        self.time_system = time.time()
        self._preprocessing_3d__()
        self.time_total_train = time.time()
        self._training__()
        self.time_total_train = round(time.time() - self.time_total_train, 4)
        self.time_epoch = round(self.time_total_train / self.epoch, 4)
        self.time_predict = time.time()
        y_actual, y_predict, y_actual_normalized, y_predict_normalized = self._forecasting__()
        self.time_predict = round(time.time() - self.time_predict, 8)
        self.time_system = round(time.time() - self.time_system, 4)
        if self.test_type == "normal":
            self._save_results__(y_actual, y_predict, y_actual_normalized, y_predict_normalized, self.loss_train)
        elif self.test_type == "stability":
            self._save_results_ntimes_run__(y_actual, y_predict, y_actual_normalized, y_predict_normalized)




