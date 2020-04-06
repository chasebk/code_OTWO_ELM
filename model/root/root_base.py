#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:10, 06/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from sklearn.preprocessing import MinMaxScaler
from utils.PreprocessingUtil import TimeSeries
from utils.MeasureUtil import MeasureTimeSeries
from utils.IOUtil import save_all_models_to_csv, save_prediction_to_csv, save_loss_train_to_csv
from utils.GraphUtil import draw_predict_with_error


class RootBase:
    """
        This is root of all networks.
    """
    def __init__(self, root_base_paras=None):
        self.dataset = root_base_paras["dataset"]
        self.data_idx = root_base_paras["data_idx"]
        self.sliding = root_base_paras["sliding"]
        self.output_idx = root_base_paras["output_idx"]
        self.method_statistic = root_base_paras["method_statistic"]
        self.scaler = MinMaxScaler()

        self.n_runs = root_base_paras["n_runs"]
        self.path_save_result = root_base_paras["path_save_result"]
        self.log_filename = root_base_paras["log_filename"]
        self.multi_output = root_base_paras["multi_output"]
        self.draw = root_base_paras["draw"]
        self.print_train = root_base_paras["print_train"]

        self.model, self.solution, self.loss_train, self.filename = None, None, [], None
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = None, None, None, None, None, None
        self.time_total_train, self.time_epoch, self.time_predict, self.time_system = None, None, None, None

    def _preprocessing_2d__(self):
        ts = TimeSeries(self.dataset, self.data_idx, self.sliding, self.output_idx, self.method_statistic, self.scaler)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.scaler = ts._preprocessing_2d__()

    def _preprocessing_3d__(self):
        ts = TimeSeries(self.dataset, self.data_idx, self.sliding, self.output_idx, self.method_statistic, self.scaler)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.scaler = ts._preprocessing_3d__()

    def _save_results__(self, y_true=None, y_pred=None, y_true_scaled=None, y_pred_scaled=None, loss_train=None, n_runs=1):
        if self.multi_output:
            measure_scaled = MeasureTimeSeries(y_true_scaled, y_pred_scaled, "raw_values", number_rounding=4)
            measure_scaled._fit__()
            data1 = "CPU_"
            data2 = "RAM_"
            item = {'model_name': self.filename, 'total_time_train': self.time_total_train, 'time_epoch': self.time_epoch,
                    'time_predict': self.time_predict, 'time_system': self.time_system,
                    data1 + 'scaled_EV': measure_scaled.score_ev[0], data1 + 'scaled_MSLE': measure_scaled.score_msle[0],
                    data1 + 'scaled_R2': measure_scaled.score_r2[0], data1 + 'scaled_MAE': measure_scaled.score_mae[0],
                    data1 + 'scaled_MSE': measure_scaled.score_mse[0], data1 + 'scaled_RMSE': measure_scaled.score_rmse[0],
                    data1 + 'scaled_MAPE': measure_scaled.score_mape[0], data1 + 'scaled_SMAPE': measure_scaled.score_smape[0],

                    data2 + 'scaled_EV': measure_scaled.score_ev[1], data2 + 'scaled_MSLE': measure_scaled.score_msle[1],
                    data2 + 'scaled_R2': measure_scaled.score_r2[1], data2 + 'scaled_MAE': measure_scaled.score_mae[1],
                    data2 + 'scaled_MSE': measure_scaled.score_mse[1], data2 + 'scaled_RMSE': measure_scaled.score_rmse[1],
                    data2 + 'scaled_MAPE': measure_scaled.score_mape[1], data2 + 'scaled_SMAPE': measure_scaled.score_smape[1]}

            if n_runs == 1:
                save_prediction_to_csv(y_true[:, 0:1], y_pred[:, 0:1], self.filename, self.path_save_result + data1)
                save_prediction_to_csv(y_true[:, 1:2], y_pred[:, 1:2], self.filename, self.path_save_result + data2)
                save_loss_train_to_csv(loss_train, self.filename, self.path_save_result + "Error-")

                if self.draw:
                    draw_predict_with_error(1, [y_true[:, 0:1], y_pred[:, 0:1]], [measure_scaled.score_rmse[0], measure_scaled.score_mae[0]], self.filename,
                                            self.path_save_result + data1)
                    draw_predict_with_error(2, [y_true[:, 1:2], y_pred[:, 1:2]], [measure_scaled.score_rmse[1], measure_scaled.score_mae[1]], self.filename,
                                            self.path_save_result + data2)
                if self.print_train:
                    print('Predict DONE - CPU - RMSE: %f, RAM - RMSE: %f' % (measure_scaled.score_rmse[0], measure_scaled.score_rmse[1]))
            save_all_models_to_csv(item, self.log_filename, self.path_save_result)

        else:
            measure_scaled = MeasureTimeSeries(y_true_scaled, y_pred_scaled, None, number_rounding=4)
            measure_scaled._fit__()
            measure_unscaled = MeasureTimeSeries(y_true, y_pred, None, number_rounding=4)
            measure_unscaled._fit__()

            item = {'model_name': self.filename, 'total_time_train': self.time_total_train, 'time_epoch': self.time_epoch,
                    'time_predict': self.time_predict, 'time_system': self.time_system,
                    'scaled_EV': measure_scaled.score_ev, 'scaled_MSLE': measure_scaled.score_msle, 'scaled_R2': measure_scaled.score_r2,
                    'scaled_MAE': measure_scaled.score_mae, 'scaled_MSE': measure_scaled.score_mse, 'scaled_RMSE': measure_scaled.score_rmse,
                    'scaled_MAPE': measure_scaled.score_mape, 'scaled_SMAPE': measure_scaled.score_smape,
                    'unscaled_EV': measure_unscaled.score_ev, 'unscaled_MSLE': measure_unscaled.score_msle, 'unscaled_R2': measure_unscaled.score_r2,
                    'unscaled_MAE': measure_unscaled.score_mae, 'unscaled_MSE': measure_unscaled.score_mse, 'unscaled_RMSE': measure_unscaled.score_rmse,
                    'unscaled_MAPE': measure_unscaled.score_mape, 'unscaled_SMAPE': measure_unscaled.score_smape}

            if n_runs == 1:
                save_prediction_to_csv(y_true, y_pred, self.filename, self.path_save_result)
                save_loss_train_to_csv(loss_train, self.filename, self.path_save_result + "Error-")
                if self.draw:
                    draw_predict_with_error([y_true, y_pred], [measure_unscaled.score_rmse, measure_unscaled.score_mae], self.filename, self.path_save_result)
                if self.print_train:
                    print('Predict DONE - RMSE: %f, MAE: %f' % (measure_unscaled.score_rmse, measure_unscaled.score_mae))

            save_all_models_to_csv(item, self.log_filename, self.path_save_result)

    def _forecasting__(self):
        pass

    def _training__(self):
        pass

    def _running__(self):
        pass
