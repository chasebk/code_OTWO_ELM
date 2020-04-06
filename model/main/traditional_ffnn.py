#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:51, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from keras.models import Sequential
from keras.layers import Dense
from model.root.traditional.root_mlnn import RootMlnn


class Mlnn1HL(RootMlnn):
    def __init__(self, root_base_paras=None, root_mlnn_paras=None):
        RootMlnn.__init__(self, root_base_paras, root_mlnn_paras)
        self.filename = "MLNN-1H-sliding_{}-{}".format(root_base_paras["sliding"], root_mlnn_paras["paras_name"])

    def _training__(self):
        self.model = Sequential()
        self.model.add(Dense(units=self.hidden_sizes[0], input_dim=self.X_train.shape[1], activation=self.activations[0]))
        self.model.add(Dense(1, activation=self.activations[1]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]


class Mlnn2HL(RootMlnn):
    def __init__(self, root_base_paras=None, root_mlnn_paras=None):
        RootMlnn.__init__(self, root_base_paras, root_mlnn_paras)
        self.filename = "MLNN-2H-sliding_{}-{}".format(root_base_paras["sliding"], root_mlnn_paras["paras_name"])

    def _training__(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_sizes[0], input_dim=self.X_train.shape[1], activation=self.activations[0]))
        self.model.add(Dense(self.hidden_sizes[1], activation=self.activations[1]))
        self.model.add(Dense(1, activation=self.activations[2]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]

