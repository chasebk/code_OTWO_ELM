from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from model.root.traditional.root_rnn import RootRnn

class Rnn1HL(RootRnn):
    def __init__(self, root_base_paras=None, root_rnn_paras=None):
        RootRnn.__init__(self, root_base_paras, root_rnn_paras)
        self.filename = "RNN-1HL-sliding_{}-net_para_{}".format(root_base_paras["sliding"], [self.hidden_sizes, self.epoch,
                            self.batch_size, self.learning_rate, self.activations, self.optimizer, self.loss])

    def _training__(self):
        #  The RNN architecture
        self.model = Sequential()
        self.model.add(LSTM(units=self.hidden_sizes[0], activation=self.activations[0], input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(self.dropouts[0]))
        self.model.add(Dense(units=1, activation=self.activations[1]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]


class Rnn2HL(RootRnn):
    def __init__(self, root_base_paras=None, root_rnn_paras=None):
        RootRnn.__init__(self, root_base_paras, root_rnn_paras)
        self.filename = "RNN-2HL-sliding_{}-net_para_{}".format(root_base_paras["sliding"], [self.hidden_sizes, self.epoch,
                            self.batch_size, self.learning_rate, self.activations, self.optimizer, self.loss])

    def _training__(self):
        #  The RNN architecture
        self.model = Sequential()
        self.model.add(LSTM(units=self.hidden_sizes[0], return_sequences=True, input_shape=(self.X_train.shape[1], 1), activation=self.activations[0]))
        self.model.add(Dropout(self.dropouts[0]))
        self.model.add(LSTM(units=self.hidden_sizes[1], activation=self.activations[1]))
        self.model.add(Dropout(self.dropouts[1]))
        self.model.add(Dense(units=1, activation=self.activations[2]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]




class Lstm1HL(RootRnn):
    def __init__(self, root_base_paras=None, root_rnn_paras=None):
        RootRnn.__init__(self, root_base_paras, root_rnn_paras)
        self.filename = "LSTM-1HL-sliding_{}-net_para_{}".format(root_base_paras["sliding"], [self.hidden_sizes, self.epoch,
                            self.batch_size, self.learning_rate, self.activations, self.optimizer, self.loss])
    def _training__(self):
        #  The LSTM architecture
        self.model = Sequential()
        self.model.add(LSTM(units=self.hidden_sizes[0], input_shape=(None, 1), activation=self.activations[0]))
        self.model.add(Dense(units=1, activation=self.activations[1]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]



class Lstm2HL(RootRnn):
    def __init__(self, root_base_paras=None, root_rnn_paras=None):
        RootRnn.__init__(self, root_base_paras, root_rnn_paras)
        self.filename = "LSTM-2HL-sliding_{}-net_para_{}".format(root_base_paras["sliding"], [self.hidden_sizes, self.epoch,
                            self.batch_size, self.learning_rate, self.activations, self.optimizer, self.loss])
    def _training__(self):
        #  The LSTM architecture
        self.model = Sequential()
        self.model.add(LSTM(units=self.hidden_sizes[0], return_sequences=True, input_shape=(None, 1), activation=self.activations[0]))
        self.model.add(LSTM(units=self.hidden_sizes[1], activation=self.activations[1]))
        self.model.add(Dense(units=1, activation=self.activations[2]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]





class Gru1HL(RootRnn):
    def __init__(self, root_base_paras=None, root_rnn_paras=None):
        RootRnn.__init__(self, root_base_paras, root_rnn_paras)
        self.filename = "GRU-1HL-sliding_{}-net_para_{}".format(root_base_paras["sliding"], [self.hidden_sizes, self.epoch,
                            self.batch_size, self.learning_rate, self.activations, self.optimizer, self.loss])
    def _training__(self):
        #  The GRU architecture
        self.model = Sequential()
        self.model.add(GRU(units=self.hidden_sizes[0], input_shape=(self.X_train.shape[1], 1), activation=self.activations[0]))
        self.model.add(Dropout(self.dropouts[0]))
        self.model.add(Dense(units=1, activation=self.activations[1]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]



class Gru2HL(RootRnn):
    def __init__(self, root_base_paras=None, root_rnn_paras=None):
        RootRnn.__init__(self, root_base_paras, root_rnn_paras)
        self.filename = "GRU-2HL-sliding_{}-net_para_{}".format(root_base_paras["sliding"], [self.hidden_sizes, self.epoch,
                            self.batch_size, self.learning_rate, self.activations, self.optimizer, self.loss])
    def _training__(self):
        #  The GRU architecture
        self.model = Sequential()
        self.model.add(GRU(units=self.hidden_sizes[0], return_sequences=True, input_shape=(self.X_train.shape[1], 1), activation=self.activations[0]))
        self.model.add(Dropout(self.dropouts[0]))
        self.model.add(GRU(units=self.hidden_sizes[1],  activation=self.activations[1]))
        self.model.add(Dropout(self.dropouts[1]))
        self.model.add(Dense(units=1, activation=self.activations[2]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]









