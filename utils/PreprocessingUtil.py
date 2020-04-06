import numpy as np
from copy import deepcopy


class TimeSeries(object):
    def __init__(self, dataset=None, data_idx=None, sliding=None, output_index=None,
                 method_statistic=0, minmax_scaler=None):
        '''
        :param data_idx:
        :param sliding:
        :param output_index:
        :param method_statistic:
        :param minmax_scaler:
        '''
        self.original_dataset = dataset
        self.dimension = dataset.shape[1]       # The real number of features
        self.original_dataset_len = len(dataset)
        self.dataset_len = self.original_dataset_len - sliding

        self.train_idx = int(data_idx[0] * self.dataset_len)
        self.train_len = self.train_idx
        self.valid_idx = self.train_idx + int(data_idx[1] * self.dataset_len)
        self.valid_len = self.valid_idx - self.train_idx
        self.test_idx = self.dataset_len
        self.test_len = self.dataset_len - self.train_len - self.valid_len
        self.sliding = sliding
        self.output_index = output_index
        self.method_statistic = method_statistic
        self.minmax_scaler = minmax_scaler

    def __get_dataset_X__(self, list_transform=None):
        """
        :param list_transform: [ x1 | t1 ] => Make a window slides
        :return: dataset_sliding = [ x1 | x2 | x3| t1 | t2 | t3 | ... ]
        """
        dataset_sliding = np.zeros(shape=(self.test_idx, 1))        #[ 0 | x1 | x2 | x3| t1 | t2 | t3 | ... ]
        for i in range(self.dimension):
            for j in range(self.sliding):
                temp = np.array(list_transform[j: self.test_idx + j, i:i + 1])
                dataset_sliding = np.concatenate((dataset_sliding, temp), axis=1)
        dataset_sliding = dataset_sliding[:, 1:]                    #[ x1 | x2 | x3| t1 | t2 | t3 | ... ]

        ## Find the dataset_X by using different statistic method on above window slides
        if self.method_statistic == 0:                      # default
            dataset_X = deepcopy(dataset_sliding)
        else:
            dataset_X = np.zeros(shape=(self.test_idx, 1))
            if self.method_statistic == 1:
                """
                mean(x1, x2, x3, ...), mean(t1, t2, t3,...) 
                """
                for i in range(self.dimension):
                    meanx = np.reshape(np.mean(dataset_sliding[:, i * self.sliding:(i + 1) * self.sliding], axis=1), (-1, 1))
                    dataset_X = np.concatenate((dataset_X, meanx), axis=1)

            if self.method_statistic == 2:
                """
                min(x1, x2, x3, ...), mean(x1, x2, x3, ...), max(x1, x2, x3, ....)
                """
                for i in range(self.dimension):
                    minx = np.reshape(np.amin(dataset_sliding[:, i * self.sliding:(i + 1) * self.sliding], axis=1), (-1, 1))
                    meanx = np.reshape(np.mean(dataset_sliding[:, i * self.sliding:(i + 1) * self.sliding], axis=1), (-1, 1))
                    maxx = np.reshape(np.amax(dataset_sliding[:, i * self.sliding:(i + 1) * self.sliding], axis=1), (-1, 1))
                    dataset_X = np.concatenate((dataset_X, minx, meanx, maxx), axis=1)

            if self.method_statistic == 3:
                """
                min(x1, x2, x3, ...), median(x1, x2, x3, ...), max(x1, x2, x3, ....), min(t1, t2, t3, ...), median(t1, t2, t3, ...), max(t1, t2, t3, ....)
                """
                for i in range(self.dimension):
                    minx = np.reshape(np.amin(dataset_sliding[:, i * self.sliding:(i + 1) * self.sliding], axis=1), (-1, 1))
                    medix = np.reshape(np.median(dataset_sliding[:, i * self.sliding:(i + 1) * self.sliding], axis=1), (-1, 1))
                    maxx = np.reshape(np.amax(dataset_sliding[:, i * self.sliding:(i + 1) * self.sliding], axis=1), (-1, 1))
                    dataset_X = np.concatenate((dataset_X, minx, medix, maxx), axis=1)
            dataset_X = dataset_X[:, 1:]
        return dataset_X

    def _preprocessing_2d__(self):
        """
            output_index = None
                + single input => single output
                + multiple input => multiple output

            output_index = number (index)
                + single input => single output index
                + multiple input => single output index

            valid_idx = 0 ==> No validate data ||  cpu(t), cpu(t-1), ..., ram(t), ram(t-1),...
        """

        if self.output_index is None:
            list_transform = self.minmax_scaler.fit_transform(self.original_dataset)
            #    print(preprocessing.MinMaxScaler().data_max_)
            dataset_y = deepcopy(list_transform[self.sliding:])             # Now we need to find dataset_X
        else:
            # Example : data [0, 1, 2, 3]
            # output_index = 2          ==>  Loop scale through 3, 0, 1, 2
            # [ cpu, ram, disk_io, disk_space ]
            # list_transform:   [ 0, disk_space, cpu, ram, disk_io ]
            # Cut list_transform:   [ disk_space, cpu, ram, disk_io ]
            # Dataset y = list_transform[-1]

            list_transform = np.zeros(shape=(self.original_dataset_len, 1))
            for i in range(0, self.dimension):
                t = self.output_index - (self.dimension - 1) + i
                d1 = self.minmax_scaler.fit_transform(
                    self.original_dataset[:self.original_dataset_len, t].reshape(-1, 1))
                list_transform = np.concatenate((list_transform, d1), axis=1)
                # print(minmax_scaler.data_max_)
            list_transform = list_transform[:, 1:]
            dataset_y = deepcopy(list_transform[self.sliding:, -1:])         # Now we need to find dataset_X

        dataset_X = self.__get_dataset_X__(list_transform)

        ## Split data to set train and set test
        if self.valid_len == 0:
            X_train, y_train = dataset_X[0:self.train_idx], dataset_y[0:self.train_idx]
            X_test, y_test = dataset_X[self.train_idx:self.test_idx], dataset_y[self.train_idx:self.test_idx]
            # print("Processing data done!!!")
            return X_train, y_train, None, None, X_test, y_test, self.minmax_scaler
        else:
            X_train, y_train = dataset_X[0:self.train_idx], dataset_y[0:self.train_idx]
            X_valid, y_valid = dataset_X[self.train_idx:self.valid_idx], dataset_y[self.train_idx:self.valid_idx]
            X_test, y_test = dataset_X[self.valid_idx:self.test_idx], dataset_y[self.valid_idx:self.test_idx]
            # print("Processing data done!!!")
            return X_train, y_train, X_valid, y_valid, X_test, y_test, self.minmax_scaler

    def _preprocessing_3d__(self):
        if self.output_index is None:
            list_transform = self.minmax_scaler.fit_transform(self.original_dataset)
            dataset_y = deepcopy(list_transform[self.sliding:])             # Now we need to find dataset_X
        else:
            list_transform = np.zeros(shape=(self.original_dataset_len, 1))
            for i in range(0, self.dimension):
                t = self.output_index - (self.dimension - 1) + i
                d1 = self.minmax_scaler.fit_transform(
                    self.original_dataset[:self.original_dataset_len, t].reshape(-1, 1))
                list_transform = np.concatenate((list_transform, d1), axis=1)
            list_transform = list_transform[:, 1:]
            dataset_y = deepcopy(list_transform[self.sliding:, -1:])         # Now we need to find dataset_X
        dataset_X = self.__get_dataset_X__(list_transform)
        ## Split data to set train and set test
        if self.valid_len == 0:
            X_train, y_train = dataset_X[0:self.train_idx], dataset_y[0:self.train_idx]
            X_valid, y_valid = None, None
            X_test, y_test = dataset_X[self.train_idx:self.test_idx], dataset_y[self.train_idx:self.test_idx]

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # y_train = y_train.flatten()
            # y_test = y_test.flatten()
        else:
            X_train, y_train = dataset_X[0:self.train_idx], dataset_y[0:self.train_idx]
            X_valid, y_valid = dataset_X[self.train_idx:self.valid_idx], dataset_y[self.train_idx:self.valid_idx]
            X_test, y_test = dataset_X[self.valid_idx:self.test_idx], dataset_y[self.valid_idx:self.test_idx]

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # y_train = y_train.flatten()
            # y_valid = y_valid.flatten()
            # y_test = y_test.flatten()
        return X_train, y_train, X_valid, y_valid, X_test, y_test, self.minmax_scaler
