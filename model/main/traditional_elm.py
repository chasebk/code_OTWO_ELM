from model.root.traditional.root_elm import RootElm
import numpy as np

class Elm(RootElm):
    """
        Amazing tutorial: https://www.kaggle.com/robertbm/extreme-learning-machine-example
    """
    def __init__(self, root_base_paras=None, root_elm_paras=None):
        RootElm.__init__(self, root_base_paras, root_elm_paras)
        self.filename = "ELM-sliding_{}-net_para_{}".format(root_base_paras["sliding"], root_elm_paras)

    def _training__(self):
        """
        1. Random weights between input and hidden layer
        2. Calculate output of hidden layer
        3. Calculate weights between hidden and output layer based on matrix multiplication
        """
        self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]
        w1 = np.random.uniform(size=[self.input_size, self.hidden_size])
        b = np.random.uniform(size=[1, self.hidden_size])
        H = self._activation__(np.add(np.matmul(self.X_train, w1), b))
        w2 = np.dot(np.linalg.pinv(H), self.y_train)
        self.model = {"w1": w1, "b": b, "w2": w2}

    def _forecasting__(self):
        hidd = self._activation__(np.add(np.matmul(self.X_test, self.model["w1"]), self.model["b"]))
        y_pred = np.matmul(hidd, self.model["w2"])
        real_inverse = self.scaler.inverse_transform(self.y_test)
        pred_inverse = self.scaler.inverse_transform(np.reshape(y_pred, self.y_test.shape))
        return real_inverse, pred_inverse, self.y_test, y_pred
