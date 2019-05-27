from model.optimizer.evolutionary import GA, DE
from model.optimizer.swarm import PSO, BFO
from model.optimizer.physics import QSO, TWO
from model.root.hybrid.root_hybrid_elm import RootHybridElm

class GaElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.ga_paras = ga_paras
        self.filename = "GA_ELM-sliding_{}-nets_{}-ga_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], ga_paras)

    def _training__(self):
        ga = GA.BaseGA(root_algo_paras=self.root_algo_paras, ga_paras = self.ga_paras)
        self.solution, self.loss_train = ga._train__()


class DeElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.de_paras = de_paras
        self.filename = "DE_ELM-sliding_{}-nets_{}-de_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], de_paras)

    def _training__(self):
        md = DE.BaseDE(root_algo_paras=self.root_algo_paras, de_paras = self.de_paras)
        self.solution, self.loss_train = md._train__()


class PsoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.pso_paras = pso_paras
        self.filename = "PSO_ELM-sliding_{}-nets_{}-pso_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], pso_paras)

    def _training__(self):
        pso = PSO.BasePSO(root_algo_paras=self.root_algo_paras, pso_paras = self.pso_paras)
        self.solution, self.loss_train = pso._train__()


class BfoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, bfo_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.bfo_paras = bfo_paras
        self.filename = "BFO_ELM-sliding_{}-nets_{}-bfo_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], bfo_paras)

    def _training__(self):
        md = BFO.BaseBFO(root_algo_paras=self.root_algo_paras, bfo_paras = self.bfo_paras)
        self.solution, self.loss_train = md._train__()


class ABfoLSElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, abfols_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.abfols_paras = abfols_paras
        self.filename = "ABfoLS_ELM-sliding_{}-nets_{}-abfols_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], abfols_paras)

    def _training__(self):
        md = BFO.ABFOLS(root_algo_paras=self.root_algo_paras, abfols_paras=self.abfols_paras)
        self.solution, self.loss_train = md._train__()


class QsoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, qso_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.qso_paras = qso_paras
        self.filename = "QSO_ELM-sliding_{}-nets_{}-QSO_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], qso_paras)

    def _training__(self):
        md = QSO.BaseQSO(root_algo_paras=self.root_algo_paras, qso_paras=self.qso_paras)
        self.solution, self.loss_train = md._train__()



class TwoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, two_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.two_paras = two_paras
        self.filename = "TWO_ELM-sliding_{}-nets_{}-TWO_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], two_paras)

    def _training__(self):
        md = TWO.BaseTWO(root_algo_paras=self.root_algo_paras, two_paras=self.two_paras)
        self.solution, self.loss_train = md._train__()


class OTwoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, two_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.two_paras = two_paras
        self.filename = "OppoTwo_ELM-sliding_{}-nets_{}-OTWO_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], two_paras)

    def _training__(self):
        md = TWO.OppoTWO(root_algo_paras=self.root_algo_paras, two_paras=self.two_paras)
        self.solution, self.loss_train = md._train__()


class LTwoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, two_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.two_paras = two_paras
        self.filename = "LevyTwo_ELM-sliding_{}-nets_{}-LTWO_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], two_paras)

    def _training__(self):
        md = TWO.LevyTWO(root_algo_paras=self.root_algo_paras, two_paras=self.two_paras)
        self.solution, self.loss_train = md._train__()


class ITwoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, two_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.two_paras = two_paras
        self.filename = "ImprovedQSO_ELM-sliding_{}-nets_{}-IQSO_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], two_paras)

    def _training__(self):
        md = TWO.ITWO(root_algo_paras=self.root_algo_paras, two_paras=self.two_paras)
        self.solution, self.loss_train = md._train__()

