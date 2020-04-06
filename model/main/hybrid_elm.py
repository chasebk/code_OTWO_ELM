#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 03:51, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from model.root.hybrid.root_hybrid_elm import RootHybridElm
from mealpy.evolutionary_based import GA, DE
from mealpy.swarm_based import PSO, BFO
from mealpy.physics_based import TWO
from mealpy.human_based import QSA


class GaElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = ga_paras["epoch"]
        self.pop_size = ga_paras["pop_size"]
        self.pc = ga_paras["pc"]
        self.pm = ga_paras["pm"]
        self.filename = "GA_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = GA.BaseGA(self._objective_function__, self.problem_size, self.domain_range, self.print_train, self.epoch, self.pop_size, self.pc, self.pm)
        self.solution, self.best_fit, self.loss_train = md._train__()


class DeElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = de_paras["epoch"]
        self.pop_size = de_paras["pop_size"]
        self.wf = de_paras["wf"]
        self.cr = de_paras["cr"]
        self.filename = "DE_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = DE.BaseDE(self._objective_function__, self.problem_size, self.domain_range, self.print_train, self.epoch, self.pop_size, self.wf, self.cr)
        self.solution, self.best_fit, self.loss_train = md._train__()


class PsoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = pso_paras["epoch"]
        self.pop_size = pso_paras["pop_size"]
        self.c1 = pso_paras["c_minmax"][0]
        self.c2 = pso_paras["c_minmax"][1]
        self.w_min = pso_paras["w_minmax"][0]
        self.w_max = pso_paras["w_minmax"][1]
        self.filename = "PSO_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = PSO.BasePSO(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                         self.epoch, self.pop_size, self.c1, self.c2, self.w_min, self.w_max)
        self.solution, self.best_fit, self.loss_train = md._train__()


class BfoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, bfo_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.pop_size = bfo_paras["pop_size"]
        self.Ci = bfo_paras["Ci"]
        self.Ped = bfo_paras["Ped"]
        self.Ns = bfo_paras["Ns"]
        self.Ned = bfo_paras["Ned"]
        self.Nre = bfo_paras["Nre"]
        self.Nc = bfo_paras["Nc"]
        self.attract_repels = bfo_paras["attract_repels"]
        self.filename = "BFO_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = BFO.BaseBFO(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                         self.pop_size, self.Ci, self.Ped, self.Ns, self.Ned, self.Nre, self.Nc, self.attract_repels)
        self.solution, self.best_fit, self.loss_train = md._train__()


class ABfoLSElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, abfols_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = abfols_paras["epoch"]
        self.pop_size = abfols_paras["pop_size"]
        self.Ci = abfols_paras["Ci"]
        self.Ped = abfols_paras["Ped"]
        self.Ns = abfols_paras["Ns"]
        self.N_minmax = abfols_paras["N_minmax"]
        self.filename = "ABFOLS_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = BFO.ABFOLS(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                        self.epoch, self.pop_size, self.Ci, self.Ped, self.Ns, self.N_minmax)
        self.solution, self.best_fit, self.loss_train = md._train__()


class QsaElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, qsa_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = qsa_paras["epoch"]
        self.pop_size = qsa_paras["pop_size"]
        self.filename = "QSA_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = QSA.BaseQSA(self._objective_function__, self.problem_size, self.domain_range, self.print_train, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = md._train__()


class TwoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, two_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = two_paras["epoch"]
        self.pop_size = two_paras["pop_size"]
        self.filename = "Two_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = TWO.BaseTWO(self._objective_function__, self.problem_size, self.domain_range, self.print_train, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = md._train__()


class OTwoElm(RootHybridElm):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, two_paras=None):
        RootHybridElm.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = two_paras["epoch"]
        self.pop_size = two_paras["pop_size"]
        self.filename = "OppoTwo_ELM-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = TWO.OppoTWO(self._objective_function__, self.problem_size, self.domain_range, self.print_train, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = md._train__()
