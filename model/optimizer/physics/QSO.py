import numpy as np
from math import gamma
from model.root.root_algo import RootAlgo

class BaseQSO(RootAlgo):
    ID_POS = 0
    ID_FIT = 1
    def __init__(self, root_algo_paras=None, qso_paras = None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch =  qso_paras["epoch"]
        self.pop_size = qso_paras["pop_size"]

    def _calculate_queue_length__(self, t1, t2 , t3):
        """
        calculate length of each queue based on  t1, t2,t3
        """
        n1 = (1/t1)/((1/t1) + (1/t2) + (1/t3))
        n2 = (1/t2)/((1/t1) + (1/t2) + (1/t3))
        n3 = (1/t3)/((1/t1) + (1/t2) + (1/t3))
        q1 = int(n1*self.pop_size)
        q2 = int(n2*self.pop_size)
        q3 = self.pop_size - q1 - q2
        return q1, q2, q3

    def _update_bussiness_1__(self, pop, current_iter, max_iter):
        sorted_pop = sorted(pop, key = lambda x: x[self.ID_FIT][self.ID_ERROR])
        s1, s2, s3 = sorted_pop[0:3]
        A1, A2 , A3 = s1[self.ID_POS], s2[self.ID_POS], s3[self.ID_POS]
        t1, t2 , t3 = s1[self.ID_FIT][self.ID_ERROR], s2[self.ID_FIT][self.ID_ERROR], s3[self.ID_FIT][self.ID_ERROR]
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        for i in range(self.pop_size):
            if i < q1:
                if i == 0:
                    case = 1
                A = A1
            elif i >= q1 and  i < q1 + q2:
                if i == q1 :
                    case = 1
                A = A2
            else:
                if i == q1 + q2 :
                    case = 1
                A = A3
            beta = np.power(current_iter, np.power(current_iter/max_iter, 0.5))
            alpha = np.random.uniform(-1,1)
            solution_shape = pop[0][0].shape
            E = np.random.exponential(0.5, solution_shape)
            e = np.random.exponential(0.5)
            F1 = beta*alpha*(E*np.abs(A - pop[i][0])) + e*A - e*pop[i][0]
            F2 = beta*alpha*(E*np.abs(A - pop[i][0]))
            if case == 1:
                X_new = A + F1
                new_fit = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
                if new_fit[self.ID_ERROR] < pop[i][self.ID_FIT][self.ID_ERROR]:
                    pop[i] = [X_new, new_fit]
                    case = 1
                else:
                    case = 2
            else:
                X_new = pop[i][0] + F2
                new_fit = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
                if new_fit[self.ID_ERROR] < pop[i][self.ID_FIT][self.ID_ERROR]:
                    pop[i] = [X_new, new_fit]
                    case = 2
                else:
                    case = 1
        return pop

    def _update_bussiness_2__(self, pop):
        sorted_pop = sorted(pop, key=lambda x:x[self.ID_FIT][self.ID_ERROR])
        s1, s2, s3 = sorted_pop[0:3]
        A1, A2, A3 = s1[self.ID_POS], s2[self.ID_POS], s3[self.ID_POS]
        t1, t2, t3 = s1[self.ID_FIT][self.ID_ERROR], s2[self.ID_FIT][self.ID_ERROR], s3[self.ID_FIT][self.ID_ERROR]
        #print("t1 {} , t2 {} , t3 {}".format(t1,t2,t3))
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        pr = [i/self.pop_size for i in range(1,self.pop_size+1)]
        cv = t1/(t2+t3)
        for i in range(self.pop_size):
            if i < q1:
                A = A1
            elif i >= q1 and i < q1 + q2:
                A = A2
            else:
                A = A3
            if np.random.random() < pr[i]:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                X1 = pop[i1][0]
                X2 = pop[i2][0]
                e = np.random.exponential(0.5)
                F1 = e*(X1-X2)
                F2 = e*(A-X1)
                if np.random.random() < cv:
                    X_new = sorted_pop[i][self.ID_POS] + F1
                    fit = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
                else:
                    X_new = sorted_pop[i][self.ID_POS] + F2
                    fit = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
                if fit[self.ID_ERROR] < sorted_pop[i][self.ID_FIT][self.ID_ERROR]:
                    sorted_pop[i] = [X_new, fit]
        return sorted_pop      
    
    def _update_bussiness_3__(self, pop):
        sorted_pop = sorted(pop, key=lambda x: x[self.ID_FIT][self.ID_ERROR])
        pr = [ i/self.pop_size for i in range(1, self.pop_size + 1)]
        for i in range(self.pop_size):
            X_new = np.zeros_like(pop[0][0])
            for j in range(self.problem_size):
                if np.random.random() > pr[i]:
                    i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                    e = np.random.exponential(0.5)
                    X1 = pop[i1][self.ID_POS]
                    X2 = pop[i2][self.ID_POS]
                    X_new[j] = X1[j] + e*(X2[j]- sorted_pop[i][self.ID_POS][j])
                else:
                    X_new[j] = sorted_pop[i][self.ID_POS][j]
            fit = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
            if fit[self.ID_ERROR] < sorted_pop[i][self.ID_FIT][self.ID_ERROR]:
                sorted_pop[i] = [X_new, fit]
        return sorted_pop

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROBLEM) for _ in range(self.pop_size)]
        sorted_pop = None
        for current_iter in range(self.epoch):
            pop = self._update_bussiness_1__(pop, current_iter, self.epoch)
            pop = self._update_bussiness_2__(pop)
            pop = self._update_bussiness_3__(pop)
            sorted_pop = sorted(pop, key=lambda x:x[self.ID_FIT][self.ID_ERROR])
            self.loss_train.append(sorted_pop[0][self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, Best fit: {1}".format(current_iter+1, sorted_pop[0][self.ID_FIT][self.ID_ERROR]))
        return sorted_pop[0][self.ID_POS], self.loss_train


class LevyQSO(BaseQSO):
    def __init__(self, root_algo_paras=None, qso_paras = None):
         BaseQSO.__init__(self, root_algo_paras, qso_paras)

    def _levy_flight__(self, solution, A, current_iter):
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        beta = 1
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy)
        v = np.random.normal(0, sigma_v)
        s = muy / np.power(np.abs(v), 1 / beta)
        D = self._create_solution__(minmax=self.ID_MIN_PROBLEM)[self.ID_POS]
        LB = 0.01 * s * (solution - A)
        levy = D * LB
        # X_new = solution + 0.01*levy
        # X_new = solution + 1.0/np.sqrt(current_iter+1)*np.sign(np.random.random()-0.5)*levy
        return levy

    def _update_bussiness_2__(self, pop=None, current_iter=None):
        sorted_pop = sorted(pop, key=lambda x:x[self.ID_FIT][self.ID_ERROR])
        s1, s2, s3 = sorted_pop[0:3]
        A1, A2, A3 = s1[self.ID_POS], s2[self.ID_POS], s3[self.ID_POS]
        t1, t2, t3 = s1[self.ID_FIT][self.ID_ERROR], s2[self.ID_FIT][self.ID_ERROR], s3[self.ID_FIT][self.ID_ERROR]
        #print("t1 {} , t2 {} , t3 {}".format(t1,t2,t3))
        q1, q2, q3 = self._calculate_queue_length__(t1, t2, t3)
        pr = [i/self.pop_size for i in range(1,self.pop_size+1)]
        cv = t1/(t2+t3)
        for i in range(self.pop_size):
            if i < q1:
                A = A1
            elif i >= q1 and i < q1 + q2:
                A = A2
            else:
                A = A3
            if np.random.random() < pr[i]:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                X1 = pop[i1][self.ID_POS]
                X2 = pop[i2][self.ID_POS]
                e = np.random.exponential(0.5)
                F1 = e*(X1-X2)
                F2 = e*(A-X1)
                if np.random.random() < cv:
                    X_new = self._levy_flight__(sorted_pop[i][self.ID_POS], A, current_iter)
                    fit = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
                else:
                    X_new = sorted_pop[i][0] + F2
                    fit = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
                if fit[self.ID_ERROR] < sorted_pop[i][self.ID_FIT][self.ID_ERROR]:
                    sorted_pop[i] = [X_new, fit]
        return sorted_pop

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROBLEM) for _ in range(self.pop_size)]
        sorted_pop = None
        for current_iter in range(self.epoch):
            pop = self._update_bussiness_1__(pop, current_iter, self.epoch)
            pop = self._update_bussiness_2__(pop, current_iter)
            pop = self._update_bussiness_3__(pop)
            sorted_pop = sorted(pop, key=lambda x:x[self.ID_FIT][self.ID_ERROR])
            self.loss_train.append(sorted_pop[0][self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, Best fit: {1}".format(current_iter + 1, sorted_pop[0][self.ID_FIT][self.ID_ERROR]))
        return sorted_pop[0][self.ID_POS], self.loss_train


class OppQSO(BaseQSO):
    def __init__(self, root_algo_paras=None, qso_paras = None):
        BaseQSO.__init__(self, root_algo_paras, qso_paras)

    def apply_opposition_based(self, sorted_pop, best):    
        a = 0.3
        num_change = int(self.pop_size*a)
        for i in range(self.pop_size-num_change,self.pop_size):
            X_new = self._create_opposition_solution__(sorted_pop[i][self.ID_POS], best)
            fitness = self._fitness_model__(model=X_new, minmax=self.ID_MIN_PROBLEM)
            if fitness[self.ID_ERROR] < sorted_pop[i][self.ID_FIT][self.ID_ERROR]:
                sorted_pop[i] = [X_new, fitness]
        return sorted_pop

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROBLEM) for _ in range(self.pop_size)]
        sorted_pop = None
        for current_iter in range(self.epoch):
            pop = self._update_bussiness_1__(pop, current_iter, self.epoch)
            pop = self._update_bussiness_2__(pop)
            pop = self._update_bussiness_3__(pop)
            sorted_pop = sorted(pop, key=lambda x:x[self.ID_FIT][self.ID_ERROR])
            sorted_pop = self.apply_opposition_based(sorted_pop, sorted_pop[0][self.ID_POS])
            self.loss_train.append(sorted_pop[0][self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, Best fit: {1}".format(current_iter + 1, sorted_pop[0][self.ID_FIT][self.ID_ERROR]))
        return sorted_pop[0][self.ID_POS], self.loss_train


class LevyOppQSO(OppQSO, LevyQSO):
    def __init__(self, root_algo_paras=None, qso_paras = None):
        OppQSO.__init__(self, root_algo_paras, qso_paras)
        LevyQSO.__init__(self, root_algo_paras, qso_paras)

    def _train__(self):
        pop = [self._create_solution__(minmax=self.ID_MIN_PROBLEM) for _ in range(self.pop_size)]
        sorted_pop = None
        for current_iter in range(self.epoch):
            pop = self._update_bussiness_1__(pop, current_iter, self.epoch)
            pop = self._update_bussiness_2__(pop, current_iter)
            pop = self._update_bussiness_3__(pop)
            sorted_pop = sorted(pop, key=lambda x:x[self.ID_FIT][self.ID_ERROR])
            sorted_pop = self.apply_opposition_based(sorted_pop, sorted_pop[0][self.ID_POS])
            self.loss_train.append(sorted_pop[0][self.ID_FIT])
            if self.print_train:
                print("Generation : {0}, Best fit: {1}".format(current_iter + 1, sorted_pop[0][self.ID_FIT][self.ID_ERROR]))
        return sorted_pop[0][self.ID_POS], self.loss_train