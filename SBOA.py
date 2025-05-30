# Secretary Bird Optimization Algorithm (SBOA) - Python Template
# Converted to follow the structure of the GWO algorithm

import numpy as np
import time
from solution import solution  # assumes a 'solution' class like in GWO


def Levy(dim):
    beta = 1.5
    sigma = (
        np.math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / np.abs(v) ** (1 / beta)


def SBOA(objf, lb, ub, dim, pop_size, Max_iter, ProbParam):
    n, m, ETC, VM, SP, ETime, DTime = ProbParam

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    lb = np.array(lb)
    ub = np.array(ub)

    X = np.random.rand(pop_size, dim) * (ub - lb) + lb
    fitness = np.array([objf(ind, n, m, ETC, VM, SP, ETime, DTime) for ind in X])

    s = solution()
    Convergence_curve = np.zeros(Max_iter)

    print("SBOA is optimizing")
    start_time = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for t in range(Max_iter):
        CF = (1 - t / Max_iter) ** (2 * t / Max_iter)

        best_idx = np.argmin(fitness)
        best_fit = fitness[best_idx]
        best_pos = X[best_idx].copy()

        for i in range(pop_size):
            if t < Max_iter / 3:
                idx1, idx2 = np.random.randint(0, pop_size, 2)
                R1 = np.random.rand(dim)
                X1 = X[i] + (X[idx1] - X[idx2]) * R1
            elif t < 2 * Max_iter / 3:
                RB = np.random.randn(dim)
                X1 = best_pos + np.exp((t / Max_iter) ** 4) * (RB - 0.5) * (best_pos - X[i])
            else:
                RL = 0.5 * Levy(dim)
                X1 = best_pos + CF * X[i] * RL

            X1 = np.clip(X1, lb, ub)
            f_new = objf(X1, n, m, ETC, VM, SP, ETime, DTime)

            if f_new < fitness[i]:
                X[i] = X1
                fitness[i] = f_new

        r = np.random.rand()
        k = np.random.randint(pop_size)
        Xrand = X[k]

        for i in range(pop_size):
            if r < 0.5:
                RB = np.random.rand(dim)
                X2 = best_pos + (1 - t / Max_iter) ** 2 * (2 * RB - 1) * X[i]
            else:
                K = round(1 + np.random.rand())
                R2 = np.random.rand(dim)
                X2 = X[i] + R2 * (Xrand - K * X[i])

            X2 = np.clip(X2, lb, ub)
            f_new = objf(X2, n, m, ETC, VM, SP, ETime, DTime)

            if f_new < fitness[i]:
                X[i] = X2
                fitness[i] = f_new

        Convergence_curve[t] = np.min(fitness)

    end_time = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = end_time - start_time
    s.bestIndividual = X[np.argmin(fitness)]
    s.convergence = Convergence_curve
    s.optimizer = "SBOA"
    s.objfname = objf.__name__

    return s, X
