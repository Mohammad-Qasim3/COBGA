# Greylag Goose Optimization (GGO) - Python Template

import numpy as np
import time
from solution import solution

def GGO(objf, lb, ub, dim, pop_size, Max_iter, ProbParam):
    n, m, ETC, VM, SP, ETime, DTime = ProbParam

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    lb = np.array(lb)
    ub = np.array(ub)

    # Initialize population
    population = np.random.rand(pop_size, dim) * (ub - lb) + lb
    fitness = np.array([objf(ind, n, m, ETC, VM, SP, ETime, DTime) for ind in population])

    s = solution()
    Convergence_curve = np.zeros(Max_iter)

    print("GGO is optimizing")
    start_time = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for t in range(Max_iter):
        for i in range(pop_size):
            best_idx = np.argmin(fitness)
            best_agent = population[best_idx]

            # Exploration and exploitation combined move
            new_solution = population[i] + np.random.rand(dim) * (best_agent - population[i])
            new_solution = np.clip(new_solution, lb, ub)

            new_fitness = objf(new_solution, n, m, ETC, VM, SP, ETime, DTime)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        Convergence_curve[t] = np.min(fitness)

    end_time = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = end_time - start_time
    s.bestIndividual = population[np.argmin(fitness)]
    s.convergence = Convergence_curve
    s.optimizer = "GGO"
    s.objfname = objf.__name__

    return s, population
