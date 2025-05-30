import random
import numpy
import time
from solution import solution


# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]
def DE(objf, lb, ub, dim, pop_size , Max_iter, ProbParam):
    n = ProbParam[0]
    m = ProbParam[1]
    ETC = ProbParam[2]
    VM = ProbParam[3]
    SP = ProbParam[4]
    ETime = ProbParam[5]
    DTime = ProbParam[6]

    

    mutation_factor = 0.5
    crossover_ratio = 0.7
    stopping_func = None

    # convert lb, ub to array
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]

    # solution
    s = solution()

    s.best = float("inf")

    # initialize population
    population = []

    population_fitness = numpy.array([float("inf") for _ in range(pop_size)])

    for p in range(pop_size):
        sol = []
        for d in range(dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        population.append(sol)

    population = numpy.array(population)

    # calculate fitness for all the population
    for i in range(pop_size):
        fitness = objf(population[i, :], n, m, ETC, VM, SP, ETime, DTime)
        population_fitness[p] = fitness
        # s.func_evals += 1

        # is leader ?
        if fitness < s.best:
            s.best = fitness
            s.bestIndividual = population[i, :]

    convergence_curve = numpy.zeros(Max_iter)
    # start work
    print('DE is optimizing')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 0
    while t < Max_iter:
        # should i stop
        if stopping_func is not None and stopping_func(s.best, s.bestIndividual, t):
            break

        # loop through population
        for i in range(pop_size):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in range(pop_size) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(dim):
                d_val = population[id_1, d] + mutation_factor * (
                    population[id_2, d] - population[id_3, d]
                )

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = numpy.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = objf(mutant_sol, n, m, ETC, VM, SP, ETime, DTime)
            # s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < s.best:
                    s.best = mutant_fitness
                    s.bestIndividual = mutant_sol

        convergence_curve[t] = s.best
        #if t % 1 == 0:
         #   print(["At iteration " + str(t + 1) + " the best fitness is " + str(s.best)])

        # increase iterations
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "DE"
    s.objfname = objf.__name__

    # return solution
    return s, population
