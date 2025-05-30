import random
import numpy
import math
from solution import solution
import time


def SSA(objf, lb, ub, Dim, pop_size, Max_iter, ProbParam):

    n = ProbParam[0]
    m = ProbParam[1]
    ETC = ProbParam[2]
    VM = ProbParam[3]
    SP = ProbParam[4]
    ETime = ProbParam[5]
    DTime = ProbParam[6]

    # Max_iteration=1000
    # lb=-100
    # ub=100
    # dim=30
    #N = 50  # Number of search agents
    N = pop_size
    if not isinstance(lb, list):
        lb = [lb] * Dim
    if not isinstance(ub, list):
        ub = [ub] * Dim
    Convergence_curve = numpy.zeros(Max_iter)

    # Initialize the positions of salps
    SalpPositions = numpy.zeros((N, Dim))
    for i in range(Dim):
        SalpPositions[:, i] = numpy.random.uniform(0, 1, N) * (ub[i] - lb[i]) + lb[i]
    SalpFitness = numpy.full(N, float("inf"))

    FoodPosition = numpy.zeros(Dim)
    FoodFitness = float("inf")
    # Moth_fitness=numpy.fell(float("inf"))

    s = solution()

    print('SSA is optimizing')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for i in range(0, N):
        # evaluate moths
        SalpFitness[i] = objf(SalpPositions[i, :],n, m, ETC, VM, SP, ETime, DTime)

    sorted_salps_fitness = numpy.sort(SalpFitness)
    I = numpy.argsort(SalpFitness)

    Sorted_salps = numpy.copy(SalpPositions[I, :])

    FoodPosition = numpy.copy(Sorted_salps[0, :])
    FoodFitness = sorted_salps_fitness[0]

    Iteration = 0

    # Main loop
    while Iteration < Max_iter:

        # Number of flames Eq. (3.14) in the paper
        # Flame_no=round(N-Iteration*((N-1)/Max_iteration));

        c1 = 2 * math.exp(-((4 * Iteration / Max_iter) ** 2))
        # Eq. (3.2) in the paper

        for i in range(0, N):

            SalpPositions = numpy.transpose(SalpPositions)

            if i < N / 2:
                for j in range(0, Dim):
                    c2 = random.random()
                    c3 = random.random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition[j] + c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )
                    else:
                        SalpPositions[j, i] = FoodPosition[j] - c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )

                    ####################

            elif i >= N / 2 and i < N + 1:
                point1 = SalpPositions[:, i - 1]
                point2 = SalpPositions[:, i]

                SalpPositions[:, i] = (point2 + point1) / 2
                # Eq. (3.4) in the paper

            SalpPositions = numpy.transpose(SalpPositions)

        for i in range(0, N):

            # Check if salps go out of the search spaceand bring it back
            for j in range(Dim):
                SalpPositions[i, j] = numpy.clip(SalpPositions[i, j], lb[j], ub[j])

            SalpFitness[i] = objf(SalpPositions[i, :],n, m, ETC, VM, SP, ETime, DTime)

            if SalpFitness[i] < FoodFitness:
                FoodPosition = numpy.copy(SalpPositions[i, :])
                FoodFitness = SalpFitness[i]

        # Display best fitness along the iteration
        #if Iteration % 1 == 0:
         #   print(
          #      [
           #         "At iteration "
            #        + str(Iteration+1)
             #       + " the best fitness is "
              #      + str(FoodFitness)
               # ]
            #)

        Convergence_curve[Iteration] = FoodFitness

        Iteration = Iteration + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.bestIndividual = FoodPosition
    s.convergence = Convergence_curve
    s.optimizer = "SSA"
    s.objfname = objf.__name__

    return s, SalpPositions
