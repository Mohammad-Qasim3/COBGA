import pyDecision
from pyDecision.algorithm import moora_method
import numpy as np 


def sort_list_Angle(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)] 
    return z

def MOORA(DATASET):
    #print("Hello, MOORA")
    #Weights
    weights = [0.25, 0.25, 0.25, 0.25]
    #Load Criterion Type: 'max' or 'min'
    criterion_type = ['min', 'min', 'min', 'min']

    Size = len(DATASET)
    RANK = []
    for i in range(Size):
        # Dataset
        #print(DATASET[i])
        dataset = np.array(DATASET[i])
        #print("dataset", i, dataset, type(dataset));input()
        
        # Call MOORA Function
        rank = moora_method(dataset, weights, criterion_type, graph=False, verbose = False)
        #print(rank)
        score = list(map(float, rank[:, 1]))
        ScoreIndex = list(range(0, len(score)))
        ActualRank = sort_list_Angle(ScoreIndex, score)
        ActualRank.reverse()
        #print("score", score)
        #print("ScoreIndex", ScoreIndex)
        #print("ActualRank", ActualRank);input() 
        RANK.append(ActualRank)

    #print(RANK)
    return RANK
