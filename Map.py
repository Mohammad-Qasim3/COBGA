import random
import math
import numpy as np
import matplotlib.pyplot as plt

def cosinecube(x,r,size):
    Z = []
    for i in range (size):
        x = abs(np.cos(np.pi*x*(r-x**2)*10**5))
        Z.append(x)

    return Z


def pwlcm_map(dim, p):
    x = random.random()
    outputs = []

    for _ in range(dim):
        if 0 <= x < p:
            x = x / p
        elif p <= x < 0.5:
            x = (x - p) / (0.5 - p)
        elif 0.5 <= x < 1 - p:
            x = (1 - p - x) / (0.5 - p)
        else: # 1 - p <= x < 1
            x = (1 - x) / p
        outputs.append(x)
    
    return outputs





def combined_sine_logistic_map(dim):
    x = random.random()
    inputs = []
    outputs = []
    x_input = []

    for i in range(0, dim):
        x_input.append(i)
        inputs.append(x)
        x = np.sin(np.pi * 4 * x * (1 - x) * 10**5)
        outputs.append(x)
    return list(outputs)


def bernoulli_map(dim):
    x = random.random()
    inputs = []
    outputs = []
    x_input = []

    for i in range(0, dim):
        if 0 <= x <= (1 - 0.603):
            x = x / (1 - 0.603)
        else:
            x = (x - 1 + 0.603) / 0.603
        outputs.append(x)
    
    return list(outputs)




'''
def tent_map(dim,):
    x = random.random()
    outputs = []

    for _ in range(dim):
        if 0 <= x <= 0.5:
            x = x / 0.5
        else:
            x = (1 - x) / (1 - 0.5)
        outputs.append(x)
    
    return outputs
'''




def tent_map(dim):
    """iterate a tent map with x as the initial condition in the domain of x in (0,1), with n steps (default n=5)"""
    x = random.random()
    inputs = []
    outputs = []
    x_input = []
    #print(x)
    for i in range(0, dim):
        x_input.append(i)
        inputs.append(x)
        if x < 0.5:
            x = 2 * x
        elif x > 0.5:
            x = 2 - 2 * x
        
        outputs.append(x)
    #print("Input", inputs)
    #print("Output", outputs)
    #plt.plot(inputs, outputs)
    #plt.show()
    #plt.plot(x_input, outputs)
    #plt.show()
    return list(outputs)


def chebyshev_map(dim):
    x = random.random()
    inputs = []
    outputs = []
    x_input = []

    for i in range(0, dim):
        x_input.append(i)
        inputs.append(x)
        x = math.cos(i * math.acos(x))
        outputs.append(x)
    return list(outputs)



def logistic_map(dim, r=4):
    x = random.random()
    outputs = []

    for _ in range(dim):
        x = r * x * (1 - x)
        outputs.append(x)
    
    return outputs



def sine_map(dim):
    x = random.random()
    outputs = []

    for _ in range(dim):
        x = math.sin(math.pi * x)
        outputs.append(x)
    
    return outputs




def ImprovedCircleChaotic_map(dim):
    xxx= random.random()
    YY = [1] * dim
    X = []
    Y = []
    for i in range(dim):
        X.append(i)
        Y.append(xxx)
        xxx = (xxx + 0.4204 - (0.0305 / math.pi) * math.sin(2 * math.pi * xxx)) % 1
        YY[i] = xxx

    #print(YY)
    #plt.plot(Y, YY)
    #plt.show()
    #plt.plot(X, YY)
    #plt.show()
    return list(YY)
#example
#tent_map(2048)
#ImprovedCircleChaotic_map(2048)
