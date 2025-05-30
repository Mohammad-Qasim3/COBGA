from GGO import GGO
from SBOA import SBOA
from SSA import SSA
from DE import DE
from GA import GA
from COBGA import COBGA
from PSO import PSO
from GWO import GWO
from WOA import WOA
import random
import math
import statistics
import pyDecision
from Objectives import CompObj, Fitness
import copy
import numpy as np
import MCDM
import Plot
import matplotlib.pyplot as plt


#Function to generate the MI for IIoT tasks
def GenerateMI(n, minMI, maxMI):
    ETC= [0]*n
    AI = 3#Average Index
    for i in range(AI):
        for j in range(n):
            ETC[j] = ETC[j] + random.randint(minMI, maxMI)

    for k in range(n):
        ETC[k] = int(ETC[k]/AI)
    return ETC

# Setting Parameters
#+++++++Virtual Machine (VM) Parameters++++++++#
#number of VMs
m = 120 # Must be a Mutiple of 6
#VM speed MIPS 
VM_speed = [10, 20, 40, 60, 80, 100]
VM_temp = int(m/6)
VM = []
for i in range(6):
    VM.extend([VM_speed[i]]*VM_temp)
random.shuffle(VM)

#print(VM)
#Cost per second = [#Value of Speed/100] 
#Power = [#value of speed] watt 
#Security Policy Set [RC4, RC5, BLOWFISH, IDEA, SKIPJACK, 3DES]
ETime=[0.0063,0.0125,0.0170,0.0196,0.0217,0.0654]
DTime=[0.00572,0.0175,0.0238,0.0196,0.0248,0.106]

#+++++++Industrial IoT Tasks Parameters++++++++#
#IIoT tasks in the set T = {200,400,600,800,1000,1200,1400,1600,1800,2000}
#MI range MI= [10-100]
#number of Tasks
n = 500
MinMI = 1000
MaxMI = 2000
ETC = GenerateMI(n, MinMI, MaxMI)

#Security Policy Chosen by IIoT tasks 
SP = []
for i in range(n):
    SP.append(random.randint(0, 5))

#MH Parameters
lb= -500
ub = 500
dim = n
pop_size = 100
Max_iter = 50
#Problem Parameters
ProbParam = [n, m, ETC, VM, SP, ETime, DTime]
PSOPop = []; PSOConv = []
GWOPop = []; GWOConv = []
SSAPop = []; SSAConv = []
DEPop = []; DEConv = []
WOAPop = []; WOAConv = []
GAPop = []; GAConv = []
COBGAPop = []; COBGAConv = []
GGOPop = []; GGOConv = []
SBOAPop = []; SBOAConv = []

#Average Index (must be minimum 2)
AvgIndex = 10
for i in range(AvgIndex):
    PSOsol, pop1 = PSO(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)
    GWOsol, pop2 = GWO(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)
    SSAsol, pop3 = SSA(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)
    DEsol, pop4 = DE(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)
    WOAsol, pop5 = WOA(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)
    GAsol, pop6 = GA(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam) 
    COBGAsol, pop7 = COBGA(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)
    GGOsol, pop8 = GGO(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)
    SBOAsol, pop9 = SBOA(CompObj, lb, ub, dim, pop_size, Max_iter, ProbParam)

    PSOPop.append(PSOsol.bestIndividual)
    PSOConv.append(PSOsol.convergence)

    GWOPop.append(GWOsol.bestIndividual)
    GWOConv.append(GWOsol.convergence)
    
    SSAPop.append(SSAsol.bestIndividual)
    SSAConv.append(SSAsol.convergence)
    
    DEPop.append(DEsol.bestIndividual)
    DEConv.append(DEsol.convergence)
    
    WOAPop.append(WOAsol.bestIndividual)
    WOAConv.append(WOAsol.convergence)
    
    GAPop.append(GAsol.bestIndividual)
    GAConv.append(GAsol.convergence)
    
    COBGAPop.append(COBGAsol.bestIndividual)
    COBGAConv.append(COBGAsol.convergence)

    GGOPop.append(GGOsol.bestIndividual)
    GGOConv.append(GGOsol.convergence)

    SBOAPop.append(SBOAsol.bestIndividual)
    SBOAConv.append(SBOAsol.convergence)
    
    Result1 = Fitness(PSOsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result2 = Fitness(GWOsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result3 = Fitness(SSAsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result4 = Fitness(DEsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result5 = Fitness(WOAsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result6 = Fitness(GAsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result7 = Fitness(COBGAsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result8 = Fitness(GGOsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    Result9 = Fitness(SBOAsol.bestIndividual, n, m, ETC, VM, SP, ETime, DTime)
    print("The result for", "\nPSO is", Result1, 
          "\nGWO is", Result2, "\nSSA is", Result3,
          "\nDE is", Result4, "\nWOA is", Result5,
          "\nGA is", Result6, "\nCOBGA is", Result7,
          "\nGGO is", Result8, "\nSBOA is", Result9)

#Optimization Results(Convergence Curve)
PSOAv = np.array(PSOConv)
GWOAv = np.array(GWOConv)
SSAAv = np.array(SSAConv)
DEAv = np.array(DEConv)
WOAAv = np.array(WOAConv)
GAAv = np.array(GAConv)
COBGAAv = np.array(COBGAConv)
GGOAv = np.array(GGOConv)
SBOAAv = np.array(SBOAConv)
#print("The array is:\n", len(PSOAv), list(PSOAv))

#Calculate the sum of each column
PSOcs = np.average(PSOAv, axis=0)
GWOcs= np.average(GWOAv, axis=0)
SSAcs= np.average(SSAAv, axis=0)
DEcs= np.average(DEAv, axis=0)
WOAcs= np.average(WOAAv, axis=0)
GAcs= np.average(GAAv, axis=0)
COBGAcs= np.average(COBGAAv, axis=0)
GGOcs = np.average(GGOAv, axis=0)
SBOAcs = np.average(SBOAAv, axis=0)
#print("PSOcs", len(PSOcs), list(PSOcs))

#Plot Convergence Curve
XAxis = list(range(1, Max_iter+1))
plt.ylabel('Optimization Criterion', fontsize=15, fontweight='bold')
plt.xlabel('Iterations', fontsize=15, fontweight='bold')
plt.title('Convergence Curve', fontweight='bold')
plt.plot(XAxis, PSOcs)
plt.plot(XAxis, GWOcs)
plt.plot(XAxis, SSAcs)
plt.plot(XAxis, DEcs)
plt.plot(XAxis, WOAcs)
plt.plot(XAxis, GAcs)
plt.plot(XAxis, COBGAcs)
plt.plot(XAxis, GGOcs)
plt.plot(XAxis, SBOAcs)
#font= font_manager.FontProperties(weight='bold')
plt.legend(['PSO', 'GWO', 'SSA', 'DE', 'WOA', 'GA', 'COBGA', 'GGO', 'SBOA'])#,loc='upper left' )
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.show()


#Prepare for MOORA
POP = [PSOPop, GWOPop, SSAPop, DEPop, WOAPop, GAPop, COBGAPop, GGOPop, SBOAPop]
Size = len(POP[0])
DATASET = []
for i in range(9):
    nextPop = POP[i]
    nextDataset = []
    for j in range(Size):
        Result = Fitness(nextPop[j], n, m, ETC, VM, SP, ETime, DTime)
        nextDataset.append(list(Result))
    DATASET.append(nextDataset)

    #print(len(nextDataset), len(DATASET), "DATASET");input()

#print("Results before MOORA", DATASET);input()

RANK = MCDM.MOORA(DATASET)
#print(RANK)
#for the best solution
X = []
Y = []
Z = []
C = []
#For box-plot 
MK=[]
SEC=[]
CO=[]
EN=[]
Size = len(DATASET)
for i in range(Size):
    nextData = DATASET[i]
    nextRank = RANK[i]
    nextBSValue = []
    #print("nextData", nextData)
    #print("nextRank", nextRank);input()
    mk=[];sec=[];co=[];en=[]
    for j in range(AvgIndex):
        nextBSValue.append(nextData[nextRank[j]])
        if j == 0:
            X.append(nextData[nextRank[j]][0])
            Y.append(nextData[nextRank[j]][1])
            Z.append(nextData[nextRank[j]][2])
            C.append(nextData[nextRank[j]][3])

        mk.append(nextData[nextRank[j]][0])
        sec.append(nextData[nextRank[j]][1])
        co.append(nextData[nextRank[j]][2])
        en.append(nextData[nextRank[j]][3])
        
    MK.append(mk)
    SEC.append(sec)
    CO.append(co)
    EN.append(en)

#print("MK", X, MK)
#print("SEC", Y, SEC)
#print("CO", Z, CO)
#print("EN", C, EN)

#Print the results of the best solution
print("MOORA's best results: PSO, GWO, SSA, DE, WOA, GA, COBGA, GGO, SBOA")
print("The best makepan", X)
print("The best Security Time", Y)
print("The best Cost", Z)
print("The best Energy", C)
#Plot the best solution
Labels = ['PSO','GWO', 'SSA', 'DE', 'WOA', 'GA', 'COBGA', 'GGO', 'SBOA']
Plot.PlotBest(X, Y, Z, C, Labels)


#Plot boxplot
Plot.boxPlot(MK, "Makespan", Labels)#Makepan
Plot.boxPlot(SEC, "Security Time", Labels)#Security Time
Plot.boxPlot(CO, "Cost", Labels)#Cost
Plot.boxPlot(EN, "Energy", Labels)#Energy

print("Simulation Settings")
print("Task =", n, "MI Range", MinMI, MaxMI)
print('Mean', statistics.mean(ETC), 'StdDev', statistics.stdev(ETC))
print("Min MI", min(ETC), "Max MI", max(ETC))
