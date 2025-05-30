import random
import math
import statistics
import numpy as np
import copy


def Fitness(Vector, n, m, ETC, VM, SP, ETime, DTime):
    #n -number of tasks
    #m - number of VMs
    #print("Hello, Fitness")
    #print(n, m, len(VM),VM);input()
    #print(n, m, len(ETC),ETC);input()
    #print(min(ETC), max(ETC))
    #print(min(VM), max(VM));input()
    Vector = copy.deepcopy(Vector)
    Sol =[]
    for i in range(len(Vector)):
        temp = int(Vector[i] * (math.pow(2, 16))) % m
        Sol.append(temp)
    
    #if len(Sol) < n:
     #   print(Vector)
      #  print(len(Vector), len(Sol), n)
       # print(Sol)

    #Fetching all tasks for all VMs
    A = dict()
    for i in range(n):
        task = i #ith task (index of list representis task) 
        #print(i, Sol[i]) #Sol[i] is the VM for task i
        CVM = Sol[i] #Current VM
        if CVM in A.keys():
            A[CVM].append(task)
        else:
            A[CVM] = [task]
    #print(min(list(A.keys())),max(list(A.keys())));input()
    
    #Computing Makespan
    makespan = 0
    CTVM = [0]*m #Availability time for all VMs
    ExeTime = [0]*n #Execution time of the tasks
    SecT = [0]*n #Security time for all tasks
    Cost = [0]*n #Monetry Cost = ((ExeTime + SecT)/leased period)*Cost
    Energy =[0]*n #Energy consumption for all tasks
    minSpeed = min(VM)
    #print('minSpeed', minSpeed);input() 
    for a in A:
        TaskList = A[a]
        AVT = 0
        #print(TaskList)
        for task in TaskList:
            #print(task, a, ETC[task], VM[a])
            #DecTime = ((ETC[task]*1000000*32)/8*1000)*(time[SP[task]]/1000))
            DecTime = int((ETC[task]*(4))*DTime[SP[task]]*(minSpeed/VM[a]))
            #Execution Time
            ExeTime[task] = math.ceil(ETC[task]/VM[a])
            #Encryption Time
            #Assume response consists of 70% of actual data for the given task
            EncTime = int((math.ceil(0.7*ETC[task])*(4))*ETime[SP[task]]*(minSpeed/VM[a]))
            AVT = AVT + DecTime + ExeTime[task] + EncTime
            SecT[task] = DecTime + EncTime
            #Cost is 1 unit for a leased period of 5 minutes
            Cost[task] = math.ceil((ExeTime[task]+SecT[task])/300)*(VM[a]/10)
            #print("ExeTime", ExeTime[task])
            #print("DecTime", DecTime, "EncTime", EncTime)
            #print("Cost", Cost[task])
            #print("AVT", AVT);input()

        #Availability time of VM after executing all scheduled taks
        CTVM[a] = AVT
        
    #Computing Makespan
    makespan = max(CTVM)
    #Computing Security Time
    SecurityTime = int(sum(SecT))
    #Computing Monetry Cost
    TotalCost = int(sum(Cost))
    #Computing Energy Consumption
    #VM speed is equal to the energy consumption in Watt
    TotalEnergy = 0
    maxCTVM = max(CTVM)
    VMEn = [0]*m
    for i in range(m):
        CVMEn = CTVM[i]*VM[i] + (maxCTVM- CTVM[i])*(0.7)*VM[i]
        VMEn[i] = CVMEn/1000

    TotalEnergy = int(sum(VMEn))
    #print(makespan, SecurityTime, TotalCost, TotalEnergy)
    return makespan, SecurityTime, TotalCost, TotalEnergy


def CompObj(Sol, n, m, ETC, VM, SP, ETime, DTime):
    #Result =[]
    #M = 0 #Square root of square of all obj values of makespan
    #S = 0 #similar for Security time
    #C = 0 #similar for cost
    #E = 0 #similar for energy
    #Computing fitness for all solutions in the population
    temp = Fitness(Sol, n, m, ETC, VM, SP, ETime, DTime)
    #print("temp", temp);input()

    #Normalized Makespan
    maxT = sum(ETC)/min(VM)
    NormMK = temp[0]/maxT
    
    #Normalized Security Time
    #TempList = list(temp)
    DecTime = int((sum(ETC)*(4))*DTime[5])
    #Assume response consists of 70% of actual data for the given task
    EncTime = int((math.ceil(0.7*sum(ETC))*(4))*ETime[5])
    NormST = temp[1]/(DecTime + EncTime)

    #Normalized Cost
    MaxCost = math.ceil((maxT+DecTime+EncTime)/300)*(max(VM)/10)
    NormCT = temp[2]/MaxCost

    #Normalized Energy
    MaxEn = (maxT*(max(VM))*(m))/1000
    NormEn = temp[3]/MaxEn

    '''if NormMK > 1 or NormST > 1 or NormCT > 1 or NormEn > 1:
        print("NormMK", NormMK)
        print("NormST", NormST)
        print("NormCT", NormCT)
        print("NormEn", NormEn);input()'''

    #Computing Normalized Fitness value 
    lambda1 = lambda2 = lambda3 = lambda4 = X = 0.25
    Score = X*NormMK + X*NormST + X*NormCT + X*NormEn
    #print("Score", Score);input()
    return Score
