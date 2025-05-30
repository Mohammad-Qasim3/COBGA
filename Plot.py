from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def PlotBest(x, y, z, c, Labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    S = [50, 60, 70, 80, 90, 100, 110, 120, 130]
    
    x = np.array(x);x = x/1000
    y = np.array(y);y = y/1000
    z = np.array(z);z = z/1000
    c = np.array(c);c = c/1000

    img = ax.scatter(x, y, z, s=S, c=c, cmap=plt.get_cmap('jet'))
    i=0
    for a, b, c in zip(x, y, z):
        label = Labels[i];i=i+1
        ax.text(a, b, c, label)
    fig.colorbar(img, label="Energy",location='left', orientation ='vertical') 

    # Set axis labels
    ax.set_xlabel('MK(10^3)')
    ax.set_ylabel('ST(10^3)')
    ax.set_zlabel('CO(10^3)')
    ax.set_title("The Best Solution")
    plt.show()


def boxPlot(data, objName, Labels):
    plt.boxplot(data)
    plt.ylabel(objName, fontweight = 'bold', fontsize = 12)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.xticks(x, Labels, rotation=45)
    #show plot
    plt.show()
