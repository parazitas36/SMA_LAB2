import numpy as np
import matplotlib.pyplot as plt
def fx(x):
    mat=np.array([
        x[0]**2 - 2*x[0] + 1+x[1] - 10,
        x[0]**2 + x[1]**2 + 4*x[0]*np.cos(x[0] +x[1])-40
        ])
    mat.shape=(2,1)
    mat=np.matrix(mat)
    return mat

def df(x):
    dfmat=np.array([
        2*x[0]-2, 1,
        2*x[0] - 4*x[0]*np.sin(x[0] + x[1]) + 4*np.cos(x[0] + x[1]), 2*x[1] - 4*x[0]*np.sin(x[0]+x[1])
        ], dtype=np.float64)
    dfmat.shape=(2,2)
    dfmat=np.matrix(dfmat)
    return dfmat


def newton(x, alpha):
    itmax=1000
    eps=1e-10
    for i in range(itmax):
        deltax=-alpha*np.linalg.solve(df(x), fx(x))
        x=np.matrix(x+deltax)
        prec=np.linalg.norm(deltax) / (np.linalg.norm(x) + np.linalg.norm(deltax))
        print('Iteracija: %d tikslumas: %e'%(i, prec))
        if prec < eps:
            print('Sprendinys x=')
            print(x)
            print('Funkcijos reiksme f=')
            print(fx(x))
            break
        elif i == itmax-1:
            print('Tikslumas nepasiektas, paskutinis artinys x=')
            print(x)
            print('Funkcijos reiksme f=')
            print(fx(x))

n=2
x=np.matrix(np.zeros(shape=(n,1))); 
x[0]=4
x[1]=-1

alpha=0.7
newton(x,alpha)

def draw():
    fig1=plt.figure(1,figsize=plt.figaspect(0.5));
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    plt.draw()
    xx=np.linspace(-10,10,20);yy=np.linspace(-10,10,20)
    X, Y = np.meshgrid(xx, yy)
    Z=np.zeros(shape=(len(xx),len(yy),2))
    for i in range (0,len(xx)):
        for j in range (0,len(yy)): Z[i,j,:]=fx([X[i][j],Y[i][j]]).transpose();
    surf1 = ax1.plot_surface(X, Y, Z[:,:,0], color='blue', alpha=0.4)
    CS11 = ax1.contour(X, Y, Z[:,:,0],[0],colors='b')
    surf2 = ax1.plot_surface(X, Y, Z[:,:,1], color='purple',alpha=0.4)
    CS12 = ax1.contour(X, Y, Z[:,:,1],[0],colors='g')
    CS1 = ax2.contour(X, Y, Z[:,:,0],[0],colors='b')
    CS2 = ax2.contour(X, Y, Z[:,:,1],[0],colors='g')
    plt.show()

draw()