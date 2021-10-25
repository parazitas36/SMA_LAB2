import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PyFunkcijos import *
import math
import time

#-------- Lygciu sistemos funkcija---------
def LF(x):  # grazina reiksmiu stulpeli
    s=np.array( [(x[0] - 1)**2 + x[1] - 10, 
                x[0]**2 + x[1]**2 + 4*x[0]*np.cos(x[0] + x[1]) - 40])
    s.shape=(2,1)
    s=np.matrix(s)
    return s
#----------------------------------

#
#*******************  Programa ************************************ 
# 
T=ScrollTextBox(100,20) # sukurti teksto isvedimo langa
T.insert(END,"Broideno metodas")

n=2 # lygciu skaicius
x=np.matrix(np.zeros(shape=(n,1))); x[0]=2.5;x[1]=0.3
SpausdintiMatrica(x,T)
# pradinis saknies artinys
maxiter=30  # didziausias leistinas iteraciju skaicius
eps=1e-6    # reikalaujamas tikslumas
  
#------ Grafika: funkciju LF pavirsiai -----------------------------------------------------------------------
fig1=plt.figure(1,figsize=plt.figaspect(0.5));
ax1 = fig1.add_subplot(1, 2, 1, projection='3d');ax1.set_xlabel('x');ax1.set_ylabel('y');ax1.set_ylabel('z')
ax2 = fig1.add_subplot(1, 2, 2, projection='3d');ax2.set_xlabel('x');ax2.set_ylabel('y');ax2.set_ylabel('z')
plt.draw();  #plt.pause(1);
xx=np.linspace(-10,10,20);yy=np.linspace(-10,10,20);
X, Y = np.meshgrid(xx, yy); Z=Pavirsius(X,Y,LF)

#surf1 = ax1.plot_surface(X, Y, Z[:,:,0], color='blue', alpha=0.4, linewidth=0.1, antialiased=True)
wire1 = ax1.plot_wireframe(X, Y, Z[:,:,0], color='black', alpha=1, linewidth=1, antialiased=True)
surf2 = ax1.plot_surface(X, Y, Z[:,:,1], color='purple', alpha=0.4, linewidth=0.1, antialiased=True)
CS11 =  ax1.contour(X, Y, Z[:,:,0],[0],colors='b')
CS12 =  ax1.contour(X, Y, Z[:,:,1],[0],colors='g')
CS1 =   ax2.contour(X, Y, Z[:,:,0],[0],colors='b')
CS2 =   ax2.contour(X, Y, Z[:,:,1],[0],colors='g')

XX=np.linspace(-5,5,2);  YY=XX; XX, YY = np.meshgrid(XX, YY); ZZ=XX*0
zeroplane = ax2.plot_surface(XX, YY, ZZ, color='gray', alpha=0.4, linewidth=0, antialiased=True)
#---------------------------------------------------------------------------------------------------------------

dx=0.1   # dx pradiniam Jakobio matricos iverciui
A=np.matrix(np.zeros(shape=(n,n)))
x1=np.zeros(shape=(n,1));
for i in range (0,n):   
   x1=np.matrix(x);
   x1[i]+=dx;
   A[:,i]=(LF(x1)-LF(x))/dx

ff=LF(x)
ax1.plot3D([x[0,0],x[0,0]],[x[1,0],x[1,0]],[0,ff[0,0]],"m*-")
plt.draw();plt.pause(1)


T.insert(END,"\n pradine Jakobio matrica:"); SpausdintiMatrica(A,T)
SpausdintiMatrica(ff.transpose(),T,"Pradine funkcijos reiksme")

for i in range (1,maxiter):
    deltax=-np.linalg.solve(A,ff); 
    x1=np.matrix(x+deltax); 
    ff1=LF(x1)
    A+=(ff1-ff-A*deltax)*deltax.transpose()/(deltax.transpose()*deltax);  
    tiksl=tikslumas(x,x1,ff,ff1,eps) ; SpausdintiMatrica(tiksl,T,"tiksl")
    ff= ff1; 
    x=x1; 
    if tiksl < eps: break; 
    else: 
 
        SpausdintiMatrica(x1.transpose(),T,"x1 ")
   
   #------ Grafika:  -----------------------------------------------------------------------------------------------
    ax1.plot3D([x[0,0],x1[0,0]],[x[1,0],x1[1,0]],[0,0],"ro-")  # reikia prideti antra indeksa, kadangi x yra matrica
    ax1.plot3D([x[0,0],x1[0,0]],[x[1,0],x1[1,0]],[ff[0,0],ff1[0,0]],"c-.")
    ax1.plot3D([x1[0,0],x1[0,0]],[x1[1,0],x1[1,0]],[0,ff1[0,0]],"m*-")
    ax2.plot3D([x[0,0],x1[0,0]],[x[1,0],x1[1,0]],[0,0],"ro-")
    plt.draw();plt.pause(2)
    #---------------------------------------------------------------------------------------------------------------


#------ Grafika:  ---------------------------------------
ax1.plot3D([x[0,0],x[0,0]],[x[1,0],x[1,0]],[0,0],"ks")
ax2.plot3D([x[0,0],x[0,0]],[x[1,0],x[1,0]],[0,0],"ks")
plt.draw();plt.pause(1)
#--------------------------------------------------------

SpausdintiMatrica(x1.transpose(),T,"Sprendinys")
SpausdintiMatrica(tiksl,T,"Galutinis tikslumas")

print("Plotting pre-finished");
plt.show();
print("Plotting finished");

#str1=input("Baigti darba? Press 'Y' \n")

