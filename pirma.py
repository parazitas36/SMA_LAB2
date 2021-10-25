import numpy as np
import matplotlib.pyplot as plot

#---------------------------
# 4-ta lygtis
# Matrica
#---------------------------
A = np.matrix([ [3, 7, 1, 3],
        [1,-6,6,8],
        [4, 4, -7, 1],
        [4,16,2,0] ]).astype(float)
# Vektorius
b=np.array([[40], [19], [36], [48]]).astype(float)
#---------------------------
def Amat(x):
    arr=np.array([
        [3*x[0]+7*x[1]+x[2] +3*x[3]],
        [1*x[0]-6*x[1]+6*x[2] +8*x[3]],
        [4*x[0]+4*x[1]-7*x[2] +x[3]],
        [4*x[0]+16*x[1]+2*x[2] +0*x[3]]
        ])
    arr.shape=(4,1)
    arr=np.matrix(arr)
    return arr
#---------------------------
# 21-a lygtis
# Matrica
#---------------------------
A21 = np.matrix([
    [3,7,1,3],
    [1,-6,6,8],
    [4,4,-7,1],
    [-1,3,8,2]
    ]).astype(np.float64)
b21=np.array([[11], [3], [1], [1]]).astype(np.float64)

def A21mat(x):
    arr=np.array([
        [3*x[0]+7*x[1]+x[2] +3*x[3]],
        [1*x[0]-6*x[1]+6*x[2] +8*x[3]],
        [4*x[0]+4*x[1]-7*x[2] +x[3]],
        [-1*x[0]+3*x[1]+8*x[2] +2*x[3]]
        ])
    arr.shape=(4,1)
    arr=np.matrix(arr)
    return arr
#---------------------------


# mat - matrica, vec - vektorius
def checkForZeroes(mat,vec):
    for i in range(0, np.shape(mat)[0]):
        if(mat[i,i] == 0):
            equation=mat[i].copy()
            vecVal=vec[i].copy()
            for j in range(0, np.shape(mat)[0]):
                if mat[j,i] != 0 and mat[i,j] != 0:
                    mat[i]=mat[j].copy()
                    mat[j]=equation
                    vec[i]=vec[j].copy()
                    vec[j]=vecVal
                    break
            break
    print("\nPo pakeitimo")
    print("Matrica:")
    print(mat)
    print("Vektorius:")
    print(vec)

# mat - matrica, vec - vektorius
def iteration(alpha, mat, vec):
    n=np.shape(mat)[0]
    checkForZeroes(mat,vec)
    Atld=np.diag(1./np.diag(mat)).dot(mat)-np.diag(alpha)
    btld=np.diag(1./np.diag(mat)).dot(vec)

    itmax=10000
    eps=1e-12
    x=np.zeros(shape=(n,1))
    x1=np.zeros(shape=(n,1))
    prec =[]
    for it in range(0, itmax):
        x1=((btld-Atld.dot(x)).transpose()/alpha).transpose()
        prec.append(np.linalg.norm(x1-x)/(np.linalg.norm(x)+np.linalg.norm(x1)))
        if prec[it] < eps :
           print("Tikslumas: %e, Iteraciju kiekis: %d, Gauti sprendiniai: \n"%(prec[it], it), x)
           break
        x[:]=x1[:]
    return prec, x

def qr(mat, vec):
    n=(np.shape(mat))[0]
    nb=(np.shape(vec))[1]
    A1=np.hstack((mat,vec))

    # tiesioginis etapas
    for i in range(0, n-1):
        z=A1[i:n, i]
        zp=np.zeros(np.shape(z))
        zp[0]=np.linalg.norm(z)
        omega=z-zp
        omega=omega/np.linalg.norm(omega)
        Q=np.identity(n-i) - 2*omega*omega.transpose()
        A1[i:n,:]=Q.dot(A1[i:n,:])
    # atgalinis etapas
    x=np.zeros(shape=(n, nb))
    for i in range(n-1, -1, -1):
        x[i,:]=(A1[i,n:n+nb] - A1[i, i+1:n] * x[i+1:n,:])/A1[i,i]
    return x


kuri=2
method='iter'
# Iteracinis metodas su 4-ta lygtim
if method == 'iter':
    if kuri == 1:
        print("Matrica A4: \n", A)
        print("Vektorius b4: \n", b)
        print("Sprendiniai:")
        print(np.linalg.solve(A,b))

        alpha=np.array([2.5, 2.5, 2.5, 2.5])
        prec, sprend = iteration(alpha, A, b)
        print('Patikrinimas:')
        print(Amat(sprend))
        line1 = plot.semilogy(prec[:])

        alpha=np.array([3, 3, 3, 3])
        prec, sprend = iteration(alpha, A, b)
        print('Patikrinimas:')
        print(Amat(sprend))
        line2 = plot.semilogy(prec[:])

        alpha=np.array([4.2, 4.2, 4.2, 4.2])
        prec, sprend = iteration(alpha, A, b)
        print('Patikrinimas:')
        print(Amat(sprend))
        line3 = plot.semilogy(prec[:])

        alpha=np.array([5, 5, 5, 5])
        prec, sprend = iteration(alpha, A, b)
        print('Patikrinimas:')
        print(Amat(sprend))
        line4 = plot.semilogy(prec[:])
        plot.semilogy(prec[:])

        plot.xlabel("Iteraciju kiekis")
        plot.ylabel("Tikslumas")
        plot.legend([line1, line2, line3, line4], labels=['2.5', '3', '4.2', '5'], title="Alpha reiksme")
        plot.show()
    else:
        print("Matrica A21: \n", A21)
        print("Vektorius b21: \n", b21)
        print("Sprendiniai:")
        print(np.linalg.solve(A21,b21))

        print('det:',np.linalg.det(A21))

        alpha=np.array([-4.5, -4.5, -4.5, -4.5])
        prec, sprend = iteration(alpha, A21, b21)
        print('Patikrinimas:')
        print(A21mat(sprend))
        line1 = plot.semilogy(prec[:])

        alpha=np.array([-2.7, -2.7, -2.7, -2.7])
        prec, sprend = iteration(alpha, A21, b21)
        print('Patikrinimas:')
        print(A21mat(sprend))
        line2 = plot.semilogy(prec[:])

        alpha=np.array([-2.15, -2.15, -2.15, -2.15])
        prec, sprend = iteration(alpha, A21, b21)
        print('Patikrinimas:')
        print(A21mat(sprend))
        line3 = plot.semilogy(prec[:])

        alpha=np.array([-0.355, -0.355, -0.355, -0.355])
        prec, sprend = iteration(alpha, A21, b21)
        print('Patikrinimas:')
        print(A21mat(sprend))
        line4 = plot.semilogy(prec[:])
        plot.semilogy(prec[:])

        plot.xlabel("Iteraciju kiekis")
        plot.ylabel("Tikslumas")
        plot.legend([line1, line2, line3, line4], labels=['-4.5', '-2.7', '-2.15', '-0.355'], title="Alpha reiksme")
        plot.show()
else:
    if kuri == 1:
        print("Matrica A4: \n", A)
        print("Vektorius b4: \n", b)
        print("Sprendiniai:")
        print(np.linalg.solve(A,b))
        print('\tPatikrinimas')
        print('Su NumPy sprendiniais:')
        print(Amat(np.linalg.solve(A,b)))
        sprend=qr(A, b)
        print('Su QR skaidos sprendiniais:')
        print(Amat(sprend))
        print('QR sprendiniai:')
        print(sprend)
       
    else:
        print("Matrica A21: \n", A21)
        print("Vektorius b21: \n", b21)
        print("Sprendiniai:")
        print(np.linalg.solve(A21,b21))
        print('\tPatikrinimas')
        print('Su NumPy sprendiniais:')
        print(A21mat(np.linalg.solve(A21,b21)))
        sprend=qr(A21, b21)
        print('Su QR skaidos sprendiniais:')
        print(A21mat(sprend))
        print('QR sprendiniai:')
        print(sprend)
        print('Singualiari:')
        print(A21mat(np.linalg.lstsq(A21,b21)[0]))