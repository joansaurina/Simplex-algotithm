
# ==============================================================================
# [GIA-UPC]        auxiliar_functions.py
# 
# Author/s:        Joan Saurina i Ricós , Sergi Tomás Martínez
# Date:            26 de març de 2023
# Description:     Aquest codi inclou les funcions auxiliars de la implementació
#                  de simplex: read_data, costos_minims, direccio, longitud,
#                  actualitzacio, actualitzacio_inversa i matriusn.
# ==============================================================================

#import packages
import numpy as np

def read_data(f):
    instruccio = f.readline()


    # ---------------
    # |    Read c   |
    # ---------------
    
    while not instruccio.startswith('c='):
        instruccio = f.readline()
        
    instruccio = f.readline()

    if instruccio.startswith('\n'):
        f.readline()
        f.readline()
        primersnum = f.readline().split()

        f.readline()
        f.readline()
        f.readline()

        segonsnums = f.readline().split()

        c = np.mat(primersnum+segonsnums).astype(np.int32)
    else:

        c = np.mat(instruccio.split()).astype(np.int32)


    # ---------------
    # |   Read A    |
    # ---------------

    flag = 0
    f.readline()
    f.readline()
    instruccio = f.readline()

    if instruccio.startswith('\n'):
        f.readline()
        f.readline()
        instruccio = f.readline()
        flag = 1

    A = []
    while instruccio != '\n':
        A.append(instruccio.split())
        instruccio = f.readline()

    A = np.matrix(A).astype(np.int32)

    if flag == 1:
        instruccio = f.readline()
        instruccio = f.readline()
        A_aux = []
        instruccio = f.readline()
        while instruccio != '\n':
            A_aux.append(instruccio.split())
            instruccio = f.readline()
        A_aux = np.matrix(A_aux).astype(np.int32)

        A = np.concatenate((A,A_aux),axis=1) 


    # ---------------
    # |   Read b    |
    # ---------------

    f.readline()

    b = np.matrix(f.readline().split()).astype(np.int32)
    b = b.T

    return (A,b,c)


def costos_minims(cn,cb,B_1,An):
    #Funció de càlcul de costos mínims. Apartada per estètica.
    return cn - (np.matmul(np.matmul(cb,B_1),An))


def direccio(B_1,A,columna):
    #Variable per a informar a les altres funcions de si es tracta d'un problema no acotat.
    flag = 0
    db = -np.matmul(B_1, A[:,columna])
    if np.all(db >= 0):
            #Cas de no acotació (STOP)
            flag = 1
            
    return (db, flag)


def longitud(Xb,db,base):
    #Iniciem p buida i la millor theta a infinit
    theta_estr = float('inf')
    p = None
    #Iterem pels indexos amb db negativa
    indices, = np.where(np.ravel(db)<0)

    for basica in indices:
        theta = -np.ravel(Xb)[basica]/np.ravel(db)[basica]
        
        #Si hem trobat una theta inferior, guardem p i actualitzem theta_estr
        if theta < theta_estr:
            theta_estr = theta
            p = basica
        
        #Apliquem Regla de Bland
        elif theta == theta_estr:
            if base[basica] < base[p]:
                theta_estr = theta
                p = basica
    
    return (theta_estr,p)


def actualitzacio(p, q, base, no_base, Xb, theta, db, cb, c, A, B_1):

    #Fem servir var_entrada, var_sortida per legibilitat
    var_sortida = base[p]
    var_entrada = q
    #Acutalitzem la base
    base[p] = q

    #Afegim q al final per conveniència, ja que les matrius de les no bàsiques es calculen de zero a matriusn()
    no_base.append(var_sortida)
    no_base.remove(var_entrada)

    #actualitzar Xb
    Xb = Xb + theta*db
    #Canviem el valor Xb[p] per theta. És la posició de la variable que acaba d'entrar a la base.
    Xb[p,0] = theta
    
    #Reconstrucció del vector X. Útil només per els prints.
    X = []
    for i in range(len(base)+len(no_base)):
        if i in base:
            X.append(Xb[base.index(i),0])
        else:
            X.append(0)
    X = np.array(X).T

    #actualitzar B_1
    B_1 = actualitzacio_inversa(B_1,p,db)

    #actualitzar An, cn, cb
    An, cn = matriusn(A, c, no_base)

    #Canviem el valor de cb[p] per el coeficient de la funció de la variable que acaba d'entrar a les bàsiques
    cb[p] = c[q]
    
    return base, no_base, Xb, B_1, An, cn, cb, c, X


def actualitzacio_inversa(B_1, p, db):

    E = np.identity(len(db))
    E[:,p] = (-db/db[p]).T
 
    E[p,p] = -1 / db[p]
    B_1 = np.matmul(E, B_1)

    return B_1

        
def matriusn(A, c, no_base):
    # La funció matriusn retorna les matrius que per comoditat preferim que siguin recalculades a cada iteració, hem decidit que a les matrius
    # associades a variables no basiques, l'ordre de les columnes respecterà l'ordre dels indexos i, per tant, en comptes de col·locar la columna
    # de la variable de sortida on pertoca, recalculem les matrius en la seva totalitat. La matrius associades a variables bàsiques sí que seran
    # modificades amb el pas de les iteracions. 
    
    cn = []
    An = np.matrix([0]*len(A)).T
    
    for i in no_base:
        cn.append(c[i]) 
        An = np.concatenate((An,A[:,i]), axis=1)

    cn = np.matrix(cn)
    An = An[:, 1:]

    return (An, cn)
