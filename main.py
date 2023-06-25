
# ==============================================================================
# [GIA-UPC]        main.py                                                    
#                                                                               
# Author/s:        Joan Saurina i Ricós , Sergi Tomás Martínez
# Date:            26 de març de 2023
# Description:     Aquest codi inclou la funció main de la implementació de
#                  simplex.
# ==============================================================================

#required packages
from easyinput import read
import numpy as np
from fase_1 import fase1
from fase_2 import fase2
from auxiliar_functions import matriusn, read_data

def main():

    filename = 'OPT22-23_Datos práctica 1.txt'
    f = open(filename, 'r')

    while True:
        print(f'-------------------------------- \n Inicialitzant Problema \n--------------------------------')
        print(f'*** LLegint dades... ***\n\n')
        A,b,c = read_data(f)
        f.readline()
        instruccio = f.readline()
        
        # Problem Dimension (n = Nombre de vars., m = Nombre de restriccions)
        m, n=A.shape

        print(f'Problema de minimització amb {n} variables i {m} restriccions:')
        print('Funció a minimitzar: ', end = '')
        print(f'{c[0,0]}x0', end = '')
        for i in range(1,n):
            print(f' + {c[0,i]}x{i}', end = '')

        input('\n\n\n**Intro per continuar...**')


        #Cridem a Fase 1 en busca d'una SBF inicial
        B_1, base, Xb = fase1(A,b,c,n,m)

        print(f'-------------------------------- \n Fase 1 acabada \n--------------------------------')
        if B_1 is not None:
            print(f'SBF trobada a la base {base}\n\n')
            
            #Construcció de variables no bàsiques i vector X en base a la base trobada
            #X no té repercussió sobre la solució però és útil per al feedback i saber que està fent el programa a cada moment
            no_base = []
            X = []
            for i in range(n):
                if i not in base:
                    no_base.append(i)
                    X.append(0)
                else:
                    X.append(Xb[base.index(i),0])

            #Construcció de cb en base a la base trobada
            cb = []
            for i in base:
                cb.append(c[0,i])

            #Preprocessing per a matriusn()
            X = np.array(X).T
            c = np.ravel(c)
            An, cn = matriusn(A, c, no_base)

            print(f'-------------------------------- \n Iniciant Fase 2 \n--------------------------------')
            _, _ , _ , z , X = fase2(cb, cn, c, X, Xb, B_1, A, An, base, no_base)
        

            if X is not None:
                #Cas acotat, en cas contrari, el programa saltarà o bé al següent problema, o bé al final, si es tracta de l'últim.
                print(f'-------------------------------- \n Problema Minimitzat! \n--------------------------------')
                print(f'Mínim trobat a:', end='')
                for i in range(n):
                    if X[i]>0:
                        print(f'x{i} = {round(X[i],3)},  ', end = '')
                print(f'\n  Amb valor final z = {z}')
        input('\n\nPrem qualsevol tecla per a passar al següent problema')
        if instruccio.startswith('fi'):
            break 

main()

print('He acabat!!')  