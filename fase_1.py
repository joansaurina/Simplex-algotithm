
# ==============================================================================
# [GIA-UPC]        fase_1.py                                                    
#                                                                               
# Author/s:        Joan Saurina i Ricós , Sergi Tomás Martínez
# Date:            26 de març de 2023
# Description:     Aquest codi inclou la funció de fase_1 de la implementació de
#                  simplex.
# ==============================================================================

#required packages
import numpy as np
from fase_2 import fase2
from auxiliar_functions import matriusn

def fase1(A,b,c,n,m):

    print(f"------------------------------------------ \n Inicialitzant Fase 1 en busca d'una SBF \n------------------------------------------\n\n")
    print(f'Afegint {m} variables artificials... \n\n')

    #       New A Preparation
    A = np.concatenate((A,np.identity(m)), axis=1)

    #       New C Preparation
    c = [0]*n + [1]*m
    cb = [1]*m
    
    # Artificial variables as base
    base = []
    for i in range(n,n+m):
        base.append(i)

    # Non-artificial variables as non-base
    no_base = []
    for i in range(n):
        no_base.append(i)
    
    # B_1 definition
    B_1 = np.identity(m)
    
    #Els vectors X i Xb son més fàcils de construir prèviament a llençar la fase 2, de manera que només aniran sent actualitzats
    Xb = np.matmul(B_1,b)
    X = np.concatenate((np.matrix([0]*n).T, Xb))
   
    An, cn = matriusn(A, c, no_base)

    print('Entrant a fase 2 per al problema artificial...')
    input('\n**Intro per continuar...**')
    base, B_1, Xb, z, X = fase2(cb, cn, c, X, Xb, B_1, A, An, base, no_base)
    
    
    if round(z,2) > 0:
        #Problema infactible
        print('Aquest problema és infactible')
        return None, None , None

    else:
        B = np.linalg.inv(B_1)
        while np.any(np.array(base) >= n):
            input('Aquest problema conté una variable artificial i és degenerat.')
            var_nobasica = next(x[0] for x in enumerate(base) if base[1] > n)
            
            for var_entrada in no_base:
                B[var_nobasica] = A[var_entrada]
                
                if np.allclose(np.linalg.det(B), 0) != 0:
                    base[var_nobasica], no_base[var_entrada] = base[var_entrada], no_base[var_nobasica]
                    break

    return B_1, base, Xb
