
# ==============================================================================
# [GIA-UPC]        fase_2.py                                                    
#                                                                               
# Author/s:        Joan Saurina i Ricós , Sergi Tomás Martínez
# Date:            26 de març de 2023
# Description:     Aquest codi inclou la funció de fase_2 de la implementació de
#                  simplex.
# ==============================================================================

#required packages
import numpy as np
from auxiliar_functions import costos_minims,direccio,longitud,actualitzacio

def fase2(cb, cn, c, X, Xb, B_1, A, An, base, no_base):
    iter = 1
    #La variable flag s'utilitza per a notificar a la resta de funcions si no s'ha estat capaç de resoldre el problema, mss és l'output de l'error
    flag = 0
    print(f"------------------------------------------ \n Inicialitzant Fase 2 \n------------------------------------------\n\n")

    z = np.matmul(cb,Xb)[0,0]
    #Aquesta funció impedeix que les matrius es printejin amb excessius decimals i/o notació científica.
    np.set_printoptions(suppress = True, precision = 3)
    print(f'***  Iteració {iter}  ***')
    print(f' Variables bàsiques: {base}      Variables no bàsiques: {no_base}')
    print(f"\n Vector X:\n {X.T}")
    print(f'\n Valor z: {z}')
    input('\n**Intro per continuar...**')

    print(f'--Comprovant optimalitat...--\n')
    
    #Comprovem optimalitat
    r = costos_minims(cn,cb,B_1,An)

    
    while np.any(r < 0):        

        # Seleccionem variable d'entrada (index a no_base)
        iq = None
        # Var. entrada per index general
        q = float('inf')
        # Apliquem regla de Bland i seleccionem la variable d'index menor amb ri < 0
        for i in range(len(no_base)):
            if np.ravel(r)[i] < 0 and no_base[i]<q:
                iq = i 
                q = no_base[i]

        print(f"Òptim no trobat, variable d'entrada a la base: {q}")

        print(f'--Calculant direcció de descens...--\n')

        #Calculem direcció de descens
        db, flag = direccio(B_1, A, q)
        if flag == 1:
            print(f'-------------WARNING!!!!!-------------\n')
            print(f'Els costos reduits han sigut negatius per a la variable X{q} però la direcció de descens és positiva.\n')
            print(f"És a dir, el problema és NO ACOTAT, i la solució òptima és a l'infinit.\n")
            return (None,None,None,None,None)

        #Calculs de theta i variable de sortida
        theta, p = longitud(Xb, db, base)
        print(f'Posició de sortida de la base: {p}\n')

        #Actualització de les següents matrius
        base, no_base, Xb, B_1, An, cn, cb, c, X = actualitzacio(p, q, base, no_base, Xb, theta, db, cb, c, A, B_1)
        
        #Actualització de z
        z = round(z + theta * np.ravel(r)[iq], 4)

        #Recalculem r per a la següent iteració
        r = costos_minims(cn,cb,B_1,An)
        iter += 1

        print(f'***  Iteració {iter}  ***')
        print(f' Variables bàsiques: {base}      Variables no bàsiques: {no_base}')
        print(f'\n Vector X:\n {X}')
        #De vegades, python representava z com a -0 en comptes de 0 dins de la fase 1. El "+1-1" després del round(z), serveix per a representar un 0 en aquesta situació
        print(f'\n Valor z: {round(z,3)+1-1}')
        input('\n**Intro per continuar...**')

        print(f'\n--Comprovant optimalitat...--')
    
    return(base, B_1, Xb, z, X)