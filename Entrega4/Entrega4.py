
# coding: utf-8

# In[1]:

#Programa para comparar la eficiencia y eficacia
#de los algortimos Búsqueda Tabú y Enfriamiento Simulado
#para el problema de ubicar N reinas de ajedrez en un tablero de NxN
#---Curso:Optimización 2017-01 Universidad de Antioquia
#---Estudiante: John Tapias Zarrazola

# coding: utf-8
from math import *
import random
from __future__ import division


# In[2]:

def generarNuevoCandidato(semilla,movimiento):
    
    #Copia el vector semilla
    nuevo=list(semilla)
    
    #Realiza las permutaciones
    temp= nuevo[movimiento[0]]
    nuevo[movimiento[0]] = nuevo[movimiento[1]]
    nuevo[movimiento[1]] = temp
    
    return nuevo
        
    
 


# In[3]:

def generarCandidatos(semilla,lista,posibles,memoria,N):
    print "---------------------------"
    print "Comienza la generación de nuevos candidatos,"
    print "según la semilla:"
    print semilla
    print "---------------------------"
    candidatos=[]
    while len(candidatos)<posibles:
        #Crea un nuevo movimiento
        movimiento=[]
        #El movimiento consta de dos partes
        while len(movimiento) < 2:
            #Las posiciones del vector van de 0 a N-1
            temp=random.randint(0,N-1)
            if temp not in movimiento:
                movimiento.append(temp)
        movimiento=list(sorted(movimiento))
        
        if movimiento not in lista:
            print "Permutación ",movimiento," : Ha sido aceptada"
            lista.append(movimiento)
            candidato= generarNuevoCandidato(semilla,movimiento)
            print "Se genera nuevo candidato: ",candidato
            candidatos.append(candidato)
            if len(lista) > posibles*memoria:
                #La lista funciona como cola
                #Se elimina el primero de la lista
                del lista[0]                
        else:
            print "Permutación ",movimiento," : Ha sido rechazado"
            
    
    print "---------------------------"
    print "Lista Tabú actualizada:"
    print lista
    print "---------------------------"
    return candidatos
            


# In[4]:

def evaluar(v):
    #Parejas penalizadas
    #Se almacenan para no penalizar dos veces lo mismo
    #penalizados=[]
    penalizar=0
    for i in range(0,len(v)):
        #El vector se analiza de izquierda a derecha
        #Intervalo de [1;len(V)-i)
        for j in range(1,len(v)-i):
            if v[i+j]==v[i]+j or v[i+j]==v[i]-j:
                #print "Penaliza ind:",i," ind:",(i+j)
                penalizar=penalizar+1
            #print "Fitness=",penalizar
        
    
    return penalizar
    


# In[5]:

def combinatoria(N):
    return (N*(N-1))/2


# In[6]:

def producirSemilla(N):
    semilla=[]
    #semilla={"vector":}
    #vector=[]
    while len(semilla)<N:
        temp = random.randint(0,N-1)
        if temp not in semilla:
            semilla.append(temp)
    #Se genera el vector semilla
    print "---------------------------"
    print "Solución aleatoria inicial:"
    print semilla
    print "Fitness = ",evaluar(semilla)
    print "---------------------------"
    #Se retorna el vector semilla
    return semilla


# In[7]:

def busquedaTabu(N,I):
    #Es una combinatoria de N tomados de a 2
    #Es el número máximo de elementos que puede tener la lista tabú
    #La lista viene dada por el usuario memoria*posibles
    #combinatoria(N,2) <= memoria*posibles
    textto='''
    ------------------------------------------------------
    ------------------------------------------------------
    La lista Tabú tendrá como máximo de elementos:
    maxElem = cantidad de iteraciones a recordar * número de individuos nuevos por cada generación

    Pero la lista Tabú está restringida según la cantidad de reinas N a organizar
    maxCombinaciones = Combinatoria(N,2)

    Asegurese que maxElem <= maxCombinaciones
    ------------------------------------------------------
    ------------------------------------------------------
    '''    
    continuar=False
    while not continuar:
        print textto
        maxCombinaciones=combinatoria(N)
        print "maxCombinaciones = ",maxCombinaciones
        memoria=input('Ingrese la cantidad de iteraciones que recordará la lista Tabú:')
        posibles=input('Ingrese el número de inviduos a generar en cada iteración:')

        if memoria*posibles>maxCombinaciones:
            print "\n---------------------------"
            print "  >>ALERTA<<"
            print "maxElem = ",memoria*posibles," supera a maxCombinaciones=",maxCombinaciones
            print "---------------------------\n"
        else:
            continuar=True
    
    lista=[]*memoria
    individuos=[]*posibles
    
    vector=producirSemilla(N)
    
    #Comienza el algoritmo
    finalizar= False
    i=0
    while i<I and not finalizar:
        dicVector={}
        dicVector[0]={evaluar(vector):vector}
        #Se generan los nuevos individuos
        #Restringidos por la lista de memoria
        individuos = generarCandidatos(vector,lista,posibles,memoria,N)
        #Se evaluan a los nuevos individuos
        for ind in individuos:            
            dicVector[len(dicVector)]={evaluar(ind):ind}
        #Se ordenan ascendentemente, la idea es minizar la función
        dicVector = list(sorted(dicVector.values()))
        print "\nFitness\tVector"
        print "--------\t--------"
        for elem in dicVector:
            [(k, v)] = elem.items()
            print k,"\t",v
        print "--------\t--------\n"
        #El primero es el que tiene igual o más bajo fitness
        
        vector =dicVector[0]
        print "Se selecciona nueva semilla ",vector
        #Si alcanza una penalización igual a cero
        if 0 in vector:
            finalizar=True
            print "\n\n---------------------------"
            print "Finaliza algoritmo búsqueda tabú con solución:"            
            for fit,vec in vector.iteritems():
                print vec
                print "Fitness = ",fit            
            print "---------------------------"
        else:
            print "---------------------------"
            print "Mejor solución hasta el momento"
            for fit,vec in vector.iteritems():
                print vec
                print "Fitness = ",fit
                vector=vector[fit]
                break
            print "---------------------------"
            
            i=i+1
        
    


# In[8]:

def enfriamientoSimulado(N,I):
    print "Soy enfriamiento simulado"


# In[ ]:

def finalizar():
    print "\n\n---------------------------"
    resp=raw_input("¿Desea ejecutar algún algoritmo nuevamente?:\n(S)Sí\n(N)No:\n").strip().lower()
    if respc=="s":
        return True
    
    return False


# In[ ]:

repetir=True
#Primero se decide cuál algoritmo se desea correr
texto='''
Ingrese el tipo de algoritmo:
(1) Búsqueda Tabú
(2) Enfriamiento Simulado
'''
while repetir==True:
    algoritmo=input(texto)
    if(algoritmo==1):
        nroReinas=input('Ingrese el número de reinas:')
        nroIteraciones=input('Ingrese el número de iteraciones máximas:')
        busquedaTabu(nroReinas,nroIteraciones)
        repetir=finalizar()
    elif(algoritmo==2):
        nroReinas=input('Ingrese el número de reinas:')
        nroIteraciones=input('Ingrese el número de iteraciones máximas:')        
        enfriamientoSimulado(nroReinas,nroIteraciones)
        repetir=finalizar()
    else:
        print "---------------------------"
        print "Algoritmo desconocido, favor repita su selección."
        print "---------------------------\n\n"
    


# In[ ]:

ve=[3,4,2,0,5,1,6]
print evaluar(ve)


# In[ ]:



