
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
import sys
import matplotlib.pyplot as plt
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

def crearMovimiento(N):
    #El movimiento consta de dos partes    
    movimiento=[]

    while len(movimiento) < 2:
        #Las posiciones del vector van de 0 a N-1
        temp=random.randint(0,N-1)
        if temp not in movimiento:
            movimiento.append(temp)
    movimiento=list(sorted(movimiento))  
    return movimiento


# In[4]:

def generarCandidatos(semilla,lista,posibles,memoria,N):
    print "---------------------------"
    print "Comienza la generación de nuevos candidatos,"
    print "según la semilla:"
    print semilla
    print "---------------------------"
    candidatos=[]
    while len(candidatos)<posibles:
        
        #Crea un nuevo movimiento
        movimiento=crearMovimiento(N)

        
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
            


# In[5]:

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
    


# In[6]:

def combinatoria(N):
    return (N*(N-1))/2


# In[7]:

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


# In[8]:

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
    print "Total iteraciones = ",i
    print "---------------------------\n"
        
    


# In[9]:

def enfriamientoSimulado(N,I):
    print "\n\n---------------------------"
    #Calculamos la temperatura inicial basados en el número reinas N
    #Si la solución candidata empeora el fitness N/2
    #Tiene una probabilidad del 99% de ser aceptada
    Tini = -1*((N/2)/log(0.99))
    print "Temperatura Inicial del sistema = ",Tini
    #Calculamos la temperatura final basados en el número reinas N
    #Si la solución candidata empeora el fitness N/2
    #Tiene una probabilidad del 1% de ser aceptada 
    Tfin = -1*((N/2)/log(0.01))
    print "Temperatura Final del sistema = ",Tfin
    
    #Calculamos la tasa de enfriamiento basados en las iteraciones totales
    #De tal forma que la última iteración tenga la temperatura final
    deltaT=pow((Tfin/Tini),(1/I))
    print "Delta de temperatura = ",deltaT
    print "---------------------------\n"
    
    vector=producirSemilla(N)
    
    #Inicializo Históricos
    histBestFitness=[]
    histBestFitnessEnd=[]
    histCandFitness=[]
    histProbLim=[]
    histProb=[]
    histTemp=[]
    #Comienza el algoritmo
    T=Tini
    probabilidadLimite=1
    magnitud=float('nan')
    if evaluar(vector)==0:
        finalizar=True
    else:        
        finalizar= False
    i=0
    while i<I and not finalizar:
        print "\n\n--------------------------------"
        print "--------------------------------"
        print "         Iteración #",i
        print "--------------------------------"
        print "--------------------------------"
        movimiento=crearMovimiento(N)
        candidato= generarNuevoCandidato(vector,movimiento)
        #Se evaluan los fitness del vector y del nuevo candidato
        fitVector = evaluar(vector)
        fitCandidato = evaluar(candidato)
        
        histTemp.append(T)
        
        
        print "\nFitness\tVector"
        print "--------\t--------"
        print fitVector,"\t",vector
        print fitCandidato,"\t",candidato
        print "--------\t--------"
        
        deltaIndividuos = fitCandidato-fitVector

        
        histProbLim.append(probabilidadLimite)
        
        if deltaIndividuos<=0:
            #Si es mejor la solución encontrada
            print "------------------"
            print "     Reemplaza    "
            print "------------------\n"
            vector = list(candidato)
            #Probabilidad no necesaria
            probabilidad=0
            
        else:
            #Enfriamiento Simulado
            try:
                magnitud =exp(deltaIndividuos/T)
            except OverflowError:
                print "------------------"
                print "     ALERTA"
                print "El algoritmo ha excedido la precisión de float de su computador"
                print "Para minimizar el error en el algoritmo"
                print "Se asume el máximo número permitido por su arquitectura"
                print sys.float_info[0]
                print "------------------"
                magnitud = sys.float_info[0]
            probabilidadLimite = 1/magnitud                                    
            probabilidad = random.random()
            
            if probabilidad <= probabilidadLimite:
                print "------------------"
                print "     Reemplaza    "
                print " probabilidad <= probabilidad Límite"
                print probabilidad," <= ",probabilidadLimite
                print "------------------\n"                              
                vector = list(candidato)
        histProb.append(probabilidad)
        print "Delta indivuos: ",deltaIndividuos
        print "Magnitud:",magnitud
        print "Probabilidad Límite",probabilidadLimite 
        print "Probabilidad: ",probabilidad
        
        #El rendimiento de la mejor solución
        #Al finalizar la iteración
        #Esto indica si la mejor solución se desplazó
        fitVectorFin=evaluar(vector)
        #Almacena el rendimiento de las soluciones
        histBestFitness.append(fitVector)
        histBestFitnessEnd.append(fitVectorFin)
        histCandFitness.append(fitCandidato)
        
        #Verifica si es fin del algoritmo
        if evaluar(vector)==0:
            finalizar=True
        else:
            T = deltaT*T
            print "Nueva Temperatura = ",T
            i=i+1
            
    print "---------------------------"
    if finalizar:
        
        print "Algortimo encontró un óptimo en ",i," iteraciones"
        print vector
        print "Fitness = ", evaluar(vector)
    else:
        print "Algortimo una buena solución en ",i," iteraciones"
        print vector
        print "Fitness = ", evaluar(vector)
    print "---------------------------"
    
    #Graficar
    plt.figure(1)
    plt.plot(range(0,len(histBestFitness)), histBestFitness,'r-')
    plt.plot(range(0,len(histBestFitnessEnd)), histBestFitnessEnd,'gs')
    plt.plot(range(0,len(histCandFitness)), histCandFitness,'b-')    
    plt.xlabel('Iteracion')
    plt.ylabel('Fitness')
    plt.figure(2)    
    plt.subplot(211)
    plt.plot(range(0,len(histProbLim)), histProbLim,'r-')
    plt.plot(range(0,len(histProb)), histProb,'b-')
    plt.xlabel('Iteracion')
    plt.ylabel('Probabilidad')
    plt.subplot(212)
    plt.plot(range(0,len(histTemp)), histTemp,'gs')
    plt.xlabel('Iteracion')
    plt.ylabel('Temperatura del sistema')
    plt.show()

        
            
            
        


# In[10]:

def finalizar():
    print "\n\n---------------------------"
    resp=raw_input("¿Desea ejecutar algún algoritmo nuevamente?:\n(S)Sí\n(N)No:\n").strip().lower()
    if resp=="s":
        return True
    
    return False


# In[11]:

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
        nroReinas=input('\nIngrese el número de reinas:')
        nroIteraciones=input('Ingrese el número de iteraciones máximas:')
        busquedaTabu(nroReinas,nroIteraciones)
        repetir=finalizar()
    elif(algoritmo==2):
        nroReinas=input('\nIngrese el número de reinas:')
        nroIteraciones=input('Ingrese el número de iteraciones máximas:')        
        enfriamientoSimulado(nroReinas,nroIteraciones)
        repetir=finalizar()
    else:
        print "---------------------------"
        print "Algoritmo desconocido, favor repita su selección."
        print "---------------------------\n\n"
    

