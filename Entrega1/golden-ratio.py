#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from math import *
import matplotlib.pyplot as plt
import numpy as np
import compiler
import time

portada="""
 _____       _     _             ______      _   _          ___  _                  _ _   _
|  __ \     | |   | |            | ___ \    | | (_)        / _ \| |                (_| | | |
| |  \/ ___ | | __| | ___ _ __   | |_/ /__ _| |_ _  ___   / /_\ | | __ _  ___  _ __ _| |_| |__  _ __ ___
| | __ / _ \| |/ _` |/ _ | '_ \  |    // _` | __| |/ _ \  |  _  | |/ _` |/ _ \| '__| | __| '_ \| '_ ` _ \
| |_\ | (_) | | (_| |  __| | | | | |\ | (_| | |_| | (_) | | | | | | (_| | (_) | |  | | |_| | | | | | | | |
 \____/\___/|_|\__,_|\___|_| |_| \_| \_\__,_|\__|_|\___/  \_| |_|_|\__, |\___/|_|  |_|\__|_| |_|_| |_| |_|
                                                                    __/ |
         Development By: John Tapias Zarrazola                     |___/
"""
print(portada)

def evalFun(x,func):
    return eval(func)

flag=False
tipoAlg =0
phi=0
while(not flag):

	print('\n----------------------------------------------------')
	print('Ingrese el tipo de algoritmo:\n')
	print(' (1)  Búsqueda Razón Dorada con valor positivo')
	print(' (2)  Búsqueda Razón Dorada con valor negativo')
	print('----------------------------------------------------')
	tipoAlg = input('>>')
	if(tipoAlg==1):
		phi = ((sqrt(5) +1) / 2)
		flag=True
	elif(tipoAlg==2):
		phi = ((sqrt(5) -1) / 2)
		flag=True
	else:
		print('\n\nValor inválido ...(Presione Enter)\n')
		non=raw_input(' ')

print('\n----------------------------------------------------')
print('           Ingrese numero de iteraciones')
print('----------------------------------------------------')
numIter = input('>>')

print('\n----------------------------------------------------')
print('              Ingrese umbral de error')
print('----------------------------------------------------')
umbral = input('>>')

#print('\n----------------------------------------------------')
#print('              Ingrese ecuacion')
#print('              Ejemplo: -1*x*x + 8*x - 12')
#print('----------------------------------------------------')
#funcion = input('>>')
#func = "-1*x*x + 8*x - 12 "
func= "2*sin(x) - ((x**2)/10)"
eq= compiler.parse( func )

print('\n----------------------------------------------------')
print('Dado el dominio de busqueda: a<=x<=b')
print('----------------------------------------------------')
a = input('Ingrese a:\n>>')
b = input('Ingrese b:\n>>')

flag=False
maximizar =0
while(not flag):

	print('----------------------------------------------------')
	print('Ingrese :\n')
	print(' (1)  Para hallar máximo')
	print(' (2)  Para hallar mínimo')
	print('----------------------------------------------------')
	maximizar = input('>>')
	if(maximizar==1 or maximizar==2):
		flag=True
	else:
		print('\n\nValor inválido ...(Presione Enter)\n')
		a=raw_input(' ')

#-----------Graficar función
dt = ((abs(a-b))/200)
y=[]
for i in range(0,200):
    y.append(evalFun(a+(dt * i),func))
t= np.arange(a, b, dt)
plt.ion()
#---------------

terminar =False
iterActual=0

datosFinal = []
Xl = a
Xu = b
while(not terminar):
    iterActual=iterActual +1

    if(tipoAlg==1):#Positivo
        X1=Xl+ ((Xu-Xl)/(1+phi))
        X2=Xl + (Xu-Xl)
    else:#Negativo
        d = phi*(Xu-Xl)
        X1 = Xl + d
        X2 = Xu - d


    #Evaluamos los puntos en la función
    fXl = evalFun(Xl,func)
    fXu = evalFun(Xu,func)
    fX1 = evalFun(X1,func)
    fX2 = evalFun(X2,func)

    #Almacenamos los datos
    datos = {}
    datos['Xl'] ={'x':Xl,'f(x)':fXl}
    datos['Xu'] ={'x':Xu,'f(x)':fXu}
    datos['X1'] ={'x':X1,'f(x)':fX1}
    datos['X2'] ={'x':X2,'f(x)':fX2}


    #Graficamos
    plt.ylabel(func)
    plt.xlabel('x')
    plt.title('ITERACION #'+str(iterActual))
    plt.plot(t, y,'b--')
    plt.plot(Xl, fXl,'rs')
    plt.plot((Xl, Xl), ((-1)*fXl, fXl), 'r-')
    plt.plot(Xu, fXu,'rs')
    plt.plot((Xu, Xu), ((-1)*fXu, fXu), 'r-')
    plt.plot(X1, fX1,'g^')
    plt.plot((X1, X1), ((-1)*fX1, fX1), 'g-')
    plt.plot(X2, fX2,'g^')
    plt.plot((X2, X2), ((-1)*fX2, fX2), 'g-')
    plt.show()

    #if(maximizar==1):#maximizar

    #temp = {'Xl':fXl, 'Xu': fXu, 'X1': fX1 , 'X2': fX2 }
    #temp = {fXl:'Xl',fXu: 'Xu' , fX1:'X1'  , fX2:'X2' }
    temp = [('Xl',Xl,'fXl',fXl), ('Xu',Xu,'fXu',fXu), ('X1',X1,'fX1',fX1), ('X2',X2,'fX2',fX2) ]
    sortedTemp = sorted(temp, key= lambda punto:punto[1]) #Ascendente por el x evaluado en f

    print('----------------------------------------------------')
    print('                  ITERACIÓN #'+str(iterActual))
    print('----------------------------------------------------')
    #print('                  SIN ORDENAR')
    #print(temp)
    print('             ORDENADO RESPECTO A x')
    print(sortedTemp)
    print('----------------------------------------------------')
    if(maximizar==1):#maximizar
        if(fX1<fX2):
            print('Se descarta Xu='+str(Xu)+ '-> ahora Xu=X2='+str(X2))
            Xu=X2
        else:
            print('Se descarta Xl='+str(Xl)+ '-> ahora Xl=X1='+str(X1))
            Xl=X1

        #Seleccionamos el menor para ser eliminado
        #print("Se elimina el menor  "+ str(sortedTemp[0]))
        #sortedTemp.remove(sortedTemp[0])
    else:#minimizar
        if(fX1<fX2):
            print('Se descarta Xl='+str(Xl)+ '-> ahora Xl=X1='+str(X1))
            Xl=X1
        else:
            print('Se descarta Xu='+str(Xu)+ '-> ahora Xu=X2='+str(X2))
            Xu=X2


        #Seleccionamos el mayor para ser eliminado
        #print("Se elimina el mayor  "+ str(sortedTemp[3]))
        #sortedTemp.remove(sortedTemp[3])


    #Ahora se ordenan respecto a X
#    sortedTemp = sorted(sortedTemp, key= lambda punto:punto[1]) #Ascendente por el x evaluado en f
#    print('             ORDENADO RESPECTO A x')
#    print(sortedTemp)

    print('----------------------------------------------------')
    print('                   ACTUALIZA Xl y Xu')

#    if(maximizar==1):#maximizar
    	#Actualizamos Xu
#        if(sortedTemp[2][1]<=b):
#            Xu = sortedTemp[2][1]
#        elif(sortedTemp[1][1]<=b):
#            Xu = sortedTemp[1][1]
#        elif(sortedTemp[0][1]<=b):
#            Xu = sortedTemp[0][1]
#        else:
#            print('>>>Xu se sale del intervalo<<<')

    	#Actualizamos Xl
#        if(sortedTemp[0][1]>=a):
#            Xl= sortedTemp[0][1]
#        elif(sortedTemp[1][1]>=a):
#            Xl= sortedTemp[1][1]
#        elif(sortedTemp[2][1]>=a):
#            Xl= sortedTemp[2][1]
#        else:
#            print('>>>Xl se sale del intervalo<<<')

#    else:#minimizar
    	#Actualizamos Xu
        #if(sortedTemp[2][1]<=b):
        #    Xu = sortedTemp[2][1]
        #elif(sortedTemp[1][1]<=b):
        #    Xu = sortedTemp[1][1]
        #elif(sortedTemp[0][1]<=b):
        #    Xu = sortedTemp[0][1]
        #else:
        #    print('>>>Xu se sale del intervalo<<<')

    	#Actualizamos Xl
        #if(sortedTemp[0][1]>=a):
        #    Xl= sortedTemp[0][1]
        #elif(sortedTemp[1][1]>=a):
        #    Xl= sortedTemp[1][1]
        #elif(sortedTemp[2][1]>=a):
        #    Xl= sortedTemp[2][1]
        #else:
        #    print('>>>Xl se sale del intervalo<<<')


    #Xu = sortedTemp[3][1] #Xu toma el mayor de los tres
    #Xl = sortedTemp[1][1] #Xl toma el menor de los tres
    print("Xu =  "+ str(Xu))
    print("Xl =  "+ str(Xl))
    print('----------------------------------------------------')


    #Esperamos un tiempo proporcional al número de iteraciones
    #time.sleep((10/numIter))
    plt.pause((10/numIter))

    datos['error'] ={'e':((Xu -Xl)/2)}
    datosFinal.append(datos)

    if iterActual >= numIter:
	       terminar=True
    elif abs(Xu-Xl) <= umbral:
        terminar=True
    else:
        plt.clf()


print('\n\n----------------------------------------------------')
#-----------MOSTRAR DATOS
print('Iteración #\t| Xl\t| Xu\t| X1\t| X1\t| X2\t| f(Xl)\t| f(Xu)\t| f(X1)\t| f(X1)\t| f(X2)')
for elemento in range(len(datosFinal)):
    cadena = str(elemento+1)
    cadena = cadena + "\t| " + str(datosFinal[elemento]['Xl']['x'])
    cadena = cadena + "\t| " + str(datosFinal[elemento]['Xu']['x'])
    cadena = cadena + "\t| " + str(datosFinal[elemento]['X1']['x'])
    cadena = cadena + "\t| " + str(datosFinal[elemento]['X2']['x'])
    cadena = cadena + "\t| " + str(datosFinal[elemento]['Xl']['f(x)'])
    cadena = cadena + "\t| " + str(datosFinal[elemento]['Xu']['f(x)'])
    cadena = cadena + "\t| " + str(datosFinal[elemento]['X1']['f(x)'])
    cadena = cadena + "\t| " + str(datosFinal[elemento]['X2']['f(x)'])
    print( cadena )
print('----------------------------------------------------')
print('Iteración #\t| Error')
for elemento in range(len(datosFinal)):
    cadena = str(elemento+1)
    cadena = cadena + "\t| " + str(datosFinal[elemento]['error']['e'])
    print( cadena )
print('----------------------------------------------------')

print('\n----------------------------------------------------')
print('               EJECUCIÓN FINALIZADA')
print('----------------------------------------------------')
print('                valor encontrado x = '+str((Xu+Xl)/2))
print('                valor encontrado f(x) = ' + str(evalFun(((Xu+Xl)/2),func)))
print('                iteraciones = '+str(iterActual))
print('----------------------------------------------------')
non=raw_input('(Presione Enter )')
