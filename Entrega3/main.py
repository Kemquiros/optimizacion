#!/usr/bin/env python
# -*- coding: utf-8 -*
import random
from polynomial import Polinomio
from gd1deterministico import gd1
#from aleatorio import aleat


#Imprimir gradiente descendete
#
#
#Imprimir gradiente descendete

#Adaptativo del polinomio
#Aleatorio
#Conjugado
resultado={}
corridasTotales=30
flag=False
isMax=True
numIter=0
error=0
a=0
b=0


"""
print('\n----------------------------------------------------')
print('           Ingrese número de iteraciones')
print('----------------------------------------------------')
numIter = input('>>')
"""
numIter=40
"""
print('\n----------------------------------------------------')
print('              Ingrese umbral de error')
print('----------------------------------------------------')
error = input('>>')
"""
error=0.0001
"""
flag=False
while(not flag):
    print('\n----------------------------------------------------')
    print('Dado el dominio de búsqueda: a<=x<=b')
    print('----------------------------------------------------')
    a = input('Ingrese a:\n>>')
    b = input('Ingrese b:\n>>')
    if a<b:
        flag=True
    else:
        print('\n\nValor inválido: a debe ser mayor que b ...(Presione Enter)\n')
        non=raw_input(' ')
"""
a=-100
b=100

resul={}

#---------Establecer polinomio
#Se requieren 5 polinomio
#3 de grado 3
#2 de grado 4
polinomios=[]
for i in range(5):
    if(i>3):
        grado=2
    else:
        grado=3
    pol = Polinomio(grado)
    pol.establecer()
    #Derivamos
    pol.derivar()
    #Almacenamos ambos polinomios
    polinomios.append(pol)
"""
if(isMin):
    pol = pol.invertirPolinomio()
    polDeriv1 = Polinomio(grado-1)
    polDeriv1.setPolinomio(pol.derivar())
"""
"""
    Gradiente descendete con alfa fijo
"""
#Se genera un alfa Aleatorio fijo
#alfa= random.uniform(0,1)
alfa=0.02
print "Alfa: "+str(alfa)
for pol in polinomios:
    print pol.pol
    #Se genera un vector inicial X0
    #perteneciente al intervalo
    X0 =random.uniform(a,b)

    #Polinomio a maximizar
    #Número de iteraciones
    #Umbral de error
    #a: límite inferior del intervalo de búsqueda
    #b: límite superior del intervalo de búsqueda
    #alfa
    algoritmo = gd1()
    blabla= algoritmo.calcular(pol,numIter,error,a,b,alfa,X0)
    print blabla
