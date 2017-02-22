#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bisectionMethod import Bisection
from newtonRaphson import NewtonRaphson
from polynomial import Polinomio

flag=False
algoritmo=[]
tipoAlg=0
while(not flag):

	print('\n----------------------------------------------------')
	print('\n----   ENCONTRAR RAÍCES DE UNA FUNCIÓN   -----------')
	print('\n----------------------------------------------------')
	print('Ingrese el tipo de algoritmo:\n')
	print(' (1)  Método de Bisección')
	print(' (2)  Método de Newton-Raphson')
	print('----------------------------------------------------')
	tipoAlg = input('>>')
	if(tipoAlg==1):
		algoritmo = Bisection()
		flag=True
	elif(tipoAlg==2):
		algoritmo = NewtonRaphson()
		flag=True
	else:
		print('\n\nValor inválido ...(Presione Enter)\n')
		non=raw_input(' ')

algoritmo.imprimirPortada()
print('\n----------------------------------------------------')
print('           Ingrese número de iteraciones')
print('----------------------------------------------------')
numIter = input('>>')

print('\n----------------------------------------------------')
print('              Ingrese umbral de error')
print('----------------------------------------------------')
umbral = input('>>')

print('\n----------------------------------------------------')
print('Dado el dominio de búsqueda: a<=x<=b')
print('----------------------------------------------------')
a = input('Ingrese a:\n>>')
b = input('Ingrese b:\n>>')


flag=False
grado=0
while(not flag):

	print('\n----------------------------------------------------')
	print('Ingrese el grado del polinomio:\n')
	print('----------------------------------------------------')
	grado = input('>>')
	if(grado>0):
		flag=True
	else:
		print('\n\nValor inválido ...(Presione Enter)\n')
		non=raw_input(' ')

#---------Establecer polinomio
pol = Polinomio(grado)
pol.establecer()
print('\n----------------------------------------------------')
print('------------------POLINOMIO------------------------')
print('----------------------------------------------------')
print(pol.imprimir())
polDeriv1 = Polinomio(grado-1)
polDeriv1.setPolinomio(pol.derivar())
print('\n----------------------------------------------------')
print('---------------PRIMERA DERIVADA---------------------')
print('----------------------------------------------------')
print(polDeriv1.imprimir())

polDeriv2=""
if(algoritmo.nombre== "newton-raphson"):
	polDeriv2 = Polinomio(grado-2)
	polDeriv2.setPolinomio(polDeriv1.derivar())
	print('\n----------------------------------------------------')
	print('---------------SEGUNDA DERIVADA---------------------')
	print('----------------------------------------------------')
	print(polDeriv2.imprimir())
	algoritmo.calcular(pol,polDeriv1,polDeriv2,umbral,numIter,a,b)
else:
	algoritmo.calcular(pol,polDeriv1,umbral,numIter,a,b)

non=raw_input('(Presione Enter )')	

