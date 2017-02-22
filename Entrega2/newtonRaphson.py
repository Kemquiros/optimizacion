#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from math import *
import matplotlib.pyplot as plt
import numpy as np
import random

class NewtonRaphson:

	portada="""
 _       _______        ________________ _             _______ _______ _______         _______ _______ _
( (    /(  ____ \       \__   __(  ___  | (    /|     (  ____ |  ___  |  ____ )\     /(  ____ (  ___  | (    /|
|  \  ( | (    \/ )   ( |  ) (  | (   ) |  \  ( |     | (    )| (   ) | (    )| )   ( | (    \/ (   ) |  \  ( |
|   \ | | (__   | | _ | |  | |  | |   | |   \ | |_____| (____)| (___) | (____)| (___) | (_____| |   | |   \ | |
| (\ \) |  __)  | |( )| |  | |  | |   | | (\ \) (_____)     __)  ___  |  _____)  ___  (_____  ) |   | | (\ \) |
| | \   | (     | || || |  | |  | |   | | | \   |     | (\ (  | (   ) | (     | (   ) |     ) | |   | | | \   |
| )  \  | (____/\ () () |  | |  | (___) | )  \  |     | ) \ \_| )   ( | )     | )   ( /\____) | (___) | )  \  |
|/    )_|_______(_______)  )_(  (_______)/    )_)     |/   \__//     \|/      ||    | \_______|_______)/    )_)

             Desarrollado por: John Tapias Zarrazola

"""
	def __init__(self):
		self.nombre="newton-raphson"
	def imprimirPortada(self):
		print(self.portada)
	def calcular(self,pol,deriv1,deriv2,error,iteraciones,A,B):
		itera=0
		termina=False
		resul = []

		#------------GRAFICAR
		#Dado el rango graficamos 200 puntos
		dt = ((abs(A-B))/200)
		t= np.arange(A, B, dt)

		#Polinomio
		y=[]
		for i in range(0,200):
    			y.append(pol.evalPol(A+(dt * i)))

		#Derivada primera del polinomio
		y1=[]
		for i in range(0,200):
    			y1.append(deriv1.evalPol(A+(dt * i)))

		#Derivada segunda del polinomio
		y2=[]
		for i in range(0,200):
    			y2.append(deriv2.evalPol(A+(dt * i)))


		#Thread no bloqueante de Pyplot
		plt.ion()

		x0 = random.uniform(A,B)
		while not termina:
			temp=[]
			errorTemp=0
			itera=itera+1
			temp.append(['Iteracion',itera])
			#Escogemos un X0 al azar perteneciente al intervalo [A;B]			
			f0 = deriv1.evalPol(x0)#Se evalúa x0 en la primera derivada
			ff0 = deriv2.evalPol(x0)#Se evalúa x0 en la segunda derivada
			x1 = x0 - (f0/ff0)
			f1 = deriv1.evalPol(x1)
			temp.append(['x0',x0])
			temp.append(['x1',x1])
			temp.append(['f(x0)',f0])
			temp.append(['f(x1)',f1])
			temp.append(["f'(x0)",ff0])
			errorTemp = abs(x0-x1)
			temp.append(['Error',errorTemp])
			resul.append(temp)

			if errorTemp <= error:
				termina = True
			elif itera> iteraciones:
				termina=True

			#Graficamos
			plt.figure(1)

			#Cálculo de puntos
			Y0p = pol.evalPol(x0)
			Y1p = pol.evalPol(x1)

			Y0d1 = deriv1.evalPol(x0)
			Y1d1 = deriv1.evalPol(x1)

			Y0d2 = deriv2.evalPol(x0)
			Y1d2 = deriv2.evalPol(x1)

			#Subplot polinomio
			plt.subplot(221)
			plt.title('Funcion')
    			plt.ylabel(pol.imprimir())
    			plt.xlabel('x')
			plt.plot(t, y,'b--')
    			plt.plot(x0, Y0p,'rs')
    			plt.plot((x0, x0), ((-2)*Y0p, 2*Y0p), 'r-')
    			plt.plot(x1, Y1p,'g^')
    			plt.plot((x1, x1), ((-2)*Y1p, 2*Y1p), 'g-')
			plt.grid(True)
			#Subplot primera derivada
			plt.subplot(223)
			plt.title('Primera Derivada')
    			plt.ylabel(deriv1.imprimir())
    			plt.xlabel('x')
			plt.plot(t, y1,'b--')
    			plt.plot(x0, Y0d1,'rs')
    			plt.plot((x0, x0), ((-2)*Y0d1, 2*Y0d1), 'r-')
    			plt.plot(x1, Y1d1,'g^')
    			plt.plot((x1, x1), ((-2)*Y1d1, 2*Y1d1), 'g-')
			plt.grid(True)
			#Subplot segunda derivada
			plt.subplot(224)
			plt.title('Segunda Derivada')
    			plt.ylabel(deriv2.imprimir())
    			plt.xlabel('x')
			plt.plot(t, y2,'b--')
    			plt.plot(x0, Y0d2,'rs')
    			plt.plot((x0, x0), ((-2)*Y0d2, 2*Y0d2), 'r-')
    			plt.plot(x1, Y1d2,'g^')
    			plt.plot((x1, x1), ((-2)*Y1d2, 2*Y1d2), 'g-')
			plt.grid(True)

			plt.show()
			plt.pause((20/iteraciones))

			#Si no termina
			#Actualice el valor de x0
			if not termina:
				x0=x1
				plt.clf()

		print('\n----------------------------------------------------')
		print('-------------------HISTÓRICO------------------------')
		print('----------------------------------------------------')
		#print(resul)
		cadena=""
		print("Iteración #\t| x0\t| x1\t| f(x0)\t| f(x1)\t| f'(x0)\t| Error")
		for elemento in resul:
    			cadena = str(elemento[0][1])
			cadena = cadena + "\t| " + str(elemento[1][1])
			cadena = cadena + "\t| " + str(elemento[2][1])
			cadena = cadena + "\t| " + str(elemento[3][1])
			cadena = cadena + "\t| " + str(elemento[4][1])
			cadena = cadena + "\t| " + str(elemento[5][1])
			cadena = cadena + "\t| " + str(elemento[6][1])
			print(cadena)
		#Obtener el último X1
		ultimo = resul[len(resul)-1]
		respuesta = ultimo[2][1]
		print('\n----------------------------------------------------')
		print('-------------------RESULTADO------------------------')
		print('----------------------------------------------------')
		print("Cero real aproximado: "+str(respuesta))
		print('----------------------------------------------------')
