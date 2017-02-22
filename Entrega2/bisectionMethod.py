#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from math import *
import random
import matplotlib.pyplot as plt
import numpy as np

class Bisection:
	
	portada="""
______ ________________ _______ _______________________________ _          _______ ________________        _______ ______  
(  ___ \|__   __(  ____ (  ____ (  ____ \__   __|__   __(  ___  | (    /|  (       |  ____ \__   __/\     /(  ___  |  __  \ 
| (   ) )  ) (  | (    \/ (    \/ (    \/  ) (     ) (  | (   ) |  \  ( |  | () () | (    \/  ) (  | )   ( | (   ) | (  \  )
| (__/ /   | |  | (_____| (__   | |        | |     | |  | |   | |   \ | |  | || || | (__      | |  | (___) | |   | | |   ) |
|  __ (    | |  (_____  )  __)  | |        | |     | |  | |   | | (\ \) |  | |(_)| |  __)     | |  |  ___  | |   | | |   | |
| (  \ \   | |        ) | (     | |        | |     | |  | |   | | | \   |  | |   | | (        | |  | (   ) | |   | | |   ) |
| )___) )__) (__/\____) | (____/\ (____/\  | |  ___) (__| (___) | )  \  |  | )   ( | (____/\  | |  | )   ( | (___) | (__/  )
|/ \___/\_______|_______|_______(_______/  )_(  \_______(_______)/    )_)  |/     \(_______/  )_(  |/     \(_______|______/ 

      Desarrollado por: John Tapias Zarrazola  
                                                                                                                 
								"""

	def __init__(self):
		self.nombre="biseccion"
	def imprimirPortada(self):
		print(self.portada)
	def calcular(self,pol,deriv1,error,iteraciones,A,B):
		itera=0
		a=A
		b=B
		mit = (a+b)/2
		termina=False
		azar=False
		resul = []
		#------------GRAFICAR
		#Dado el rango graficamos 200 puntos
		dt = ((abs(A-B))/200)
		t= np.arange(A, B, dt)
		#Polinomio
		y1=[]
		for i in range(0,200):
    			y1.append(pol.evalPol(A+(dt * i)))
			
		#Derivada del polinomio
		y2=[]
		for i in range(0,200):
    			y2.append(deriv1.evalPol(A+(dt * i)))

		plt.ion()
		
		
		while not termina:
			errorTemp=0
			azar = False
			itera=itera+1
			mit = (a+b)/2
			Ya=deriv1.evalPol(a)
			Yb=deriv1.evalPol(b)
			Ymit = deriv1.evalPol(mit)
			temp=[]
			temp.append(['Iteracion',itera])
			temp.append(['a',a,'f(a)',Ya])
			temp.append(['p',mit,'f(p)',Ymit])
			temp.append(['b',b,'f(b)',Yb])
			
			if Ymit ==0:
				print('Una raiz del polinomio se encuentra en x='+str(mit))
				termina=True
				errorTemp=0
				temp.append(['Error',errorTemp])
				
			elif Ya*Ymit >0:
				errorTemp = abs(a-mit)
				temp.append(['Error',errorTemp])
				if errorTemp<=error:
					termina=True					
				a=mit
			elif Yb*Ymit>0:
				errorTemp = abs(b-mit)
				temp.append(['Error',errorTemp])
				if errorTemp<=error:
					termina=True					
				b=mit
			else:
				azar= True
				mit=random.uniform(A,B)
				temp.append(['Error','Azar'])

			resul.append(temp)

			#Graficamos
			plt.figure(1)
			
			#Subplot polinomio
			Yap = pol.evalPol(a)
			Ybp = pol.evalPol(b)
			Ymitp = pol.evalPol(mit)
			plt.subplot(211)
			plt.title('Funcion')
    			plt.ylabel(pol.imprimir())
    			plt.xlabel('x')
			plt.plot(t, y1,'b--')
    			plt.plot(a, Yap,'rs')
    			plt.plot((a, a), ((-2)*Yap, 2*Yap), 'r-')
    			plt.plot(b, Ybp,'rs')
    			plt.plot((b, b), ((-2)*Ybp, 2*Ybp), 'r-')
    			plt.plot(mit, Ymitp,'g^')
    			plt.plot((mit, mit), ((-2)*Ymitp, 2*Ymitp), 'g-')	
			plt.grid(True)		
			#Subplot derivada
			plt.subplot(212)
			plt.title('Derivada')
    			plt.ylabel(deriv1.imprimir())
    			plt.xlabel('x')
			plt.plot(t, y2,'b--')
    			plt.plot(a, Ya,'rs')
    			plt.plot((a, a), ((-2)*Ya, 2*Ya), 'r-')
    			plt.plot(b, Yb,'rs')
    			plt.plot((b, b), ((-2)*Yb, 2*Yb), 'r-')
    			plt.plot(mit, Ymit,'g^')
    			plt.plot((mit, mit), ((-2)*Ymit, 2*Ymit), 'g-')	
			plt.grid(True)	

    			plt.show()
			plt.pause((20/iteraciones))

			if not azar:
				mit = (a+b)/2
			if itera>iteraciones:
				termina=True
			if not termina:
				plt.clf()

		print('\n----------------------------------------------------')
		print('-------------------HISTÓRICO------------------------')
		print('----------------------------------------------------')
		#print(resul)
		cadena=""
		print('Iteración #\t| a\t| p\t| b\t| f(a)\t| f(p)\t| f(b)\t| Error')
		for elemento in resul:
    			cadena = str(elemento[0][1])
			cadena = cadena + "\t| " + str(elemento[1][1])
			cadena = cadena + "\t| " + str(elemento[2][1])
			cadena = cadena + "\t| " + str(elemento[3][1])
			cadena = cadena + "\t| " + str(elemento[1][3])
			cadena = cadena + "\t| " + str(elemento[2][3])
			cadena = cadena + "\t| " + str(elemento[3][3])
			cadena = cadena + "\t| " + str(elemento[4][1])		
			print(cadena)	
		print('\n----------------------------------------------------')
		print('-------------------RESULTADO------------------------')
		print('----------------------------------------------------')
		print("Cero real aproximado: "+str(mit))
		print('----------------------------------------------------')


			
			
				
