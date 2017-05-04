#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math
import random


class Polinomio:

	def __init__(self,grado_):
		self.grado=grado_
		self.pol = []
		self.polD =[]

	def establecer(self):
		for i in range(self.grado,0,-1):

			self.pol.append([random.uniform(-10,10),i])

		self.pol.append([random.uniform(-10,10),0])

	#Esta funcion se utiliza para invertir
	#el signo de los coeficientes
	def invertirPolinomio(self):
		polInv = []
		for t in self.pol:
			polInv.append([(-1)*t[0],t[1]])
		return polInv

	def setPolinomio(self,pol_):
		self.pol = pol_

	def derivar(self):
		for termino in self.pol:
			#termino[0] coeficiente
			#termino[1] exponente
			if termino[1] > 0:
				nuevoCoef = termino[0]*termino[1]
				nuevoExp = termino[1]-1
				self.polD.append([nuevoCoef,nuevoExp])


	def imprimir(self):
		primero=True
		cadena=""
		for termino in self.pol:
			if termino[0] != 0:
				cadena=cadena+" "
				if termino[0] > 0 and primero==False: #Al primer término no se le imprime +
					cadena= cadena+"+"
				cadena= cadena + str(termino[0])
				if termino[1] != 0:#Al término independiente no se le imprime parte literal
					cadena=cadena+ "X^"+str(termino[1])
			primero=False
		return cadena


	def evalPol(self,x):
		resul = 0
		for termino in self.pol:
			resul = resul+ (termino[0] * pow(x,termino[1]) )
		return resul

	def evalPolD(self,x):
		resul = 0
		for termino in self.polD:
			resul = resul+ (termino[0] * pow(x,termino[1]) )
		return resul
