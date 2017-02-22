#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

class Polinomio:
	
	def __init__(self,grado_):
		self.grado=grado_
		self.pol = []

	def establecer(self):
		for i in range(self.grado,0,-1):
			valor = input('>> Ingrese el coeficiente del término con grado '+str(i)+" :  ")
			self.pol.append([valor,i])
		valor = input(">> Ingrese el término independiente :  ")
		self.pol.append([valor,0])

	def setPolinomio(self,pol_):
		self.pol = pol_
	
	def derivar(self):		
		polDeriv=[]
		for termino in self.pol:
			#termino[0] coeficiente
			#termino[1] exponente
			if termino[1] > 0:
				nuevoCoef = termino[0]*termino[1]
				nuevoExp = termino[1]-1
				polDeriv.append([nuevoCoef,nuevoExp])
		return polDeriv

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

		
			
