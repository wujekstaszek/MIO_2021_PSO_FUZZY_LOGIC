import numpy as np
import pyswarms as ps
import math as m
from skfuzzy import control as ctrl
import skfuzzy as fuzz

class hmm:
	def __init__(self,dataset,input_vars,fis,words,result_word,dim,consq_number):
		self.fis = fis
		self.words = words
		self.result_word = result_word
		self.dim=dim
		self.dataset = dataset
		self.input_vars = input_vars
		self.consq_number = consq_number
		return PSO_WLASCIWE()

	def fun(self,pos):
		anwser = []
		for pop in pos:
			#TODO: AKTUALIZACJA PARAMETRÓW
			for i in range(self.consq_number):
				temp = fuzz.trimf(iris_class.universe, [pop[3*i], pop[3*i+1], pop[3*i+2]]) #NIE WIEM CZY TO DZIALA!!!!!
				ctrl.fis.Consequent = temp
			sim = ctrl.ControlSystemSimulation(self.fis, flush_after_run=len(data))
			sim.compute()
			for elem in dataset:
				i=0
				for word in words:
	                inputs[word]=elem[i]
	                i+=1	
        		sim.inputs(inputs)
        		results.append(sim.output[self.result_word])
        		#TODO: PORÓWNANIE

			anwser.append(result_w_procentach)
		print(anwser)
		return  anwser

	def PSO_WLASCIWE(self):
		options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
		optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=self.consq_number*3, options=options)
		cost, pos = optimizer.optimize(self.fun, iters=1000)
		return cost,pos