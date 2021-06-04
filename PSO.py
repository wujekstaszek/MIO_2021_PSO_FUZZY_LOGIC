import numpy as np
import pyswarms as ps
import math as m


class hmm:
	def __init__(self,fis,words,result_word,dim):
		self.fis = fis
		self.words = words
		self.result_word = result_word
		self.dim=self.dim

	def fun(self,pos):
		anwser = []
		self.fis.compute()
		for pop in pos:
			#TODO: AKTUALIZACJA PARAMETRÓW

			i=0
			for word in words:
                inputs[word]=elem[i]
                i+=1
        	self.fis.inputs(inputs)
			anwser.append(self.fis.output[self.result_word])
		print(anwser)
		return  anwser

	def PSO_WLASCIWE(self):
		options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
		optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=self.dim, options=options)
		cost, pos = optimizer.optimize(self.fun, iters=1000)
		return cost,pos