import numpy as np
import pyswarms as ps
import math as m


class hmm:
	def __init__(self,dataset,function):
		self.dataset = dataset
		self.function = function
	def fun(self,pos):
		anwser = []
		for pop in pos:
			anwser.append(self.function(self.dataset,*pop))
		print(anwser)
		return  anwser

if __name__ == "__main__":
	function = lambda dataset,x,y:m.sin(x)+m.cos(y)
	test = hmm(0,function)
	options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
	optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
	cost, pos = optimizer.optimize(test.fun, iters=1000)
	print(cost, pos)