import numpy as np
import pyswarms as ps
import math as m

def fun(pos):
	anwser = []
	for pop in pos:
		x = pop[0]
		y = pop[1]
		anwser.append(m.sin(x)+m.cos(y))
	print(anwser)
	return  anwser

if __name__ == "__main__":
	options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
	optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
	cost, pos = optimizer.optimize(fun, iters=1000)
	print(cost, pos)