import math


def PSO(params):
    # Dekodowanie Parametrów
    fun = params("function")
    pop_quantity = params("population")
    c1 = params("c1")
    c2 = params("c2")
    a = params("a")
    b = params("b")
    iter_quantity = params("iterations")
    dim = params("dimensions");
    Clerc = params("clerc")
    # Inicjalizacja Populacji
    pops = zeros(pop_quantity,dim)
    for i in range(dim):
        random_pops = rand(pop_quantity,1)
        random_pops = random_pops*(b(i)-a(i))+a(i)
        pops(i)=random_pops
    velocities = rand(pop_quantity,dim)
    #Iteracje
    gbest = pops(1)
    pbest = pops
    for i in range(iter_quantity):
        for j in range(pop_quantity):
            if fun(pbest(j,:)) >= fun(pops(j,:))
                pbest(j,:) = pops(j,:)
                if fun(pbest(j,:)) > fun(gbest):
                    gbest = pbest(j,:)
         velocities(j,:) = velocities(j,:) + rand(1,dim)*c1*(pbest(j,:)-pops(j,:)) +  rand(1,dim)*c2*(gbest-pops(j,:))
         pops(j,:) = pops(j,:) + velocities(j,:)

    result = [gbest,fun(gbest)]
    return result

















pop=10
c1=2
c2=2
a=[-10,-10]
b=[10,10]
fun = lambda x, y: x*x+y*y-20*(cos(x*math.pi)+cos(y*math.pi)-2)
iterations = 2000
 
keys=["population","c1","c2","a","b","function","iterations","clerc","dimensions"]
values={pop,c1,c2,a,b,fun,iterations,0,2}

options = containers.Map(keys,values)

print(options)
PSO(options)
