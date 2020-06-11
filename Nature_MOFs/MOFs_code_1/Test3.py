import multiprocessing
import math
def function(x):
    return print(math.pow(x,100))
cores=multiprocessing.cpu_count()
print(cores)
pool=multiprocessing.Pool(processes=cores)
for i in range(1000000000000):
    pool.apply_async(function(5))