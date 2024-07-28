import time
import numpy as np
import numexpr as ne

a= np.random.rand(10000000)
b= np.random.rand(10000000)

start = time.time()
result_np = a+b+np.sin(a)
end = time.time()

print('NumPy time: ', end-start)

########################

start = time.time()
result_ne =ne.evaluate('a * b + sin(a)')
end = time.time()

print('NumExpr time: ', end-start)
