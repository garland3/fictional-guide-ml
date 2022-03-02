from p3_optimization import *
import time

tic = time.perf_counter()
n = 300
update_fre = 20
for i in range(n):
    xpoints, ypoints = get_data(False)
    f = make_inter_function(xpoints,ypoints)
    x, xs = find_a_min_grad(f)
    x, xs = find_a_min_momentum(f)
    x, xs = find_a_min_rmsp(f)
    x, xs = find_a_min_adam(f)
    if i % update_fre == 0:
        print("--"*20)
        print(f"-----------On iteration {i}  -----------------")
        print("--"*20)

toc = time.perf_counter()
print(f"total time is {toc - tic}")
