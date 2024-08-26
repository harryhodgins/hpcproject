import numpy
import matplotlib.pyplot as plt

problem_sizes = [512,1024,2048,4096,8192,16384]
num_procs = [1,2,4,8,16,32]

#start with time for 512 on 1 proc
#then 1024 on 2, 2048 on 4, 4096 on 8, 8192 on 16

#timings = [0.009153,0.018243,0.035703,0.079271,0.173805,0.685033]
timings = [0.002492,0.007830,0.011968,0.033846,0.055949,0.230160]
#timings = [0.004467,0.008739,0.015420,0.028996,0.058923,0.231640]

# Calculate normalized runtime
base_time = timings[0]
normalized_runtimes = [base_time / t for t in timings]

def line(x):
    res = []
    for i in x:
        res.append(1)
    return res
myaxis = [1,32]
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(num_procs, normalized_runtimes, marker='o', linestyle='-', color='blue',label='Measured')
plt.plot(num_procs,line(num_procs),color='black',linestyle='--',label='Ideal Scaling')
plt.xlabel('Number of Processes', fontsize=16)
plt.ylabel('Efficiency', fontsize=16)
plt.title('PA1 v2 Weak Scaling', fontsize=20)
plt.legend(loc = 'lower left',fontsize = 14)
plt.grid(True)
plt.show() 