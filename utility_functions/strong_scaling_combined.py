import pandas as pd
import matplotlib.pyplot as plt

def read_and_process_data(filename):
    with open(filename, 'r') as file:
        first_line = file.readline()
        num_vectors, matrix_dimension = map(int, first_line.split())

    data = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=['Processes', 'Runtime'])
    sequential_runtime = data[data['Processes'] == min(data['Processes'])]['Runtime'].iloc[0]

    # Calculate speedup
    data['Speedup'] = sequential_runtime / data['Runtime']
    data['IdealSpeedup'] = data['Processes']
    data['Matrix Dimension'] = matrix_dimension

    return data


def plot_combined_speedup(file_list):
    plt.figure(figsize=(12, 8))

    for idx, filename in enumerate(file_list):
        data = read_and_process_data(filename)
        plt.plot(data['Processes'], data['Speedup'],marker='o',
                 linestyle='-',
                 label=f'n = {data["Matrix Dimension"].iloc[0]}')

        if idx == 0:
            plt.plot(data['Processes'], data['IdealSpeedup'], linestyle='--', color='k', label='Ideal Speedup')

    plt.xlabel('Number of Processes', fontsize=16)
    plt.ylabel('Speedup', fontsize=16)
    plt.title('PA1 v1 Speedup Plots for Various Problem Sizes', fontsize=22)
    plt.grid(True)
    plt.xlim(0,33)
    plt.legend(fontsize=16,loc='upper left')

def plot_combined_runtime(file_list):
    plt.figure(figsize=(12, 8))

    for filename in file_list:
        data = read_and_process_data(filename)
        plt.plot(data['Processes'], data['Runtime'], marker='o',
                 linestyle='-',
                 label=f'n = {data["Matrix Dimension"].iloc[0]}')

    plt.xlabel('Number of Processes', fontsize=16)
    plt.ylabel('Runtime (s)', fontsize=16)
    plt.title('PA1 v1 Runtime Plots for Various Problem Sizes', fontsize=22)
    plt.grid(True)
    plt.yscale('log')
    plt.xlim(0, 33)
    plt.legend(fontsize=16, loc='upper right')
    plt.show()
    
file_list = [
    'pa1v1_512.txt',
    'pa1v1_1024.txt',
    'pa1v1_2048.txt',
    'pa1v1_4096.txt',
    'pa1v1_8192.txt',
    'pa1v1_16384.txt'
]

plot_combined_speedup(file_list)
plot_combined_runtime(file_list)


n_vals = [1, 2, 4, 8, 16, 32,64]
s_vals = [1, 0.9995,0.995,0.99,0.95, 0.8, 0.5, 0.1]

def amdahl(p, n):
    return 1 / ((1 - p) + p / n)
plt.figure(figsize=(12, 8))
for p in s_vals:
    speedups = [amdahl(p, n) for n in n_vals]
    
    plt.plot(n_vals, speedups, label=f'p= {p}')

plt.xlabel('Number of Processors (n)',fontsize=16)
plt.ylabel('Speedup',fontsize=16)
plt.title("Amdahl's Law",fontsize=22)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()
