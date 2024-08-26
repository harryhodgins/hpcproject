import numpy as np
import scipy.io

# Define the input and output file paths
input_txt_file = "C:\\Users\Harry Hodgins\Documents\Python Scripts\dwt_512.txt"
output_mat_file = "dwt_512.mat"

# Read the text file
with open(input_txt_file, "r") as f:
    # Read the dimensions from the first line
    first_line = f.readline().strip()
    rows, cols = map(int, first_line.split())
    
    # Read the matrix data
    matrix_data = np.loadtxt(f, delimiter=" ")

# Verify that the read dimensions match the actual data
assert matrix_data.shape == (rows, cols), "Dimensions do not match the actual matrix data"

# Save the matrix data to a .mat file
scipy.io.savemat(output_mat_file, {'matrix': matrix_data})

print(f"Matrix data has been saved to {output_mat_file}")
