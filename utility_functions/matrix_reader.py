import scipy.io
import scipy.sparse
import numpy as np
# Read the .mtx file
matrix = scipy.io.mmread(r"C:\\Users\Harry Hodgins\Documents\hpc_project\matrices\delaunay_n10\delaunay_n10\delaunay_n10.mtx")

dense_matrix = matrix.toarray()

rows,cols = dense_matrix.shape

header = np.array([[rows, cols]])

output_file = "delaunay_n10_1024.txt"

with open(output_file, "w") as f:
    # Write the dimensions in the first row
    f.write(f"{rows} {cols}\n")
    # Write the dense matrix data
    np.savetxt(f, dense_matrix, delimiter=" ",fmt="%.1f")

# selected_rows = dense_matrix[:, :16]

# # Get the dimensions of the selected matrix
# selected_rows_count, cols = selected_rows.shape

# Define the output file path
# output_file = "delaunay_n12_4096.txt"

# with open(output_file, "w") as f:
#     # Write the dimensions in the first row
#     f.write(f"{selected_rows_count} {cols}\n")
#     np.savetxt(f, selected_rows, delimiter=" ", fmt="%.1f")

print(dense_matrix)

# Create a random vector
vector = np.random.rand(dense_matrix.shape[1])

# Perform matrix-vector multiplication
result = dense_matrix.dot(vector)

# Print the result

import matplotlib.pyplot as plt

# Visualize the sparsity pattern of the matrix
plt.figure(figsize=(6, 6))
plt.spy(dense_matrix, markersize=1)
plt.title("SuiteSparse delaunay_n10 Matrix")
plt.show()


