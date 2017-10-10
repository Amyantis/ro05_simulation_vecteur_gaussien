import numpy as np

# m mean
m = [1, 1, 1]
# K covariance
K = [[2, 1, 1], [1, 3, -1], [1, -1, 2]]

# Y ~ N3(m, K)
Y = np.random.multivariate_normal(m, K)
print("Y:", Y.shape, Y, sep="\n")

# We'll try to generate a new Y (gauss d-dimensional)
# without np.random.multivariate_normal

# 1 - use Box-Muller Method to generate a gauss vector Z ~ N3(0, I3)
U1 = np.random.uniform(low=1e-10, high=1, size=3)
U2 = np.random.uniform(low=1e-10, high=1, size=3)

Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
Z1 = np.sqrt(-2 * np.log(U2)) * np.sin(2 * np.pi * U1)

# Z0 ~ N3(0, I3) and Z1 ~ N3(0, I3)
print("Z0", Z0, "Z1", Z1, sep="\n")

Z = Z0

# 2 using Cholesky refactor K with AAt
A = np.linalg.cholesky(K)
# check that decomposition is correct
K_from_decomposition = np.dot(A, A.T.conj())
np.testing.assert_almost_equal(K_from_decomposition, K)
print("A", A)

# 3. Donner une réalisation de ce vecteur Y ~ N3(m, K)
Yprime = np.dot(A, Z) + m
print("Y ~ N3(m, K)", Yprime, sep="\n")

# Bonus
import matplotlib.pyplot as plt

n = 100  # number of realisation of Y ~ N2(mean, cov)
mean = [-1, 2]
cov = [[1, 0], [0, 100]]

# step 1 - generate Z ~ N2 (0, I2)
Z = []
for _ in range(int(n / 2)):
    U1 = np.random.uniform(low=1e-10, high=1, size=2)
    U2 = np.random.uniform(low=1e-10, high=1, size=2)
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U2)) * np.sin(2 * np.pi * U1)
    Z.append(Z0)
    Z.append(Z1)

# step 2 - decompose cov as cov = AA*
A = np.linalg.cholesky(cov)

# step 3 - compute Y ~ N2(mean, cov)
Y = []
for Z_ in Z:
    Y.append(np.dot(A, Z_) + mean)
Y = np.matrix(Y)
plt.plot(Y[:, 0], Y[:, 1], linestyle="", marker="o")
plt.plot(mean[0], mean[1], color='red', marker="o")
plt.axis('equal')
plt.show()
