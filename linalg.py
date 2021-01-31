import scipy.linalg as linalg
import numpy as np


## 矩阵求逆

A = np.array([[1,3,5],[2,5,1],[2,3,8]])
A_inv = linalg.inv(A)
A_inv


## 求解线性方程组

A = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])
sol = linalg.inv(A).dot(b)
A.dot(sol) - b  # 验证

## 计算行列式

A = np.array([[1,2],[3,4]])
A_det = linalg.det(A)

## 计算范数

A = np.array([[1, 2],[3, 4]])
linalg.norm(A)
linalg.norm(A,'fro') # frobenius norm is the default
linalg.norm(A,1) # L1 norm (max column sum)
linalg.norm(A,-1)
linalg.norm(A,np.inf) # L inf norm (max row sum)

## 求解线性最小二乘问题

c1, c2 = 5.0, 2.0
i = np.r_[1:11]
xi = 0.1*i
yi = c1*np.exp(-xi) + c2*xi
zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi)) # 围绕标准函数做随机点
A = np.c_[np.exp(-xi)[:, np.newaxis], xi[:, np.newaxis]] # 增加维度合成两列作为两个自变量取值
c, resid, rank, sigma = linalg.lstsq(A, zi) # 计算方程 ax = b 的最小二乘解

# resid: Square of the 2-norm for each column in b - a
# sigma: Singular values of a

## 特征值分解

A = np.array([[1, 2], [3, 4]])
la, v = linalg.eig(A)
l1, l2 = la
print(l1, l2)   # eigenvalues
print(v[:, 0])   # first eigenvector
print(v[:, 1])   # second eigenvector

## 奇异值分解

A = np.array([[1,2,3],[4,5,6]])

M,N = A.shape
U,s,Vh = linalg.svd(A)
Sig = linalg.diagsvd(s,M,N) #获取矩阵
Sig
Vh
U.dot(Sig.dot(Vh)) #check computation

## LU分解

# # LU分解

A = np.arange(1, 17).reshape(4, 4)
p, l, u = linalg.lu(a=A, permute_l=False, overwrite_a=False, check_finite=True)

print('原矩阵\n', A)
print('p矩阵\n', p)
print('l矩阵\n', l)
print('u矩阵\n', u)




