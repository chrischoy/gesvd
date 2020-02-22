import torch
import gesvd
import numpy as np

R = np.array([[0.41727819, -0.87345426, 0.25091147],
              [0.32246181, 0.40043949, 0.85771009],
              [-0.84964539, -0.27699435, 0.44875031]])
thR = torch.from_numpy(R)

U, S, V = np.linalg.svd(R)
thU, thS, thV = gesvd.forward(thR, True, True)

print(np.allclose(U, thU))
print(np.allclose(V, thV.t()))
print(np.allclose(S, thS))
print(torch.allclose(torch.matmul(thU, thV.t()), thR))
