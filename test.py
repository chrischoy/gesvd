import torch
from gesvd import GESVDFunction
import numpy as np

R = np.array([[[0.41727819, -0.87345426, 0.25091147],
              [0.32246181, 0.40043949, 0.85771009],
              [-0.84964539, -0.27699435, 0.44875031]],
             [[0.41727819, -0.87345426, 0.25091147],
              [0.32246181, 0.40043949, 0.85771009],
              [-0.84964539, -0.27699435, 0.44875031]]])
thR = torch.from_numpy(R)
thR.requires_grad_()

print(thR)

U, S, V = np.linalg.svd(R[0])
svd = GESVDFunction()
# Batch SVD
thU, thS, thV = svd.apply(thR)
(thS).sum().backward()
print(thR.grad)

print(np.allclose(U, thU[0].detach()))
print(np.allclose(V, thV[0].detach().t()))
print(np.allclose(S, thS[0].detach()))
print(torch.allclose(torch.matmul(thU[0], thV[0].t()), thR[0]))
