# Pytorch SVD using LAPACK GESVD

## Background

Matrix decomposition is an expensive operation [1](https://mathoverflow.net/questions/161252/what-is-the-time-complexity-of-truncated-svd) which results in various algorithms that can speed up the process.
LAPACK provides two methods for SVD: 1) GESVD and 2) GESDD. The second method, GESDD, is faster and more scalable as it uses divide and conquer to decompose a large matrix. However, it also introduces from numerical instability which could be potentially devastating for some applications that require precision.

Pytorch, (currently Feb 2020, v1.4), uses gesdd by default for SVD and currently there is no option to use GESVD.

## Installing the package

```
git clone https://github.com/chrischoy/gesvd.git
cd gesvd
python setup.py install
```

## Usage

```
from gesvd import GESVDFunction
svd = GESVDFunction()

# SVD
A = torch.randn(4, 5)
U, S, V = svd.apply(A)

# Batched SVD
A = torch.randn(3, 4, 5)
U, S, V = svd.apply(A)
```


## References

- [1]: Time complexity of SVD: [https://mathoverflow.net/questions/161252/what-is-the-time-complexity-of-truncated-svd](https://mathoverflow.net/questions/161252/what-is-the-time-complexity-of-truncated-svd)
- [2]: Discussion on GESDD vs. GESVD: [https://discourse.julialang.org/t/svd-better-default-to-gesvd-instead-of-gesdd/20603/17](https://discourse.julialang.org/t/svd-better-default-to-gesvd-instead-of-gesdd/20603/17)
