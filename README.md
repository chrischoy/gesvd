# Pytorch SVD using LAPACK GESVD

## Background

Matrix decomposition is an expensive operation that can be approximated with various algorithms for speed [[1]](https://mathoverflow.net/questions/161252/what-is-the-time-complexity-of-truncated-svd). Some provide more numerically inaccurate results, but the LAPACK provides two numerically stable methods for SVD: 1) GESVD and 2) GESDD. The second method, GESDD, is faster and more scalable as it uses the divide-and-conquer method to decompose a large matrix, but it also introduces numerical errors that could be potentially devastating for applications that require precision.

Pytorch, (currently Feb 2020, v1.4), uses GESDD by default for SVD and currently there is no option to choose the GESVD backend. This package is simply a drop-in replacement for differentiable SVD with the GESVD backend if you prefer accuracy over speed.


## Installing the package

```
git clone https://github.com/chrischoy/gesvd.git
cd gesvd
python setup.py install
```

## Usage

```
from gesvd import GESVD
svd = GESVD()

# SVD
A = torch.randn(4, 5)
U, S, V = svd(A)

# Batched SVD
A = torch.randn(3, 4, 5)
U, S, V = svd(A)
```


## References

- [1]: Time complexity of SVD: [https://mathoverflow.net/questions/161252/what-is-the-time-complexity-of-truncated-svd](https://mathoverflow.net/questions/161252/what-is-the-time-complexity-of-truncated-svd)
- [2]: Discussion on GESDD vs. GESVD: [https://discourse.julialang.org/t/svd-better-default-to-gesvd-instead-of-gesdd/20603/17](https://discourse.julialang.org/t/svd-better-default-to-gesvd-instead-of-gesdd/20603/17)
