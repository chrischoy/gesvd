#include <torch/extension.h>

#include <iostream>
#include <vector>

#define PRIVATE_CASE_TYPE(enum_type, type, ...)                                \
  case enum_type: {                                                            \
    using scalar_t = type;                                                     \
    return __VA_ARGS__();                                                      \
  }

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                               \
  [&] {                                                                        \
    const auto &the_type = TYPE;                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                      \
    switch (_st) {                                                             \
      PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)           \
      PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)             \
    default:                                                                   \
      AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");           \
    }                                                                          \
  }()

extern "C" void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a,
                        int *lda, double *s, double *u, int *ldu, double *vt,
                        int *ldvt, double *work, int *lwork, int *info);
extern "C" void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a,
                        int *lda, float *s, float *u, int *ldu, float *vt,
                        int *ldvt, float *work, int *lwork, int *info);

/*
 * Given a vector of int64_t infos, obtained after a batch operations,
 * this function checks if the computation over all these batches has been
 * successful (info = 0) or not, and report in case of the latter.
 */
static inline void batchCheckErrors(std::vector<int64_t> &infos,
                                    const char *name,
                                    bool allow_singular = false) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info,
               " has illegal value");
    } else if (info > 0) {
      if (strstr(name, "svd")) {
        AT_ERROR(name,
                 ": the updating process of SBDSDC did not converge (error: ",
                 info, ")");
      } else if (strstr(name, "symeig")) {
        AT_ERROR(name, ": For batch ", i,
                 ": the algorithm failed to converge; ", info,
                 " off-diagonal elements of an intermediate tridiagonal form "
                 "did not converge to zero.");
      } else if (!allow_singular) {
        AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info,
                 ") is zero, singular U.");
      }
    }
  }
}

/*
 * This is an overloaded case of the previous function for a tensor of infos.
 */
static inline void batchCheckErrors(const at::Tensor &infos, const char *name,
                                    bool allow_singular = false) {
  auto batch_size = infos.numel();
  auto infos_cpu = infos.to(at::kCPU);
  auto infos_data = infos_cpu.data_ptr<int>();
  for (int64_t i = 0; i < batch_size; i++) {
    auto info = infos_data[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info,
               " has illegal value");
    } else if (!allow_singular && info > 0) {
      AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info,
               ") is zero, singular U.");
    }
  }
}

/*
 * Given a info int, obtained after a single operation, this function check if
 * the computation
 * has been successful (info = 0) or not, and report in case of the latter.
 */
static inline void singleCheckErrors(int64_t info, const char *name,
                                     bool allow_singular = false) {
  if (info < 0) {
    AT_ERROR(name, ": Argument ", -info, " has illegal value");
  } else if (info > 0) {
    if (strstr(name, "svd")) {
      AT_ERROR(name,
               ": the updating process of SBDSDC did not converge (error: ",
               info, ")");
    } else if (strstr(name, "symeig")) {
      AT_ERROR(name, ": the algorithm failed to converge; ", info,
               " off-diagonal elements of an intermediate tridiagonal form did "
               "not converge to zero.");
    } else if (!allow_singular) {
      AT_ERROR(name, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

/*
 * Given batches of matrices with arbitrary batch dim,
 * computes the number of batches.
 */
static inline int64_t batchCount(const at::Tensor &batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

// Computes the number of elements of a matrix in a batched matrix tensor
static inline int64_t matrixStride(const at::Tensor &batched_matrices) {
  return batched_matrices.size(-1) * batched_matrices.size(-2);
}

/*
 * Clones a Tensor so that the following conditions hold:
 * If we think of a Tensor of having size (B, M, N), where B is any number
 * of batch dimensions, then:
 * - Each (M, N) matrix is in column major form
 * - Let Tensor P have size (B, M, N) and Q have size (B, M', N').
 *   Then when laid out in memory, the M by N matrix starting at
 *   P.data_ptr()[b * M * N] is of the same corresponding batch as the M' by N'
 *   matrix starting at Q.data_ptr()[b * M' * N'].
 */
static inline at::Tensor cloneBatchedColumnMajor(const at::Tensor &src) {
  // If src is already in batched column major format, then
  // this will be efficient (no reordering of the data will occur)
  // because the first transpose will make the tensor contiguous,
  // and cloning a contiguous tensor is fast.
  auto result = src.transpose(-2, -1).clone().contiguous();
  result.transpose_(-2, -1);
  return result;
}

template <class scalar_t, class value_t = scalar_t>
void lapackSvd(char jobu, char jobvt, int m, int n, scalar_t *a, int lda,
               value_t *s, scalar_t *u, int ldu, scalar_t *vt, int ldvt,
               scalar_t *work, int lwork, int *info);

template <>
void lapackSvd<double>(char jobu, char jobvt, int m, int n, double *a, int lda,
                       double *s, double *u, int ldu, double *vt, int ldvt,
                       double *work, int lwork, int *info) {
  dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
          info);
}

template <>
void lapackSvd<float>(char jobu, char jobvt, int m, int n, float *a, int lda,
                      float *s, float *u, int ldu, float *vt, int ldvt,
                      float *work, int lwork, int *info) {
  sgesvd_(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
          info);
}

// Function to generate empty tensors of required size, strides and dtype for
// the SVD operation
static inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
_create_U_S_VT(const at::Tensor &input, bool some, bool compute_uv) {
  auto sizes = input.sizes().vec();
  int64_t m = input.size(-2), n = input.size(-1);

  sizes[input.dim() - 1] = (compute_uv && some) ? std::min(m, n) : m;
  auto strides = at::detail::defaultStrides(sizes);
  // U should be a column-major or a batch of column-major matrices
  // ... x m x ucol will have strides: ...., ucol, 1
  // We require: ...., 1, m
  strides[input.dim() - 1] = m;
  strides[input.dim() - 2] = 1;

  at::Tensor U_empty;
  if (!input.is_cuda()) {
    U_empty = at::empty_strided(sizes, strides, input.options());
  } else {
    // NB: U_empty is an empty tensor created on the CPU intentionally, because
    // magma_(d/s)gesdd
    // (which is the driver routine for the divide and conquer SVD operation)
    // takes in arrays on the CPU as input. This routine is a hybrid CPU-GPU
    // routine that
    // moves the inputs between devices internally.
    U_empty =
        at::empty_strided(sizes, strides, input.options().device(at::kCPU));
  }

  sizes[input.dim() - 2] = n;
  sizes[input.dim() - 1] = n;
  // VT should be a row-major or a batch of row-major matrices
  at::Tensor VT_empty;
  if (!input.is_cuda()) {
    VT_empty = at::empty(sizes, input.options());
  } else {
    // NB: VT_empty is an empty tensor created on the CPU intentionally, because
    // magma_(d/s)gesdd
    // (which is the driver routine for the divide and conquer SVD operation)
    // takes in arrays on the CPU as input. This routine is a hybrid CPU-GPU
    // routine that
    // moves the inputs between devices internally.
    VT_empty = at::empty(sizes, input.options().device(at::kCPU));
  }

  sizes.pop_back();
  sizes[input.dim() - 2] = std::min(m, n);
  at::Tensor S_empty;
  at::ScalarType dtype = at::typeMetaToScalarType(input.dtype());
  if (!input.is_cuda()) {
    S_empty = at::empty(sizes, input.options().dtype(dtype));
  } else {
    // NB: S_empty is an empty tensor created on the CPU intentionally, because
    // magma_(d/s)gesdd
    // (which is the driver routine for the divide and conquer SVD operation)
    // takes in arrays on the CPU as input. This routine is a hybrid CPU-GPU
    // routine that
    // moves the inputs between devices internally.
    S_empty = at::empty(sizes, input.options().dtype(dtype).device(at::kCPU));
  }
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(U_empty, S_empty,
                                                        VT_empty);
}

template <typename scalar_t>
static void apply_svd(at::Tensor &self, at::Tensor &U, at::Tensor &S,
                      at::Tensor &VT, char job, std::vector<int64_t> &infos) {
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<scalar_t>();
  auto VT_data = VT.data_ptr<scalar_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = matrixStride(U);
  auto S_stride = S.size(-1);
  auto VT_stride = matrixStride(VT);
  auto batchsize = batchCount(self);

  int info;
  auto m = self.size(-2);
  auto n = self.size(-1);

  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this
  // outside
  // the loop saves (batch_size - 1) workspace queries which would provide the
  // same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using
  // at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackSvd<scalar_t>(job, job, m, n, self_data, m, S_data, U_data, m, VT_data,
                      n, &wkopt, lwork, &info);
  lwork = wkopt;
  at::Tensor work = at::empty({lwork}, self.options());
  auto work_data = work.data_ptr<scalar_t>();

  for (int64_t i = 0; i < batchsize; i++) {
    scalar_t *self_working_ptr = &self_data[i * self_stride];
    scalar_t *S_working_ptr = &S_data[i * S_stride];
    scalar_t *U_working_ptr = &U_data[i * U_stride];
    scalar_t *VT_working_ptr = &VT_data[i * VT_stride];

    // Compute S, U (optionally) and VT (optionally)
    lapackSvd<scalar_t>(job, job, m, n, self_working_ptr, m, S_working_ptr,
                        U_working_ptr, m, VT_working_ptr, n, work_data, lwork,
                        &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ge_svd_helper(const at::Tensor &self, bool some, bool compute_uv) {
  std::vector<int64_t> infos(batchCount(self), 0);
  int64_t m = self.size(-2), n = self.size(-1);
  int64_t k = std::min(m, n);

  char job = compute_uv ? (some ? 'S' : 'A') : 'N';

  at::Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) =
      _create_U_S_VT(self, some, compute_uv);

  if (self.numel() > 0) {
    auto self_working_copy = cloneBatchedColumnMajor(self);

    DISPATCH_FLOATING_TYPES(self.scalar_type(), "svd_cpu", [&] {
      apply_svd<scalar_t>(self_working_copy, U_working_copy, S_working_copy,
                          VT_working_copy, job, infos);
    });

    if (self.dim() > 2) {
      batchCheckErrors(infos, "svd_cpu");
    } else {
      singleCheckErrors(infos[0], "svd_cpu");
    }

    if (compute_uv) {
      if (some) {
        VT_working_copy = VT_working_copy.narrow(-1, 0, k);
      }
    } else {
      VT_working_copy.zero_();
      U_working_copy.zero_();
    }
  } else {
    U_working_copy.zero_();
    VT_working_copy.zero_();
  }
  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
svd_forward(const at::Tensor &self, bool some = true, bool compute_uv = true) {
  TORCH_CHECK(self.dim() >= 2,
              "self should have at least 2 dimensions, but has ", self.dim(),
              " dimensions instead");
  return ge_svd_helper(self, some, compute_uv);
}

at::Tensor svd_backward(const std::vector<torch::autograd::Variable> &grads,
                        const at::Tensor &self, bool some, bool compute_uv,
                        const at::Tensor &raw_u, const at::Tensor &sigma,
                        const at::Tensor &raw_v) {
  TORCH_CHECK(compute_uv, "svd_backward: Setting compute_uv to false in "
                          "torch.svd doesn't compute singular matrices, ",
              "and hence we cannot compute backward. Please use "
              "torch.svd(compute_uv=True)");

  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = sigma.size(-1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(-1, 0, k);
    v = raw_v.narrow(-1, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(-1, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(-1, 0, k);
    }
  }
  auto vt = v.transpose(-2, -1);

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    sigma_term = at::matmul(
        u, at::matmul(gsigma.diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1),
                      vt));
  } else {
    sigma_term = at::zeros_like(self).contiguous();
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto ut = u.transpose(-2, -1);
  auto im = at::eye(m, self.options());
  auto in = at::eye(n, self.options());
  auto sigma_mat = sigma.diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  auto sigma_mat_inv =
      sigma.pow(-1).diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  auto sigma_sq = sigma.pow(2);
  auto F = sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(INFINITY);
  F = F.pow(-1);

  at::Tensor u_term, v_term;

  if (gu.defined()) {
    u_term =
        at::matmul(u, at::matmul(F.mul(at::matmul(ut, gu) -
                                       at::matmul(gu.transpose(-2, -1), u)),
                                 sigma_mat));
    if (m > k) {
      u_term = u_term + at::matmul(im - at::matmul(u, ut),
                                   at::matmul(gu, sigma_mat_inv));
    }
    u_term = at::matmul(u_term, vt);
  } else {
    u_term = at::zeros_like(self).contiguous();
  }

  if (gv.defined()) {
    auto gvt = gv.transpose(-2, -1);
    v_term = at::matmul(
        sigma_mat,
        at::matmul(F.mul(at::matmul(vt, gv) - at::matmul(gvt, v)), vt));
    if (n > k) {
      v_term = v_term + at::matmul(sigma_mat_inv,
                                   at::matmul(gvt, in - at::matmul(v, vt)));
    }
    v_term = at::matmul(u, v_term);
  } else {
    v_term = at::zeros_like(self).contiguous();
  }

  return u_term + sigma_term + v_term;
}

template static void apply_svd<float>(at::Tensor &self, at::Tensor &U,
                                      at::Tensor &S, at::Tensor &VT, char job,
                                      std::vector<int64_t> &infos);
template static void apply_svd<double>(at::Tensor &self, at::Tensor &U,
                                       at::Tensor &S, at::Tensor &VT, char job,
                                       std::vector<int64_t> &infos);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &svd_forward, "svd forward");
  m.def("backward", &svd_backward, "svd backward");
}
