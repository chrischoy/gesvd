import torch
from torch import nn
from torch.autograd import Function

import gesvd_cpp


class GESVDFunction(Function):

  @staticmethod
  def forward(ctx, input):
    U, S, V = gesvd_cpp.forward(input, True, True)
    ctx.save_for_backward(input, U, S, V)
    return U, S, V

  @staticmethod
  def backward(ctx, grad_u, grad_s, grad_v):
    A, U, S, V = ctx.saved_variables
    grad_A = gesvd_cpp.backward([grad_u, grad_s, grad_v], A, True, True, U, S, V)
    return grad_A


class GESVD(nn.Module):

  def __init__(self):
    nn.Module.__init__(self)
    self.svd = GESVDFunction()

  def forward(self, input):
    return self.svd.apply(input)
