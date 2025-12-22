import torch
import soft_rank_numpy


def wrap_class(cls, **kwargs):

    class NumpyOpWrapper(torch.autograd.Function):

        @staticmethod
        def forward(ctx, values):
            obj = cls(values.detach().numpy(), **kwargs)
            ctx.numpy_obj = obj
            return torch.from_numpy(obj.compute())

        @staticmethod
        def backward(ctx, grad_output):
            return torch.from_numpy(ctx.numpy_obj.vjp(grad_output.numpy()))

    return NumpyOpWrapper


def map_tensor(map_fn, tensor):
    return torch.stack([map_fn(tensor_i) for tensor_i in torch.unbind(tensor)])


def soft_rank(values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"):
    if len(values.shape) != 2:
        raise ValueError("'values' should be a 2d-tensor " "but got %r." % values.shape)

    wrapped_fn = wrap_class(soft_rank_numpy.SoftRank, regularization_strength=regularization_strength, direction=direction, regularization=regularization)
    return map_tensor(wrapped_fn.apply, values)


def soft_sort(values, direction="ASCENDING", regularization_strength=1.0, regularization="l2"):
    if len(values.shape) != 2:
        raise ValueError("'values' should be a 2d-tensor " "but got %s." % str(values.shape))

    wrapped_fn = wrap_class(soft_rank_numpy.SoftSort, regularization_strength=regularization_strength, direction=direction, regularization=regularization)

    return map_tensor(wrapped_fn.apply, values)
