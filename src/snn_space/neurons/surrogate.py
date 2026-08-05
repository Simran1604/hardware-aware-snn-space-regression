import torch


class ATanSurrogate(torch.autograd.Function):
    """
    Arctangent surrogate gradient.
    """

    alpha = 2.0

    @staticmethod
    def forward(ctx, x):

        ctx.save_for_backward(x)

        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):

        (x,) = ctx.saved_tensors

        alpha = ATanSurrogate.alpha

        grad = alpha / (
            2 * (1 + (torch.pi * alpha * x / 2).pow(2))
        )

        return grad_output * grad


spike_fn = ATanSurrogate.apply