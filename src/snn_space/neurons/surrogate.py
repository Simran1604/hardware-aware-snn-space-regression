import torch


class SurrogateSpike(torch.autograd.Function):
    """
    Straight-through surrogate gradient.

    Forward:
        Hard threshold.

    Backward:
        Smooth derivative.
    """

    @staticmethod
    def forward(ctx, membrane, threshold):

        ctx.save_for_backward(membrane, threshold)

        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):

        membrane, threshold = ctx.saved_tensors

        alpha = 5.0

        grad = alpha * torch.exp(
            -alpha * torch.abs(membrane - threshold)
        )

        return grad_output * grad, None


surrogate_spike = SurrogateSpike.apply