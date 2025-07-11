import torch
from torch.optim.optimizer import Optimizer
from torch.optim import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, AdamW, ASGD
from torch_optimizer import (
    AccSGD,
    AdaBound,
    AdaMod,
    DiffGrad,
    Lamb,
    NovoGrad,
    PID,
    QHAdam,
    QHM,
    RAdam,
    SGDW,
    Yogi,
    Ranger,
    RangerQH,
    RangerVA,
)

__all__ = [
    "AccSGD",
    "AdaBound",
    "AdaMod",
    "DiffGrad",
    "Lamb",
    "NovoGrad",
    "PID",
    "QHAdam",
    "QHM",
    "RAdam",
    "SGDW",
    "Yogi",
    "Ranger",
    "RangerQH",
    "RangerVA",
    "Adam",
    "RMSprop",
    "SGD",
    "Adadelta",
    "Adagrad",
    "Adamax",
    "AdamW",
    "ASGD",
    "Lion",
    "make_optimizer",
    "get",
]


def make_optimizer(params, optim_name="adam", **kwargs):
    """

    Args:
        params (iterable): Output of `nn.Module.parameters()`.
        optimizer (str or :class:`torch.optim.Optimizer`): Identifier understood
            by :func:`~.get`.
        **kwargs (dict): keyword arguments for the optimizer.

    Returns:
        torch.optim.Optimizer
    Examples
        >>> from torch import nn
        >>> model = nn.Sequential(nn.Linear(10, 10))
        >>> optimizer = make_optimizer(model.parameters(), optimizer='sgd',
        >>>                            lr=1e-3)
    """
    return get(optim_name)(params, **kwargs)


def register_optimizer(custom_opt):
    """Register a custom opt, gettable with `optimzers.get`.

    Args:
        custom_opt: Custom optimizer to register.

    """
    if (
        custom_opt.__name__ in globals().keys()
        or custom_opt.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Activation {custom_opt.__name__} already exists. Choose another name."
        )
    globals().update({custom_opt.__name__: custom_opt})


def get(identifier):
    """Returns an optimizer function from a string. Returns its input if it
    is callable (already a :class:`torch.optim.Optimizer` for example).

    Args:
        identifier (str or Callable): the optimizer identifier.

    Returns:
        :class:`torch.optim.Optimizer` or None
    """
    if isinstance(identifier, Optimizer):
        return identifier
    elif isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret optimizer : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret optimizer : {str(identifier)}")


class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss
