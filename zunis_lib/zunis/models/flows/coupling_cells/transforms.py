"""Abstract definition of invertible transforms for coupling cells"""
from better_abc import ABC,abstractmethod
import torch


class InvertibleTransform(ABC):
    """Callable invertible transform which calls self.forward or self.backward depending on self.inverse

    Attributes
    ----------
    inverse: bool
        whether calling the object will use the forward or backward transform
    """
    def __init__(self):
        """Constructor for InvertibleTransform

        No arguments, no returns, this is just encapsulation for a function and its inverse
        """
        self.inverse = False

    @abstractmethod
    def forward(self, y, T, compute_jacobian=True):
        """Abstract forward transform

        Parameters
        ----------
        y: torch.Tensor
            batch of points to transform
        T: torch.Tensor
            batch of parameters for the transformation
        compute_jacobian: bool
            whether to compute the jacobian or not

        Returns
        -------
        tuple of torch.Tensor, torch.Tensor
            `(z,j)` where the first tensor is the batch of transformed points and the second is the batch of jacobians
        """
        pass

    @abstractmethod
    def backward(self, y, T, compute_jacobian=True):
        """Abstract backward transform

        Parameters
        ----------
        y: torch.Tensor
            batch of points to transform
        T: torch.Tensor
            batch of parameters for the transformation
        compute_jacobian: bool
            whether to compute the jacobian or not

        Returns
        -------
        tuple of torch.Tensor, torch.Tensor
            `(z,j)` where the first tensor is the batch of transformed points and the second is the batch of jacobians
        """
        pass

    def __call__(self,y , T, compute_jacobian=True):
        """A transform is callable and either performs the forward or inverse transformation
        depending on the attribute `self.inverse`.

        Parameters
        ----------
        y: torch.Tensor
            batch of points to transform
        T: torch.Tensor
            batch of parameters for the transformation
        compute_jacobian: bool
            whether to compute the jacobian or not


        Returns
        -------
        tuple of torch.Tensor, torch.Tensor
            `(z,j)` where the first tensor is the batch of transformed points and the sec
        """
        if self.inverse:
            return self.backward(y, T)
        else:
            return self.forward(y, T)

    def invert(self):
        """Flip the `inverse` attribute

        Returns
        -------
        None
        """
        self.inverse = not self.inverse

