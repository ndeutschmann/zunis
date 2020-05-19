"""Abstract definition of invertible transforms for coupling cells"""
from better_abc import ABC,abstractmethod


class InvertibleTransform(ABC):
    """Callable invertible transform which calls self.forward or self.backward depending on self.inverse"""
    def __init__(self):
        self.inverse = False

    @abstractmethod
    def forward(self, y, T, compute_jacobian=True):
        pass

    @abstractmethod
    def backward(self, y, T, compute_jacobian=True):
        pass

    def __call__(self,y , T, compute_jacobian=True):
        if self.inverse:
            return self.backward(y, T)
        else:
            return self.forward(y, T)

    def invert(self):
        self.inverse = not self.inverse

