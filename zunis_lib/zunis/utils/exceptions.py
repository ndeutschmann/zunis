"""Custom exceptions for the ZüNIS library"""


class AvertedCUDARuntimeError(RuntimeError):
    """Prevents the raising of a RuntimeError cause by a CUDA device-side assert failure
    which whould lock the GPU in a failed state."""


class NoCheckpoint(RuntimeError):
    """Raised to signify no checkpoint is available for reloading"""


class TrainingInterruption(RuntimeError):
    """Raised when a Trainer determines that the model should no longer be trained"""
