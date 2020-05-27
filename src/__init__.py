"""Library of the Pytorch-Flows project"""
import logging

# Ensure that no submodule loggers outputs anything, unless explicitly setup by the user
logging.getLogger(__name__).addHandler(logging.NullHandler())
