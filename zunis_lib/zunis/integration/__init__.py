"""Integration package using neural importance sampling

The main API is exposed as :py:func:`zunis.integration.Integrator <zunis.integration.default_integrator.Integrator>`
"""
import logging
from zunis.integration.default_integrator import Integrator

integration_logger = logging.getLogger(__name__)
"""Overall parent logger for all integration operations"""
