import logging

import torch
import vegas
import datetime
import numpy as np
from dictwrapper import NestedMapping

from utils.benchmark.benchmarker import Benchmarker, GridBenchmarker, RandomHyperparameterBenchmarker, \
    SequentialBenchmarker, GridBenchmarkerN, SequentialBenchmarkerN
from zunis.utils.config.loaders import get_default_integrator_config, create_integrator_args
from utils.flat_integrals import evaluate_integral_flat
from utils.integral_validation import compare_integral_result
from utils.integrator_integrals import evaluate_integral_integrator
from utils.vegas_integrals import evaluate_integral_vegas
from zunis.integration import Integrator

logger = logging.getLogger(__name__)


class VegasBenchmarker(Benchmarker):
    """Benchmark by comparing with VEGAS"""

    def benchmark_method(self, d, integrand, integrator_config=None, integrand_params=None, n_batch=100000,
                        keep_history=False, device=torch.device("cpu")):
        logger.debug("=" * 72)
        logger.info("Defining integrand")
        if integrand_params is None:
            integrand_params = dict()
        f = integrand(d=d, device=device, **integrand_params)
        vf = f.vegas(device=device)

        logger.debug("=" * 72)
        logger.info("Defining integrator")
        if integrator_config is None:
            integrator_config = get_default_integrator_config()
        integrator_args = create_integrator_args(integrator_config)
        integrator = Integrator(f=f, d=d, device=device, **integrator_args)

        vintegrator = vegas.Integrator([[0, 1]] * d, max_nhcube=1)
        time1=datetime.datetime.utcnow()
        integrator_result = evaluate_integral_integrator(f, integrator, n_batch=n_batch, keep_history=keep_history)
        time2=datetime.datetime.utcnow()
        vegas_result = evaluate_integral_vegas(vf, vintegrator, n_batch=n_batch,
                                            n_batch_survey=integrator_args["n_points_survey"])
        time3=datetime.datetime.utcnow()
        flat_result = evaluate_integral_flat(f, d, n_batch=n_batch, device=device)

        result = compare_integral_result(integrator_result, vegas_result, sigma_cutoff=3, keep_history=keep_history)
        result["flat_variance_ratio"] = (flat_result["value_std"] / result["value_std"]) ** 2

        if isinstance(integrator_config, NestedMapping):
            result.update(integrator_config.as_flat_dict())
        else:
            result.update(integrator_config)

        result.update(integrand_params)

        result["d"] = d
        result["time_zunis"]=(time2-time1).total_seconds()
        result["time_vegas"]=(time3-time2).total_seconds()
        zunis_relerr=1
        vegas_relerr=1
        batches=1
        continuer=True
        time_zunis=datetime.datetime.utcnow()
        error_zunis_sum=0
        zunis01_found=False
        zunis001_found=False
        error_vegas_sum=0
        vegas01_found=False
        vegas001_found=False
        while(continuer):
            result_zunis_relerr=evaluate_integral_integrator(f, integrator, n_batch=batches*5000, train=False, keep_history=keep_history)
            error_zunis_sum+=(1/result_zunis_relerr["value_std"]**2)
            zunis_relerr=np.sqrt(1/error_zunis_sum)/result["value"]
            logger.info(zunis_relerr)
            if(zunis_relerr<0.1 and not zunis01_found):
                result["time_zunis_01"]=(datetime.datetime.utcnow()-time_zunis).total_seconds()
                zunis01_found=True
                batches=10
                logger.info("Zunis 0.1")
            elif(zunis_relerr<0.01 and not zunis001_found):
                result["time_zunis_001"]=(datetime.datetime.utcnow()-time_zunis).total_seconds()
                zunis001_found=True
                batches=100
                logger.info("Zunis 0.01")
            elif(zunis_relerr<0.001):
                result["time_zunis_0001"]=(datetime.datetime.utcnow()-time_zunis).total_seconds()
                continuer=False
                logger.info("Zunis 0.001")
        batches=1
        time_vegas=datetime.datetime.utcnow()
        continuer=True
        while(continuer):
            result_vegas_relerr=evaluate_integral_vegas(vf, vintegrator, n_batch=batches*15000,n_batch_survey=0, train=False)
            error_vegas_sum+=(1/result_vegas_relerr["value_std"]**2)
            vegas_relerr=np.sqrt(1/error_vegas_sum)/result["value"]
            logger.info(vegas_relerr)
            if(vegas_relerr<0.1 and not vegas01_found):
                result["time_vegas_01"]=(datetime.datetime.utcnow()-time_vegas).total_seconds()
                vegas01_found=True
                batches=10
                logger.info("Vegas 0.1")
            elif(vegas_relerr<0.01 and not vegas001_found):
                result["time_vegas_001"]=(datetime.datetime.utcnow()-time_vegas).total_seconds()
                vegas001_found=True
                batches=100
                logger.info("Vegas 0.01")
            elif(vegas_relerr<0.001):
                result["time_vegas_0001"]=(datetime.datetime.utcnow()-time_vegas).total_seconds()
                continuer=False
                logger.info("Vegas 0.001")
        
    
        #idea: add the evaluation routine here. then I could set it up the same way as before, let it run and write on the paper meanwhile.
        #problem: is the training saved?->apparently how to do the fixed precision integration?

        return result, integrator


class VegasGridBenchmarker(GridBenchmarker, VegasBenchmarker):
    """Benchmark against VEGAS by sampling parameters on a grid"""

class VegasGridBenchmarkerN(GridBenchmarkerN, VegasBenchmarker):
    """Benchmark against VEGAS by sampling parameters on a grid"""
    
class VegasRandomHPBenchmarker(RandomHyperparameterBenchmarker, VegasBenchmarker):
    """Benchmark against VEGAS by sampling integrator hyperparameters randomly"""


class VegasSequentialBenchmarker(SequentialBenchmarker, VegasBenchmarker):
    """Benchmark against VEGAS by testing on a sequence of (dimension, integrand, integrator) triplets"""
    

class VegasSequentialBenchmarkerN(SequentialBenchmarkerN, VegasBenchmarker):
    """Benchmark against VEGAS by testing on a sequence of (dimension, integrand, integrator) triplets n times"""
