import logging
import pickle

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
from utils.known_integrals import evaluate_known_integral
from utils.integrands.abstract import KnownIntegrand
from zunis.integration import Integrator

logger = logging.getLogger(__name__)


class VegasBenchmarker(Benchmarker):
    """Benchmark by comparing with VEGAS"""

    def __init__(self, stratified=False):
        self.stratified = stratified

    def benchmark_method(self, d, integrand, integrator_config=None, integrand_params=None, n_batch=100000,
                         keep_history=False, device=torch.device("cpu")):
        """Benchmarking class for comparing with VEGAS

        Parameters
        ----------
        d: int
        integrand: utils.integrands.abstract.Integrand
        integrator_config: dict
        integrand_params: dict
        n_batch: int
        keep_history: bool
        device: torch.device

        Returns
        -------
            utils.benchmark.vegas_benchmarks.VegasBenchmarker
        """

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

        vintegrator = vegas.Integrator([[0, 1]] * d)

        # Preparing VEGAS arguments
        n_survey_steps = None
        if 'n_iter' in integrator_config:
            n_survey_steps = integrator_config["n_iter"]
        if 'n_iter_survey' in integrator_config:
            n_survey_steps = integrator_config["n_iter_survey"]

        n_batch_survey = None
        if 'n_points' in integrator_config:
            n_batch_survey = integrator_config["n_points"]
        if 'n_points_survey' in integrator_config:
            n_batch_survey = integrator_config["n_points_survey"]

        vegas_checkpoint_path = None
        if "checkpoint_path" in integrator_config:
            vegas_checkpoint_path = integrator_config['checkpoint_path']
        if vegas_checkpoint_path is not None:
            vegas_checkpoint_path = vegas_checkpoint_path + ".vegas"

        # Starting integrals
        start_time_zunis = datetime.datetime.utcnow()
        integrator_result = evaluate_integral_integrator(f, integrator, n_batch=n_batch, keep_history=keep_history)
        end_time_zunis = datetime.datetime.utcnow()

        start_time_vegas = datetime.datetime.utcnow()
        vegas_result = evaluate_integral_vegas(vf, vintegrator, n_batch=n_batch,
                                               n_batch_survey=n_batch_survey,
                                               n_survey_steps=n_survey_steps,
                                               stratified=self.stratified)

        end_time_vegas = datetime.datetime.utcnow()
        if vegas_checkpoint_path is not None:
            try:
                with open(vegas_checkpoint_path, "xb") as vegas_checkpoint_file:
                    pickle.dump(vintegrator, vegas_checkpoint_file)
            except "FileExistsError" as e:
                logger.error("Error while saving VEGAS checkpoint: File exists")
                logger.error(vegas_checkpoint_path)
                logger.error(e)


        flat_result = evaluate_integral_flat(f, d, n_batch=n_batch, device=device)

        if isinstance(f, KnownIntegrand):
            exact_result = evaluate_known_integral(f)

        logger.info("")
        logger.info("Integrator (Result 1) vs Flat integrator (Result 2)")
        result_vs_flat = compare_integral_result(integrator_result, flat_result, sigma_cutoff=3,
                                                 keep_history=keep_history)
        logger.info("")
        logger.info("Integrator (Result 1) vs VEGAS (Result 2)")
        result_vs_vegas = compare_integral_result(integrator_result, vegas_result, sigma_cutoff=3)
        if isinstance(f, KnownIntegrand):
            logger.info("")
            logger.info("Integrator (Result 1) vs exact result (Result 2)")
            result_vs_exact = compare_integral_result(integrator_result, exact_result, sigma_cutoff=3)

        target_keys = ['target', 'target_std', 'sigma_cutoff', 'sigmas_off', 'percent_difference', 'variance_ratio',
                       'match']

        for key in target_keys:
            result_vs_flat["flat_" + key] = result_vs_flat.pop(key)
            result_vs_vegas["vegas_" + key] = result_vs_vegas.pop(key)
            if isinstance(f, KnownIntegrand):
                result_vs_exact["exact_" + key] = result_vs_exact.pop(key)

        common_keys = ["value", "value_std"]

        for key in common_keys:
            result_vs_vegas.pop(key)
            if isinstance(f, KnownIntegrand):
                result_vs_exact.pop(key)
        result = result_vs_flat
        result.update(result_vs_vegas)
        if isinstance(f, KnownIntegrand):
            result.update(result_vs_exact)

        if isinstance(integrator_config, NestedMapping):
            result.update(integrator_config.as_flat_dict())
        else:
            result.update(integrator_config)

        result.update(integrand_params)

        result["d"] = d

        result["time_zunis"] = (end_time_zunis - start_time_zunis).total_seconds()
        result["time_vegas"] = (end_time_vegas - start_time_vegas).total_seconds()
        result["stratified"] = self.stratified

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

    def __init__(self, n=5, stratified=False):
        RandomHyperparameterBenchmarker.__init__(self, n=n)
        VegasBenchmarker.__init__(self, stratified=stratified)


class VegasSequentialBenchmarker(SequentialBenchmarker, VegasBenchmarker):
    """Benchmark against VEGAS by testing on a sequence of (dimension, integrand, integrator) triplets"""
    

class VegasSequentialBenchmarkerN(SequentialBenchmarkerN, VegasBenchmarker):
    """Benchmark against VEGAS by testing on a sequence of (dimension, integrand, integrator) triplets n times"""
