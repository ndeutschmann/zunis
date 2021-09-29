import torch
import datetime
import numpy as np
import logging

from utils.integrator_integrals import evaluate_integral_integrator
from utils.vegas_integrals import evaluate_integral_vegas

logger = logging.getLogger(__name__)

def run_time_benchmark(f, vf, integrator, vintegrator, result):
    """ Benchmark the evaluation time of a trained ZuNIS model against an
    adapted VEGAS model

    Parameters
    ----------
    f : utils.integrands.abstract.Integrand
    vf : vegas.batchintegrand
    integrator : utils.integrator_integrals.IntegratorSampler
    vintegrator : utils.vegas_integrals.VegasSampler
    result : utils.record.Record

    Returns
    ---------
    result : utils.record.Record
    """

    zunis_relerr = 1
    vegas_relerr = 1
    batches =1
    continuer=True
    error_zunis_sum = 0
    error_vegas_sum = 0
    zunis01_found = False
    zunis001_found = False
    vegas01_found = False
    vegas001_found = False
    time_zunis=datetime.datetime.utcnow()
    while(continuer):
        result_zunis_relerr=evaluate_integral_integrator(f, integrator, n_batch=batches*2500, train=False, keep_history=False)
        error_zunis_sum+=(1/result_zunis_relerr["value_std"]**2)
        zunis_relerr=np.sqrt(1/error_zunis_sum)/result["value"]
        logger.info(zunis_relerr)
        if(zunis_relerr<0.1 and not zunis01_found):
            result["time_zunis_01"]=(datetime.datetime.utcnow()-time_zunis).total_seconds()
            zunis01_found=True
            batches=5
            if(result["time_zunis_01"]>200):
                continuer=False
                result["time_zunis_001"]=-1
                result["time_zunis_0005"]=-1
            logger.info("Reaching relative precision of 0.1 with ZuNIS took "+str(result["time_zunis_01"]))
        elif(zunis_relerr<0.01 and not zunis001_found):
            result["time_zunis_001"]=(datetime.datetime.utcnow()-time_zunis).total_seconds()
            zunis001_found=True
            batches=50
            logger.info("Reaching relative precision of 0.01 with ZuNIS took "+str(result["time_zunis_001"]))
            if(result["time_zunis_001"]>300):
                continuer=False
                result["time_zunis_0005"]=-1
        elif(zunis_relerr<0.005):
            result["time_zunis_0005"]=(datetime.datetime.utcnow()-time_zunis).total_seconds()
            continuer=False
            logger.info("Reaching relative precision of 0.005 with ZuNIS took "+str(result["time_zunis_0005"]))
    batches=1
    continuer=True
    time_vegas=datetime.datetime.utcnow()
    while(continuer):
        result_vegas_relerr=evaluate_integral_vegas(vf, vintegrator, n_batch=batches*2500,n_batch_survey=2500, train=False)
        error_vegas_sum+=(1/result_vegas_relerr["value_std"]**2)
        vegas_relerr=np.sqrt(1/error_vegas_sum)/result["value"]
        logger.info(vegas_relerr)
        if(vegas_relerr<0.1 and not vegas01_found):
            result["time_vegas_01"]=(datetime.datetime.utcnow()-time_vegas).total_seconds()
            vegas01_found=True
            batches=5
            logger.info("Reaching relative precision of 0.1 with Vegas took "+str(result["time_vegas_01"]))
            if(result["time_vegas_01"]>200):
                result["time_vegas_001"]=-1
                result["time_vegas_0005"]=-1
                continuer=False
        elif(vegas_relerr<0.01 and not vegas001_found):
            result["time_vegas_001"]=(datetime.datetime.utcnow()-time_vegas).total_seconds()
            vegas001_found=True
            batches=50
            logger.info("Reaching relative precision of 0.01 with Vegas took "+str(result["time_vegas_001"]))
            if(result["time_vegas_001"]>300):
                result["time_vegas_0005"]=-1
                continuer=False
        elif(vegas_relerr<0.005):
            result["time_vegas_0005"]=(datetime.datetime.utcnow()-time_vegas).total_seconds()
            continuer=False
            logger.info("Reaching relative precision of 0.005 with Vegas took "+str(result["time_vegas_0005"]))

    return result;
