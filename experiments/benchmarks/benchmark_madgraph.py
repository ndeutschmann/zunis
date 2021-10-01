"""Comparing ZuNIS to VEGAS on madgraph integrals"""
import click
import sys

from utils.benchmark.vegas_benchmarks import VegasRandomHPBenchmarker, VegasGridBenchmarker, VegasSequentialBenchmarker
from utils.command_line_tools import PythonLiteralOption
from utils.config.loaders import get_sql_types
from utils.integrands.madgraph import CrossSection


def benchmark_madgraph(e_cm=None,pdf=None, delr_cut=None,pt_cut=None, rap_maxcut=None,process=None,pdf_type=None,
                       pdf_dir=None, lhapdf_dir=None, db=None, experiment_name=None, debug=None, cuda=None, keep_history=None, config=None, n_search=100, stratified=False, benchmark_time=False):

    dtypes = get_sql_types()

    # Integrand specific defaults
    base_integrand_params = {
        "e_cm": 1000,
        "pdf": True,
        "delr_cut": 0.4,
        "pt_cut":0.1,
        "rap_maxcut":2.4,
        "process":"",
        "pdf_type":"",
        "pdf_dir":"",
        "lhapdf_dir":""
    }

    benchmarker = VegasSequentialBenchmarker(n_repeat=n_search, stratified=stratified,benchmark_time=benchmark_time)

    benchmark_config = benchmarker.set_benchmark_grid_config(config=config,dimensions=1, #will be overwritten
                                                             keep_history=keep_history,
                                                             dbname=db, experiment_name=experiment_name, cuda=cuda,
                                                             debug=debug,
                                                             base_integrand_params=base_integrand_params)

    # Integrand specific CLI argument mapped to standard API
    if e_cm is not None:
        benchmark_config["integrand_params_grid"]["e_cm"] = [e_cm]
    if pdf is not None:
        benchmark_config["base_integrand_params"]["pdf"] = pdf
    if delr_cut is not None:
        benchmark_config["integrand_params_grid"]["delr_cut"] = delr_cut
    if pt_cut is not None:
        benchmark_config["integrand_params_grid"]["pt_cut"] = pt_cut
    if rap_maxcut is not None:
        benchmark_config["integrand_params_grid"]["rap_maxcut"] = rap_maxcut
    if process is not None:
        benchmark_config["base_integrand_params"]["process"] = process
    if pdf_type is not None:
        benchmark_config["base_integrand_params"]["pdf_type"] = pdf_type
    if pdf_dir is not None:
        benchmark_config["base_integrand_params"]["pdf_dir"] = pdf_dir
    if lhapdf_dir is not None:
        benchmark_config["base_integrand_params"]["lhapdf_dir"] = lhapdf_dir

    #The integrand is initialised once in order to get the right number of dimensions needed to integrate the process
    CS=CrossSection(pdf=benchmark_config["base_integrand_params"]["pdf"], pdf_dir=benchmark_config["base_integrand_params"]["pdf_dir"], lhapdf_dir=benchmark_config["base_integrand_params"]["lhapdf_dir"], process=benchmark_config["base_integrand_params"]["process"],pdf_type=benchmark_config["base_integrand_params"]["pdf_type"])

    benchmark_config["dimensions"]=[CS.d]


    benchmarker.run(integrand=CrossSection, sql_dtypes=dtypes,
                    **benchmark_config)


cli = click.Command("cli", callback=benchmark_madgraph, params=[
    PythonLiteralOption(["--e_cm"], default=None),
    PythonLiteralOption(["--delr_cut"], default=None),
    PythonLiteralOption(["--pt_cut"], default=None),
    PythonLiteralOption(["--rap_maxcut"], default=None),
    PythonLiteralOption(["--process"], default=None, type=str),
    PythonLiteralOption(["--pdf_type"], default=None, type=str),
    click.Option(["--debug/--no-debug"], default=None, type=bool),
    click.Option(["--pdf/--no-pdf"], default=None, type=bool),
    click.Option(["--pdf_dir"], default=None, type=click.Path()),
    click.Option(["--lhapdf_dir"], default=None, type=click.Path()),
    click.Option(["--cuda"], default=None, type=int),
    click.Option(["--db"], default=None, type=str),
    click.Option(["--experiment_name"], default=None, type=str),
    click.Option(["--config"], default=None, type=str),
    click.Option(["--n_search"], default=5, type=int),
    click.Option(["--stratified/--not-stratified"], default=None, type=bool),
    click.Option(["--benchmark_time/--not-benchmark_time"], default=None, type=bool)
])

if __name__ == '__main__':
    cli()
