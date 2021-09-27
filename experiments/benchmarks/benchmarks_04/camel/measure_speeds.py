import sys
import pandas as pd
import numpy as np
import torch
import time
import pickle

from pathlib import Path

from sqlalchemy import PickleType

from zunis.utils.config.loaders import get_default_integrator_config
from zunis.utils.config.loaders import create_integrator_args
from zunis.integration import Integrator

from utils.data_storage.dataframe2sql import read_pkl_sql
from utils.config.loaders import get_sql_types
from utils.integrands.camel import KnownSymmetricCamelIntegrand
from utils.vegas_integrals import VegasSampler
from utils.integrator_integrals import IntegratorSampler

here = Path(__file__).parent.resolve()
benchmarks_04 = here.parent
db_path = benchmarks_04 / 'benchmarks.db'

benchmarks = benchmarks_04.parent
experiments = benchmarks.parent

sys.path.append(str(benchmarks))


def get_df():
    sigma_info = experiments / 'exploration' / 'gaussian_camel_integrands.csv'

    dtypes = get_sql_types()
    dtypes["value_history"] = PickleType

    df = read_pkl_sql(db_path, "camel_defaults", dtypes=dtypes)
    df.columns = df.columns.astype(str)
    d_sigmas = pd.read_csv(sigma_info)
    d_sigma_camel = d_sigmas[['d', 'sigma_camel', 'sigma_1d', 'relative_std_camel']].rename(columns=
    {
        "sigma_camel": 's',
        'relative_std_camel': 'relative_std'
    })

    df = df.merge(d_sigma_camel, on=["d", "s"], how="left")
    return df


def create_integrators(row, device=None):
    d = row['d']
    s = row['s']
    f = KnownSymmetricCamelIntegrand(d=d, s=s, device=device)

    config = get_default_integrator_config()
    kw = create_integrator_args(config)

    integrator = Integrator(d=d, f=f, device=device, **kw)

    ckpt = torch.load(benchmarks / row['checkpoint_path'])
    integrator.model_trainer.flow.load_state_dict(ckpt)

    ckpt_vegas = benchmarks / (row['checkpoint_path'] + '.vegas')
    with open(ckpt_vegas, 'rb') as ckpt_vegas_file:
        vintegrator = pickle.load(ckpt_vegas_file)

    return integrator, vintegrator


def evaluate_integrator(integrator, f, npoints_max=1000000, npoints_min=100, n_splits=10, n_repeat=10, how='geom'):
    spacers = {
        'geom': np.geomspace,
        'lin': np.linspace
    }
    data = pd.DataFrame()
    for n in spacers[how](npoints_min, npoints_max, n_splits, dtype=np.int):
        for rep in range(n_repeat):
            with torch.no_grad():
                try:
                    start = time.time()
                    _, px, fx = integrator.sample(f=f, n_batch=n)
                    end = time.time()
                    mean, std = torch.std_mean(fx / px)
                    mean = mean.cpu().item()
                    std = std.cpu().item() / np.sqrt(n)
                    data = data.append(
                    {'n_points': n,
                     'mean': mean,
                     'std': std,
                     'sample_time': end - start},
                    ignore_index=True
                    )
                except Exception:
                    pass

    return data


def evaluate_row(row, npoints_max=1000000, npoints_min=100, n_splits=10, n_repeat=10, how='geom', device=None):
    zintegrator, vintegrator = create_integrators(row, device=device)
    f = zintegrator.f
    vf = f.vegas(device=device)

    zunis_integrator = IntegratorSampler(zintegrator, train=False)
    data = evaluate_integrator(zunis_integrator, f=f, npoints_max=npoints_max, npoints_min=npoints_min,
                               n_splits=n_splits,
                               n_repeat=n_repeat, how=how)
    data['d'] = row['d']
    data['s'] = row['s']
    data['sigma_1d'] = row['sigma_1d']
    data['relative_std'] = row['relative_std']
    data['integrator'] = 'ZuNIS'

    vegas_integrator = VegasSampler(vintegrator, integrand=None, train=False, stratified=False)
    datav = evaluate_integrator(vegas_integrator, f=vf, npoints_max=npoints_max, npoints_min=npoints_min,
                                n_splits=n_splits,
                                n_repeat=n_repeat, how=how)
    datav['d'] = row['d']
    datav['s'] = row['s']
    datav['sigma_1d'] = row['sigma_1d']
    datav['relative_std'] = row['relative_std']
    datav['integrator'] = 'VEGAS'

    return data.append(datav, ignore_index=True)


def evaluate_benchmarks(npoints_max=1000000, npoints_min=100, n_splits=10, n_repeat=10, how='geom', device=None):
    df = get_df()
    results = pd.DataFrame()
    for _, row in df.iterrows():
        print(row[['d', 's']])
        data = evaluate_row(row, npoints_max=npoints_max, npoints_min=npoints_min, n_splits=n_splits, n_repeat=n_repeat,
                            how=how, device=device)
        results = results.append(data, ignore_index=True)

    results.to_csv(here / 'zunis_speed.csv')
    return results


if __name__ == '__main__':
    evaluate_benchmarks(device='cuda:7')
