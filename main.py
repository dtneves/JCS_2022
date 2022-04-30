########################################################################################################################
# Research Centers
# ----------------
# Medical Informatics Group
# BIH - Berlin Institute of Health
# Charité - Universitätsmedizin Berlin
# https://www.bihealth.org/en/research/research-groups/fabian-prasser/
#
# Centro ALGORITMI - School of Engineering – University of Minho
# Braga - Portugal
# http://algoritmi.uminho.pt/
#
#
# Description
# -----------
# This Python script is the entry point of the code partially described in [1].
#
#
# Moto
# ----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
#
#
# References
# ----------
#  [1] Diogo Telmo Neves, João Alves, Marcel Ganesh Naik, Alberto José Proença, Fabian Praßer.
#      "From Missing Data Imputation to Data Generation."
#      Journal of Computational Science (JCS), 2022.
#  [2] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença.
#      "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation."
#      International Conference on Computational Science (ICCS), 2021.
#  [3] Jinsung Yoon, James Jordon, Mihaela van der Schaar,
#      "GAIN: Missing Data Imputation using Generative Adversarial Nets."
#      International Conference on Machine Learning (ICML), 2018.
#  [4] Xu, Lei, et al.
#      "Modeling Tabular data using Conditional GAN."
#      Advances in Neural Information Processing Systems (NIPS), 2019.
#
#
# Authors
# -------
# diogo telmo neves -- {dneves@di.uminho.pt, diogo-telmo.neves@charite.de, tada.science@gmail.com}
#
#
# Copyright
# ---------
# Copyright (c) 2020 diogo telmo neves.
# All rights reserved.
#
#
# Conditions
# ----------
# This code is free/open source code but the following conditions must be met:
#   * Redistributions of source code must retain the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#
# DISCLAIMER
# ----------
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Date
# ----
# September 2021
########################################################################################################################

import _io
import logging

import numpy as np
import pandas as pd

from ordered_set import OrderedSet

from socket import gethostname
from argparse import ArgumentParser, Namespace

# this can be removed, it was used to distinguish between the local machine and
# the (remote) machine where we run the final experiments
if gethostname() != 'BigMedilytics':
    from ctgan import CTGANSynthesizer

from purify.dataset.metadata import Metadata
from purify.dataset.profiling import profiler
from purify.dataset.processors import PreProcessor
from purify.encoders import label_encoders_fit_transform, label_encoders_inverse_transform
from purify.encoders import get_dummies_fit_transform, get_dummies_inverse_transform
from purify.generation.tabulator import TabularDataGenerator

from typing import List, Dict, Tuple, Union

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

DATASETS: Dict[str, int] = {
    'adult': 30162, 'breast': 569, 'cover': 581012, 'credit': 30000, 'eeg': 14980, 'iris': 150, 'iris_sample': 12,
    'letter': 20000, 'mushroom': 8124, 'news': 39644, 'spam': 4601, 'wine-red': 1599, 'wine-white': 4898, 'yeast': 1484
}


def run_CTGAN(dataset: str = 'adult',
              n_epochs: int = 10,
              n_samples: int = 100,
              in_folder: str = './datasets',
              out_folder: str = './experiments',
              verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # df_raw: pd.DataFrame = load_demo() if dataset == 'adult' else pd.read_csv(
    #     filepath_or_buffer=f"./datasets/{dataset}.csv", na_values='?')
    df_raw: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=f"{in_folder}/{dataset}.csv", skipinitialspace=True, na_values='?', skip_blank_lines=True)
    df_pre: pd.DataFrame  # pandas DataFrame to hold preprocessed data
    synthesizer: CTGANSynthesizer  # the CTGAN data synthesizer
    df_sam: pd.DataFrame  # to store the samples (i.e., the synthetic data)
    filename: str = f"{dataset}_CTGAN_{n_epochs}"

    # data preprocessing
    df_pre = PreProcessor.drop_vars(dataset=dataset, df=df_raw)
    df_pre = PreProcessor.replace_miss_values_by_nans(df=df_pre, dataset=dataset)
    df_pre = PreProcessor.drop_nans(df=df_pre)
    # create the synthesizer
    synthesizer = CTGANSynthesizer(epochs=n_epochs)
    # learn from the data distribution
    synthesizer.fit(train_data=df_pre, discrete_columns=Metadata.discrete_vars(dataset=dataset, df=df_pre))
    # generate synthetic data
    df_sam = synthesizer.sample(n=n_samples)
    if verbose:
        log: _io.TextIOWrapper = open(file=f"{out_folder}/{filename}.txt", mode='w+')

        log.write(f"{'--- CTGAN ---' * 3}\n")
        log.write(f"dataset: {dataset}\n")
        log.write(f"raw shape: {df_raw.shape}\n")
        log.write(f"preprocessing shape: {df_pre.shape}\n")
        log.write(f"n_epochs: {n_epochs}\n")
        log.write(f"n_samples: {n_samples}\n")
        # log.write("samples:\n")
        # log.write(df_sam.head())
        # log.write("\n")
        # log.write("...\n")
        # log.write(df_sam.tail())
        # log.write("\n")
        log.write(f"profiling: {profiler(df=df_sam, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_sam))}")
        log.write("\n")
        # log.write(f"are the pandas dataframes equals? {df_pre.equals(other=df_sam)}\n")
    # open(file=f"{out_folder}/{filename}_pre.txt", mode='w+').write(
    #     f"profiling: {profiler(df=df_pre, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_pre))}")
    # open(file=f"{out_folder}/{filename}_sam.txt", mode='w+').write(
    #     f"profiling: {profiler(df=df_sam, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_sam))}")
    df_pre.to_csv(path_or_buf=f"{out_folder}/{filename}_pre.csv", index=False)
    df_sam.to_csv(path_or_buf=f"{out_folder}/{filename}_sam.csv", index=False)
    return df_pre, df_sam


def run_tabulator(dataset: str = 'adult',
                  ampu_rate: float = 0.2,
                  encoder_type: str = 'label',
                  algo: str = 'tabulator',
                  loss: str = 'mse',
                  batch_size: int = 128,
                  n_iterations: int = 1000,
                  n_samples: int = 100,
                  in_folder: str = './datasets',
                  out_folder: str = './experiments',
                  verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # df_raw: pd.DataFrame = load_demo() if dataset == 'adult' else pd.read_csv(
    #     filepath_or_buffer=f"./datasets/{dataset}.csv")
    df_raw: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=f"{in_folder}/{dataset}.csv", skipinitialspace=True, na_values='?', skip_blank_lines=True)
    df_pre: pd.DataFrame  # pandas DataFrame to hold preprocessed data
    df_enc: pd.DataFrame  # pandas DataFrame to hold the encoded and the non-encoded data
    df_sam: pd.DataFrame  # to store the samples (i.e., the synthetic data) in a pandas DataFrame
    samples: np.ndarray  # the samples (i.e., the synthetic data)
    generator: TabularDataGenerator
    new_discrete_vars: Union[List[str], List[int]]
    filename: str = f"{dataset}_{ampu_rate}_{encoder_type}_{algo}_{batch_size}_{loss}_{n_iterations}"

    # data preprocessing
    df_pre = PreProcessor.drop_vars(dataset=dataset, df=df_raw)
    df_pre = PreProcessor.replace_miss_values_by_nans(df=df_pre, dataset=dataset)
    df_pre = PreProcessor.drop_nans(df=df_pre)
    # encoding the discrete variables
    if encoder_type == 'one-hot':
        # data transformation that looks like one-hot encoding
        df_enc = get_dummies_fit_transform(
            data=df_pre, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_pre))
        # list of the new discrete variables, which came from the original discrete variables
        new_discrete_vars = [var for var in df_enc if var not in df_pre.columns]
        # replace each zero of one-hot encoding with minus one
        df_enc = PreProcessor.replace_values(
            df=df_enc, to_replace={new_discrete_var: {0: -1} for new_discrete_var in new_discrete_vars})
    else:  # 'label' --> default encoder
        df_enc, label_encoders = label_encoders_fit_transform(
            data=df_pre, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_pre))
    # create an instance of the generator
    generator = TabularDataGenerator(
        data=df_enc.to_numpy(),
        algo=algo,
        algo_parameters={'miss_rate': ampu_rate, 'batch_size': batch_size, 'loss': loss, 'n_iterations': n_iterations})
    # logging some execution info
    if verbose:
        logging.basicConfig(filename=f"{out_folder}/{filename}.txt", level=logging.INFO)
        logging.info(f"{'--- tabulatorSGAIN ---' * 3}")
        logging.info(f"dataset: {dataset}")
        logging.info(f"amputation rate: {ampu_rate}")
        logging.info(f"encoder type: {encoder_type}")
        logging.info(f"raw shape: {df_raw.shape}")
        logging.info(f"preprocessing shape: {df_pre.shape}")
        logging.info(f"encoded shape: {df_enc.shape}")
        logging.info(f"algorithm: {algo}")
        logging.info(f"batch size: {batch_size}")
        logging.info(f"loss: {loss}")
        logging.info(f"n_iterations: {n_iterations}")
        logging.info(f"n_samples: {n_samples}")
    # sampling (i.e., get the samples)
    samples = generator.sampler(n_samples=n_samples)
    # decoding the discrete variables
    if encoder_type == 'one-hot':
        df_sam = pd.DataFrame(data=samples, columns=df_enc.columns)
        # invert the replacement of each zero of one-hot encoding with minus one
        df_sam = PreProcessor.replace_values(
            df=df_sam, to_replace={new_discrete_var: {-1: 0} for new_discrete_var in new_discrete_vars})
        # data transformation to invert (i.e., to revert) the one that looks line one-hot encoding
        df_sam = get_dummies_inverse_transform(
            dataset=dataset,
            # data=pd.DataFrame(data=samples, columns=df_enc.columns),
            data=df_sam,
            discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_pre),
            vars_order=df_pre.columns)
    else:  # 'label' --> default encoder
        df_sam = label_encoders_inverse_transform(
            dataset=dataset,
            data=pd.DataFrame(data=samples, columns=df_enc.columns),
            label_encoders=label_encoders)
    if verbose:
        # logging.info("samples:")
        # logging.info(df_sam.head())
        # logging.info("...")
        # logging.info(df_sam.tail())
        logging.info(
            f"profiling: {profiler(df=df_sam, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_sam))}")
        # logging.info(f"are the pandas dataframes equals? {df_pre.equals(other=df_sam)}")
    # open(file=f"{out_folder}/{filename}_pre.txt", mode='w+').write(
    #     f"profiling: {profiler(df=df_pre, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_pre))}")
    # open(file=f"{out_folder}/{filename}_sam.txt", mode='w+').write(
    #     f"profiling: {profiler(df=df_sam, discrete_vars=Metadata.discrete_vars(dataset=dataset, df=df_sam))}")
    df_pre.to_csv(path_or_buf=f"{out_folder}/{filename}_pre.csv", index=False)
    df_sam.to_csv(path_or_buf=f"{out_folder}/{filename}_sam.csv", index=False)
    return df_pre, df_sam


def main(args: Namespace) -> None:
    algos: OrderedSet[str] = OrderedSet([algo.strip() for algo in args.algos.split(',')])
    # TODO: GET RID OF HARDCODED
    algos_set: OrderedSet[str] = OrderedSet(['CTGAN', 'tabulator', 'tabulator-CP', 'tabulator-GP'])
    datasets: OrderedSet[str] = OrderedSet([dataset.strip() for dataset in args.datasets.split(',')])
    # TODO: GET RID OF HARDCODED
    datasets_set: OrderedSet[str] = OrderedSet(['breast', 'credit', 'eeg', 'iris', 'letter', 'news',
                                                'spam', 'wine-red', 'wine-white', 'yeast'])
    ampu_rates: OrderedSet[str] = OrderedSet([ampu_rate.strip() for ampu_rate in args.ampu_rates.split(',')])
    dataset: str
    algo: str
    ampu_rate: str
    ampu_rate_tmp: float
    df_pre: pd.DataFrame
    df_sam: pd.DataFrame

    if not algos.issubset(algos_set):
        raise ValueError(f"In terms of algorithms, expecting a subset of {algos_set} but got: {algos}.")
    if not datasets.issubset(datasets_set):
        raise ValueError(f"In terms of datasets, expecting a subset of {datasets_set} but got: {datasets}.")
    for dataset in datasets:
        for ampu_rate in ampu_rates:
            try:
                ampu_rate_tmp = float(ampu_rate)
                if ampu_rate_tmp < 0.00 or ampu_rate_tmp > 1.00:
                    raise ValueError(
                        f"Expecting an amputation rate within the interval [0.00, 1.00] but got: {ampu_rate_tmp}.")
            except ValueError:
                print(f"Expecting an amputation rate within the interval [0.00, 1.00] but got: {ampu_rate}.")
                exit(1)
            for algo in algos:
                print(f"{dataset} :: {ampu_rate} :: {algo}")
                df_pre, df_sam = run_tabulator(dataset=dataset,
                                               ampu_rate=ampu_rate_tmp,
                                               encoder_type='one-hot',
                                               algo=algo,
                                               batch_size=128,
                                               n_iterations=1000,
                                               n_samples=DATASETS[dataset],
                                               verbose=False)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        '--algos',
        help="a csv list of the algorithms to run (e.g., 'CTGAN, tabulator, tabulator-CP, tabulator-GP')",
        # choices=['CTGAN', 'tabulator', 'tabulator-CP', 'tabulator-GP'],
        default='tabulator',
        type=str)
    parser.add_argument(
        '--datasets',
        help="a csv list of datasets short names",
        # choices=['breast', 'cover-type', 'credit', 'eeg', 'iris', 'letter', 'mushroom',
        #          'news', 'spam', 'wine-red', 'wine-white', 'yeast'],
        default='letter',
        type=str)
    parser.add_argument(
        '--ampu_rates',
        help="a csv list of amputation rates ([0.00, 1.00])",
        default='0.20',
        type=str)
    parser.add_argument(
        '--batch_size',
        help="number of samples in mini-batch",
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',  # NOTE: the algorithms SGAIN, WSGAIN-CP, and WSGAIN-GP do NOT use this parameter,
        help='hint probability',  # it is here just because the GAIN algorithm requires the `hint_rate` parameter
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help="hyper-parameter to compute generator's loss",
        default=100,
        type=float)
    parser.add_argument(
        '--lambd',
        help="hyper-parameter to compute critic's loss",
        default=10,
        type=float)
    parser.add_argument(
        '--clip_value',
        help="clip (penalty) value",
        default=0.01,
        type=float)
    parser.add_argument(
        '--optimizer',
        help="solvers' optimizer",
        choices=['Adam', 'GDA', 'RMSProp'],
        default='Adam',
        type=str)
    parser.add_argument(
        '--learn_rate',
        help="optimizer's learning rate",
        default=1e-3,
        type=float)
    parser.add_argument(
        '--beta_1',
        help="Adam optimizer's hyper-parameter (1st moment estimates)",
        default=0.900,
        type=float)
    parser.add_argument(
        '--beta_2',
        help="Adam optimizer's hyper-parameter (2nd moment estimates)",
        default=0.999,
        type=float)
    parser.add_argument(
        '--decay',
        help="RMSProp optimizer's hyper-parameter (discounting factor for the history/coming gradient)",
        default=0.900,
        type=float)
    parser.add_argument(
        '--momentum',
        help="RMSProp optimizer's hyper-parameter (a scalar tensor)",
        default=0.000,
        type=float)
    parser.add_argument(
        '--epsilon',
        help="Adam hyper-parameter to ensure numerical stability or RMSProp hyper-parameter to avoid zero denominator",
        default=1e-08,
        type=float)
    parser.add_argument(
        '--n_iterations',
        help="number of training iterations",
        default=10000,
        type=int)
    parser.add_argument(
        '--n_critic',
        help="number of additional iterations to train the critic",
        default=5,
        type=int)
    parser.add_argument(
        '--n_runs',
        help="number of runs",
        default=3,
        type=int)
    parser.add_argument(
        '--verbose',
        help="to control verbosity",
        choices=['False', 'True'],  # `bool` type does NOT work as expected
        default='False',  # `bool` type does NOT work as expected
        type=str)  # `bool` type does NOT work as expected

    main(args=parser.parse_args())  # rock 'n roll

# python main.py --algos="GAIN,SGAIN,WSGAIN-CP,WSGAIN-GP" --datasets="iris,yeast" --ampu_rates="0.20" --optimizer=GDA --learn_rate=0.001 --n_iterations=1000 --n_runs=3
