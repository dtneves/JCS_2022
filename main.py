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

from socket import gethostname

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
    df_pre: pd.DataFrame            # pandas DataFrame to hold preprocessed data
    synthesizer: CTGANSynthesizer   # the CTGAN data synthesizer
    df_sam: pd.DataFrame            # to store the samples (i.e., the synthetic data)
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
                  miss_rate: float = 0.2,
                  encoder_type: str = 'label',
                  algo: str = 'tabulator',
                  loss: str = 'both',
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
    df_pre: pd.DataFrame    # pandas DataFrame to hold preprocessed data
    df_enc: pd.DataFrame    # pandas DataFrame to hold the encoded and the non-encoded data
    df_sam: pd.DataFrame    # to store the samples (i.e., the synthetic data) in a pandas DataFrame
    samples: np.ndarray     # the samples (i.e., the synthetic data)
    generator: TabularDataGenerator
    new_discrete_vars: Union[List[str], List[int]]
    filename: str = f"{dataset}_{miss_rate}_{encoder_type}_{algo}_{batch_size}_{loss}_{n_iterations}"

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
        algo_parameters={'miss_rate': miss_rate, 'batch_size': batch_size, 'loss': loss, 'n_iterations': n_iterations})
    # logging some execution info
    if verbose:
        logging.basicConfig(filename=f"{out_folder}/{filename}.txt", level=logging.INFO)
        logging.info(f"{'--- tabulatorSGAIN ---' * 3}")
        logging.info(f"dataset: {dataset}")
        logging.info(f"missing rate: {miss_rate}")
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


if __name__ == "__main__":
    dataset: str
    df_pre: pd.DataFrame
    df_sam: pd.DataFrame

    # if len(sys.argv) == 4 and sys.argv[1] == 'CTGAN':
    #     dataset = sys.argv[2]
    #     df_pre, df_sam = CTGAN(dataset=dataset, n_epochs=int(sys.argv[3]), n_samples=DATASETS[dataset], verbose=False)
    #     # report_correlations(original_data=df_pre, synthetic_data=df_sam, dataset=dataset, algorithm="CTGAN")
    # elif len(sys.argv) == 7 and sys.argv[1] in tabulator.tabulators:
    #     dataset = sys.argv[2]
    #     df_pre, df_sam = tabulatorSGAIN(dataset=sys.argv[2],
    #                                      miss_rate=float(sys.argv[3]),
    #                                      encoder_type=sys.argv[4],  # ['label', 'one-hot']
    #                                      algo=sys.argv[1],
    #                                      loss=sys.argv[5],  # ['both', 'corr', 'mse']
    #                                      batch_size=128,
    #                                      n_iterations=int(sys.argv[6]),
    #                                      n_samples=DATASETS[dataset],
    #                                      verbose=False)
    #     # main(_code=sys.argv[1], _dataset=sys.argv[2], _miss_rate=float(sys.argv[3]), _n_iterations=int(sys.argv[4]))
    # else:
    #     raise RuntimeError("Bad usage!")
    #     # raise RuntimeError(f"Usage:\npython {sys.argv[0]} <code> <dataset> <miss_rate> <n_iterations>")

    # ['adult', 'breast', 'cover', 'credit', 'eeg', 'iris', 'letter', 'mushroom', 'news', 'spam', 'wine-red', 'wine-white', 'yeast']
    # ['adult', 'breast', 'credit', 'eeg', 'letter', 'news', 'spam']
    # ['iris', 'wine-red', 'wine-white', 'yeast']

    for dataset in ['iris_sample']:
        df_pre, df_sam = run_CTGAN(dataset=dataset, n_epochs=10, n_samples=DATASETS[dataset], verbose=False)
        for miss_rate in [0.25]:  # [0.20, 0.40, 0.60, 0.80]:
            for algo in ['tabulator']:  # ['tabulator', 'tabulator-CP', 'tabulator-GP']:
                for loss in ['mse']:  # ['both', 'corr', 'mse']:
                    print(f"{dataset} :: {miss_rate} :: {algo} :: {loss}")
                    df_pre, df_sam = run_tabulator(dataset=dataset,
                                                    miss_rate=miss_rate,
                                                    encoder_type='one-hot',
                                                    algo=algo,
                                                    loss=loss,
                                                    batch_size=128,
                                                    n_iterations=1000,
                                                    n_samples=DATASETS[dataset],
                                                    verbose=False)

