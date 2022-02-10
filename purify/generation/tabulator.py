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
# tabulator -- Tabul(ar Data Gener)ator --, is a Python module that allows to perform synthetic data generation
# for tabular datasets.
# Specific details about tabulator are provided in [1].
# For the sake of simplicity and for the moment,
# the amputation procedure follows the MCAR -- Missing Completely at Random -- setting.
# Therefore, each amputated value does not depend on the data values.
# The amputation rate can range from 0% (0.00) to 99% (0.99).
# On the one hand, it is NOT recommended to use high values for the amputation rate
# since they will lead to a severe loss on data utility.
# On the other hand, very low amputation rates will likely lead to synthetic data that is highly identifiable.
# Thus, it is up to the user to use metrics that allow her/him to assess
# the trade-off between data utility and data lesser identifiable.
# The implementation of advanced techniques to generate synthetic data and,
# yet, maintain as much as possible data privacy is on our research roadmap,
# those will be based on metadata, domain knowledge, differential privacy, amongst other aspects.
# Finally, one should be aware that exception handling to take care of incorrect data types,
# incorrect parameters' values, and so forth is, typically, NOT performed, the rule is:
# We are all grown up (Python) programmers!
#
#
# Moto
# ----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
#
#
# Related Work
# ------------
#   * https://link.springer.com/chapter/10.1007/978-3-030-77961-0_10  -->  SGAIN + WSGAIN-CP + WSGAIN-GP paper
#   * https://github.com/dtneves/ICCS_2021
#   * https://arxiv.org/pdf/1907.00503  -->  CTGAN paper
#   * https://github.com/sdv-dev/CTGAN
#   * https://github.com/jsyoon0823/GAIN
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
# Copyright (c) 2021 diogo telmo neves.
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

import numpy as np

from purify.imputation.gain import SGAIN, WSGAIN_CP, WSGAIN_GP

import random

import math

from typing import Any, Callable, Dict, List, Tuple


class tabulator(SGAIN):
    """This class is a facade of the SGAIN algorithm [1, 2].

    References
    ----------
    [1] Diogo Telmo Neves, João Alves, Marcel Ganesh Naik, Alberto José Proença, Fabian Praßer.
        "From Missing Data Imputation to Data Generation."
        Journal of Computational Science (JCS), 2022.
    [2] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença.
        "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation."
        International Conference on Computational Science (ICCS), 2021.
    """
    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any] = {}):
        super().__init__(data=data, algo_parameters=algo_parameters)


class tabulator_CP(WSGAIN_CP):
    """This class is a facade of the WSGAIN-CP algorithm [1, 2].

    References
    ----------
    [1] Diogo Telmo Neves, João Alves, Marcel Ganesh Naik, Alberto José Proença, Fabian Praßer.
        "From Missing Data Imputation to Data Generation."
        Journal of Computational Science (JCS), 2022.
    [2] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença.
        "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation."
        International Conference on Computational Science (ICCS), 2021.
    """

    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any] = {}):
        super().__init__(data=data, algo_parameters=algo_parameters)


class tabulator_GP(WSGAIN_GP):
    """This class is a facade of the WSGAIN-GP algorithm [1, 2].

    References
    ----------
    [1] Diogo Telmo Neves, João Alves, Marcel Ganesh Naik, Alberto José Proença, Fabian Praßer.
        "From Missing Data Imputation to Data Generation."
        Journal of Computational Science (JCS), 2022.
    [2] Diogo Telmo Neves, Marcel Ganesh Naik, Alberto Proença.
        "SGAIN, WSGAIN-CP and WSGAIN-GP: Novel GAN Methods for Missing Data Imputation."
        International Conference on Computational Science (ICCS), 2021.
    """

    def __init__(self, data: np.ndarray, algo_parameters: Dict[str, Any] = {}):
        super().__init__(data=data, algo_parameters=algo_parameters)


class TabularDataGenerator:

    TABULAR_DATA_GENERATORS: Dict[str, Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = {
        'tabulator': tabulator,
        'tabulator-CP': tabulator_CP,
        'tabulator-GP': tabulator_GP
    }
    """The supported tabular data generators."""

    def __init__(self, data: np.ndarray, algo: str = 'tabulator', algo_parameters: Dict[str, Any] = {}):
        self.data: np.ndarray = data.copy()  # to NOT mess up with the given `data`
        # if algo not in Generator.GENERATORS:
        #     raise ValueError("Expecting one of the supported tabular data generators -- "
        #                      f"{' ,'.join([TabularDataGenerator.TABULAR_DATA_GENERATORS])} -- "
        #                      f"as the algorithm to be used for tabular data generation but got: {algo}.")
        self.algo: Callable[[np.ndarray, Dict[str, Any]], np.ndarray] = \
            TabularDataGenerator.TABULAR_DATA_GENERATORS[algo] \
                if algo in TabularDataGenerator.TABULAR_DATA_GENERATORS else tabulator
        self.algo_parameters: Dict[str, Any] = algo_parameters
        ################################################################################################################
        # TODO: VERIFY IF THIS CAN BE REMOVED --> LOOK AT THE `SGAIN` IMPLEMENTATION
        self.verbose: bool = algo_parameters['verbose'] if 'verbose' in algo_parameters else False
        self.n_obs: int = self.data.shape[0]
        self.dim: int = self.data.shape[1]
        ################################################################################################################

    def _execute(self, n_samples: int = 100) -> np.ndarray:
        # TODO: THIS ALGORITHM IS EXTREMELY SLOW DUE TO ITERATIONS OVER THE CELLS OF THE pandas DataFrame
        #       POSSIBLE OPTIMIZATIONS ARE:
        #           1. CHANGE THE ORDER OF LOOPS FOM ROW..COLUMN TO COLUMN..ROW
        #           2. USE MASK MATRICES AND "LINEAR ALGEBRA"
        #           3. USE DICTIONARIES AND SORTING
        synthetic_data: np.ndarray = self.data.copy()
        indices: Dict[int, Tuple[int, int]] = {}
        row: int
        col: int
        key: int = 0

        self.verbose = True

        for row in range(self.n_obs):
            for col in range(self.dim):
                indices[key] = (row, col)
                key += 1
        if self.verbose:
            print()
            print("purify.generation.tabulator.TabularDataGenerator :: _execute()")
        while indices:
            indices_sample: List[int] = random.sample(
                population=list(indices.keys()),
                k=min(int(math.ceil(self.n_obs * self.dim * self.algo_parameters['miss_rate'])), len(indices)))
            # for each run there is the need of using a fresh copy of the original data
            # (i.e., the synthetic data will always be generated from the original data)
            # additionally, after the preprocessing stage, the original data is only composed by numeric data
            # (i.e., each variable is either an `int` or a `float` data type) yet there is the need to ensure
            # that it is only a `float` data type, otherwise there will be a data type mismatch
            # when introducing missing values into an `int` variable
            data: np.ndarray = self.data.copy().astype(dtype=float)
            positions: List[int, int] = []

            # remove each index in `indices_sample` from `indices` and
            # perform the amputation of the `data` (i.e., of the numpy ndarray)
            for index in indices_sample:
                row, col = indices.pop(index)   # remove the index
                data[row, col] = np.NaN         # ampute the cell mapped by the index
                positions.append((row, col))
            data = self.algo(data=data, algo_parameters=self.algo_parameters).execute()
            for row, col in positions:
                synthetic_data[row, col] = data[row, col]
            if self.verbose:
                print()
                print(f"first {min(5, n_samples)} row(s) of synthetic data:")
                print(synthetic_data[0:min(5, n_samples), :])
                print("...")
                print(f"shape: {synthetic_data.shape}")

        self.verbose = False

        return synthetic_data

    def sampler(self, n_samples: int = 100) -> np.ndarray:
        synthetic_data_list: List[np.ndarray] = [
            self._execute(n_samples=n_samples) for _ in range(n_samples // self.n_obs)]
        synthetic_data: np.ndarray

        if n_samples % self.n_obs > 0:
            random_indices: np.ndarray = np.random.choice(a=self.n_obs, size=(n_samples % self.n_obs), replace=False)

            # slice from the output of this final run using random indices of it
            synthetic_data_list.append(self._execute()[random_indices, :])
        synthetic_data = np.concatenate(synthetic_data_list, axis=0)
        if self.verbose:
            print()
            print("purify.generation.tabulator.TabularDataGenerator :: sampler()")
            print(f"algo:  {self.algo}")
            print(f"first {min(5, n_samples)} row(s) of synthetic data:")
            print(synthetic_data[0:min(5, n_samples), :])
            print("...")
            print(f"shape: {synthetic_data.shape}")
        return synthetic_data

