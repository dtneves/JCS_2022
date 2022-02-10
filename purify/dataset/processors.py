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
# This module defines two classes -- PreProcessor and PostProcessor -- which are useful to perform
# a few data transformations.
# One should be aware that exception handling to take care of incorrect data types, incorrect parameters' values, and
# so forth is, typically, NOT performed, the rule is: We are all grown up (Python) programmers!
#
#
# Moto
# ----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
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

import numpy as np
import pandas as pd

from purify.dataset.metadata import Metadata

from typing import Any, Dict, List, Union


class PreProcessor:
    """This class provides a few methods that leverage a few data preprocessing operations.
    One should be aware that the implementation of this class is dependent (i.e., it relies) on
    the implementation of the :class:`purify.dataset.metadata.Metadata` class.
    """

    @classmethod
    def drop_vars(cls, dataset: str, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Drop each variable (i.e., column/feature) that is marked in its metadata to be dropped.
        One should be aware that this implementation is only for datasets that are supported through
        the :class:`purify.dataset.metadata.Metadata` class.

        Parameters
        ----------
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported through
            the :class:`purify.dataset.metadata.Metadata` class.
        df : DataFrame
            A pandas DataFrame with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output, which is useful, for instance, to debug and
            to trace the execution.

        Returns
        -------
        df_drop : DataFrame
            A copy of the given pandas DataFrame (i.e., `df`) without the variables (i.e., columns/features)
            that are marked in their metadata to be dropped.

        See Also
        --------
        :func:`purify.dataset.metadata.Metadata.vars_to_drop`
        """
        df_drop: pd.DataFrame = df.drop(columns=Metadata.vars_to_drop(dataset=dataset, df=df))

        if verbose:
            print()
            print("purify.dataset.processors.PreProcessor :: drop_vars()")
            print("Before dropping variables:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After dropping variables:")
            print(df_drop.head())
            print("...")
            print(df_drop.tail())
        return df_drop

    @classmethod
    def drop_nans(cls, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Drop each observation (i.e., row) that has at least one missing value (i.e., a numpy NaN).
        One should be aware that this implementation is only for datasets that are supported through
        the :class:`purify.dataset.metadata.Metadata` class.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame with data of the given `dataset`.
        verbose : bool, optional
            If True some info will be sent to the standard output, which is useful, for instance, to debug and
            to trace the execution.

        Returns
        -------
        df_drop : DataFrame
            A copy of the given pandas DataFrame (i.e., `df`) in which each observation (i.e., row)
            that has at least one missing value (i.e., a numpy NaN) is dropped.
        """
        df_drop: pd.DataFrame = df.dropna()

        if verbose:
            print()
            print("purify.dataset.processors.PreProcessor :: drop_nans()")
            print("Before dropping NaNs:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After dropping NaNs:")
            print(df_drop.head())
            print("...")
            print(df_drop.tail())
        return df_drop

    @classmethod
    def replace_miss_values_by_nans(cls, df: pd.DataFrame, dataset: str, verbose: bool = False) -> pd.DataFrame:
        """Replace the missing values of each variable (i.e., column/feature) by numpy NaNs.
        One should be aware that this implementation is only for datasets that are supported through
        the :class:`purify.dataset.metadata.Metadata` class.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame with data of the given `dataset`.
        dataset : str
            Dataset's (short) name, has to be one of the datasets supported through
            the :class:`purify.dataset.metadata.Metadata` class.
        verbose : bool, optional
            If True some info will be sent to the standard output, which is useful, for instance, to debug and
            to trace the execution.

        Returns
        -------
        df_rep : DataFrame
            A copy of the given pandas DataFrame (i.e., `df`) in which
            the missing values of each variable (i.e., column/feature) are replaced by numpy NaNs.
        """
        df_rep: pd.DataFrame = df.copy(deep=True)

        for variable in Metadata.DATASETS[dataset].keys():
            # it could happen that, for instance, the :func:`purify.dataset.processors.PreProcessor.drop_vars`
            # method has been invoked before this one and, if so, the current `variable` may NOT be part of
            # the given pandas DataFrame (i.e., may NOT be a column of `df`),
            # thus, it is required to perform the following test
            if variable in df_rep.columns:
                for value in Metadata.DATASETS[dataset][variable]['missing_values']:
                    df_rep[variable] = df_rep[variable].replace(to_replace=value, value=np.NaN)
        if verbose:
            print()
            print("purify.dataset.processors.PreProcessor :: replace_miss_values_by_nans()")
            print("Before replacing missing values:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After replacing missing values:")
            print(df_rep.head())
            print("...")
            print(df_rep.tail())
        return df_rep

    @classmethod
    def replace_values(cls,
                       df: pd.DataFrame,
                       to_replace: Dict[Union[List[str], List[int]], Dict[Any, Any]],
                       verbose: bool = False) -> pd.DataFrame:
        """TODO: ADD DOCUMENTATION"""
        df_rep: pd.DataFrame = df.replace(to_replace=to_replace)

        if verbose:
            print()
            print("purify.dataset.processors.PreProcessor :: replace()")
            print("Before replacing values:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After replacing values:")
            print(df_rep.head())
            print("...")
            print(df_rep.tail())
        return df_rep


class PostProcessor:
    """This class provides one method that allows to set the proper data type of each variable (i.e., feature/column).
    One should be aware that the implementation of this class is dependent (i.e., it relies) on
    the implementation of the :class:`purify.dataset.metadata.Metadata` class.
    """

    @classmethod
    def set_data_types(cls, dataset: str, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """TODO: ADD DOCUMENTATION"""
        df_copy: pd.DataFrame = df.copy()  # to NOT mess up with the given pandas DataFrame

        for col in df_copy.columns:
            # if the variable's data type is `int` then get rid of the decimal part, if any
            if Metadata.DATASETS[dataset][col]['data_type'] == int:
                df_copy[col] = df_copy[col].round()
            df_copy[col] = df_copy[col].astype(dtype=Metadata.DATASETS[dataset][col]['data_type'])
        if verbose:
            print()
            print("purify.dataset.processors.PostProcessor :: set_data_types()")
            print("Before setting data types:")
            print(df.head())
            print("...")
            print(df.tail())
            print("After setting data types:")
            print(df_copy.head())
            print("...")
            print(df_copy.tail())
        return df_copy

