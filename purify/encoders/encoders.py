########################################################################################################################
# Research Centers:
# -----------------
# Centro ALGORITMI - School of Engineering – University of Minho
# Braga - Portugal
# http://algoritmi.uminho.pt/
#
# Medical Informatics Group
# BIH - Berlin Institute of Health
# Charité - Universitätsmedizin Berlin
# https://www.bihealth.org/en/research/research-groups/fabian-prasser/
#
# Description:
# ------------
# This module allows to perform basic data encoders operations -- label encoding and one-hot encoding -- and
# to inverse (i.e., to revert) those data transformations.
# One should be aware that exception handling to take care of incorrect data types, incorrect parameters' values, and
# so forth is, typically, NOT performed, the rule is: We are all grown up (Python) programmers!
#
#
# Moto:
# -----
# "We think too much and feel too little. More than machinery we need humanity."
#                         -- Excerpt of the final speech from The Great Dictator
#
#
# Related Work:
# -------------
#   * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
#   * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#   * https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
#
#
# Authors:
# --------
# diogo telmo neves -- {dneves@di.uminho.pt, diogo-telmo.neves@charite.de, tada.science@gmail.com}
#
#
# Copyright:
# ----------
# Copyright (c) 2021 diogo telmo neves.
# All rights reserved.
#
#
# Conditions:
# -----------
# This code is free/open source code but the following conditions must be met:
#   * Redistributions of source code must retain the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#     the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#
# DISCLAIMER:
# -----------
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
# Date:
# -----
# September 2021
########################################################################################################################

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from purify.dataset.metadata import Metadata
from purify.dataset.processors import PostProcessor

from typing import Dict, List, Tuple, Union


# TODO: TURN `label_encoders_fit_transform()` AND `label_encoders_inverse_transform()` MEMBERS OF A CLASS
def label_encoders_fit_transform(data: pd.DataFrame,
                                 discrete_vars: Union[List[str], List[int]],
                                 verbose: bool = False) -> Tuple[pd.DataFrame, Dict[Union[str, int], LabelEncoder]]:
    """Applies label encoding to each of the given `discrete_vars` of the given `data`,
    following the paradigm 'fit and transform'.

    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param discrete_vars: The discrete variables (aka the categorical columns/features) of the given `data`.
    It is important to notice that the list is either a list of columns' names (i.e., a list of `str`) or
    a list of columns indices (i.e., a list of `int`).
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: A tuple that is composed by the result of the data transformation as well as
    by a dictionary that maps each column's name or column's index to an instance of `LabelEncoder`,
    which allows to invert (i.e., to revert) the transformation.
    The dictionary can be depicted as follows: {<column's name> | <column's index>: <label encoder>, ...}
    """
    df: pd.DataFrame
    label_encoders: Dict[Union[str, int], LabelEncoder]

    # more than just sanity checks
    if isinstance(data, pd.DataFrame):  # pd.DataFrame
        df = data.copy(deep=True)  # to NOT mess up the given data structure
    else:  # NOT a pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")
    if set(discrete_vars) - set(df.columns):
        raise ValueError("Bad list of discrete columns, "
                         "at least one of them does NOT belong to the columns of the given `data`.")
    label_encoders = {}
    if verbose:
        print("\nBefore applying label encoder:")
        print(df.head())
        print("...")
        print(df.tail())
    for discrete_var in discrete_vars:
        label_encoders[discrete_var] = LabelEncoder()
        df[discrete_var] = label_encoders[discrete_var].fit_transform(y=df[discrete_var])
    if verbose:
        print("\nAfter applying label encoder:")
        print(df.head())
        print("...")
        print(df.tail())
    return df, label_encoders


# TODO: TURN `label_encoders_fit_transform()` AND `label_encoders_inverse_transform()` MEMBERS OF A CLASS
def label_encoders_inverse_transform(dataset: str,
                                     data: pd.DataFrame,
                                     label_encoders: Dict[Union[str, int], LabelEncoder],
                                     verbose: bool = False) -> pd.DataFrame:
    """Applies an inverse transformation to the given `data` using the given `label_encoders`.

    :param dataset: The short name of a dataset.
    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param label_encoders: A dictionary that maps each column's name or column's index to an instance of `LabelEncoder`,
    which allows to invert (i.e., to revert) the (previously applied data) transformation.
    The dictionary can be depicted as follows: {<column's name> | <column's index>: <label encoder>, ...}
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: An instance of `pd.DataFrame` with the result of the inverse (i.e., the reverse) transformation.
    """
    df: pd.DataFrame

    # more than just sanity checks
    if isinstance(data, pd.DataFrame):  # pd.DataFrame
        df = data.copy(deep=True)  # to NOT mess up the given data structure
    else:  # NOT an pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")
    if set(label_encoders.keys()) - set(df.columns):
        raise ValueError("At least one of the discrete columns does NOT belong to the columns of the given `data`.")

    if verbose:
        print("\nBefore applying inverse transform:")
        print(df.head())
        print("...")
        print(df.tail())
    for discrete_var, label_encoder in label_encoders.items():
        df[discrete_var] = label_encoder.inverse_transform(y=data[discrete_var].astype(int))
    # enforce the original data types
    df = PostProcessor.set_data_types(dataset=dataset, df=df, verbose=verbose)
    if verbose:
        print("\nAfter applying inverse transform:")
        print(df.head())
        print("...")
        print(df.tail())
    return df


# TODO: TURN `get_dummies_fit_transform()` AND `get_dummies_inverse_transform()` MEMBERS OF A CLASS
def get_dummies_fit_transform(data: pd.DataFrame,
                              discrete_vars: Union[List[str], List[int]],
                              verbose: bool = False) -> pd.DataFrame:
    """Applies a data transformation that is identically to the well known One-Hot Encoding data transformation.

    DISCLAIMER: THIS FUNCTION DOES NOT PERFORM A `FIT AND TRANSFORM` OPERATION,
                ITS BEHAVIOR LOOKS LIKE A `FIT AND TRANSFORM` OPERATION BUT IT IS NOT.

    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param discrete_vars: The discrete variables (aka the categorical columns/features) of the given `data`.
    It is important to notice that the list is either a list of columns' names (i.e., a list of `str`) or
    a list of columns indices (i.e., a list of `int`).
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: An instance of `pd.DataFrame` with the result of the (alike) 'fit and transform' data transformation.
    """
    df: pd.DataFrame

    # just a sanity check
    if not isinstance(data, pd.DataFrame):  # NOT an pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")
    df = pd.get_dummies(data=data, columns=discrete_vars)
    if verbose:
        print("\nBefore applying fit transform:")
        print(data.head())
        print("...")
        print(data.tail())
        print("\nAfter applying fit transform:")
        print(df.head())
        print("...")
        print(df.tail())
    return df


# TODO: TURN `get_dummies_fit_transform()` AND `get_dummies_inverse_transform()` MEMBERS OF A CLASS
def get_dummies_inverse_transform(dataset: str,
                                  data: pd.DataFrame,
                                  discrete_vars: Union[List[str], List[int]],
                                  vars_order: Union[List[str], List[int]] = None,
                                  verbose: bool = False) -> pd.DataFrame:
    """Applies a data transformation that is identically to the well known inverse transformation that is possible to
    apply after applying the One-Hot Encoding data transformation based on the 'fit and transform' paradigm.

    DISCLAIMER: THIS FUNCTION DOES NOT PERFORM TRULY AN `INVERSE TRANSFORM` OPERATION,
                HOWEVER ITS BEHAVIOR LOOKS LIKE INDEED AN `INVERSE TRANSFORM` OPERATION.

    :param dataset: The short name of a dataset.
    :param data: The data (as a `pd.DataFrame`) to be transformed.
    :param discrete_vars: The discrete variables (aka the categorical columns/features) of the given `data`.
    It is important to notice that the list is either a list of columns' names (i.e., a list of `str`) or
    a list of columns indices (i.e., a list of `int`).
    :param vars_order: The desired variables' order of the result (i.e., pandas DataFrame).
    :param verbose: To control the verbosity -- amongst other usages, it is useful to debug.
    :return: An instance of `pd.DataFrame` with the result of the (alike) inverse (i.e., the reverse) transformation.
    """
    df: pd.DataFrame
    df_final: pd.DataFrame

    # just a sanity check
    if not isinstance(data, pd.DataFrame):  # NOT an pd.DataFrame
        raise ValueError(f"Expecting a pandas DataFrame but got: {type(data)}.")
    df = data.copy(deep=True)  # to NOT mess up the given data structure
    df_final = pd.DataFrame()
    if verbose:
        print("\nBefore applying inverse transform:")
        print(df.head())
        print("...")
        print(df.tail())
    for variable in discrete_vars:
        # finds the columns associated with the current `variable`
        columns: List[str] = [column for column in df.columns if column.startswith(f"{variable}_")]
        # creates a pandas DataFrame with the columns from which is possible to invert the transformation
        df_tmp: pd.DataFrame = pd.DataFrame(data=df[columns])
        # most important statement of this algorithm,
        # the first part -- `df_tmp.idxmax(axis=1)` -- does the magic of inverting the transformation,
        # whereas the second part removes the unwanted string from the head of the string of each cell
        df_tmp[variable] = df_tmp.idxmax(axis=1).map(lambda x: x.replace(f"{variable}_", ""))
        # drops the unwanted columns and concatenates both pandas DataFrames
        df_final = pd.concat(objs=[df_final, df_tmp.drop(columns=columns)], axis=1)
        # ensure that the discrete variable has the original data type
        # TODO: CALL A TRANSFORMATION --> SEE LINE #262
        df_final[variable] = df_final[variable].astype(dtype=Metadata.DATASETS[dataset][variable]['data_type'])
    # adds also the continuous variables (i.e. columns/features) and, then, put them all in the original order
    df_final = pd.concat(objs=[df[set(vars_order) - set(discrete_vars)], df_final], axis=1)[vars_order]
    # enforce the original data types
    # TODO: NO NEED TO DO IT FOR ALL --> ONLY FOR THE CONTINUOUS
    df_final = PostProcessor.set_data_types(dataset=dataset, df=df_final, verbose=verbose)
    if verbose:
        print("\nAfter applying inverse transform:")
        print(df_final.head())
        print("...")
        print(df_final.tail())
    return df_final

