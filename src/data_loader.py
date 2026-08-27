"""
Data Loader Module for Credit Risk Intelligence System
------------------------------------------------------
Provides high-performance data ingestion, automated dtype downcasting for 
memory optimization, and comprehensive structural/missingness reporting for 
credit risk datasets (Home Credit Default Risk).

Author: Portfolio Project
Phase: 1 - Stage 1
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataLoader")


def optimize_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Downcasts numeric columns (int64 -> int32/int16/int8, float64 -> float32)
    to optimize memory footprint without loss of numeric precision for risk calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.
    verbose : bool, default=False
        Whether to print memory reduction metrics.

    Returns
    -------
    pd.DataFrame
        Memory-optimized dataframe.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                # Float32 is sufficient for financial amounts and normalized indices
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Memory footprint reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df


class DataLoader:
    """
    Handles robust ingestion, path resolution, and metadata inspection
    for credit risk datasets.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.app_train_path = os.path.join(self.data_dir, "application_train.csv")
        self.installments_path = os.path.join(self.data_dir, "installments_payments.csv")

    def load_application_train(
        self,
        nrows: Optional[int] = None,
        optimize_memory: bool = True
    ) -> pd.DataFrame:
        """
        Loads application_train.csv containing loan application records and default labels.

        Parameters
        ----------
        nrows : int, optional
            Number of rows to load (useful for rapid testing/prototyping).
        optimize_memory : bool, default=True
            Whether to downcast dtypes for memory optimization.

        Returns
        -------
        pd.DataFrame
            Application train dataset.
        """
        if not os.path.exists(self.app_train_path):
            raise FileNotFoundError(f"Application dataset not found at: {self.app_train_path}")

        logger.info(f"Loading application dataset from {self.app_train_path} (nrows={nrows})...")
        df = pd.read_csv(self.app_train_path, nrows=nrows)
        logger.info(f"Successfully loaded application data. Initial shape: {df.shape}")

        if optimize_memory:
            df = optimize_memory_usage(df, verbose=True)

        return df

    def load_installments(
        self,
        nrows: Optional[int] = None,
        optimize_memory: bool = True,
        usecols: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Loads installments_payments.csv containing granular transaction and repayment records.

        Parameters
        ----------
        nrows : int, optional
            Number of rows to load.
        optimize_memory : bool, default=True
            Whether to downcast dtypes.
        usecols : list, optional
            Subset of columns to load.

        Returns
        -------
        pd.DataFrame
            Installment payment transactions.
        """
        if not os.path.exists(self.installments_path):
            raise FileNotFoundError(f"Installments dataset not found at: {self.installments_path}")

        logger.info(f"Loading installments data from {self.installments_path} (nrows={nrows})...")
        df = pd.read_csv(self.installments_path, nrows=nrows, usecols=usecols)
        logger.info(f"Successfully loaded installments data. Initial shape: {df.shape}")

        if optimize_memory:
            df = optimize_memory_usage(df, verbose=True)

        return df

    @staticmethod
    def get_missing_summary(df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Generates a comprehensive missing value summary table.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to evaluate.
        top_n : int, optional
            Limit to top N missing columns.

        Returns
        -------
        pd.DataFrame
            Dataframe with Missing_Count, Missing_Percentage, and Data_Type.
        """
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        dtypes = df.dtypes

        summary = pd.DataFrame({
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_pct,
            "Dtype": dtypes
        })

        # Filter only columns that have missing values and sort descending
        missing_summary = summary[summary["Missing_Count"] > 0].sort_values(
            by="Missing_Percentage", ascending=False
        )

        if top_n is not None:
            return missing_summary.head(top_n)
        return missing_summary

    @staticmethod
    def inspect_dataset(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
        """
        Computes structural metrics including dimensions, column types, memory,
        and missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to inspect.
        name : str
            Descriptive name of the dataset.

        Returns
        -------
        dict
            Dictionary containing summary metrics.
        """
        total_rows, total_cols = df.shape
        mem_mb = df.memory_usage().sum() / (1024 ** 2)
        dtype_counts = df.dtypes.value_index = df.dtypes.value_counts().to_dict()
        missing_cols_count = (df.isnull().sum() > 0).sum()
        total_missing_cells = df.isnull().sum().sum()
        total_cells = total_rows * total_cols
        missing_cell_pct = (total_missing_cells / total_cells) * 100 if total_cells > 0 else 0.0

        print("=" * 80)
        print(f"[DATASET INSPECTION REPORT] {name.upper()}")
        print("=" * 80)
        print(f"  * Shape                     : {total_rows:,} rows x {total_cols:,} columns")
        print(f"  * Memory Usage              : {mem_mb:.2f} MB")
        print(f"  * Columns with Missing Data : {missing_cols_count} / {total_cols} ({(missing_cols_count/total_cols)*100:.1f}%)")
        print(f"  * Total Missing Cells       : {total_missing_cells:,} ({missing_cell_pct:.2f}% of all cells)")
        print(f"  * Data Type Breakdown       : {dtype_counts}")
        print("=" * 80)

        return {
            "name": name,
            "shape": (total_rows, total_cols),
            "memory_mb": mem_mb,
            "missing_columns_count": missing_cols_count,
            "total_missing_cells": total_missing_cells,
            "missing_cell_pct": missing_cell_pct,
            "dtype_counts": dtype_counts
        }


if __name__ == "__main__":
    loader = DataLoader(data_dir="data")
    app_df = loader.load_application_train(nrows=5000)
    DataLoader.inspect_dataset(app_df, name="Application Train (Sample)")
    missing_sum = DataLoader.get_missing_summary(app_df, top_n=10)
    print("\nTop 10 Missing Columns:")
    print(missing_sum)
