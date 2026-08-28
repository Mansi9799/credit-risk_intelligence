import pandas as pd

def calculate_dti(df):
    """Calculates Debt-to-Income Ratio."""
    df = df.copy()
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['DEBT_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    return df

def calculate_employment_ratio(df):
    """Calculates Employment-to-Age Ratio."""
    df = df.copy()
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['EMPLOYMENT_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df
