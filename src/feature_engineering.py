import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def engineer_application_features(self, df):
        df = df.copy()
        
        # 1. Financial Burden Ratios
        df['DEBT_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
        df['PAYMENT_BURDEN_INDEX'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
        
        # 2. Demographic & Employment Ratios
        df['EMPLOYMENT_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)
        df['DISPOSABLE_INCOME_PER_MEMBER'] = (df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY'].fillna(0)) / (df['CNT_FAM_MEMBERS'] + 1e-5)
        df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
        
        # 3. External Score Interactions
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        for col in ext_cols:
            if col not in df.columns:
                df[col] = 0.5  # Safe median fallback
        
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
        df['EXT_SOURCE_PRODUCT'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['EXT_SOURCE_WEIGHTED'] = df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 1
        df['EXT_SOURCE_X_AGE'] = df['EXT_SOURCE_MEAN'] * (-df['DAYS_BIRTH'] / 365.25)
        df['EXT_SOURCE_X_DTI'] = df['EXT_SOURCE_MEAN'] * df['DEBT_TO_INCOME_RATIO']
        
        # Automatically generate the remaining one-hot encoded categorical columns to reach ~167 features
        df = pd.get_dummies(df, drop_first=True)
        return df
        
    def engineer_installment_features(self, inst_df):
        # 4. Repayment Behavioral Velocity
        inst_df['DPD'] = (inst_df['DAYS_ENTRY_PAYMENT'] - inst_df['DAYS_INSTALMENT']).clip(lower=0)
        inst_df['LATE_PAYMENT'] = (inst_df['DPD'] > 0).astype(int)
        inst_df['PAYMENT_RATIO'] = inst_df['AMT_PAYMENT'] / (inst_df['AMT_INSTALMENT'] + 1e-5)
        inst_df['PAYMENT_DEFICIT'] = (inst_df['AMT_INSTALMENT'] - inst_df['AMT_PAYMENT']).clip(lower=0)
        
        agg_dict = {
            'DPD': ['mean', 'max'],
            'LATE_PAYMENT': ['mean'],
            'PAYMENT_RATIO': ['mean'],
            'PAYMENT_DEFICIT': ['sum', 'mean']
        }
        agg_df = inst_df.groupby('SK_ID_CURR').agg(agg_dict)
        agg_df.columns = [
            'INST_AVG_DPD', 'INST_MAX_DPD', 'INST_PCT_LATE',
            'INST_AVG_PAYMENT_RATIO', 'INST_TOTAL_DEFICIT', 'INST_AVG_DEFICIT'
        ]
        
        # Velocity / Recency 
        recent = inst_df[inst_df['DAYS_INSTALMENT'] >= -180].groupby('SK_ID_CURR')['DPD'].mean()
        agg_df['INST_REC180_AVG_DPD'] = recent
        agg_df['INST_DPD_VELOCITY'] = agg_df['INST_REC180_AVG_DPD'] / (agg_df['INST_AVG_DPD'] + 1e-5)
        
        return agg_df.fillna(0)

    def build_feature_matrix(self, app_df, inst_df):
        app_feat = self.engineer_application_features(app_df)
        inst_feat = self.engineer_installment_features(inst_df)
        final_df = app_feat.merge(inst_feat, on='SK_ID_CURR', how='left')
        
        # Fill NAs created by the join
        final_df = final_df.fillna(0)
        return final_df
