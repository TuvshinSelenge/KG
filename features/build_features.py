import pandas as pd
import numpy as np

def build_features(df_countries, merged_df):
    actors = list({*merged_df.Actor1Code, *merged_df.Actor2Code})
    features_df = df_countries.set_index('country_code')[
        ['gdp_million_usd','trade_balance_billion_usd']
    ]
    features_df['pol_enc'] = pd.Categorical(df_countries.pol_index).codes
    for col in ['gdp_million_usd','trade_balance_billion_usd']:
        features_df[col] = (
            features_df[col].astype(str).str.replace(r'[^0-9\.-]','',regex=True)
        )
        features_df[col] = pd.to_numeric(features_df[col],errors='coerce').fillna(0.0)
    feat_list = []
    for code in actors:
        if code in features_df.index:
            row = features_df.loc[code]
            feat_list.append([row['gdp_million_usd'], row['trade_balance_billion_usd'], row['pol_enc']])
        else:
            feat_list.append([0.0,0.0,0.0])
    return np.array(feat_list), actors
