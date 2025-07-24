import pandas as pd
import country_converter as coco

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, col)).strip() for col in df.columns]
    return df

def fetch_country_data():
    # 1) URLs
    gdp_url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
    cab_url = 'https://en.wikipedia.org/wiki/List_of_sovereign_states_by_current_account_balance'
    dem_url = 'https://en.wikipedia.org/wiki/The_Economist_Democracy_Index'

    # 2) GDP (IMF-Forecast 2025 in Mio. USD)
    gdp_raw = pd.read_html(gdp_url, decimal=',', thousands='.')[2]
    gdp_raw = flatten_columns(gdp_raw)
    gdp_df = gdp_raw.iloc[:, [0, 1]].copy()
    gdp_df.columns = ['country_name', 'gdp_million_usd']

    # 3) Current Account Balance
    cab_raw = pd.read_html(cab_url, decimal=',', thousands='.')[0]
    cab_raw = flatten_columns(cab_raw)
    cab_df = cab_raw.iloc[:, [0, 1]].copy()
    cab_df.columns = ['country_name', 'trade_balance_million_usd']
    cab_df['trade_balance_million_usd'] = (
        cab_df['trade_balance_million_usd']
        .astype(str)
        .str.replace(r'[^0-9\.-]', '', regex=True)
    )
    cab_df['trade_balance_million_usd'] = pd.to_numeric(
        cab_df['trade_balance_million_usd'], errors='coerce'
    )
    cab_df['trade_balance_billion_usd'] = cab_df['trade_balance_million_usd'] / 1000
    cab_df.drop(columns='trade_balance_million_usd', inplace=True)

    # 4) Democracy Index
    dem_tables = pd.read_html(dem_url)
    raw_dem    = flatten_columns(dem_tables[5])
    dem_df     = raw_dem[['Country', 'Regime type']].copy()
    dem_df.columns = ['country_name', 'pol_index']

    # 5) Merge all three
    df_all = (
        gdp_df
        .merge(cab_df[['country_name', 'trade_balance_billion_usd']],
               on='country_name', how='outer')
        .merge(dem_df[['country_name', 'pol_index']],
               on='country_name', how='outer')
    )

    cc = coco.CountryConverter()
    df_all['country_code'] = cc.pandas_convert(
        series=df_all['country_name'],
        to='ISO3',
        not_found=None
    )
    df_all = df_all[[
        'country_code',
        'country_name',
        'gdp_million_usd',
        'trade_balance_billion_usd',
        'pol_index'
    ]]
    # Nur Länder mit gültigem Code und Index
    df_countries = df_all.dropna(subset=['country_code', 'pol_index'])
    return df_countries

if __name__ == "__main__":
    df = fetch_country_data()
    print(df.head())
