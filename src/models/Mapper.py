import pandas as pd
    
def upper_case_df(df):
    for col in df.columns:
        if df[col].dtype in (str, object):
            df[col] = df[col].str.upper()
    return df


def strip_whitespaces(df):
    for col in df.columns:
        if df[col].dtype in (str, object):
            df[col] = df[col].str.strip()
    return df


def df_ISO3_mapper(df, mapper):
    df = upper_case_df(df)
    df = strip_whitespaces(df)
    return df.replace(mapper)
