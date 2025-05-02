import pandas as  pa



def perc_indicator(df):
    exclude = [x for x in df.columns if 'rank' in x]
    df = df[list(set(df.columns) -set(exclude) )]
    columns = [x for x in df.columns if 'percentile' in x]
    
    for col in columns:
        df.loc[((df[col] >= 75) ), col.replace('percentile','indic')] = 'Q_75_100'
        df.loc[((df[col] >= 50) & (df[col] < 75)), col.replace('percentile','indic')] = 'Q_50_75'
        df.loc[((df[col] >= 25) & (df[col] < 50)), col.replace('percentile','indic')] = 'Q_25_50'
        df.loc[((df[col] <25) ), col.replace('percentile','indic')] = 'Q_0_25'
    return df
    

def census_data_aggregate(df):
    columns = [x for x in df.columns if 'per' in x and 'percentile' not in x and 'rank' not in x]
    for col in columns:
        df[col.replace('per','TOTAL')] = df[col]*df['total_pop']
    return df
    
    
  
