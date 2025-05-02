import pandas as pa

from data_prep.add_demogs import add_county

def rank_percentile_calc(df):
    from scipy.stats import rankdata, percentileofscore
    columns = [x for x in df.columns if '_act' in x]
    for col in columns:
        df[col.replace('_act','_per')] = (df[col]/df.total_pop)*100
        df[col.replace('_act','_rank')] = rankdata(df[col])
        df[col.replace('_act','_percentile')] = ((rankdata(df[col])-1)/df.shape[0] ) *100
    return df


def census_ins():
    df = pa.read_csv('../Data/zip_codes_census_insurance.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'rate' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    df.fillna(0,inplace=True)
    for per in per_columns:
        df[per.replace('rate','ins_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'rate' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'ins_' in x]+['total_pop']:
        measures[metric] ='sum'
    
    dma = pa.read_csv('../Data/nielsen_dma_data.csv')
    dma['zipcode'] = dma['zipcode'].map(lambda x: int(x))
    df = pa.merge(df, dma, left_on= 'zip', right_on='zipcode', how='left')
    df.drop('zipcode', axis=1, inplace=True)
    
   
    
    df = df.groupby(['data_source', 'dma_code', 'state'], as_index=False).agg(measures)
    df = rank_percentile_calc(df)
    df.to_csv('../Processed Data/dma/dma_ins.csv')
    return df



def census_demogs_state():
    df = pa.read_csv('../Data/zip_codes_census_demographics.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'per' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    for per in per_columns:
        df[per.replace('per','cen_dem_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'per' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'cen_dem' in x]+['total_pop']:
        measures[metric] ='sum'
    
    dma = pa.read_csv('../Data/nielsen_dma_data.csv')
    dma['zipcode'] = dma['zipcode'].map(lambda x: int(x))
    df = pa.merge(df, dma, left_on= 'zip', right_on='zipcode', how='left')
    df.drop('zipcode', axis=1, inplace=True)
    
   
    
    df = df.groupby(['data_source', 'state'], as_index=False).agg(measures)
    rank_percentile_calc(df)
    df.to_csv('../Processed Data/state_cen_demo.csv')
    return df
    
    
def census_demogs_income_state():
    df = pa.read_csv('../Data/zip_codes_census_income_education.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'per' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    for per in per_columns:
        df[per.replace('per','cen_inc_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'per' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'cen_inc' in x]+['total_pop']:
        measures[metric] ='sum'
    
    dma = pa.read_csv('../Data/nielsen_dma_data.csv')
    dma['zipcode'] = dma['zipcode'].map(lambda x: int(x))
    df = pa.merge(df, dma, left_on= 'zip', right_on='zipcode', how='left')
    df.drop('zipcode', axis=1, inplace=True)
    
   
    
    df = df.groupby(['data_source', 'state'], as_index=False).agg(measures)
    rank_percentile_calc(df)
    df.to_csv('../Processed Data/state_cen_inc.csv')
    return df



def census_demogs_dma():
    df = pa.read_csv('../Data/zip_codes_census_demographics.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'per' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    for per in per_columns:
        df[per.replace('per','cen_dem_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'per' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'cen_dem' in x]+['total_pop']:
        measures[metric] ='sum'
    
    dma = pa.read_csv('../Data/nielsen_dma_data.csv')
    dma['zipcode'] = dma['zipcode'].map(lambda x: int(x))
    df = pa.merge(df, dma, left_on= 'zip', right_on='zipcode', how='left')
    df.drop('zipcode', axis=1, inplace=True)
    
   
    
    df = df.groupby(['data_source', 'dma_code', 'state'], as_index=False).agg(measures)
    rank_percentile_calc(df)
    df.to_csv('../Processed Data/dma/dma_cen_demo.csv')
    return df
    
    
def census_demogs_income():
    df = pa.read_csv('../Data/zip_codes_census_income_education.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'per' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    for per in per_columns:
        df[per.replace('per','cen_inc_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'per' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'cen_inc' in x]+['total_pop']:
        measures[metric] ='sum'
    
    dma = pa.read_csv('../Data/nielsen_dma_data.csv')
    dma['zipcode'] = dma['zipcode'].map(lambda x: int(x))
    df = pa.merge(df, dma, left_on= 'zip', right_on='zipcode', how='left')
    df.drop('zipcode', axis=1, inplace=True)
    
   
    
    df = df.groupby(['data_source', 'dma_code', 'state'], as_index=False).agg(measures)
    rank_percentile_calc(df)
    df.to_csv('../Processed Data/dma/dma_cen_inc.csv')
    return df





def census_ins_county():
    df = pa.read_csv('../Data/zip_codes_census_insurance.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'rate' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    df.fillna(0,inplace=True)
    for per in per_columns:
        df[per.replace('rate','ins_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'rate' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'ins_' in x]+['total_pop']:
        measures[metric] ='sum'
    
    df = add_county(df, zipvar='zip')
    
   
    
    df = df.groupby(['data_source', 'county', 'state'], as_index=False).agg(measures)
    df = rank_percentile_calc(df)
    df.to_csv('../Processed Data/county/county_ins.csv')
    return df


def census_demogs_county():
    df = pa.read_csv('../Data/zip_codes_census_demographics.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'per' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    for per in per_columns:
        df[per.replace('per','cen_dem_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'per' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'cen_dem' in x]+['total_pop']:
        measures[metric] ='sum'
    
    df = add_county(df, zipvar='zip')
    
   
    
    df = df.groupby(['data_source', 'county', 'state'], as_index=False).agg(measures)
    rank_percentile_calc(df)
    df.to_csv('../Processed Data/county/county_cen_demo.csv')
    return df
    
    
def census_demogs_income_county():
    df = pa.read_csv('../Data/zip_codes_census_income_education.csv')
    columns = [x for x in df.columns if 'percentile' not in x and 'rank' not in x]
    per_columns =[x for x in columns if 'per' in x]
    #df['cen_dem_total_pop'] = df['total_pop']
    #df.drop('total_pop', axis=1, inplace=True)
    for per in per_columns:
        df[per.replace('per','cen_inc_act')] =df[per]* df['total_pop']/ 100
    df = df[[x for x in df.columns if 'per' not in x and 'rank' not in x] ]
    
    df['zip'] = df['zip'].map(lambda x: int(x))
    
    measures = {}
    
    for metric in [x for x in df.columns if 'cen_inc' in x]+['total_pop']:
        measures[metric] ='sum'
    
    df = add_county(df, zipvar='zip')
    
   
    
    df = df.groupby(['data_source', 'county', 'state'], as_index=False).agg(measures)
    rank_percentile_calc(df)
    df.to_csv('../Processed Data/county/county_cen_inc.csv')
    return df