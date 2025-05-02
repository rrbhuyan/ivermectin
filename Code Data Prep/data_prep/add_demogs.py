import gc
import pandas as pa
from data_prep.census_data_aggregate import perc_indicator

def before_after(df):
    print('before: ', df.shape )
    print('after: ', df.dropna().shape )

def unmatched(df,var):
    if isinstance(var,list):
        var=var
    else:
        var=[var]
    print('unmatch %: ', (df.shape[0] - df.dropna(subset =var).shape[0] ) / df.shape[0]  )

def rename_cols(df, prefix, exclude): 
    columns = list(set(df.columns) - set(exclude))
    rename_dict = {}
    for x in columns:
        rename_dict[x] = prefix + x
    df = df.rename(columns =rename_dict)
    return df

def add_trump(df, year=2016):
    state_level = state_trump(year=year)
    df= pa.merge(df, state_level, on='state', how='inner')
    return df


def add_trump_old(df, year=2016):
    
    
    election_results = pa.read_csv("../Data/countypres_2000-2020.csv")
    if year==2020:
        election_results_2020 = election_results[election_results.year == 2020]
        election_results_2020.isna().any()
        election_results_2020_T = election_results_2020.pivot(values = "candidatevotes", columns="candidate")
        election_results_2020_T["state"] = election_results_2020.state
        election_results_2020_T.fillna(0, inplace=True)
        election_results_2020_final = election_results_2020_T.groupby("state", as_index=False).sum()
        election_results_2020_final["Trump_percent_20"] = election_results_2020_final['DONALD J TRUMP'] / election_results_2020_final.sum(axis=1, numeric_only=True)
        election_results_2020_final=election_results_2020_final[["state","Trump_percent_20"]]
        df= pa.merge(df, election_results_2020_final, on='state', how='inner')
        #election_results_2020_final.head()
    if year==2016:
        election_results_2016 = election_results[election_results.year == 2016]
        election_results_2016.isna().any()
        election_results_2016_T = election_results_2016.pivot(values = "candidatevotes", columns="candidate")
        election_results_2016_T["state"] = election_results_2016.state
        election_results_2016_T.fillna(0, inplace=True)
        election_results_2016_final = election_results_2016_T.groupby("state", as_index=False).sum()
        election_results_2016_final["Trump_percent_16"] = (election_results_2016_final['DONALD TRUMP'] / election_results_2016_final.sum(axis=1, numeric_only=True))#- (1 - election_results_2016_final['DONALD TRUMP'] / election_results_2016_final.sum(axis=1, numeric_only=True))
        election_results_2016_final=election_results_2016_final[["state","Trump_percent_16"]]
        df= pa.merge(df, election_results_2016_final, on='state', how='inner')
    return df
    #election_results_2016_final.head()

def add_politics(df):
    print('adding politics')
    govs = pa.read_csv('../Processed Data/governor_blue_red_state_year.csv')
    print('p0litics min', govs['year'].min())
    print('politics max', govs['year'].max())
    before_after(govs)
    
    govs = rename_cols(govs, 'pol_',['state','year'])
    df = pa.merge(df, govs, on=['state','year'], how='left')
    
    return df



def add_census_demo(df):
    print('adding demo')
    govs = pa.read_csv('../Data/zip_codes_census_demographics.csv')
    govs.state = govs.state.map(lambda x:x.upper())
    before_after(govs)
    keep = ['zip','total_pop','per_female_percentile','per_white_percentile','per_hispanic_percentile','per_black_percentile','per_asian_percentile',  'per_under18_percentile','per_over65_percentile']
    govs = govs[keep]
    govs['zip'] = govs['zip'].map(lambda x: int(x))
    govs = rename_cols(govs, 'cen_dem_',['zip'])
    govs = perc_indicator(govs)
    df = pa.merge(df, govs, left_on= ['state','pharmacy_zip_code'], right_on=['state','zip'], how='left')
    df.drop('zip', axis=1, inplace=True)
    unmatched(df,'cen_dem_total_pop')
    return df


def add_census_income(df):
    print('adding income')
    govs = pa.read_csv('../Data/zip_codes_census_income_education.csv')
    govs.state = govs.state.map(lambda x:x.upper())
    before_after(govs)
    keep = ['zip','total_pop','gini_percentile',  'med_income_percentile' ,
           'per_hh_income_below_poverty_percentile',
    
    'per_hh_income_1xto2x_poverty_percentile',
    
    'per_hh_income_2xto3x_poverty_percentile',
    
    'per_hh_income_3xto4x_poverty_percentile',
    
    'per_hh_income_4xto5x_poverty_percentile',
    
    'per_hh_income_5x_poverty_percentile'

           
           ]
    
    govs = govs[keep]
    govs['zip'] = govs['zip'].map(lambda x: int(x))
    govs = rename_cols(govs, 'cen_inc_',['zip'])
    govs = perc_indicator(govs)
    df = pa.merge(df, govs, left_on= ['state','pharmacy_zip_code'], right_on=['state','zip'], how='left')
    unmatched(df,'cen_inc_total_pop')
    df.drop('zip', axis=1, inplace=True)
    return df



def add_census_insurance(df):
    print('adding insurance')
    govs = pa.read_csv('../Data/zip_codes_census_insurance.csv')
    govs.state = govs.state.map(lambda x:x.upper())
    keep = ['zip','total_pop','uninsured_percentile']
    govs = govs[keep]
    before_after(govs)
    govs['zip'] = govs['zip'].map(lambda x: int(x))
    govs = rename_cols(govs, 'cen_ins_',['zip'])
    govs = perc_indicator(govs)
    df = pa.merge(df, govs, left_on= ['state','pharmacy_zip_code'], right_on=['state','zip'], how='left')
    unmatched(df,'cen_ins_total_pop')
    df.drop('zip', axis=1, inplace=True)
    return df


def add_dma(df):
    print('adding dma')
    govs = pa.read_csv('../Data/nielsen_dma_data.csv')
    govs['zipcode'] = govs['zipcode'].map(lambda x: int(x))
    df = pa.merge(df, govs, left_on= 'pharmacy_zip_code', right_on='zipcode', how='left')
    unmatched(df,'dma_code')
    df.drop('zipcode', axis=1, inplace=True)
    return df


def add_flags(df):
    print('add_flags')
    #from add_demogs import add_politics,add_census_demo, add_census_income, add_census_insurance,add_dma
    df = add_politics(df)
    df = add_census_demo(df)
    gc.collect()
    df = add_census_income(df)
    gc.collect()
    df = add_census_insurance(df)
    gc.collect()
    df = add_dma(df)
    gc.collect()
    return df


def add_census_dma(df):
    print('adding demo at dma')
    govs = pa.read_csv('../Processed Data/dma/dma_cen_demo.csv')
    
    
    govs['dma_code'] = govs['dma_code'].map(lambda x: int(x))
    df['dma_code'] = df['dma_code'].map(lambda x: int(x))
    #govs.drop('total_pop', axis=1, inplace=True)
    govs.state = govs.state.map(lambda x:x.upper())
    df = pa.merge(df, govs, on= ['dma_code','state'], how='left')
    #df.drop('zip', axis=1, inplace=True)
    #unmatched(df,'cen_dem_total_pop')
    return df


def add_census_insurance_dma(df):
    print('adding ins at dma')
    govs = pa.read_csv('../Processed Data/dma/dma_ins.csv')
    before_after(govs)
    
    govs['dma_code'] = govs['dma_code'].map(lambda x: int(x))
    df['dma_code'] = df['dma_code'].map(lambda x: int(x))
    govs.drop('total_pop', axis=1, inplace=True)
    govs.state = govs.state.map(lambda x:x.upper())
    df = pa.merge(df, govs, on= ['dma_code','state'], how='left')
    #df.drop('zip', axis=1, inplace=True)
    #unmatched(df,'cen_dem_total_pop')
    return df


def add_flags_dma(df):
    print('add_flags')
    #from add_demogs import add_politics,add_census_demo, add_census_income, add_census_insurance,add_dma
    df = add_politics(df)
    
    df = add_census_dma(df)
    gc.collect()
    
    df = add_census_insurance_dma(df)
    gc.collect()
    
    return df


def add_county(df, zipvar ='pharmacy_zip_code'):
    print('adding county')
    from data_prep.state_map import abbrev_us_state
    dma_county= pa.read_csv('../Data/amain_state_dma_county.csv')
    dma_county['zipcode'] = dma_county['zipcode'].map(lambda x: int(x))
    dma_county['county'] = dma_county['county'].map(lambda x: x.upper())
    dma_county['state'] = dma_county['st'].map(lambda x: abbrev_us_state[x].upper() if x in abbrev_us_state else 'none' )

    dma_county = dma_county[dma_county['state']!='none']
    dma_county = dma_county.drop_duplicates(['county','fips','zipcode','state','dma_code','dma_name','state'])
    
    df.state= df.state.map(lambda x: x.upper())
    df = pa.merge(df, dma_county, left_on= ['state',zipvar], right_on=['state','zipcode'], how='left')
    unmatched(df,['state','fips','county'])
    df.drop('zipcode', axis=1, inplace=True)
    return df


def add_census_county(df):
    print('adding demo at county')
    govs = pa.read_csv('../Processed Data/county/county_cen_demo.csv')
    
    
    govs['county'] = govs['county'].map(lambda x: int(x))
    df['county'] = df['county'].map(lambda x: int(x))
    #govs.drop('total_pop', axis=1, inplace=True)
    govs.state = govs.state.map(lambda x:x.upper())
    df = pa.merge(df, govs, on= ['county','state'], how='left')
    #df.drop('zip', axis=1, inplace=True)
    #unmatched(df,'cen_dem_total_pop')
    return df


def add_census_insurance_county(df):
    print('adding ins at county')
    govs = pa.read_csv('../Processed Data/dma/dma_ins.csv')
    before_after(govs)
    
    govs['county'] = govs['county'].map(lambda x: int(x))
    df['county'] = df['county'].map(lambda x: int(x))
    govs.drop('total_pop', axis=1, inplace=True)
    govs.state = govs.state.map(lambda x:x.upper())
    df = pa.merge(df, govs, on= ['county','state'], how='left')
    #df.drop('zip', axis=1, inplace=True)
    #unmatched(df,'cen_dem_total_pop')
    return df


def add_flags_county(df):
    print('add_flags')
    #from add_demogs import add_politics,add_census_demo, add_census_income, add_census_insurance,add_dma
    df = add_politics(df)
    
    df = add_census_county(df)
    gc.collect()
    
    df = add_census_insurance_county(df)
    gc.collect()
    
    return df



def state_trump(year=2016):
    # prep_county
    import pandas as pa

    county_vote = pa.read_csv('../Data/countypres_2000-2020.csv')
    county_vote=county_vote[county_vote.year==year]
    aggregate = county_vote#pa.merge(county_vote, county_dma, on =["state","county_name"], how="left")
    dma_level = aggregate.groupby(["state","candidate"], as_index=False).agg({"candidatevotes":"sum"})
    dma_level_T = dma_level.pivot(values = "candidatevotes", columns="candidate",index=["state"])
    dma_level_T.reset_index(inplace=True)#[election_results_2020_T["county_fips"] == 0]
    
    if year == 2016:
        dma_level_T["trump_share_{}".format(year)] = dma_level_T["DONALD TRUMP"] /  dma_level_T[["HILLARY CLINTON","OTHER"]].max(axis=1)#.sum(axis=1)   
        dma_level_T["trump_binary_{}".format(year)] = (dma_level_T["DONALD TRUMP"] >= dma_level_T[["DONALD TRUMP","HILLARY CLINTON","OTHER"]].max(axis=1)).astype(int)
        
    else:
        dma_level_T["trump_share_{}".format(year)] = dma_level_T["DONALD J TRUMP"] /  dma_level_T[["JOSEPH R BIDEN JR",'JO JORGENSEN',"OTHER"]].max(axis=1)#.sum(axis=1)   
        dma_level_T["trump_binary_{}".format(year)] = (dma_level_T["DONALD J TRUMP"] >= dma_level_T[["DONALD J TRUMP","JOSEPH R BIDEN JR",'JO JORGENSEN',"OTHER"]].max(axis=1)).astype(int)
    return dma_level_T#[["state","dma_code","trump_share","trump_binary"]]
        

    
def county_trump():
    # prep_county
    county_dma = pa.read_csv('../Data/amain_state_dma_county.csv')
    from data_prep.state_map import abbrev_us_state
    county_dma["county_name"] = county_dma.county.map(lambda x: x.upper())
    county_dma["state"] = county_dma.st.map(lambda x: abbrev_us_state[x].upper())
    #prep shares
    county_vote = pa.read_csv('../Data/countypres_2000-2020.csv')
    county_vote=county_vote[county_vote.year==2016]
    
    # merge:
    aggregate = pa.merge(county_vote, county_dma, on =["state","county_name"], how="left")
    
    # dma
    dma_level = aggregate.groupby(["state","county_name","county_fips","candidate"], as_index=False).agg({"candidatevotes":"sum"})
    dma_level_T = dma_level.pivot(values = "candidatevotes", columns="candidate",index=["state","county_name","county_fips"])
    dma_level_T.reset_index(inplace=True)#[election_results_2020_T["county_fips"] == 0]
    dma_level_T["trump_share"] = dma_level_T["DONALD TRUMP"] /  dma_level_T[["DONALD TRUMP","HILLARY CLINTON","OTHER"]].sum(axis=1)   
    dma_level_T["trump_binary"] = (dma_level_T["DONALD TRUMP"] >= dma_level_T[["DONALD TRUMP","HILLARY CLINTON","OTHER"]].max(axis=1)).astype(int)
    return dma_level_T[["state","county_name","county_fips","trump_share","trump_binary"]]
    

def dma_trump():
    # prep_county
    county_dma = pa.read_csv('../Data/amain_state_dma_county.csv')
    from data_prep.state_map import abbrev_us_state
    county_dma["county_name"] = county_dma.county.map(lambda x: x.upper())
    county_dma["state"] = county_dma.st.map(lambda x: abbrev_us_state[x].upper())
    #prep shares
    county_vote = pa.read_csv('../Data/countypres_2000-2020.csv')
    county_vote=county_vote[county_vote.year==2016]
    
    # merge:
    aggregate = pa.merge(county_vote, county_dma, on =["state","county_name"], how="left")
    
    # dma
    dma_level = aggregate.groupby(["state","dma_code","candidate"], as_index=False).agg({"candidatevotes":"sum"})
    dma_level_T = dma_level.pivot(values = "candidatevotes", columns="candidate",index=["state","dma_code"])
    dma_level_T.reset_index(inplace=True)#[election_results_2020_T["county_fips"] == 0]
    dma_level_T["trump_share"] = dma_level_T["DONALD TRUMP"] /  dma_level_T[["DONALD TRUMP","HILLARY CLINTON","OTHER"]].sum(axis=1)   
    dma_level_T["trump_binary"] = (dma_level_T["DONALD TRUMP"] >= dma_level_T[["DONALD TRUMP","HILLARY CLINTON","OTHER"]].max(axis=1)).astype(int)
    return dma_level_T[["state","dma_code","trump_share","trump_binary"]]




def rank_percentile_calc(df):
    from scipy.stats import rankdata, percentileofscore
    columns = [x for x in df.columns if '_act' in x]
    for col in columns:
        df[col.replace('_act','_per')] = (df[col]/df.total_pop)*100
        df[col.replace('_act','_rank')] = rankdata(df[col])
        df[col.replace('_act','_percentile')] = ((rankdata(df[col])-1)/df.shape[0] ) *100
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
    df.state=df.state.map(lambda x: x.upper())
    df.to_csv('../Processed Data/state_cen_inc.csv')
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
    df.state=df.state.map(lambda x: x.upper())
    df.to_csv('../Processed Data/state_cen_demo.csv')
    return df



def add_state_demogs(df):
    df1 = census_demogs_state()
    df= pa.merge(df, df1, on='state', how='left')
    df2 = census_demogs_income_state()
    df2.drop("total_pop", axis=1, inplace=True)
    df= pa.merge(df, df2, on='state', how='left')
    
    return df