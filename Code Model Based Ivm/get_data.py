import pandas as pa
from sklearn.preprocessing import StandardScaler, RobustScaler


def check_zeros(array):
    zero_count = sum(1 for element in array if element == 0)
    zero_percentage = zero_count / len(array)
    
    if zero_percentage > 0.35:
        return True
    else:
        return False
    
def census_demogs_state():
    df = pa.read_csv('../Data/zip_codes_census_demographics.csv')[['state', 'total_pop']]
    df = df.groupby(['state'], as_index=False).sum()
    df.state=df.state.map(lambda x: x.upper())
    return df
    
def population_by_trump_support(year=2016, pro_trump=True):
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
    
    population = census_demogs_state()
    
    dma_level_T = dma_level_T.merge(population, on='state', how='left')
    
    
    dma_level_T = dma_level_T.groupby(by="trump_binary_{}".format(year), as_index=False).sum()[["trump_binary_{}".format(year), "total_pop"]]
    
    if pro_trump:
        res_population = dma_level_T[dma_level_T["trump_binary_{}".format(year)] == 1]["total_pop"].values[0]
        
    else:
        res_population = dma_level_T[dma_level_T["trump_binary_{}".format(year)] == 0]["total_pop"].values[0]
        
    return res_population

def covid_cases_redblue(year=2016, pro_trump=True):
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
        
    covid_df = pa.read_csv("../Data/Covid_state_20_21.csv")
    # covid_df = fix_dates(covid_df)
    covid_df.rename(columns ={"new_cases":"covid_cases"}, inplace=True)
    
    covid_df.drop(["week","year"],axis=1,inplace=True)
    
    covid_df["state"] = list(map(lambda x: x.upper(), covid_df["state"]))
    
    covid_df = covid_df.merge(dma_level_T[["trump_binary_{}".format(year), "state"]], on="state")
    
    covid_df = covid_df.groupby(["time_index", "trump_binary_{}".format(year)], as_index=False).sum()
    
    if pro_trump:
        covid_cases = covid_df[covid_df["trump_binary_{}".format(year)] == 1].covid_cases.values
    
    else:
        covid_cases = covid_df[covid_df["trump_binary_{}".format(year)] == 0].covid_cases.values
        
    # print(covid_cases)
    return covid_cases

def covid_cases_by_state(state):
    import pandas as pa
        
    covid_df = pa.read_csv("../Data/Covid_state_20_21.csv")
    # covid_df = fix_dates(covid_df)
    covid_df.rename(columns ={"new_cases":"covid_cases"}, inplace=True)
    
    covid_df.drop(["week","year"],axis=1,inplace=True)
    
    covid_df["state"] = list(map(lambda x: x.upper(), covid_df["state"]))
    
    covid_cases = covid_df[covid_df.state == state].covid_cases.values

    return covid_cases
    
    


import numpy as np
def get_data(fname, scale=False, poly=False, remove={'time', 'pre_period', 'y','Unnamed: 0', 'covid_cases'},model_type='Linear'):
    df = pa.read_csv(fname)
    df= df[df.year>2018]
    df= df[df.week!=53]
    #df['const']=1
    print(df.columns)
    #df['y'] =  df['y']+1000
    time_labels = df['week_date'].to_list()# df[['year','week']].apply(lambda row: row.year*100+row.week, axis=1 ).to_list()
    #print(time_labels, len(time_labels))
    if model_type=='Poisson':
        df['y'] = df['y'].astype(int)
    
    if "covid_cases" in df.columns:
        covid_cases = df["covid_cases"].values

    else:
        covid_cases = None
    
    for c in df.columns:
        if 'lag' in c:
            remove.add(c)
    #print(remove)
    train_df = df[df.pre_period==1]

    test_df = df[df.pre_period==0]

    X = df[list(set(df.columns) - remove)].values
    
    y = df['y'].values #.flatten()
    
    y= y.reshape(-1,1)
    X_train = train_df[list(set(df.columns) - remove)].values
    
    y_train = train_df['y'].values #.flatten()
    #y_train= y_train.reshape(-1,1)
    skip=False
    
    
    
    if np.median(y_train)<25:
        skip=True
    
    x_scaler=None
    y_scaler=None
    
    if scale:
        #x_scaler = RobustScaler(with_centering=True, with_scaling = True, unit_variance= True)
        #y_scaler = RobustScaler(with_centering=True, with_scaling = True, unit_variance = True )
        x_scaler = StandardScaler(with_mean=True, with_std=True)
        y_scaler = StandardScaler(with_mean=True, with_std=True)
        
    if not scale:
        x_scaler = StandardScaler(with_mean=False, with_std=False)
        y_scaler = StandardScaler(with_mean=False, with_std=False)
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
    

    X_test = test_df[list(set(df.columns) - remove)].values
    
    y_test = test_df['y'].values
    y_test= y_test.reshape(-1,1)
    y_train= y_train.reshape(-1,1)
    
    
    if scale:
        X_test = x_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test)
        X = x_scaler.transform(X)
        y = y_scaler.transform(y)
    
    original_dim = X_train.shape[1]
    num_train = X_train.shape[0]
    #from sklearn.model_selection import train_test_split
    
    """
    if poly ==True:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(2)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        
    """
    
    
    
    #X_train_, X_val, y_train_, y_val = train_test_split( X_train, y_train, test_size=0.2,shuffle=False, random_state=42)
    """
    print("data_import X_val")
    print(X_val)
    print("data_import y_val")
    print(y_val)
    """
    
    return  X,y.flatten(),X_train, y_train.flatten(),y_scaler, time_labels, covid_cases, skip  #, X_train_,y_train_,X_val,y_val,X_test,y_test, x_scaler,y_scaler
def fix_dates(df):
    df_ = df.copy()
#   for the year 2021, add 1 to the weeks, and then change 54 to 1, also change the time index
    # print(type(df_.year.values[0]))
    # print(type(df_.week.values[0]))
    df_["week"] = df_["week"].astype(int)
    df_["year"] = df_["year"].astype(int)
    df_.loc[df_["year"] == 2021, 'week'] = df_.loc[df_["year"] == 2021, 'week'] + 1
    # print(df_.loc[(df_["week"]==54) & (df_["year"] == 2021), 'week'])
    df_.loc[((df_["week"]==54) & (df_["year"] == 2021)), 'week'] = 1
    df_['time_index'] =  df_[['year','week']].apply(lambda row: int(row[0])*100+int(row[1]), axis=1)
    df_.sort_values("time_index", inplace=True)
    df_.reset_index(inplace=True, drop=True)
    # print(df_[["year", "week", "time_index"]].tail())
    return df_

def get_data_withcovid(fname, scale=False, poly=False, remove={'time', 'pre_period', 'y','Unnamed: 0', 'covid_cases'}):
    # print("New Code")
    
    df = pa.read_csv(fname)
    df= df[df.year>2018]
    # df= df[df.week!=53]
    df = fix_dates(df)
    # df = pa.merge(df, covid_df, how="left", on ="time_index")
    df.fillna(0, inplace=True)
    if "trump_1_2016" in fname:
        covid_cases = covid_cases_redblue(pro_trump=True)
        
    elif "trump_0_2016" in fname:
        covid_cases = covid_cases_redblue(pro_trump=False)
    
    elif "trump" not in fname and "all" not in fname:
        state = fname.split('/')[-1].split('_')[-2]
        covid_cases = covid_cases_by_state(state)
        
    elif "all" in fname:
        covid_df = pa.read_csv("../Data/Covid_national_20_21.csv")
        # covid_df = fix_dates(covid_df)
        covid_df.rename(columns ={"new_cases":"covid_cases"}, inplace=True)
    
        covid_df.drop(["week","year"],axis=1,inplace=True)
        df = pa.merge(df, covid_df, how="left", on ="time_index")
        df.fillna(0, inplace=True)
        covid_cases = df["covid_cases"].values
    
    #df['const']=1
    #print(df.head())
    #df['y'] =  df['y']+1000
    
    time_labels = df['week_date'].to_list()
    #time_labels = df[['year','week']].apply(lambda row: row.year*100+row.week, axis=1 ).to_list()
    # print(time_labels, len(time_labels))

    for c in df.columns:
        if 'lag' in c:
            remove.add(c)
    #print(remove)
    train_df = df[df.pre_period==1]

    test_df = df[df.pre_period==0]

    X = df[list(set(df.columns) - remove)].values
    
    y = df['y'].values #.flatten()
    
    y= y.reshape(-1,1)
    X_train = train_df[list(set(df.columns) - remove)].values
    
    y_train = train_df['y'].values #.flatten()
    #y_train= y_train.reshape(-1,1)
    skip=False
    
    #if check_zeros(y_train):
    #    skip=True
    #if np.median(y_train)<25:
    #    skip=True
    
    x_scaler=None
    y_scaler=None
    
    if scale:
        #x_scaler = RobustScaler(with_centering=True, with_scaling = True, unit_variance= True)
        #y_scaler = RobustScaler(with_centering=True, with_scaling = True, unit_variance = True )
        x_scaler = StandardScaler(with_mean=True, with_std=True)
        y_scaler = StandardScaler(with_mean=True, with_std=True)
        
    if not scale:
        x_scaler = StandardScaler(with_mean=False, with_std=False)
        y_scaler = StandardScaler(with_mean=False, with_std=False)
    
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
    

    X_test = test_df[list(set(df.columns) - remove)].values
    
    y_test = test_df['y'].values
    y_test= y_test.reshape(-1,1)
    y_train= y_train.reshape(-1,1)
    
    
    if scale:
        X_test = x_scaler.transform(X_test)
        y_test = y_scaler.transform(y_test)
        X = x_scaler.transform(X)
        y = y_scaler.transform(y)
        
    if "trump_1_2016" in fname:
        population = population_by_trump_support(pro_trump=True)
        X = X / population * 100000
        y = y / population * 100000
        covid_cases = covid_cases / population * 100000
        
    elif "trump_0_2016" in fname:
        population = population_by_trump_support(pro_trump=False)
        X = X / population * 100000
        y = y / population * 100000
        covid_cases = covid_cases / population * 100000
        
    elif "trump" not in fname and "all" not in fname:
        state = fname.split('/')[-1].split('_')[-2]
        population = census_demogs_state()
        population = population[population.state == state]["total_pop"].values[0]
        X = X / population * 100000
        y = y / population * 100000
        covid_cases = covid_cases / population * 100000
    
    else:
        covid_cases = covid_cases / 1000000
        
    original_dim = X_train.shape[1]
    num_train = X_train.shape[0]

    #from sklearn.model_selection import train_test_split
    
    """
    if poly == True:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(2)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        
    """
    
    
    
    #X_train_, X_val, y_train_, y_val = train_test_split( X_train, y_train, test_size=0.2,shuffle=False, random_state=42)
    """
    print("data_import X_val")
    print(X_val)
    print("data_import y_val")
    print(y_val)
    """
    
    return  X,y.flatten(),X_train, y_train.flatten(),y_scaler, time_labels, covid_cases, skip  #, 

