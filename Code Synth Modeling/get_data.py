import pandas as pa
from sklearn.preprocessing import StandardScaler, RobustScaler


def check_zeros(array):
    zero_count = sum(1 for element in array if element == 0)
    zero_percentage = zero_count / len(array)
    
    if zero_percentage > 0.35:
        return True
    else:
        return False

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
    print(type(df_.year.values[0]))
    print(type(df_.week.values[0]))
    df_["week"] = df_["week"].astype(int)
    df_["year"] = df_["year"].astype(int)
    df_.loc[df_["year"] == 2021, 'week'] = df_.loc[df_["year"] == 2021, 'week'] + 1
    print(df_.loc[(df_["week"]==54) & (df_["year"] == 2021), 'week'])
    df_.loc[((df_["week"]==54) & (df_["year"] == 2021)), 'week'] = 1
    df_['time_index'] =  df_[['year','week']].apply(lambda row: int(row[0])*100+int(row[1]), axis=1)
    df_.sort_values("time_index", inplace=True)
    df_.reset_index(inplace=True, drop=True)
    print(df_[["year", "week", "time_index"]].tail())
    return df_

def get_data_withcovid(fname, scale=False, poly=False, remove={'time', 'pre_period', 'y','Unnamed: 0', 'covid_cases'},model_type='Linear'):
    print("New Code")
    covid_df = pa.read_csv("../Data/Covid_national_20_21.csv")
    # covid_df = fix_dates(covid_df)
    covid_df.rename(columns ={"new_cases":"covid_cases"}, inplace=True)
    
    covid_df.drop(["week","year"],axis=1,inplace=True)
    
    
    df = pa.read_csv(fname)
    df= df[df.year>2018]
    # df= df[df.week!=53]
    df = fix_dates(df)
    df = pa.merge(df, covid_df, how="left", on ="time_index")
    df.fillna(0, inplace=True)
    
    #df['const']=1
    #print(df.head())
    #df['y'] =  df['y']+1000
    
    time_labels = df['week_date'].to_list()
    #time_labels = df[['year','week']].apply(lambda row: row.year*100+row.week, axis=1 ).to_list()
    print(time_labels, len(time_labels))
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
    
    return  X,y.flatten(),X_train, y_train.flatten(),y_scaler, time_labels, covid_cases, skip  #, 

