import pandas as pa

def standardize(df, group, week_date, dv):
    """
    This code uses the pre- treatment window to standardize each groups values.
    df: data frame
    group: list of grouping variables - dont include time.
    week_date: string of %y-%m-%d
    dv: dependent variable to be standardized
    """
    
    week_date= pa.to_datetime(week_date, format='%Y-%m-%d')
    
    df["week_date_f"]= pa.to_datetime(df.week_date, format='%Y-%m-%d')
    
    train = df[df.week_date_f<= week_date]
    
    means = train.groupby(group,as_index= False).agg({dv:"mean"}).rename(columns={dv:"mean_{}".format(dv)})
    print(means)
    sd = train.groupby(group, as_index= False).agg({dv:"std"}).rename(columns={dv:"sd_{}".format(dv)})
    print(sd)
    df = pa.merge(df, means, on =group, how ='left')
    df = pa.merge(df, sd, on = group, how ='left')
    
    df["std_{}".format(dv)] = (df[dv]-df["mean_{}".format(dv)])/ (df["sd_{}".format(dv)])
    
    return df
    