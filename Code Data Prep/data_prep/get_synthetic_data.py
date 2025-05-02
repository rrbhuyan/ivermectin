import numpy as np

def rotate_df(temporary, var ="all", index_var='state',metric="claims_per100k", placebo=False, ivermectin=False):
    """
    Rotates data for synthetic control format
    
    """
    
    df = temporary.pivot(index=['time_index', 'year','week', 'week_date'] , columns=index_var, values=metric)
    df.reset_index(inplace=True)
    df.sort_values(["year","week"], inplace=True)
    df["trend"] = np.arange(0, df.shape[0])
    if placebo and ivermectin==False:
        df["treat_post"] = ((df.year >= 2020)|( (df.year == 2019) & (df.week >= 35)) ).astype(int)
    elif placebo==False and ivermectin==False:
        df["treat_post"] = (((df.year == 2020) & (df.week >= 9) ) | (df.year == 2021)  ).astype(int)
    elif placebo==False and ivermectin:
        df["treat_post"] = (((df.year == 2020) & (df.week >= 13) ) | (df.year == 2021)  ).astype(int)
    else:
        df["treat_post"] = (((df.year == 2020) | (df.year == 2021) ) ).astype(int)
    
    df["pre_period"] = 1 - df["treat_post"]
    df.rename(columns={var: "y"}, inplace=True)
    df.fillna(0,inplace=True)
    return df