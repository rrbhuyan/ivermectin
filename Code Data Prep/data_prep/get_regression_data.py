import numpy as np
import pandas as pd
import pandas as pa
import matplotlib.pyplot as plt
import os
import glob
from data_prep.add_demogs import add_trump

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

def get_dates(drug):
    dates = pa.read_csv('../Processed Data/state_week/synth_data/{}/actual_claim_count_CALIFORNIA_gpi.csv'.format(drug))
    dates = fix_dates(dates)
    dates = dates[dates.treat_post ==1]
    # dates = dates[dates.week !=53]
    dates =dates[['time_index','year','week','trend','week_date']]
    dates.reset_index(inplace=True)
    return dates




def get_flat_regression_data( drug , standardized="standardized"):
    
    path = "../Processed Data/state_week/synth_data/{}/Results/*.csv".format(drug)#"../data_preprocessed_full_dp/states/Results/"
    files = glob.glob(path)
    files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' not in x and "claim" in x and standardized in x]
    dates = get_dates(drug)
    
    results=[]


    # extract values
    for file in files:
        df_state = pd.read_csv( file)
        state = file.split("_")[-2].replace(".csv","")
        df = pd.DataFrame(columns=["state", "att", "att_sd", "att_per","att_per_sd", "att_efctsz","att_efctsz_sd"])
        for week in range(dates.shape[0]):
            att_per = df_state["att_per_" + str(week)][0]
            att_per_sd = df_state["att_per_sd_" + str(week)][0]
            att_efctsz = df_state["att_efctsz_" + str(week)][0]
            att_efctsz_sd = df_state["att_efctsz_sd_" + str(week)][0]
            att = df_state["att_" + str(week)][0]
            att_sd = df_state["att_sd_" + str(week)][0]

            df = df.append({"state": state, "att": att,  "att_sd":att_sd, "att_per": att_per,"att_per_sd":att_per_sd,  "att_efctsz": att_efctsz,"att_efctsz_sd":att_efctsz_sd}, ignore_index=True)

            df["eff_sz"] = df["att"]/df["att_sd"]
            df["per_eff_sz"] = df["att_per"]/df["att_per_sd"]
        df = pa.concat([dates, df],axis=1)

        #df.reset_index(inplace=True)
        results.append(df)
    df=pa.concat(results, axis=0)
    df.reset_index(inplace=True)
    df = df.groupby(["state"], as_index=False).agg({"att_efctsz":"mean","eff_sz":"mean"})
    
    #covid
    covid = pd.read_csv("../Data/Covid_state_20_21.csv")
    covid.state= covid.state.map(lambda x: x.upper())
    covid = covid.groupby(["state"], as_index=False).agg({"new_cases":"sum"})
    df["cases"] = pa.merge(df, covid, on=["state"], how="left")["new_cases"].values
    df.fillna(0, inplace=True)
    
    # population
    population = pd.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    df["population"] = df.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    
    from data_prep.get_modeling_data import add_policy
    from data_prep.add_demogs import  add_politics, add_trump, add_state_demogs
    df = add_flat_policy(df)
    df = add_flat_politics(df)
    df = add_trump(df)
    df = add_state_demogs(df)
    
    drug_dict ={"ivermectin":"IVM", "hydroxychloroquine":"HCQ"}
    df.to_csv("../Processed Data/regression/{}_Flat_Regression_Dataset_per_{}.csv".format(drug_dict[drug],standardized), index=False)
    print(df.shape)
    return df




def get_regression_data_all( drug , standardized="standardized", metric="claim", orig=False, use_gpi='', filtered=False):
    # changed argument gpi to use_gpi for consistency with other functions
    #"../data_preprocessed_full_dp/states/Results/"
    path = "../Processed Data/state_week/synth_data/{}/Results/*.csv".format(drug)
    
        
    files = glob.glob(path)
    
    if use_gpi=='gpi':
        files =[ f for f in files  if 'gpi' in f]
    elif use_gpi=='covid':
        files =[ f for f in files  if 'covid' in f]
        
    if filtered:
        files = [x for x in files if "filtered" in x]
    else:
        files = [x for x in files if "filtered" not in x]
        
    if metric == "claim":
        files = [x for x in files if "person" not in x]
    
    #print(files)
    if orig:
        files=[x for x in files if "placebo" not in x and 'all' in x and 'trump' not in x and 'orig' in x and metric in x and standardized in x]
    else:
        files=[x for x in files if "placebo" not in x and 'all' in x and 'trump' not in x and 'orig' not in x and metric in x and standardized in x]
    print(files)
    dates = get_dates(drug)
    
    results=[]


    # extract values
    for file in files:
        df_state = pd.read_csv( file)
        state = file.split("_")[-2].replace(".csv","")
        df = pd.DataFrame(columns=["state", "att", "att_sd", "lower", "upper", "att_per","att_per_sd", "lower_per", "upper_per", "att_efctsz","att_efctsz_sd"])
        for week in range(dates.shape[0]):
            att_per = df_state["att_per_" + str(week)][0]
            att_per_sd = df_state["att_per_sd_" + str(week)][0]
            lower_per = df_state["lower_per_" + str(week)][0]
            upper_per = df_state["upper_per_" + str(week)][0]
            att_efctsz = df_state["att_efctsz_" + str(week)][0]
            att_efctsz_sd = df_state["att_efctsz_sd_" + str(week)][0]
            att = df_state["att_" + str(week)][0]
            att_sd = df_state["att_sd_" + str(week)][0]
            lower = df_state["lower_" + str(week)][0]
            upper = df_state["upper_" + str(week)][0]

            df = df.append({"state": state, "att": att,  "att_sd":att_sd, "lower": lower, "upper": upper, "att_per": att_per,"att_per_sd":att_per_sd, "lower_per": lower_per, "upper_per": upper_per, "att_efctsz": att_efctsz,"att_efctsz_sd":att_efctsz_sd}, ignore_index=True)

            df["eff_sz"] = df["att"]/df["att_sd"]
            df["per_eff_sz"] = df["att_per"]/df["att_per_sd"]
        df = pa.concat([dates, df],axis=1)

        #df.reset_index(inplace=True)
        results.append(df)
    df=pa.concat(results, axis=0)
    df.reset_index(inplace=True)
    
    #covid
    covid = pd.read_csv("../Data/Covid_national_20_21.csv")
    #covid.state= covid.state.map(lambda x: x.upper())
    df["cases"] = pa.merge(df, covid, on=["week","year"], how="left")["new_cases"].values
    df.fillna(0, inplace=True)
    
    # population
    #population = pd.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    #population = population[["State", "state_population", "Category"]].drop_duplicates()
    #df["population"] = df.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    
    #from data_prep.get_modeling_data import add_policy
    #from data_prep.add_demogs import  add_politics, add_trump, add_state_demogs
    #df = add_policy(df)
    #df = add_politics(df)
    #df = add_trump(df)
    #df = add_state_demogs(df)
    
    drug_dict ={"ivermectin":"IVM", "hydroxychloroquine":"HCQ"}
    
    if filtered:
        filtered = "_filtered"
    else:
        filtered = ""
        
    if orig:
        orig = "_orig"
    else:
        orig = ""
    
    df.to_csv("../Processed Data/regression/{}_Regression_Dataset_per_{}_{}_national_{}{}{}.csv".format(drug_dict[drug],standardized,metric, str(orig),str(use_gpi), filtered), index=False)
    print(df.shape)
    return df




def get_meta_analysis_data( drug , standardized="standardized",metric="claim", orig=False):
    
    path = "../Processed Data/state_week/synth_data/{}/Results/*.csv".format(drug)#"../data_preprocessed_full_dp/states/Results/"
    files = glob.glob(path)
    if orig:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' not in x and  "orig" in x and metric in x and standardized in x]
    else:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' not in x and  "orig" not in x and metric in x and standardized in x]
    dates = get_dates(drug)
    
    results=[]
    if drug =="hydroxychloroquine":
        att_date = '2020-03-22'
    elif drug =='ivermectin':
        att_date = '2021-08-22'

    # extract values
    for file in files:
        #print(file)
        df_state = pd.read_csv( file)
        state = file.split("_")[-2].replace(".csv","")
        df = pd.DataFrame(columns=["state", "att", "att_sd", "att_per","att_per_sd", "att_efctsz","att_efctsz_sd"])
        for week in range(dates.shape[0]-1):
            att_per = df_state["att_per_" + str(week)][0]
            att_per_sd = df_state["att_per_sd_" + str(week)][0]
            att_efctsz = df_state["att_efctsz_" + str(week)][0]
            att_efctsz_sd = df_state["att_efctsz_sd_" + str(week)][0]
            att = df_state["att_" + str(week)][0]
            att_sd = df_state["att_sd_" + str(week)][0]

            df = df.append({"state": state, "att": att,  "att_sd":att_sd, "att_per": att_per,"att_per_sd":att_per_sd,  "att_efctsz": att_efctsz,"att_efctsz_sd":att_efctsz_sd}, ignore_index=True)

            df["eff_sz"] = df["att"]/df["att_sd"]
            df["per_eff_sz"] = df["att_per"]/df["att_per_sd"]
        df = pa.concat([dates, df],axis=1)
           
        #df.reset_index(inplace=True)
        results.append(df)
    df=pa.concat(results, axis=0)
    
    df = df[ df['week_date']== att_date]
    
    df.reset_index(inplace=True)
    
    #covid
    covid = pd.read_csv("../Data/Covid_state_20_21.csv")
    covid.state= covid.state.map(lambda x: x.upper())
    df["cases"] = pa.merge(df, covid, on=["state", "week","year"], how="left")["new_cases"].values
    df.fillna(0, inplace=True)
    
    # population
    population = pd.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    df["population"] = df.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    
    from data_prep.get_modeling_data import add_policy
    from data_prep.add_demogs import  add_politics, add_trump, add_state_demogs
    df = add_policy(df)
    df = add_politics(df)
    df = add_trump(df, year=2016)
    df = add_trump(df, year=2020)
    df = add_state_demogs(df)
    
    drug_dict ={"ivermectin":"IVM", "hydroxychloroquine":"HCQ"}
    df.to_csv("../Processed Data/regression/{}_Meta_Dataset_per_{}_{}_{}.csv".format(drug_dict[drug],standardized,metric, str(orig)), index=False)
    print(df.shape)
    return df


def get_regression_data( drug , standardized="standardized",metric="claim", orig=False, use_gpi='gpi', filtered=False):
    
    path = "../Processed Data/state_week/synth_data/{}/Results/*.csv".format(drug)#"../data_preprocessed_full_dp/states/Results/"
    files = glob.glob(path)
    
    files = [x for x in files if "SPR" not in x and "SPU" not in x and "UMPR" not in x and "UMPU" not in x and "LMP" not in x and "WP" not in x]
    
    if use_gpi=='gpi':
        files = [x for x in files if "gpi" in x]
    elif use_gpi=='covid':
        files = [x for x in files if "covid" in x]
        
    if filtered:
        files = [x for x in files if "filtered" in x]
    else:
        files = [x for x in files if "filtered" not in x]
    
    if orig:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' not in x and  "orig" in x and metric in x and standardized in x]
    else:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' not in x and  "orig" not in x and metric in x and standardized in x] 
    dates = get_dates(drug)
    
    print(files)
    results=[]


    # extract values
    for file in files:
        #print(file)
        df_state = pd.read_csv( file)
        #if df_state['divergent'][0]>0:
        #    continue
        state = file.split("_")[-2].replace(".csv","")
        #if use_gpi=='gpi':
        state = file.split("_")[-3].replace(".csv","")
        
        if filtered:
            state = file.split("_")[-4].replace(".csv","")
            
        
        df = pd.DataFrame(columns=["state", "att", "att_sd", "lower", "upper", "att_per","att_per_sd", "lower_per", "upper_per", "att_efctsz","att_efctsz_sd"])
        for week in range(dates.shape[0]):
            att_per = df_state["att_per_" + str(week)][0]
            lower_per = df_state["lower_per_" + str(week)][0]
            upper_per = df_state["upper_per_" + str(week)][0]
            att_per_sd = df_state["att_per_sd_" + str(week)][0]
            att_efctsz = df_state["att_efctsz_" + str(week)][0]
            att_efctsz_sd = df_state["att_efctsz_sd_" + str(week)][0]
            att = df_state["att_" + str(week)][0]
            att_sd = df_state["att_sd_" + str(week)][0]
            lower = df_state["lower_" + str(week)][0]
            upper = df_state["upper_" + str(week)][0]

            df = df.append({"state": state, "att": att, "lower": lower, "upper": upper, "att_sd": att_sd, "att_per": att_per,"att_per_sd": att_per_sd, "lower_per": lower_per, "upper_per": upper_per, "att_efctsz": att_efctsz, "att_efctsz_sd": att_efctsz_sd}, ignore_index=True)

            df["eff_sz"] = df["att"]/df["att_sd"]
            df["per_eff_sz"] = df["att_per"]/df["att_per_sd"]
        df = pa.concat([dates, df],axis=1)

        #df.reset_index(inplace=True)
        results.append(df)
    df=pa.concat(results, axis=0)
    df.reset_index(inplace=True)
    
    print(df.head())
    #covid
    covid = pd.read_csv("../Data/Covid_state_20_21.csv")
    covid.state= covid.state.map(lambda x: x.upper())
    df["cases"] = pa.merge(df, covid, on=["state", "week","year"], how="left")["new_cases"].values
    df.fillna(0, inplace=True)
    
    # population
    population = pd.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    df["population"] = df.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    
    from data_prep.get_modeling_data import add_policy
    from data_prep.add_demogs import  add_politics, add_trump, add_state_demogs
    df = add_policy(df)
    df = add_politics(df)
    df = add_trump(df, year=2016)
    df = add_trump(df, year=2020)
    df = add_state_demogs(df)
    
    drug_dict ={"ivermectin":"IVM", "hydroxychloroquine":"HCQ"}
    
    if filtered:
        filtered = "_filtered"
    else:
        filtered = ""
    
    df.to_csv("../Processed Data/regression/{}_Regression_Dataset_per_{}_{}_{}_{}{}.csv".format(drug_dict[drug],standardized,metric, str(orig),str(use_gpi), filtered), index=False)
    print(df.shape)
    return df



def covid_trump():
    
    df = pd.read_csv("../Data/Covid_state_20_21.csv")
    df.state= df.state.map(lambda x: x.upper())
    df = add_trump(df, year=2016)
    measures={}
    
    for m in  [ 'new_cases']:
        measures[m] ='sum'

    df= df.groupby(['trump_binary_2016','year','week'], as_index=False).agg(measures)
    
    df["cases"] = df["new_cases"].values
    df.fillna(0, inplace=True)
    
    return df
    


def get_regression_data_trump( drug , standardized="standardized",metric="claim", orig=False, use_gpi=False):
    
    path = "../Processed Data/state_week/synth_data/{}/Results/*trump*2016*.csv".format(drug)#"../data_preprocessed_full_dp/states/Results/"
    files = glob.glob(path)
    print(files)
    if use_gpi:
        files = [x for x in files if "gpi" in x]
    else:
        files = [x for x in files if "gpi" not in x]
        
    print(files)
    #'../Processed Data/state_week/synth_data/ivermectin/Results/actual_claim_count_orig_trump_1_2016_standardized.csv'
    if orig:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' in x and  "orig" in x and metric in x and standardized in x]
    else:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' in x and  "orig" not in x and metric in x and standardized in x]
    dates = get_dates(drug)
    
    results=[]

    
    # extract values
    for file in files:
        #print(file)
        df_state = pd.read_csv( file)
        #if df_state['divergent'][0]>0:
        #    continue
        print(file)
        state = '_'.join(file.replace(".csv","").split("_")[-2:-5])
        print(state)
        trump_indic = int('_'.join(file.split("_")[-4].replace(".csv","")))
        print(trump_indic)
        df = pd.DataFrame(columns=["state", "att", "att_sd", "att_per","att_per_sd", "att_efctsz","att_efctsz_sd"])
        for week in range(dates.shape[0]-1):
            att_per = df_state["att_per_" + str(week)][0]
            att_per_sd = df_state["att_per_sd_" + str(week)][0]
            att_efctsz = df_state["att_efctsz_" + str(week)][0]
            att_efctsz_sd = df_state["att_efctsz_sd_" + str(week)][0]
            att = df_state["att_" + str(week)][0]
            att_sd = df_state["att_sd_" + str(week)][0]

            df = df.append({"state": state,'trump_binary_2016':trump_indic, "att": att,  "att_sd":att_sd, "att_per": att_per,"att_per_sd":att_per_sd,  "att_efctsz": att_efctsz,"att_efctsz_sd":att_efctsz_sd}, ignore_index=True)

            df["eff_sz"] = df["att"]/df["att_sd"]
            df["per_eff_sz"] = df["att_per"]/df["att_per_sd"]
        df = pa.concat([dates, df],axis=1)

        #df.reset_index(inplace=True)
        results.append(df)
    df=pa.concat(results, axis=0)
    df.reset_index(inplace=True)
    
    #covid
    df_covid = covid_trump()
    print(df_covid.head())
    print(df.head())
    
    df['cases'] = pa.merge(df, df_covid, on=["trump_binary_2016", "week","year"], how="left")["cases"].values
    df.fillna(0, inplace=True)
    
    # population
    population = pd.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    population.rename(columns= {'State':'state'}, inplace=True)
    population = add_trump(population, year=2016)
    population = population.groupby(['trump_binary_2016'], as_index=False).agg({"state_population":'sum'})
    df = pa.merge(df, population, on=["trump_binary_2016"], how="left")
    
    df["population"] = df["state_population"].values
    
    #from data_prep.get_modeling_data import add_policy
    #from data_prep.add_demogs import  add_politics, add_trump, add_state_demogs
    #df = add_policy(df)
    #df = add_politics(df)
    #df = add_trump(df, year=2016)
    #df = add_trump(df, year=2020)
    #df = add_state_demogs(df)
    
    drug_dict ={"ivermectin":"IVM", "hydroxychloroquine":"HCQ"}
    df.to_csv("../Processed Data/regression/{}_Regression_Dataset_per_{}_{}_{}_trump".format(drug_dict[drug],standardized,metric, str(orig)), index=False)
    print(df.shape)
    return df
def get_regression_data_trump_( drug , standardized="standardized",metric="claim", orig=False, use_gpi=False):
    
    path = "../Processed Data/state_week/synth_data/{}/Results/*trump*2016*.csv".format(drug)#"../data_preprocessed_full_dp/states/Results/"
    files = glob.glob(path)
    print(files)
    if use_gpi:
        files = [x for x in files if "gpi" in x]
    else:
        files = [x for x in files if "gpi" not in x]
        
    print(files)
    #'../Processed Data/state_week/synth_data/ivermectin/Results/actual_claim_count_orig_trump_1_2016_standardized.csv'
    if orig:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' in x and  "orig" in x and metric in x and standardized in x]
    else:
        files=[x for x in files if "placebo" not in x and 'all' not in x and 'trump' in x and  "orig" not in x and metric in x and standardized in x]
    dates = get_dates(drug)
    
    results=[]

    
    # extract values
    for file in files:
        #print(file)
        df_state = pd.read_csv( file)
        #if df_state['divergent'][0]>0:
        #    continue
        print(file)
        state = '_'.join(file.replace(".csv","").split("_")[-2:-5])
        print(state)
        trump_indic = int('_'.join(file.split("_")[-4].replace(".csv","")))
        print(trump_indic)
        df = pd.DataFrame(columns=["state", "att", "att_sd", "att_per","att_per_sd", "att_efctsz","att_efctsz_sd"])
        for week in range(dates.shape[0]-1):
            att_per = df_state["att_per_" + str(week)][0]
            att_per_sd = df_state["att_per_sd_" + str(week)][0]
            att_efctsz = df_state["att_efctsz_" + str(week)][0]
            att_efctsz_sd = df_state["att_efctsz_sd_" + str(week)][0]
            att = df_state["att_" + str(week)][0]
            att_sd = df_state["att_sd_" + str(week)][0]

            df = df.append({"state": state,'trump_binary_2016':trump_indic, "att": att,  "att_sd":att_sd, "att_per": att_per,"att_per_sd":att_per_sd,  "att_efctsz": att_efctsz,"att_efctsz_sd":att_efctsz_sd}, ignore_index=True)

            df["eff_sz"] = df["att"]/df["att_sd"]
            df["per_eff_sz"] = df["att_per"]/df["att_per_sd"]
        df = pa.concat([dates, df],axis=1)

        #df.reset_index(inplace=True)
        results.append(df)
    df=pa.concat(results, axis=0)
    df.reset_index(inplace=True)
    
    #covid
    df_covid = covid_trump()
    print(df_covid.head())
    print(df.head())
    
    df['cases'] = pa.merge(df, df_covid, on=["trump_binary_2016", "week","year"], how="left")["cases"].values
    df.fillna(0, inplace=True)
    
    # population
    population = pd.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    population.rename(columns= {'State':'state'}, inplace=True)
    population = add_trump(population, year=2016)
    population = population.groupby(['trump_binary_2016'], as_index=False).agg({"state_population":'sum'})
    df = pa.merge(df, population, on=["trump_binary_2016"], how="left")
    
    df["population"] = df["state_population"].values
    
    #from data_prep.get_modeling_data import add_policy
    #from data_prep.add_demogs import  add_politics, add_trump, add_state_demogs
    #df = add_policy(df)
    #df = add_politics(df)
    #df = add_trump(df, year=2016)
    #df = add_trump(df, year=2020)
    #df = add_state_demogs(df)
    
    drug_dict ={"ivermectin":"IVM", "hydroxychloroquine":"HCQ"}
    df.to_csv("../Processed Data/regression/{}_Regression_Dataset_per_{}_{}_{}_trump".format(drug_dict[drug],standardized,metric, str(orig)), index=False)
    print(df.shape)
    return df