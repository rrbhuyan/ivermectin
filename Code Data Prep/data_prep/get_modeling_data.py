from data_prep.utils import year_week_date
from data_prep.policy_week import get_policy_week
from glob import glob
import numpy as np
import pandas as pa
import gc
import time



def get_policy(delta=4):
    from datetime import datetime, timedelta
    policy = get_policy_week()
    policy = policy[policy.Action_Date!='?']
    policy['Action_Date'] = policy['Action_Date'].map(lambda x:pa.to_datetime(x))
    policy['Reverse_Date'] = policy['Reverse_Date'].map(lambda x:pa.to_datetime(x))

    df = policy.copy()
    
    #df['date'] = pd.to_datetime(df['date'], format='%m\\%d\\%Y')  # Convert date column to datetime format

    target_date = '06\\21\\2020'  # Target date to filter around
    target_date = datetime.strptime(target_date, '%m\\%d\\%Y')  # Convert target date to datetime format
    
    start_date = target_date - timedelta(weeks=delta)  # Calculate the start date of the time interval
    end_date = target_date + timedelta(weeks=delta)  # Calculate the end date of the time interval

    filtered_df = df[(df['Reverse_Date'] >= start_date) & (df['Reverse_Date'] <= end_date)] 
    filtered_df_ = df[(df['policy'] == 'WP') ] 
    
    print(start_date, end_date)
    
    return filtered_df.state.tolist(), filtered_df_.state.tolist()
    
    
    
    


def add_policy(dfs):
    policy = get_policy_week()
    policy = policy[policy.Action_Date!='?']
    policy['Action_Date'] = policy['Action_Date'].map(lambda x:pa.to_datetime(x))
    policy['Reverse_Date'] = policy['Reverse_Date'].map(lambda x:pa.to_datetime(x))

    dfs['week_date'] = dfs['week_date'].map(lambda x:pa.to_datetime(x))
    #print(dfs[["week_date","week_end","week_begin"]].head(3))
    #print(policy.head())
    
    #dfs['week_date'] = dfs[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )   
    #print(dfs.head())
    dfs = pa.merge(dfs,policy, on='state', how='inner')
    #print(dfs.head())
    dfs["policy_action"]=dfs[['week_date','Action_Date',"Reverse_Date"]].apply(lambda x: 1 if x[1]<=x[0]<x[2] else 0, axis=1)
    dfs["policy_reversal"]=dfs[['week_date','Action_Date',"Reverse_Date"]].apply(lambda x: 1 if x[0]>=x[2] else 0, axis=1)

    for apolicy in policy.policy.unique():
        print(apolicy)
        opolicy = apolicy+"_orig"
        dfs[opolicy] = 0
        dfs[apolicy] = 0
        new_policy= apolicy
        if apolicy in ['UMPU','SPU',"UMPR","SPR"]:
            new_policy = apolicy[:-1]
        dfs[new_policy+"_prev"] = 0
        dfs[new_policy+"_rvsd"] = 0

    for apolicy in policy.policy.unique():
        opolicy = apolicy+"_orig"
        new_policy= apolicy
        if apolicy in ['UMPU','SPU',"UMPR","SPR"]:
            new_policy = apolicy[:-1]
        if  apolicy in ["UMPR","SPR"]:
            dfs.loc[ (dfs["policy"]==apolicy) & (dfs["policy_action"]==1), new_policy+"_prev"]=1 
            dfs.loc[ (dfs["policy"]==apolicy) & (dfs["policy_action"]==1), opolicy]=1 
            dfs.loc[ (dfs["policy"]==apolicy) & (dfs["policy_reversal"]==1), new_policy+"_rvsd"]=1 
        else:
            dfs.loc[ (dfs["policy"]==apolicy) & (dfs["policy_action"]==1), new_policy+"_prev"]=1 
            dfs.loc[ (dfs["policy"]==apolicy) & (dfs["policy_action"]==1), opolicy]=1 
            dfs.loc[ (dfs["policy"]==apolicy) & (dfs["policy_action"]==1), apolicy]=1 
    return dfs
    
    
def date_format(x):
    date = str(year_week_date(x[0],x[1]))
    year =date[:4]
    month = date[4:6]
    day =date[6:]
    return pa.to_datetime(month+"/"+day+"/"+year)   



        




def get_modeling_data_dma():
    """
    Must run raw2process at least once before.
    
    Modeling data adds : Policy,
    
    Population and Covid Cases.
    
    """
    geography =['state']
    time = ['time_index', 'year','week']
    med_code= "drug_name"
    
    from data_prep.graphs import path_creator
    
    dimension = ''

    filenames  = glob("../Processed Data/state_week/{}/state_dma_code_2*.csv".format('drug_name'))
    metrics = [
            'claim_count',
            'person_count',
            'first_time_person_count']
    dfs =[] 
    for f in filenames:
            dfs.append(pa.read_csv(f))
    dfs = pa.concat(dfs)
    dfs = dfs[dfs.year<=2021]
    dfs = dfs[dfs.year>2018]
    #dfs = dfs[dfs.week<53]
    dfs.reset_index(inplace=True)

    dfs['week_date'] = dfs.week_end#dfs[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )
    dfs = add_policy(dfs)
    #dfs["claim_count"]= (dfs["claim_count"]/12340) *100000      
    #dfs['first_time_person_count']= (dfs['first_time_person_count']/25340) *100000      
    
    #dfs['total_days_supplied']= (dfs['total_days_supplied']/6340) *100000            

    covid = pa.read_csv("../Data/covid_state_week_2020.csv")
    dfs = pa.merge(dfs, covid, on=['year','week','state'], how="left")
    dfs.fillna(0,inplace=True)
    
    

    population = pa.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    dfs["population"] = dfs.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    dfs["cases_raw"] = dfs["cases"]
    dfs["cases"] = (dfs["cases"]/dfs["population"])*100000

    dfs['ln_claim_count'] = dfs["claims_per100k"] = ( (dfs["claim_count"])/ (dfs["population"]) )*100000 
    dfs["cases_pol" ]= dfs.cases * dfs['ln_claim_count']

    #hcq=dfs[(dfs['gpi2_drug_class']==13.0)| (dfs['gpi2_drug_class']==96.0)]
    #hcq.to_csv('../Processed Data/state_week/{}/{}_hcq.csv'.format('no_covid',med_code))
    dfs.to_csv('../Processed Data/state_week/{}/dma/{}.csv'.format('modeling_data',med_code))
    
    return




def get_modeling_data(geography =['state'], time = ['time_index', 'year','week',"week_end","week_begin"], med_code= "drug_name",maintenance=False, use_gpi=False, policy_filter=False):
    """
    Must run raw2process at least once before.
    
    Modeling data adds : Policy,
    
    Population and Covid Cases.
    
    Does not ahve any other demographics!
    
    Cannot be used for Heterogeniety Analysis
    
    returns hcq (13 only) and dfs (all gpis)
    
    """
    
    
    from data_prep.graphs import path_creator

    dimension = ''

    gpi=''
    if use_gpi==True:
        gpi='_gpi'
    elif use_gpi=='covid':
        gpi='_covid'

    if maintenance:
        filenames  = glob("../Processed Data/state_week/{}/maintenance/{}_2*{}.csv".format('drug_name','_'.join(geography),gpi))
    else:
        filenames  = glob("../Processed Data/state_week/{}/{}_2*{}.csv".format('drug_name','_'.join(geography),gpi))
    metrics = [
            'claim_count',
            'person_count',
            'first_time_person_count']
    dfs =[] 
    for f in filenames:
            dfs.append(pa.read_csv(f))
    dfs = pa.concat(dfs)
    dfs = dfs[dfs.year<=2021]
    dfs = dfs[dfs.year>2018]
    #dfs = dfs[dfs.week<53]
    dfs.reset_index(inplace=True)

    dfs['week_date'] = dfs.week_end.copy() #dfs['week_date'] = dfs[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )
    
    print(dfs[["week_date","week_end","week_begin"]].head(3))
    dfs = add_policy(dfs)
    if policy_filter:
        pass
               
    dfs["claim_count_orig"] = dfs["claim_count"]   
    
    dfs["person_count_orig"] = dfs["person_count"]   
    
    dfs['first_time_person_count_orig'] = dfs['first_time_person_count']
    
    dfs['total_days_supplied_orig'] = dfs['total_days_supplied'] 
        
        
    dfs["claim_count"]= (dfs["claim_count"]/12340) *100000      
    
    dfs['first_time_person_count']= (dfs['first_time_person_count']/25340) *100000      
    
    dfs['person_count']= (dfs['person_count']/25340) *100000      
    
    dfs['total_days_supplied']= (dfs['total_days_supplied']/6340) *100000            

    covid = pa.read_csv("../Data/Covid_state_20_21.csv") #gdrive_folder/GoodRx/new_submission/Data/
    covid.state= covid.state.map(lambda x: x.upper())
    covid.drop(['time_index'], axis=1, inplace=True)
    dfs = pa.merge(dfs, covid, on=['year','week','state'], how="left")
    dfs.fillna(0,inplace=True)

    population = pa.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    dfs["population"] = dfs.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    dfs["cases_raw"] = dfs["new_cases"]
    dfs["cases"] = (dfs["new_cases"]/dfs["population"])*100000

    dfs['ln_claim_count'] = dfs["claims_per100k"] = ( (dfs["claim_count"])/ (dfs["population"]) )*100000 
    #dfs["cases_pol" ]= dfs.cases * dfs['ln_claim_count']

    #hcq=dfs[(dfs['gpi2_drug_class']==13.0)| (dfs['gpi2_drug_class']==96.0)]
    #hcq.to_csv('../Processed Data/state_week/{}/{}_hcq.csv'.format('no_covid',med_code))
    if "channel" in geography or 'retailer' in geography:
        dfs.to_csv('../Processed Data/state_week/{}/{}_{}_{}{}.csv'.format('modeling_data',med_code, '_'.join(geography) , ' '.join(dfs.state.unique()) ,gpi))
    else:
        if maintenance:
            dfs.to_csv('../Processed Data/state_week/{}/maintenance/{}{}.csv'.format('modeling_data',med_code,gpi))
        else:
            dfs.to_csv('../Processed Data/state_week/{}/{}{}.csv'.format('modeling_data',med_code,gpi))
        
    return
    
    
def get_modeling_data_gpi(med_code="drug_name"):
    """
    Must run raw2process at least once before.
    
    Modeling data adds : Policy,
    
    Population and Covid Cases.
    
    Does not ahve any other demographics!
    
    Cannot be used for Heterogeniety Analysis
    
    returns hcq (13 only) and dfs (all gpis)
    
    """
    geography =['state']
    time = ['time_index', 'year','week']
    #med_code= "drug_name"
    
    from data_prep.graphs import path_creator

    dimension = ''

    

    
    filenames  = glob("../Processed Data/state_week_gpi/{}/state_2*.csv".format(med_code))
    metrics = [
            'claim_count',
            #'person_count',
            'first_time_person_count']
    dfs =[] 
    for f in filenames:
            dfs.append(pa.read_csv(f))
    dfs = pa.concat(dfs)
    dfs = dfs[dfs.year<=2021]
    dfs = dfs[dfs.year>=2018]
    #dfs = dfs[dfs.week<53]
    dfs.reset_index(inplace=True)

    dfs['week_date'] = dfs[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )
        
    
    
    #dfs['drop'] = df['gpi2_drug_class'].map(lambda x: 'drop' if int(x) in [12,13, 15, 17, 43, 45, 90, 92, 96, 97] else 'keep')
    #dfs = dfs[dfs['drop']=='keep']
    
    
    
    dfs = add_policy(dfs)

                
    dfs["claim_count"]= (dfs["claim_count"]/12340) *100000      
    
    dfs['first_time_person_count']= (dfs['first_time_person_count']/25340) *100000      
    
    dfs['total_days_supplied']= (dfs['total_days_supplied']/6340) *100000      

    covid = pa.read_csv("../Data/Covid_state_20_21.csv")
    dfs = pa.merge(dfs, covid, on=['year','week','state'], how="left")
    dfs.fillna(0,inplace=True)

    population = pa.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    dfs["population"] = dfs.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    dfs["cases_raw"] = dfs["cases"]
    dfs["cases"] = (dfs["cases"]/dfs["population"])*100000

    dfs['ln_claim_count'] = dfs["claims_per100k"] = ( (dfs["claim_count"])/ (dfs["population"]) )*100000
    dfs["cases_pol" ]= dfs.cases * dfs['ln_claim_count']

    #hcq=dfs[(dfs['gpi2_drug_class']==13.0)| (dfs['gpi2_drug_class']==96.0)]
    #hcq.to_csv('../Processed Data/state_week/{}/{}_hcq.csv'.format('no_covid',med_code))
    dfs.to_csv('../Processed Data/state_week_gpi/{}/{}.csv'.format('modeling_data',med_code))

    
    return
    
    

def get_modeling_data_vaccine(geography =['state'], time = ['time_index', 'year','week',"week_end","week_begin"], med_code= "drug_name",maintenance=False, use_gpi=False, policy_filter=False):
    """
    Must run raw2process at least once before.
    
    Modeling data adds : Policy,
    
    Population and Covid Cases.
    
    Does not ahve any other demographics!
    
    Cannot be used for Heterogeniety Analysis
    
    returns hcq (13 only) and dfs (all gpis)
    
    """
    
    
    from data_prep.graphs import path_creator

    dimension = ''

    gpi=''
    if use_gpi==True:
        gpi=''
    elif use_gpi=='covid':
        gpi='_covid'

    if maintenance:
        filenames  = glob("../Processed Data/vaccines/state_week/{}/maintenance/{}_2*{}.csv".format('drug_name','_'.join(geography),gpi))
    else:
        filenames  = glob("../Processed Data/vaccines/state_week/{}/{}_2*{}.csv".format('drug_name','_'.join(geography),gpi))
    metrics = [
            'claim_count',
            'person_count',
            'first_time_person_count']
    dfs =[] 
    print(filenames)
    for f in filenames:
        dfs.append(pa.read_csv(f))
    dfs = pa.concat(dfs)
    dfs = dfs[dfs.year<=2021]
    dfs = dfs[dfs.year>2018]
    #dfs = dfs[dfs.week<53]
    dfs.reset_index(inplace=True)

    dfs['week_date'] = dfs.week_end.copy() #dfs['week_date'] = dfs[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )
    
    print(dfs[["week_date","week_end","week_begin"]].head(3))
    dfs = add_policy(dfs)
    if policy_filter:
        pass
               
    dfs["claim_count_orig"] = dfs["claim_count"]   
    
    dfs["person_count_orig"] = dfs["person_count"]   
    
    dfs['first_time_person_count_orig'] = dfs['first_time_person_count']
    
    dfs['total_days_supplied_orig'] = dfs['total_days_supplied'] 
        
        
    dfs["claim_count"]= (dfs["claim_count"]/12340) *100000      
    
    dfs['first_time_person_count']= (dfs['first_time_person_count']/25340) *100000      
    
    dfs['person_count']= (dfs['person_count']/25340) *100000      
    
    dfs['total_days_supplied']= (dfs['total_days_supplied']/6340) *100000            

    covid = pa.read_csv("../Data/Covid_state_20_21.csv") #gdrive_folder/GoodRx/new_submission/Data/
    covid.state= covid.state.map(lambda x: x.upper())
    covid.drop(['time_index'], axis=1, inplace=True)
    dfs = pa.merge(dfs, covid, on=['year','week','state'], how="left")
    dfs.fillna(0,inplace=True)

    population = pa.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    dfs["population"] = dfs.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    dfs["cases_raw"] = dfs["new_cases"]
    dfs["cases"] = (dfs["new_cases"]/dfs["population"])*100000

    dfs['ln_claim_count'] = dfs["claims_per100k"] = ( (dfs["claim_count"])/ (dfs["population"]) )*100000 
    #dfs["cases_pol" ]= dfs.cases * dfs['ln_claim_count']

    #hcq=dfs[(dfs['gpi2_drug_class']==13.0)| (dfs['gpi2_drug_class']==96.0)]
    #hcq.to_csv('../Processed Data/state_week/{}/{}_hcq.csv'.format('no_covid',med_code))
    if "channel" in geography or 'retailer' in geography:
        dfs.to_csv('../Processed Data/vaccines/state_week/{}/{}_{}_{}{}.csv'.format('modeling_data',med_code, '_'.join(geography) , ' '.join(dfs.state.unique()) ,gpi))
    else:
        if maintenance:
            dfs.to_csv('../Processed Data/vaccines/state_week/{}/maintenance/{}{}.csv'.format('modeling_data',med_code,gpi))
        else:
            dfs.to_csv('../Processed Data/vaccines/state_week/{}/{}{}.csv'.format('modeling_data',med_code,gpi))
        
    return
    
    
def get_modeling_data_gpi(med_code="drug_name"):
    """
    Must run raw2process at least once before.
    
    Modeling data adds : Policy,
    
    Population and Covid Cases.
    
    Does not ahve any other demographics!
    
    Cannot be used for Heterogeniety Analysis
    
    returns hcq (13 only) and dfs (all gpis)
    
    """
    geography =['state']
    time = ['time_index', 'year','week']
    #med_code= "drug_name"
    
    from data_prep.graphs import path_creator

    dimension = ''

    

    
    filenames  = glob("../Processed Data/state_week_gpi/{}/state_2*.csv".format(med_code))
    metrics = [
            'claim_count',
            #'person_count',
            'first_time_person_count']
    dfs =[] 
    for f in filenames:
            dfs.append(pa.read_csv(f))
    dfs = pa.concat(dfs)
    dfs = dfs[dfs.year<=2021]
    dfs = dfs[dfs.year>=2018]
    #dfs = dfs[dfs.week<53]
    dfs.reset_index(inplace=True)

    dfs['week_date'] = dfs[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )
        
    
    
    #dfs['drop'] = df['gpi2_drug_class'].map(lambda x: 'drop' if int(x) in [12,13, 15, 17, 43, 45, 90, 92, 96, 97] else 'keep')
    #dfs = dfs[dfs['drop']=='keep']
    
    
    
    dfs = add_policy(dfs)

                
    dfs["claim_count"]= (dfs["claim_count"]/12340) *100000      
    
    dfs['first_time_person_count']= (dfs['first_time_person_count']/25340) *100000      
    
    dfs['total_days_supplied']= (dfs['total_days_supplied']/6340) *100000      

    covid = pa.read_csv("../Data/Covid_state_20_21.csv")
    dfs = pa.merge(dfs, covid, on=['year','week','state'], how="left")
    dfs.fillna(0,inplace=True)

    population = pa.read_csv("../Data/df_aggregate/pharmacy_state_full_covid.csv")
    population = population[["State", "state_population", "Category"]].drop_duplicates()
    dfs["population"] = dfs.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
    dfs["cases_raw"] = dfs["cases"]
    dfs["cases"] = (dfs["cases"]/dfs["population"])*100000

    dfs['ln_claim_count'] = dfs["claims_per100k"] = ( (dfs["claim_count"])/ (dfs["population"]) )*100000
    dfs["cases_pol" ]= dfs.cases * dfs['ln_claim_count']

    #hcq=dfs[(dfs['gpi2_drug_class']==13.0)| (dfs['gpi2_drug_class']==96.0)]
    #hcq.to_csv('../Processed Data/state_week/{}/{}_hcq.csv'.format('no_covid',med_code))
    dfs.to_csv('../Processed Data/state_week_gpi/{}/{}.csv'.format('modeling_data',med_code))

    
    return
def get_modeling_data_testing():
    """
    Must run raw2process at least once before.
    
    Modeling data adds : Policy,
    
    Population and Covid Cases.
    
    Does not ahve any other demographics!
    
    Cannot be used for Heterogeniety Analysis
    
    returns hcq (13 only) and dfs (all gpis)
    
    """
    geography =['state']
    time = ['time_index', 'year','week']
    med_codes= ["gpi2_drug_class"]
    
    from data_prep.graphs import path_creator

    dimension = ''

    

    for med_code in med_codes:
        filenames  = glob("../Processed Data/state_week/{}/state_2*.csv".format('drug_name'))
        metrics = [
            'claim_count',
            #'person_count',
            'first_time_person_count']
        dfs =[] 
        for f in filenames:
            dfs.append(pa.read_csv(f))
        dfs = pa.concat(dfs)
        dfs = dfs[dfs.year<2021]
        dfs = dfs[dfs.year>=2018]
        dfs = dfs[dfs.week<53]
        dfs.reset_index(inplace=True)

        dfs['week_date'] = dfs[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )
        
        
        #dfs['drop'] = dfs['gpi2_drug_class'].map(lambda x: 'drop' if int(x) in [12, 15, 17, 43, 45, 90, 92, 97] else 'keep')
        #dfs = dfs[dfs['drop']=='keep']
        
        #dfs['gpi2_drug_class']= dfs['gpi2_drug_class'].map(lambda x: 13.0 if x in [13.0, 96.0] else x) 
        
        """
        keep_states = dfs[ ( dfs['year']==2019) & (dfs['gpi2_drug_class']==13.0) ].groupby("state", as_index=False).agg({'claim_count':'min'})
        keep_states = keep_states[keep_states['claim_count']>0]
        keep_states.rename(columns={'claim_count':'min_clm_cnt'}, inplace=True)
        dfs = pa.merge(dfs,keep_states, on='state',how='inner')
        """


        dfs = add_policy(dfs)

                
        dfs["claim_count"]= (dfs["claim_count"]/123400) *100000        

        covid = pa.read_csv("../Data_IB/covid_state_week_2020.csv")
        dfs = pa.merge(dfs, covid, on=['year','week','state'], how="left")
        dfs.fillna(0,inplace=True)

        population = pa.read_csv("../Data_IB/df_aggregate/pharmacy_state_full_covid.csv")
        population = population[["State", "state_population", "Category"]].drop_duplicates()
        dfs["population"] = dfs.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
        dfs["cases_raw"] = dfs["cases"]
        dfs["cases"] = (dfs["cases"]/dfs["population"])*100000

        dfs['ln_claim_count'] = dfs["claims_per100k"] = ( (dfs["claim_count"])/ (dfs["population"]) )*100000 
        dfs["cases_pol" ]= dfs.cases * dfs['ln_claim_count']

        hcq=dfs[(dfs['gpi2_drug_class']==13.0)| (dfs['gpi2_drug_class']==96.0)]
        #hcq.to_csv('../Processed Data/state_week/{}/{}_hcq.csv'.format('no_covid',med_code))
        #dfs.to_csv('../Processed Data/state_week/{}/{}.csv'.format('no_covid',med_code))

        import numpy as np
        for policy in ['UMPR','SPU','LMP','SPR','UMPU']:
            umpr_wp = hcq[((hcq.policy==policy)|(hcq.policy=='WP') ) & (hcq.year==2020) ]
            #print(set(umpr_wp['policy']))             
            #print(umpr_wp.groupby("policy").agg({"state":"unique"}))
            #umpr_wp.to_csv('../Processed Data/state_week/{}/{}_{}_hcq.csv'.format('no_covid',policy,med_code))
    
        return hcq, dfs
        
    

def get_supply_bin_modeling_data():
    """
    Must run raw2process at least once before.
    
    Modeling data adds : Policy,
    
    Population and Covid Cases.
    
    Does not ahve any other demographics!
    
    Cannot be used for Heterogeniety Analysis
    
    returns hcq (13 only) and dfs (all gpis)
    
    """
    geography =['state']
    time = ['time_index', 'year','week']
    med_codes= ["gpi2_drug_class"]
    
    from data_prep.graphs import path_creator

    dimension = ''

    hcq_list = []
    gpi_list = []

    for med_code in med_codes:
        filenames  = glob("../Processed Data/state_week/{}/{}/*state_2*.csv".format('no_covid','supply_bin'))
        metrics = [
            'claim_count',
            #'person_count',
            'first_time_person_count']
        dfs_raw =[] 
        for f in filenames:
            dfs_raw.append(pa.read_csv(f))
        dfs_raw = pa.concat(dfs_raw)
        dfs_raw = dfs_raw[dfs_raw.year<2021]
        dfs_raw = dfs_raw[dfs_raw.year>=2018]
        dfs_raw = dfs_raw[dfs_raw.week<53]
        dfs_raw.reset_index(inplace=True)

        dfs_raw['week_date'] = dfs_raw[["year","week"]].apply(lambda x: date_format(x)  , axis=1 )
        
        
        dfs_raw['drop'] = dfs_raw['gpi2_drug_class'].map(lambda x: 'drop' if int(x) in [12, 15, 17, 43, 45, 90, 92, 97] else 'keep')
        dfs_raw = dfs_raw[dfs_raw['drop']=='keep']
        
        dfs_raw['gpi2_drug_class']= dfs_raw['gpi2_drug_class'].map(lambda x: 13.0 if x in [13.0, 96.0] else x) 
        
        """
        keep_states = dfs[ ( dfs['year']==2019) & (dfs['gpi2_drug_class']==13.0) ].groupby("state", as_index=False).agg({'claim_count':'min'})
        keep_states = keep_states[keep_states['claim_count']>0]
        keep_states.rename(columns={'claim_count':'min_clm_cnt'}, inplace=True)
        dfs = pa.merge(dfs,keep_states, on='state',how='inner')
        """

        for bin_indic in dfs_raw.days_supply_bin_indic30.unique():
            dfs= dfs_raw[dfs_raw.days_supply_bin_indic30==bin_indic]    
            dfs = add_policy(dfs)


            dfs["claim_count"]= (dfs["claim_count"]) *1000    + 1000  

            covid = pa.read_csv("../Data_IB/covid_state_week_2020.csv")
            dfs = pa.merge(dfs, covid, on=['year','week','state'], how="left")
            dfs.fillna(0,inplace=True)

            population = pa.read_csv("../Data_IB/df_aggregate/pharmacy_state_full_covid.csv")
            population = population[["State", "state_population", "Category"]].drop_duplicates()
            dfs["population"] = dfs.merge(population, left_on="state", right_on="State", how="left")["state_population"].values
            dfs["cases_raw"] = dfs["cases"]
            dfs["cases"] = (dfs["cases"]/dfs["population"])*100000

            dfs['ln_claim_count'] = dfs["claims_per100k"] = ( (dfs["claim_count"])/ (dfs["population"]) )*100000 *1000
            dfs["cases_pol" ]= dfs.cases * dfs['ln_claim_count']

            hcq=dfs[(dfs['gpi2_drug_class']==13.0)| (dfs['gpi2_drug_class']==96.0)]
            hcq.to_csv('../Processed Data/state_week/{}/supply_{}_hcq.csv'.format('no_covid',bin_indic))
            dfs.to_csv('../Processed Data/state_week/{}/supply_{}.csv'.format('no_covid',bin_indic))
            hcq_list.append(hcq)
            gpi_list.append(dfs)
            import numpy as np
            for policy in ['UMPR','SPU','LMP','SPR','UMPU']:
                umpr_wp = hcq[((hcq.policy==policy)|(hcq.policy=='WP') ) & (hcq.year==2020) ]
                #print(set(umpr_wp['policy']))             
                #print(umpr_wp.groupby("policy").agg({"state":"unique"}))
                umpr_wp.to_csv('../Processed Data/state_week/{}/supply_bin_{}_{}_hcq.csv'.format('no_covid',policy,bin_indic))

    return hcq_list, gpi_list