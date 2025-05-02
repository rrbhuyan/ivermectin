
import pandas as pa
import gc
import os

from data_prep.pharmacy_map import pharmacy_map
import datetime
def end_week(curr_date) :
    end_date = datetime.datetime.strptime(curr_date, "%Y-%m-%d") + datetime.timedelta(days=7)
    return str(end_date).split(" ")[0]

from data_prep.add_demogs import add_politics, add_census_demo, add_county,add_dma

import datetime
def get_week(row):
    year =row[0]
    month =row[1]
    day=row[2]
    return datetime.date(int(year),int(month),int(day) ).isocalendar()[1]

def get_year(row):
    year =row[0]
    month =row[1]
    day=row[2]
    return datetime.date(int(year),int(month),int(day) ).isocalendar()[0]

def format_data(df, date_fix=True, add_dims=True, add_county_dim=False, drop_covid_meds=False, exclude_vaccines=True):
    from data_prep.state_map import abbrev_us_state
    
    #drop missings
    df = df.dropna(subset =['calendar_year'])
    df = df.dropna(subset =['week_begin'])
    df = df.dropna(subset =['pharmacy_zip_code'])
    if 'gpi2_drug_class' in df.columns:
        df = df.dropna(subset =['gpi2_drug_class'])
    #df = df.dropna(subset =['savings_pct_u_c'])
    df = df.dropna(subset =['parent_pharmacy_name'])
    print(df.shape)
    #df = df[df['savings_pct_u_c']!=1]
    print(df.shape)
    #geography
    df['pharmacy_state'] = df['pharmacy_state'].map(lambda x: abbrev_us_state[x].upper() if x in abbrev_us_state else 'none' )
    df = df[df['pharmacy_state']!='none']
    df['pharmacy_zip_code'] = df['pharmacy_zip_code'].astype(int)
    df['pharmacy_state'] = df['pharmacy_state'].map(lambda x: x.upper())
    df['state'] = df['pharmacy_state']
    #measures
    #df['total_copay'] = df['avg_co_pay_amount']*df['claim_count']
    #df['savings_value'] = (df['total_copay']*df['savings_pct_u_c'])/(1-df['savings_pct_u_c'])  
    
    df = df.rename(columns ={'is_mainenance_drug_clinical':'is_maintenance_drug'})
       
    # grouping variables
    #df['is_maintenance_drug'] = df['is_maintenance_drug'].map(lambda x: 'Maintenance' if x else 'Not Maintenance' )
    #df['days_supply_bin_indic'] =  df['days_supply_bin'].map(lambda x: supply_bin(x))
    #df['days_supply_bin_indic30'] =  df['days_supply_bin'].map(lambda x: 'lt30' if x == 'under 30 days' else 'gte30')
    #df['opioid'] =  df['gpi2_drug_class'].map(lambda x: 'Opioid' if int(x) == 65 else 'Non Opioid')
    #df['obesity'] =  df['gpi2_drug_class'].map(lambda x: 'Obesity' if int(x) == 61 else 'Non Obesity')
    #df['diabetics'] =  df['gpi2_drug_class'].map(lambda x: 'Diabetic' if int(x) == 27 else 'Non Diabetic')
    if 'gpi2_drug_class' in df.columns:
        df['vaccines'] =  df['gpi2_drug_class'].map(lambda x: 'vaccines' if int(x) == 17 else 'non vaccines')
    if exclude_vaccines:    
        df = df[df.vaccines=='non vaccines']
        df.reset_index(inplace=True)
    else:
        df = df[df.vaccines=='vaccines']
        df.reset_index(inplace=True)
    #df['hcq'] =  df['gpi2_drug_class'].map(lambda x: 'hcq' if int(x) == 13 else 'Non hcq')
    #df['mntl_hlth'] =  df['gpi2_drug_class'].map(lambda x: 'Mental' if int(x) in [57,58,59,60,61,62] else 'Non Mental')
    #df['generic'] = df['is_multisource'].map(lambda x: 'Generic' if int(x) == 1 else 'Branded')
    df['channel'] = df['parent_pharmacy_name'].map(lambda x: pharmacy_map[x])
    #df['channel_stock'] =df['channel']  +"_"+ df['days_supply_bin_indic30']
    #if  drop_covid_meds:
    #    df['drop'] = df['gpi2_drug_class'].map(lambda x: 'drop' if int(x) in [12, 15, 17, 43, 45, 90, 92, 96, 97] else 'keep')
    #    df = df[df['drop']=='keep']
    
    print(df.head())
    #end_date = date_1 + datetime.timedelta(days=7)
    df["week_end"] = df["week_begin"].map(lambda x: end_week(x))
    df[['year','month','day']] = df["week_begin"].str.split('-',expand=True)
    
    # time
    #df['year'] = df[['year','month','day']].apply(lambda row:  get_year(row), axis=1)
    df['week'] = df[['year','month','day']].apply(lambda row:  get_week(row), axis=1)
    df['calendar_week'] =  df['week']
    df['year'] =  df['calendar_year']
    #import datetime
    #d = "2013-W26"
    #r = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
    df['time_index'] =  df[['year','week']].apply(lambda row: int(row[0])*100+int(row[1]), axis=1)
    #df['time_index'] =  df[['calendar_year','calendar_week']].apply(lambda row: int(row[0])*100+int(row[1]), axis=1)
    #df = add_dma(df)
    if add_dims:
        #adding_politics
        df = add_politics(df)
        print(df.columns, df.shape)

        #adding_demogs
        #df = add_census_demo(df)
        #df = add_census_income(df)
        #df = add_census_insurance(df)
    if add_county_dim:
        df= add_county(df)
    print("formatted")
    return df

  


    
    
def get_data(time_window=[], state=None,path = r"../Data/"):
    """
    state takes none for all states and string value for each particular state
    """
    from glob import glob
    fnames = glob(path+'*.csv')
    state_interval=[]
    fnames_clean =[]
    for f in fnames:
        for time_unit in time_window:
            if time_unit in f:
                fnames_clean.append(f)
    for fname in  fnames_clean:
        print(fname)
        df = pa.read_csv(fname)
        df.name = fname.replace('../Data/',"").replace('.csv','.xlsx')
        if state != None: 
            state_interval_ =  df[df['pharmacy_state']==state]
        else:
            print('all')
            
            state_interval_ = df.copy()
        state_interval.append(state_interval_)
        del df 
        gc.collect()

        
    if state==None:
        state='all'
    state_interval =  pa.concat(state_interval)
    if not os.path.exists('../Processed Data/{}'.format(state)):
        os.makedirs('../Processed Data/{}'.format(state))
    state_interval.to_csv('../Processed Data/{}/{}.csv'.format(state,''.join(time_window)))
    return state_interval



def rotate_data(df, metric,  level= 'dma', state=True):
    from graphs import path_creator
    min_count= 80
    if 'dma' in level:
        grp_vars=['state','dma_code']
        lev='dma_code'
    elif 'county' in level:
        grp_vars=['state','county']
        lev='county'
    else:
        grp_vars=['state','pharmacy_zip_code']
        lev='pharmacy_zip_code'
    new_df = pa.pivot_table(df, values=metric, index=grp_vars, columns='time_index').reset_index()
    
    new_df['filter_condn'] = new_df[set(new_df.columns)-{'state',lev}].sum(axis=1)
    
    if state:
        state = df.state.to_list()[0]
        from state_map import us_state_abbrev 
        state = us_state_abbrev[  ' '.join([s.lower().capitalize() for s in state.split(' ')  ] )]
        print(state)
    else:
        state='all'
    #new_df.dropna( axis=0, thresh=min_count, inplace=True)
    
    path = 'Processed Data/{}'.format(state)
    path_creator(path)
    print(path)
    new_df.to_csv('../'+path+'/func_{}_{}_agg.csv'.format(metric, level))
    return new_df