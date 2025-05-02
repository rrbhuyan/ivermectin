import pandas as pa

from data_prep.format_data import format_data
import gc
from data_prep.graphs import path_creator
from data_prep.add_demogs import add_county
from data_prep.state_map import abbrev_us_state
from data_prep.pharmacy_map import pharmacy_map
import datetime
def end_week(curr_date) :
    end_date = datetime.datetime.strptime(curr_date, "%Y-%m-%d") + datetime.timedelta(days=7)
    return str(end_date).split(" ")[0]
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


fnames ={
    
    #'2016': '../Data/rm_claims_aggregation_research_2016.csv',
    '2017':'../Data/USC_data_extract_y2017_20230310.csv000 Jeroen van Meijgaard.gz',
    '2018':'../Data/USC_data_extract_y2018_20230310.csv000 Jeroen van Meijgaard.gz',
    '2019':'../Data/USC_data_extract_y2019_20230310.csv000 Jeroen van Meijgaard.gz',
    '2020':'../Data/USC_data_extract_y2020_20230310.csv000 Jeroen van Meijgaard.gz',
    '2021':'../Data/USC_data_extract_y2021_20230310.csv000 Jeroen van Meijgaard.gz'
}

columns=['calendar_year','week_begin','parent_pharmacy_name','pharmacy_zip_code', 'pharmacy_state', 'is_maintenance_drug_clinical','gpi2_drug_class','drug_name','is_multisource','total_days_supplied','claim_count','person_count','first_time_person_count']

def get_pharmacy_names(years=['2018','2019','2020','2021']):
    """
    Returns names of pharmacies
    """
    pharmacy_names= []
    for yr in years:
        print(yr, fnames[yr])
        df = pa.read_csv(fnames[yr], compression='gzip')
        df.columns = columns
        print(yr, "  read")
        pharmacy_names.extend(list(df["parent_pharmacy_name"].unique()))
        del df
        gc.collect()
    return pharmacy_names
    

def get_top_drugs(years=['2018','2019']):
    """
    Returns names of drugs, years and volumes
    """
    temps=[]    
    for yr in years:
        print(yr, fnames[yr])
        df = pa.read_csv(fnames[yr], compression='gzip')
        df.columns = columns
        print(yr, "  read")
        
        df = df.groupby(["calendar_year","drug_name",'gpi2_drug_class','is_maintenance_drug_clinical'], as_index=False).agg({"claim_count":"sum"})
        temps.append(df)
        del df
        gc.collect()
    df = pa.concat(temps)
    df = df.groupby(["calendar_year","drug_name",'gpi2_drug_class','is_maintenance_drug_clinical'], as_index=False).agg({"claim_count":"sum"})
    return df
    
    

    
def profile_hcq(geography, dimensions, time=[ 'time_index', 'year',"week",'week_begin'], filter_state=False, add_dims=False, years=['2019','2020','2021'], add_county_dim=False):
    """
    Simple Aggregation
    """
    
    agg=[]
    for yr in years:
        print(yr, fnames[yr])
        df = pa.read_csv(fnames[yr], compression='gzip')
        df.columns=columns
        print(yr, "  read")
        df = df[(df.drug_name=="hydroxychloroquine") |(df.drug_name=="plaquenil")]        
        #df = filter_drugs(df)
        df = filter_zips(df)
        #df = df[df.drug_name!="hydroxychloroquine"]

        if filter_state!=False:
            df['keep'] = df['pharmacy_state'].map(lambda x: 1 if x in filter_state else 0)
            df = df[df['keep']==1]        
        df = format_data(df, add_dims=add_dims,add_county_dim=False)
        
        
        print(yr, "  formatted")
        metrics = ['person_count','claim_count','total_days_supplied','first_time_person_count']
    
        measures = {}

        for m in metrics:
            measures[m] = 'sum'
        print(df.groupby(dimensions, as_index=False).agg(measures))
        del df
        gc.collect
    return 
    
def simple_agg(geography, dimensions, time=[ 'time_index', 'year',"week",'week_begin'], filter_state=False, add_dims=False, years=['2019','2020','2021'], add_county_dim=False):
    """
    Simple Aggregation
    """
    
    agg=[]
    for yr in years:
        print(yr, fnames[yr])
        df = pa.read_csv(fnames[yr], compression='gzip')
        df.columns=columns
        print(yr, "  read")
        
        #df = filter_drugs(df)
        df = filter_zips(df)
        #df = df[df.drug_name!="hydroxychloroquine"]
        df = df[df.drug_name=="hydroxychloroquine"]
        if filter_state!=False:
            df['keep'] = df['pharmacy_state'].map(lambda x: 1 if x in filter_state else 0)
            df = df[df['keep']==1]        
        df = format_data(df, add_dims=add_dims,add_county_dim=False)
        
        if add_county_dim:
            df = agg_data(df,  geography =['state', 'pharmacy_zip_code'], dimensions =['ivermectin','generic','opioid','mntl_hlth','diabetics','obesity'], time =[ 'time_index', 'year',"week",'week_begin'], filter_state=False)
            df= add_county(df)
        
        print(yr, "  formatted")
        
        dim = '_'.join( dimensions)
        path = 'Processed Data/state_week/{}'.format(dim)
        #rom graphs import path_creator
        path_creator(path)
        temp = agg_data(df,geography,dimensions, time)
        temp.to_csv('../Processed Data/state_week/{}/{}_{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr ) )
        del temp
        gc.collect
    return 


def agg_data(df,  geography =['state', 'pharmacy_zip_code'], dimensions =['is_maintenance_drug',
    'gpi2_drug_class',
    'is_multisource',
    'finance_source_level_1',
    'days_supply_bin'], time =[ 'time_index', 'year',"week",'week_begin','week_end'], filter_state=False):
    print('agg_data')
    
    if 'dma_code' in geography:
        from data_prep.add_demogs import add_dma
        df= add_dma(df)
    group_vars= geography+dimensions+time
    
    if 'parent_pharmacy_name' in df.columns:
        df['pharm_cnt'] = df['parent_pharmacy_name']
    
    
    if filter_state!=False:
        df = df[df['state']==filter_state]
    
    measures = {
    #'pharm_cnt':'nunique',
    'claim_count':'sum',
    'person_count': 'sum',
    'first_time_person_count': 'sum',
    'total_days_supplied': 'sum',
    #'savings_value':'sum'   
    }
    
    #df =df[group_vars+['claim_count','person_count','first_time_person_count','total_copay','savings_value']]
    
    print(group_vars, df.shape,df.columns)
    df = df.groupby(group_vars, as_index=False).agg(measures)
    return df



def filter_drugs_gpi(df, use_gpi=True):
    if use_gpi==True:
        filter_drugs_path = "../Processed Data/keep_drugs_100.csv"
    elif use_gpi=='covid':
        filter_drugs_path = "../Processed Data/keep_drugs_covid_drugs.csv"
    keep_drugs = pa.read_csv(filter_drugs_path)
    
    print("Before drugs",df.shape)
    keep_drugs = set(keep_drugs['drug_name'].tolist())
    
    df['drug_name'] = df[['drug_name','gpi2_drug_class']].apply(lambda x: x[0] if x[0] in keep_drugs else 'gpi_'+str(x[1]), axis=1)
    #df = pa.merge(df, keep_drugs, on=['drug_name'], how="inner")
    
    #df.drop('keep',axis=1, inplace=True)
    print("After drugs",df.shape)
    
    return df


def filter_drugs(df):
    keep_drugs = pa.read_csv("../Processed Data/keep_drugs_200.csv")
    print("Before drugss",df.shape)
    df = pa.merge(df, keep_drugs, on=['drug_name'], how="inner")
        
    print("After drugss",df.shape)
    
    return df


def filter_drugs_covid(df):
    keep_drugs = pa.read_csv("../Processed Data/keep_drugs_covid_drugs.csv")
    print('covid')
    print("Before drugss",df.shape)
    df = pa.merge(df, keep_drugs, on=['drug_name'], how="inner")
        
    print("After drugss",df.shape)
    
    return df



def filter_zips(df):
    keep_zips = pa.read_csv("../Processed Data/keep_zips.csv")
        
    df = pa.merge(df, keep_zips, on=['parent_pharmacy_name','pharmacy_zip_code', 'pharmacy_state'], how="inner")
    
    return df


def med_codes_agg_gpi(geography, dimensions,med_codes, time, filter_state=False, add_dims=False, years=['2018','2019','2020','2021'], add_county_dim=False, drop_covid_meds=False, filter_on={}):
    #agg=[]
    print(years)
    for yr in years:
        print(yr, fnames[yr])
        df = pa.read_csv(fnames[yr], compression='gzip')
        #df.columns = columns
        print(yr, "  read")
        
        if len(filter_on)>0:
            for key,value in filter_on:
                df = df[df[key]==value]

        #df= df[df['is_mainenance_drug_clinical']=='t']
        
        df = filter_zips(df)
        
        
        if "drug_name" in med_codes:
            hcq = df[df.drug_name=="hydroxychloroquine"]    
            df = df[df.drug_name!="hydroxychloroquine"]

            columns = list(hcq.columns)
            columns.remove("drug_name")


            hcq.drop(['gpi2_drug_class'], axis=1, inplace=True)
            df = df.groupby(columns, as_index=False).agg( {"claim_count":"sum", 'first_time_person_count':"sum",'total_days_supplied':'sum'})
            df.rename(columns={'gpi2_drug_class' : "drug_name"}, inplace=True)
            df = pa.concat([hcq, df], axis=0)
            df.reset_index(inplace=True, drop=True)
        
        
        if filter_state!=False:
            df['keep'] = df['pharmacy_state'].map(lambda x: 1 if x in filter_state else 0)
            df = df[df['keep']==1]
        
        df = format_data(df, add_dims=add_dims,add_county_dim=False,drop_covid_meds=drop_covid_meds)
        
        
        
                
        if add_county_dim:
            df = agg_data(df,  geography =['state', 'pharmacy_zip_code'], dimensions = med_codes, time =[ 'time_index', 'year','week',"week_begin"], filter_state=False)
            df= add_county(df)
        print(yr, "  formatted")
        
        for dim in med_codes:
            path = 'Processed Data/state_week/{}'.format(dim)
            #from graphs import path_creator
            path_creator(path)
            temp = agg_data(df,geography,dimensions+[dim], time)
            temp.to_csv('../Processed Data/state_week_gpi/{}/{}_{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr ) )
            del temp
            gc.collect
        del df
        gc.collect()
    return 



def med_codes_agg(geography, dimensions,med_codes, time, filter_state=False, add_dims=False, years=['2018','2019','2020','2021'], add_county_dim=False, drop_covid_meds=False, filter_on={},use_gpi=False):
    agg=[]
    print(years)
    for x in [geography, dimensions,med_codes, time]:
        print(x)
    for yr in years:
        print(yr, fnames[yr])
        df = pa.read_csv(fnames[yr], compression='gzip')
        #df = df[(df.week_begin=="2020-12-27") & (df.drug_name=="hydroxychloroquine")]
        filtered="unfiltered"
        if len(filter_on)>0:
            filtered="maintenance"
            df= df[df['is_mainenance_drug_clinical']=='t']
            """
            for key in filter_on.keys():
                df = df[df[key]==filter_on[key]]
            """
        #df= df[df['is_mainenance_drug_clinical']=='t']
        df.columns = columns
        print(yr, "  read")
        
        if filter_state!=False:
            df['keep'] = df['pharmacy_state'].map(lambda x: 1 if x in filter_state else 0)
            df = df[df['keep']==1]
        
        gpi =''
        if use_gpi==True:
            print('use_gpi:  ',use_gpi)
            df = filter_drugs_gpi(df, use_gpi)
            gpi='_gpi'
        elif use_gpi=='covid':
            df = filter_drugs_gpi(df, use_gpi)#filter_drugs_covid(df)
            gpi ='_covid'
        else:
            df = filter_drugs(df)
        df = filter_zips(df)
        
        df.loc[ df["drug_name"] == 'plaquenil', 'drug_name'] = "hydroxychloroquine"
        df.loc[ df["drug_name"] == 'stromectol', 'drug_name'] = "ivermectin"
        
        
        
        df = format_data(df, add_dims=add_dims,add_county_dim=False,drop_covid_meds=drop_covid_meds)
        
        
        
                
        if add_county_dim:
            df = agg_data(df,  geography =['state', 'pharmacy_zip_code'], dimensions = med_codes, time =[ 'time_index', 'year','week',"week_end", "week_begin"], filter_state=False)
            df= add_county(df)
        print(yr, "  formatted")
        
        for dim in med_codes:
            path = 'Processed Data/state_week/{}'.format(dim)
            #from graphs import path_creator
            path_creator(path)
            if filtered=="maintenance":
                path_creator(path+"/maintenance")
            temp = agg_data(df,geography,dimensions+[dim], time)
            if filter_state !=False:
                temp.to_csv('../Processed Data/state_week/{}/{}_{}{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr,filter_state[0],gpi ) )
            else:
                if filtered=="maintenance":
                    temp.to_csv('../Processed Data/state_week/{}/maintenance/{}_{}{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr ,gpi) )
                else:    
                    temp.to_csv('../Processed Data/state_week/{}/{}_{}{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr ,gpi) )
            del temp
            gc.collect
    
        #del df
        #gc.collect()
    return 


def vaccine_med_codes_agg(geography, dimensions,med_codes, time, filter_state=False, add_dims=False, years=['2018','2019','2020','2021'], add_county_dim=False, drop_covid_meds=False, filter_on={},use_gpi=False):
    agg=[]
    print(years)
    for x in [geography, dimensions,med_codes, time]:
        print(x)
    for yr in years:
        print(yr, fnames[yr])
        df = pa.read_csv(fnames[yr], compression='gzip')
        #df = df[(df.week_begin=="2020-12-27") & (df.drug_name=="hydroxychloroquine")]
        filtered="unfiltered"
        if len(filter_on)>0:
            filtered="maintenance"
            df= df[df['is_mainenance_drug_clinical']=='t']
            """
            for key in filter_on.keys():
                df = df[df[key]==filter_on[key]]
            """
        #df= df[df['is_mainenance_drug_clinical']=='t']
        df.columns = columns
        print(yr, "  read")
        
        if filter_state!=False:
            df['keep'] = df['pharmacy_state'].map(lambda x: 1 if x in filter_state else 0)
            df = df[df['keep']==1]
        
        gpi =''
        
        if 'gpi2_drug_class' in df.columns:
            df = df.dropna(subset =['gpi2_drug_class'])
        if 'gpi2_drug_class' in df.columns:
            df = df.dropna(subset =['gpi2_drug_class'])
        
        if 'gpi2_drug_class' in df.columns:
            df['vaccines'] =  df['gpi2_drug_class'].map(lambda x: 'vaccines' if int(x) == 17 else 'non vaccines')
        
        df = df[df.vaccines=='vaccines']
        df.reset_index(inplace=True)
        
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
        #df = format_data(df, add_dims=add_dims,add_county_dim=False,drop_covid_meds=drop_covid_meds)
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

        for dim in med_codes:
            path = 'Processed Data/vaccines/state_week/{}'.format(dim)
            #from graphs import path_creator
            path_creator(path)
            if filtered=="maintenance":
                path_creator(path+"/vaccines/maintenance")
            temp = agg_data(df,geography,dimensions+[dim], time)
            if filter_state !=False:
                temp.to_csv('../Processed Data/vaccines/state_week/{}/{}_{}{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr,filter_state[0],gpi ) )
            else:
                if filtered=="maintenance":
                    temp.to_csv('../Processed Data/vaccines/state_week/{}/maintenance/{}_{}{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr ,gpi) )
                else:    
                    temp.to_csv('../Processed Data/vaccines/state_week/{}/{}_{}{}.csv'.format(dim,'_'.join( dimensions)+'_'.join(geography),yr ,gpi) )
            del temp
            gc.collect
    
        #del df
        #gc.collect()
    return df
"""
def agg(geography, dimensions, time, filter_state=False, add_dims=False, years=['2017','2018','2019','2020','2021']):
    agg=[]
    columns=[]
    print(years)
    for f in years:
        print(f, fnames[f])
        if '17' in f:
            df = pa.read_csv(fnames[f],header= None,usecols=range(1,16))
            df.columns=columns
        else:
            df = pa.read_csv(fnames[f])

        if '17' not in f:
            columns = df.columns
            print(columns)
        print(f, "  read")
        
        
        if filter_state!=False:
            df['keep'] = df['pharmacy_state'].map(lambda x: 1 if x in filter_state else 0)
            df = df[df['keep']==1]
        
        df = format_data(df, add_dims=add_dims)
        
        group_vars= geography+dimensions+time
        df =df[group_vars+['claim_count','person_count','first_time_person_count','total_copay','savings_value']]
            #df = df[df['state']==filter_state]
        #df['time_index'] =  df[['calendar_year','calendar_week']].apply(lambda row: int(row[0])*100+int(row[1]), axis=1)
        print(f, "  formatted")
        ndf = agg_data(df,  geography , dimensions , time)
        print('ndf shape: ',ndf.shape )
        print(f, "  agged")
        agg.append(ndf)
        del df, ndf
        gc.collect
    df = pa.concat(agg)
    del agg
    gc.collect()
    return df




def agg_zip_week(df, drop_gpi=False, minimal=False):
    print('agg_zip_week')
    group_vars = ['state', 'pharmacy_zip_code', 'is_maintenance_drug',
    'gpi2_drug_class',
    'is_multisource',
    'finance_source_level_1',
    'days_supply_bin', 'time_index', 'year','week' ]
    
    if drop_gpi:
        group_vars.remove('gpi2_drug_class')
    
    if minimal:
         group_vars = ['state', 'pharmacy_zip_code', 'is_maintenance_drug',
                        'days_supply_bin', 'time_index', 'year','week' ]
        
    if minimal=='bare_minimum':
         group_vars = ['state', 'pharmacy_zip_code', 'time_index', 'year','week' ]
    #estimated at the lowest granularity level
    #df['total_copay'] = df['avg_co_pay_amount']*df['claim_count']
    
    #df['savings_value'] = df['total_copay']*df['savings_pct_u_c']/(1-df['savings_pct_u_c'])
    df['pharm_cnt'] = df['parent_pharmacy_name']
    measures = {
    'pharm_cnt':'count',
    'claim_count':'sum',
    'person_count': 'sum',
    'first_time_person_count': 'sum',
    'total_copay': 'sum',
    'savings_value':'sum'   
    }
    df = df.groupby(group_vars, as_index=False).agg(measures)
    return df



def agg_dma_week(df, drop_gpi=False, minimal=False):
    print('agg_dma_week')
    group_vars = ['state', 'pharmacy_zip_code', 'is_maintenance_drug',
    'gpi2_drug_class',
    'is_multisource',
    'finance_source_level_1',
    'days_supply_bin', 'time_index', 'year','week' ]
    
    if drop_gpi:
        group_vars.remove('gpi2_drug_class')
    
    if minimal:
         group_vars = ['state', 'pharmacy_zip_code', 'is_maintenance_drug',
                        'days_supply_bin', 'time_index', 'year','week' ]
        
    if minimal=='bare_minimum':
         group_vars = ['state', 'dma_code', 'time_index', 'year','week' ]
    #estimated at the lowest granularity level
    #df['total_copay'] = df['avg_co_pay_amount']*df['claim_count']
    
    #df['savings_value'] = df['total_copay']*df['savings_pct_u_c']/(1-df['savings_pct_u_c'])
    df['pharm_cnt'] = df['parent_pharmacy_name']
    measures = {
    'pharm_cnt':'count',
    'claim_count':'sum',
    'person_count': 'sum',
    'first_time_person_count': 'sum',
    'total_copay': 'sum',
    'savings_value':'sum'   
    }
    df = df.groupby(group_vars, as_index=False).agg(measures)
    return df
    
"""