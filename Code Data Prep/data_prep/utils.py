import ast
import pandas as pa

from isoweek import Week

from datetime import datetime,date
#from dateutil.relativedelta import relativedelta

MonthDict={ 1 : "Jan",
       2 : "Feb",
       3 : "Mar",
       4 : "Apr",
       5 : "May",
       6 : "Jun",
       7 : "Jul",
       8 : "Aug",
       9 : "Sep",
       10 : "Oct",
       11 : "Nov",
       12 : "Dec"
}


def year_week_date(year, week):
    w= Week(int(year), int(week) )
    str_time = str(w.sunday()).split("-")
    return int(str_time[0])*10000+int(str_time[1])*100+int(str_time[2])



def get_iso_date( gregorian_year, gregorian_month, gregorian_day):
    from datetime import date
    from isoweek import Week
    gregorian_date = date(gregorian_year, gregorian_month, gregorian_day)
    iso_year = gregorian_date.isocalendar()[0]
    iso_week= gregorian_date.isocalendar()[1]
    res = Week(iso_year, iso_week).sunday() 
    return res

def get_week_dt(x):
    gregorian_year = pa.to_datetime(x).year
    gregorian_month =pa.to_datetime(x).month
    gregorian_day =pa.to_datetime(x).day
    res = get_iso_date(gregorian_year, gregorian_month, gregorian_day)   
    return res



def count_related(x, covid):
    #print(x)
    total = 0
    val_x = ast.literal_eval(x) if '[' in x else x
    for v in val_x :
        #print(v)
        if v in covid:
            total+=1
    return total


def prep_dates_(df,date_column ='date', sep ='-'):
    df[['month','day','year']] = df[date_column].str.split(sep,expand=True)
    
    for unit in ['year','month','day']:
        df[unit] = df[unit].map(lambda x: int(x))
        
    df['clean_date'] = df[date_column].map(lambda x: pa.to_datetime(x).date())
    df['clean_date_st'] = df['clean_date'].map(lambda x: str(x.year*10000+x.month*100+x.day))
    df['week_dt'] = df['clean_date'].map(lambda x: get_week_dt(x))
    df['iso_dt_st'] = df['week_dt'].map(lambda x: str(x.year)+'-'+MonthDict[x.month]+'-'+str(x.day))
    df['iso_dt'] = df['week_dt'].map(lambda x: x.year*10000+x.month*100+x.day)
    #df['iso_dt_date'] = df['week_dt'].map(lambda x: x.year*10000+x.month*100+x.day)
    df['year_mon'] = df['clean_date'].map(lambda x: x.year*100+x.month)
    df['year_mon_st'] =  df['clean_date'].map(lambda x: str(x.year*100+x.month))
    return df

def prep_dates(df,date_column ='date', sep ='-'):
    df[['year','month','day']] = df[date_column].str.split(sep,expand=True)
    
    for unit in ['year','month','day']:
        df[unit] = df[unit].map(lambda x: int(x))
        
    df['clean_date'] = df[date_column].map(lambda x: pa.to_datetime(x).date())
    df['clean_date_st'] = df['clean_date'].map(lambda x: str(x.year*10000+x.month*100+x.day))
    df['week_dt'] = df['clean_date'].map(lambda x: get_week_dt(x))
    df['iso_dt_st'] = df['week_dt'].map(lambda x: str(x.year)+'-'+MonthDict[x.month]+'-'+str(x.day))
    df['iso_dt'] = df['week_dt'].map(lambda x: x.year*10000+x.month*100+x.day)
    #df['iso_dt_date'] = df['week_dt'].map(lambda x: x.year*10000+x.month*100+x.day)
    df['year_mon'] = df['clean_date'].map(lambda x: x.year*100+x.month)
    df['year_mon_st'] =  df['clean_date'].map(lambda x: str(x.year*100+x.month))
    return df

def get_hashtags(df):
    df['clean_hashtags'] = df.hashtags.map(lambda x:  ast.literal_eval(x) if '[' in x else x )
    hashtags = df['clean_hashtags'].to_list()
    h_temp =[]
    for x in hashtags:
        h_temp.extend(x)
    return h_temp
    
    
def get_related_hashtags(hashtag_count, related_words=['covid', 'corona', 'pandemic' ]):
    tags = set()
    total = 0
    for x in hashtag_count:
        for ht in related_words:
            if ht in x[0] and x[0] not in tags:
                tags.add(x[0])
                total+=x[1]
    return tags, total



def expander(df, clean_date_F, grp_var,freq='D'):
   
    idx = pa.date_range(df[clean_date_F].min(),  df[clean_date_F].max(), freq=freq)
    
    df.index = df[clean_date_F]
    expanded =[]
    for grp in df[grp_var].unique():
        print(grp)
        temp = df[df[grp_var]==grp]
        #print( temp.mass_indic.sum())
        temp = temp.reindex(idx, fill_value=0)
        
        temp['date']= temp.index
        temp[clean_date_F]= temp.index
        temp['year']=pa.DatetimeIndex(temp['date']).year
        temp['month']=pa.DatetimeIndex(temp['date']).month
        temp['day']=pa.DatetimeIndex(temp['date']).day
        temp['week_dt'] = temp[clean_date_F].map(lambda x: get_week_dt(x))
        temp[grp_var] = grp
        temp[clean_date_F+'_st'] = temp[clean_date_F].map(lambda x: str(x.year*10000+x.month*100+x.day))
        print(temp[grp_var].head())
        expanded.append(temp)
    expanded = pa.concat(expanded,axis=0)
    
    print('expanded')
    return expanded
