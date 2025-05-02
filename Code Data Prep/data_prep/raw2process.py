
from data_prep.aggregate_data import  med_codes_agg, med_codes_agg_gpi,vaccine_med_codes_agg


def raw2process(geography =['state'],time = ['time_index', 'year','week',"week_begin","week_end"],med_codes= ["drug_name"],dimensions = [],filter_on={}, filter_state=False,use_gpi=False):
    """
    Converts Raw data into a Processsed Data Set. This does not yield any analyzable data !
    """
    for yr in ['2020','2019']:
        if isinstance(yr,list):
            yr_=yr
        else:
            yr_=[yr]
        med_codes_agg(geography,dimensions,med_codes, time,add_dims=False,years=yr_, add_county_dim=False, drop_covid_meds=False,filter_on=filter_on, filter_state=filter_state,use_gpi=use_gpi)
        

def raw2process_vaccine(geography =['state'],time = ['time_index', 'year','week',"week_begin","week_end"],med_codes= ["drug_name"],dimensions = [],filter_on={}, filter_state=False,use_gpi=False):
    """
    Converts Raw data into a Processsed Data Set. This does not yield any analyzable data !
    """
    for yr in ['2021']:
        if isinstance(yr,list):
            yr_=yr
        else:
            yr_=[yr]
        vaccine_med_codes_agg(geography,dimensions,med_codes, time,add_dims=False,years=yr_, add_county_dim=False, drop_covid_meds=False,filter_on=filter_on, filter_state=filter_state,use_gpi=use_gpi)
        
        
def raw2process_gpi(med_codes=['gpi2_drug_class']):
    """
    Converts Raw data into a Processed Data Set. This does not yield any analyzable data !
    Note this mangles the defitionions of drugname and gpi2 - so hcq drugname rest is gpi2.
    """
    geography =['state']
    time = ['time_index', 'year','week',"week_begin"]
    #med_codes= ["drug_name"] 
    dimensions = []
    for yr in ['2021']: #'2018','2019','2020',
        if isinstance(yr,list):
            yr_=yr
        else:
            yr_=[yr]
        med_codes_agg_gpi(geography,dimensions,med_codes, time,add_dims=False,years=yr_, add_county_dim=False, drop_covid_meds=False)