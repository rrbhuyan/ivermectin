

import pandas as pa

from data_prep.utils import get_week_dt

import datetime
def get_week(row):
    year =row[2]
    month =row[0]
    day=row[1]
    year_week_day=datetime.date(int(year),int(month),int(day) ).isocalendar()
    return str(year_week_day[0]*100+year_week_day[1])

def get_time_indices(drug = 'hydroxychloroquine' ):
    
    
    if drug=='hydroxychloroquine':
        events_dict={#'2-25-2020' :"Raoult posts a video about HCQ",
        '3-20-2020' :'3-20-2020: Trump Endorses HCQ.\nFauci Counters Trump Claims',
        '4-14-2020' :'4-14-2020: 35 States Impose\nRestrictions on HCQ use',
            '5-18-2020': '5-18-2020: Trump says he is taking hydroxychloroquine',
        '6-21-2020' :'6-21-2020: FDA declares HCQ\nis no longer under EUA, NIH stops HCQ trials',
        
        #'1-14-2021' : 'NIH: Not enough Data to say Ivermectin works',
        #'3-5-2021': 'FDA warns against Ivermectin use.',
        '7-4-2021':  '7-4-2021: 67% of Adults in\nthe US receive at least one dose of vaccine.'      
        }
    elif drug=='ivermectin':
        events_dict={#'2-25-2020' :"Raoult posts a video about HCQ",
        '4-01-2020' :'4-01-2020: First Australian Study Suggesting Promise of Ivermectin',
            
         '2-4-2021': '2-4-2021: Merck Issues a warning against Ivermectin use',   
        #'4-14-2020' :'4-14-2020: 35 States Impose  \n Restrictions on HCQ use',
        #'6-21-2020' :'6-21-2020: FDA declares HCQ  \n is no longer under EUA, NIH stops HCQ trials',
        '12-8-2020': '12-8-2020: Kory testifies in favor of Ivermectin in a Senate Committee',
        '1-14-2021' : '1-14-2021: NIH Not enough Data to say Ivermectin works',
        '3-5-2021': '3-5-2021: FDA warns against Ivermectin use.',
        '7-4-2021':  '7-4-2021: 67% of Adults in  \n the US receive at least one dose of vaccine.'     ,
            '8-26-2021':'8-26-2021: CDC Issues Warning on IVM Use',
         #'9-1-2021': '9-1-2021: Medical Establishment Calls for Stopping Ivermectin Use \n Joe Rogan discloses Ivermectin use'   
        }
    
    new_dict={}
    dates = [new_dict.update({x: get_week_dt(x)})  for x in events_dict.keys()]
    new_events_dict={}
    for date in events_dict.keys():
        new_events_dict[str(new_dict[date])] = events_dict[date]
    return new_events_dict
    
   

