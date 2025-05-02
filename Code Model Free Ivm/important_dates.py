import pandas as pa

from utils import get_week_dt

import datetime
def get_week(row):
    year =row[2]
    month =row[0]
    day=row[1]
    year_week_day=datetime.date(int(year),int(month),int(day)).isocalendar()
    return str(year_week_day[0]*100+year_week_day[1])

def get_time_indices(drug = 'hydroxychloroquine' ):
    
    if drug=='hydroxychloroquine':
        events_dict={#'2-25-2020' :"Raoult posts a video about HCQ",
        '3-20-2020' :'3-20-2020: Trump Endorses HCQ. \n; Fauci Counters Trump Claims',
        '4-14-2020' :'4-14-2020: 35 States Impose  \n Restrictions on HCQ use',
        '6-21-2020' :'6-21-2020: FDA declares HCQ  \n is no longer under EUA, NIH stops HCQ trials',
        #'12-8-2020': 'Kory testifies in favor of Ivermectin in a Senate Committee',
        #'1-14-2021' : 'NIH: Not enough Data to say Ivermectin works',
        #'3-5-2021': 'FDA warns against Ivermectin use.',
        '7-4-2021':  '7-4-2021: 67% of Adults in  \n the US receive at least one dose of vaccine.'      
        }
    if drug=='chloroquine phosphate':
        events_dict={#'2-25-2020' :"Raoult posts a video about HCQ",
        '3-20-2020' :'3-20-2020: Trump Endorses HCQ. \n; Fauci Counters Trump Claims',
        '4-14-2020' :'4-14-2020: 35 States Impose  \n Restrictions on HCQ use',
        '6-21-2020' :'6-21-2020: FDA declares HCQ  \n is no longer under EUA, NIH stops HCQ trials',
        #'12-8-2020': 'Kory testifies in favor of Ivermectin in a Senate Committee',
        #'1-14-2021' : 'NIH: Not enough Data to say Ivermectin works',
        #'3-5-2021': 'FDA warns against Ivermectin use.',
        '7-4-2021':  '7-4-2021: 67% of Adults in  \n the US receive at least one dose of vaccine.'      
        }
    if drug=='ivermectin':
        events_dict={#'2-25-2020' :"Raoult posts a video about HCQ",
        '4-01-2020' : '1: Australian Study Suggests IVM Could be a Treatment for Covid-19',
        '12-8-2020': '2: Dr. Kory testifies in favor of IVM in a Senate Committee',
        '1-14-2021' : '3: NIH: Not enough data to say IVM works',
        '2-4-2021': '4: Merck Issues a warning against Ivermectin use',  
        '3-5-2021': '5: FDA warns against IVM use',
        '7-4-2021':  '6: 67\% of adults in the US receive at least one dose of vaccine',
        '7-15-2021': '7: Key studies that drew attention to IVM discredited',
        '8-26-2021': '8: CDC issues warning on IVM use'
        # '9-1-2021': '9-1-2021: Medical Establishment Calls for Stopping Ivermectin Use \n Joe Rogan discloses Ivermectin use'   
        }
        
        
    new_dict={}
    dates = [new_dict.update({x: get_week_dt(x)})  for x in events_dict.keys()]
        
    #dates = [new_dict.update({x: get_week(x.split('-'))})  for x in events_dict.keys()]
    new_events_dict={}
    for date in events_dict.keys():
        new_events_dict[str(new_dict[date])] = events_dict[date]
    return new_events_dict
    
   

