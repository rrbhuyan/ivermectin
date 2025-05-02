import arviz as az
from matplotlib import pyplot as plt
import numpy as np
import pymc as pm
import pandas as pa
from isoweek import Week
np.random.seed(seed=42)
import os
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler, RobustScaler

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    if y_true.any()==0:
        mape= np.mean( np.abs( (y_true-y_pred) / (y_true)  ) )*100
    else:
        mape = np.mean( np.abs( (y_true-y_pred) / y_true  ) )*100
    
    return mape

def model_summary(trace):
    model_summary={}
    divergences =trace.sample_stats['diverging'].sum()
    #if trace.sample_stats['diverging'].any():
    #    divergences=1
    model_summary['divergent'] = divergences.item()
   
    accept = trace.sample_stats['acceptance_rate']
    model_summary['accept'] = accept.mean().item()
    
    rhat_max_list=[]
    rhat_min_list=[]
    rhat_maxs = az.rhat(trace).max()
    rhat_mins = az.rhat(trace).min()
    for x in list(rhat_maxs):
        rhat_max_list.append(rhat_maxs[x].item())
        rhat_min_list.append(rhat_mins[x].item())

    model_summary['rhat_max'] =max(rhat_max_list)
    model_summary['rhat_min'] =min(rhat_min_list)
    
    
    return model_summary



def get_att(model_preds, post_periods,y_all,y_train):
    df = pa.DataFrame(model_preds)
    post_period = df.loc[:,y_train.shape[0]: y_all.shape[0]]
    att = (sum(y_all[y_train.shape[0]:]) - post_period.sum(axis=1) ) / (y_all.shape[0] -y_train.shape[0]) # \Sum_{post} (actual - preds)/N_post
    
    bands = az.hdi(model_preds, hdi_prob=.95).tolist()
    bands= np.stack(bands, axis=1)
    
    percentage = ( ( sum(y_all[y_train.shape[0]:]) - post_period.sum(axis=1) ) / sum(y_all[y_train.shape[0]:]) ) * 100
    ci  = az.hdi(att.values,hdi_prob=.95).tolist()
    #att = np.mean(att)
    period_dict={}
    period_dict['percentage'] = np.mean(percentage)
    period_dict['percentage_sd'] = np.std(percentage)
    ci_per = ci  = az.hdi(percentage.values,hdi_prob=.95).tolist()
    period_dict['period'] ='overall'
    period_dict['att'] = np.mean(att)
    period_dict['att_sd'] = np.std(att)
    positive = ((att > 0).sum() / att.size) * 100
    negative = ((att < 0).sum() / att.size) * 100
    period_dict['positive_prob'] = positive
    period_dict['negative_prob'] = negative
    
    period_dict['lower'] = ci[0]
    period_dict['upper'] = ci[1]
    
    # period_dict['lower_per'] = ci_per[0]
    # period_dict['upper_per'] = ci_per[1]
    
    if ci[0] <= 0 and ci[1] >= 0:
        period_dict['significant'] = 1
    else:
        period_dict['significant'] = 0
    
    

    
    if post_periods:
        ix =0
        for period in post_periods:
            mpreds= df[period].values
            att = pa.Series(y_all[period]-mpreds )
            att_per = pa.Series((y_all[period]-mpreds) / mpreds ) * 100
            efctsz = pa.Series((y_all[period]-mpreds))
            att_efctsz =pa.Series(efctsz /efctsz.std()  ) * 100
            
            ci  = az.hdi(att.values,hdi_prob=.95).tolist()
            ci_per = az.hdi(att_per.values,hdi_prob=.95).tolist()
            ci_efctsz = az.hdi(att_efctsz.values,hdi_prob=.95).tolist()
            #period_dict['period'] = period
            period_dict['actual_'+str(ix)] = np.mean(y_all[period])
            period_dict['pred_'+str(ix)] = np.mean(mpreds)
            period_dict['att_'+str(ix)] = np.mean(att)
            period_dict['att_sd_'+str(ix)] = np.std(att)
            period_dict['lower_'+str(ix)] = ci[0]
            period_dict['upper_'+str(ix)] = ci[1]
            
            period_dict['att_efctsz_'+str(ix)] = att_efctsz.mean()
            period_dict['att_efctsz_sd_'+str(ix)] = att_efctsz.std()
            period_dict['lower_efctsz_'+str(ix)] = ci_efctsz[0]
            period_dict['upper_efctsz_'+str(ix)] = ci_efctsz[1]
            
            
            period_dict['att_per_'+str(ix)] = att_per.mean()
            period_dict['att_per_sd_'+str(ix)] = att_per.std()
            period_dict['lower_per_'+str(ix)] = ci_per[0]
            period_dict['upper_per_'+str(ix)] = ci_per[1]
            
            if ci[0]<=0 and ci[1]>=0:
                period_dict['significant_'+str(ix)] = 1
            else:
                period_dict['significant_'+str(ix)]=0
            ix=ix+1
          

    return period_dict


def post_mean(trace, var_name):
    return trace["posterior"][var_name].mean(dim=("chain", "draw"))


def format_xaxis(ax,  labels, end_time_index, tick_spacing):
    """
    Handles x axis labels precisely
    ax is the axes from ax, plt
    positions - list of integers that directly map to each xtick. Play with this to define which ticks to show.
    labels  -  a list to index indices from positions. 
    returns ax
    """
    print(labels, len(labels))
    positions = [ x for x in range(0, end_time_index,tick_spacing)]+[end_time_index]
    print(positions)
    ax.xaxis.set_major_locator(MaxNLocator(len(positions)))
    ax.xaxis.set_ticks(positions)
    ax.xaxis.set_ticklabels([ labels[x] for x in positions],rotation=90)
    return ax

def format_yaxis(ax, rounding='integer'):
    if rounding=='integer':
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
    else:
        ax.set_yticklabels(['{:,}'.format(pa.np.round(x,2)) for x in ax.get_yticks().tolist()])
    return ax





def main_graph(y_pred,y_all,bands,vlinepos, covid_cases, title="", y_label="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels =[],drug_name="hydroxychloroquine",treat_pos = 0):
    
    
    font = {'family' : 'Times New Roman',
            'size'   : 14}
    plt.rc('font', **font)
    #plt.rc('axes', labelsize=28)
    fig, ax = plt.subplots(figsize=(15,8))

    
    tick_spacing=4

    colors= ['black','blue',"red","green",'yellow','brown','aqua','darkgreen']
    
    colors_= ['black',"green",'purple','saddlebrown','midnightblue','orange','forestgreen','sandybrown','midnightblue']
    # colors_[2] - yellow -> purple
    # colors_[3] - brown -> purple

    #time = [Week(t // 100, t % 100).sunday() for t in time_labels]
    
    from important_dates import get_time_indices
    
    event_lines = get_time_indices(drug=drug_name)
    
    y_max= max(y_all)
    
    """
    if drug_name=='hydroxychloroquine' or drug_name=='chloroquine phosphate':
        y_max=70
    elif drug_name=='ivermectin':
        y_max=400
    """
    labels_ = time_labels
    
    #labels_= [str(x).split(" ")[0]  for x in  time]
    #print(labels_)
    
    trend_ = [x for x in range(len(time_labels))]
    
    time = trend_
    end_time_index = max(trend_)
    
    #indices =list(new_events_dict.keys()) ## [202009,202012,202025 ]
    
    #indices_ = [Week(int(t) // 100, int(t) % 100).sunday() for t in indices]
    #print(indices_)
    #vlinepos = [ time.index(ind)  for ind in indices_]
    #print(vlinepos)
    #print(time)
    #national = national[time >= pd.to_datetime("2019-01-01")]

    #time = pa.to_datetime(time.values)

    ax.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims', linestyle="-", marker='')
    plt.plot_date(time, y_pred, lw =1.5, color=colors[1], label = 'Predicted/Counterfactual HCQ Claims', linestyle="-", marker='')

    ax.fill_between(time,bands[0],bands[1],alpha=0.2, label="95% Credible Intervals")
    
    
    max_y = y_max
    y_pos = int(max_y*0.75 )
    
    
    
    
    # for ix,vl in enumerate(list(event_lines)): 
    #     vline_position = labels_.index(vl)
    #     print(vline_position)
    #     ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
    #     ax.annotate(event_lines[vl].split(":")[0], xy =(vline_position, y_pos), xytext =(vline_position -3 , y_pos),rotation=90 , fontsize=24)
    # if "placebo" not in suffix:
    #     treatment_label = "Treatment Date:\nDidier Raoult posts his video about HCQ"    
    #     ax.axvline(treat_pos, color='black', linestyle='--', label=treatment_label, lw=2)
    if "placebo" in suffix:
        # actual_treatment_ix = 59
        treatment_label = "Placebo Treatment Date"    
        ax.axvline(treat_pos, color='darkred', linestyle='--', label=treatment_label, lw=2)
        ax.annotate(treatment_label, xy =(treat_pos-2.5, int(max_y*0.6)), rotation=90 , fontsize=14)
        
        # treatment_label = "Actual Treatment Date"    
        # ax.axvline(actual_treatment_ix, color='black', linestyle='--', label=treatment_label, lw=2)
        # ax.annotate(treatment_label, xy =(57, y_pos), rotation=90 , fontsize=14)
        
    # ax.annotate("Treatment Date", xy =(treat_pos, int(y_pos/2) ), xytext =(treat_pos -3 , int(y_pos/2)),rotation=90 , fontsize=24)
    """
    for ix,vl in enumerate(list(event_lines)):
        print(vl)
        vline_position_ =labels_.index(vl)
        vline_position = time[labels_.index(vl)]
        #print(vline_position)
        ax.vlines(vline_position , ymin=np.min(y_all), ymax=np.max(y_all), color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate(event_lines[vl].split(":")[0], xy =(vline_position, y_pos), xytext =(vline_position_ - 2.25 , y_pos),rotation=90 )
    
    print(vl)
    """

    for ix,vl in enumerate(list(event_lines)): 
        vline_position = labels_.index(vl)
        print(vline_position)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate("Event " + event_lines[vl].split(":")[0], xy =(vline_position-2.5, y_pos), rotation=90 , fontsize=14)
    
    #for ix,val in enumerate(indices_):
    #    ax.vlines(indices_[ix], ymin=np.min(y_all), ymax=np.max(y_all), color=colors_[ix], linestyle='--', label=new_events_dict[indices[ix]], lw=3)
    #ax.vlines(indices[1], ymin=np.min(y_all), ymax=np.max(y_all), color='purple', linestyle='--', label="Trump makes the first statement about HCQ", lw=3)
    #ax.vlines(indices[2], ymin=np.min(y_all), ymax=np.max(y_all), color='blue', linestyle='--', label="FDA revokes EUA for HCQ", lw=3)
    #ax.set_title('National Level Total Prescription Claims for HCQ')
    #ax.grid()
    #fig.legend(loc="upper center", ncol=6, prop=font)


    #ax.xaxis.set_ticks(np.array(time)[::5])
    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20

    ax.tick_params(axis='x', rotation=90)
    
    ax.tick_params(axis='y')

    ax.yaxis.labelpad = 5
    # ax.set_ylim(-50, 610)
    # ax.yaxis.set_ticks([0, 50] + list(np.arange(100, 750, 100)))
    # ax.set_yticklabels(np.array([0, 50] + list(np.arange(100, 750, 100))).astype(str))
    ax.set_ylabel(y_label)
    
    ax.set_title(title, fontdict={"fontsize" : 18, "fontweight" : "bold", "fontfamily":"Times New Roman"})
    # ax.grid()

    #ax.xaxis.label.set_size(20)
    #ax.yaxis.label.set_size(20)

    ax =format_yaxis(ax, rounding= "integer")
    
    ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)  
    # ax.set_ylim(-50, 610)
    # ax.yaxis.set_ticks([0, 50] + list(np.arange(100, 750, 100)))
    # ax.set_yticklabels(np.array([0, 50] + list(np.arange(100, 750, 100))).astype(str))
    font2 = {'family' : 'Times New Roman',
           'size'   : 14}
    fig.legend(loc='upper center', ncol=4, prop=font2, bbox_to_anchor=(0.5, -0.05))
    
    #fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),prop=
    #       fancybox=True, shadow=True, ncol=5 )
    
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')
    fig.savefig('{}/{}'.format(opath,suffix )+'.png',bbox_inches='tight',dpi=300)
    
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Weekly Covid Cases (in Millions)', font=font)  # we already handled the x-label with ax1
    ax2.plot_date(time, covid_cases , color = "black", lw =1, linestyle='--',marker='', label="Covid-19 Weekly Cases")
    ax2.yaxis.labelpad = 5
    ax2.xaxis.set_ticks(np.array(time)[::5])
    ax2.set_xlabel('Week (Sunday)', font=font)
    ax2.xaxis.labelpad = 20
    ax2.tick_params(axis='y')
    ax.tick_params(axis='x', rotation=90)
    ax2.xaxis.set_ticks(np.array(time)[::4])
    fig.legends = []
    # fig.legend(loc="center left", ncol=1, prop=font2,  borderaxespad= 6)
    fig.legend(loc='upper center', ncol=3, prop=font2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig('{}/{}'.format(opath,suffix )+'_covid.png',bbox_inches='tight',dpi=300)

    #plt.savefig("../Output/Pictures/main_national_indexed_final.png", bbox_inches='tight',dpi=100)
    
    return
    



def graph(y_pred,y_all,bands,vlinepos, covid, title="", y_label="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels =[]):
    #bands= np.stack(bands, axis=1)      
    colors= ['black','blue']
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import pandas as pa
    #from matplotlib.dates import DateFormatter
    import os
    font = {'family' : 'Times New Roman',
            'size'   : 16}
    plt.rc('font', **font)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots(figsize =(12,12))
    #plt.xticks(rotation = 90)
    ax.yaxis.labelpad = 5
    fig.subplots_adjust(bottom=0.1, right =1, left=0.1, top=0.7)
    time = [Week(t // 100, t % 100).sunday() for t in time_labels]
    
    
    
    ax.plot_date(time,y_all,lw =0.5, color=colors[0] ,label = 'Actual', linestyle="-", marker='')
    
    ax.plot_date(time,y_pred,lw =0.5, color=colors[1],linestyle='-' ,label = 'Predicted', marker='')
    
    v1 = bands[0]
    v2 = bands[1]
    tick_spacing =4
    end_time_index= len(y_all)-1
    ax.fill_between(time,v1,v2,alpha=0.2)
    # ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)   
    ax.xaxis.set_ticks(np.array(time)[::4])
    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20
    ax.set_ylabel(y_label, font=font)
    ax.tick_params(axis='x', rotation=90)
    #ax.set_ylim([25, 50])
    if y_all.mean() > 1:
        ax = format_yaxis(ax)

    # ax.legend(loc='upper left', ncol=1)
    #ax.grid()
    #ax.axvline(53, color='r', linestyle='--', lw=2)
    #vlinepos= len(y_all)/2
    # ax.axvline(vlinepos, color='r', linestyle='--', lw=2)
    # if len(vlinepos) == 1:
    #     ax.vlines(time[vlinepos[0]], ymin=np.min(y_all), ymax=np.max(y_all), color='r', linestyle='--', label="Campaign Start", lw=2)
    ax.vlines(time[vlinepos[0]], ymin=np.min(y_all), ymax=np.max(y_all), color='r', linestyle='--', label="Raoult posts a video about HCQ", lw=3)
    ax.vlines(time[vlinepos[0]], ymin=np.min(y_all), ymax=np.max(y_all), color='purple', linestyle='--', label="Trump makes the first statement about HCQ", lw=3)
    if len(vlinepos) > 0:
        ax.vlines(time[vlinepos[0]], ymin=np.min(y_all), ymax=np.max(y_all), color='blue', linestyle='--', label="HCQ Added to the FDA Short List", lw=3)
        ax.vlines(time[vlinepos[1]], ymin=np.min(y_all), ymax=np.max(y_all), color='g', linestyle='--', label="Short List Resolved", lw=3)
    # ax.axvline(vlinepos+20, color='r', linestyle='--', lw=2)
    ax.set_title('{}'.format(title))
    # ax.set_xlabel(xname)
    # ax.set_ylabel("Claim Count")
    
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Weekly Covid Cases')  # we already handled the x-label with ax1
    # ax2.plot(range(len(y_all) - len(covid), len(y_all)), covid.new_cases , color = "green", lw =0.7, linestyle='-', label="Covid-19 Weekly Cases")
    ax.legend(loc='upper left', ncol=1)
    # ax2.tick_params(axis='y')

    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    
    
    fig.savefig('{}/{}'.format(opath,suffix )+'.png',bbox_inches='tight',dpi=100)
    
    df = pa.DataFrame(data={"time": time, "actual" : y_all, "predicted" : y_pred, "lower" : v1, "upper" : v2})
    df.to_csv('{}/{}'.format(opath,suffix ) + '.csv', index=False)
    

    
    
def mini_summarize(trace,model, model_preds,  model_type,prior_type, y_all, y_train,estimator,selected_init_method,cv_mape,fname):
    y_pred =model_preds.mean(axis=0)
    print('model preds shape:', model_preds.shape)
    bands = az.hdi(model_preds, hdi_prob=.95).tolist()#[az.hdi(model_preds, hdi_prob=.95)[:,0], az.hdi(model_preds, hdi_prob=.95)[:,1]]
    #print(model_preds.shape, model_preds)
    #print(bands)
    bands= np.stack(bands, axis=1)   
    
    summary={}
    if estimator =='bayes':
        summary = model_summary(trace)
        summary['loo'] =-2*az.loo(trace,model)[0]
        summary['waic'] =-2*az.waic(trace,model)[0]
    
    train_pred = y_pred[:y_train.shape[0]]
    
    summary['mape_train'] =  mean_absolute_percentage_error( y_train, train_pred)
    summary['mape_test'] =  mean_absolute_percentage_error(y_all[y_train.shape[0]:], y_pred[y_train.shape[0]:])
    summary['fname'] = fname
    
    
    summary["overall_change_%"] = ((y_all[y_train.shape[0]:]-y_pred[y_train.shape[0]:]) / y_pred[y_train.shape[0]:]).mean() * 100
    summary["overall_lower_change_%"] = ((y_all[y_train.shape[0]:]-bands[1][y_train.shape[0]:]) / bands[1][y_train.shape[0]:]).mean() * 100
    summary["overall_upper_change_%"] = ((y_all[y_train.shape[0]:]-bands[0][y_train.shape[0]:]) / bands[0][y_train.shape[0]:]).mean() * 100
    """
    summary["short_run_change_%"] = ((y_all[104:110]-y_pred[104:110]) / y_pred[104:110]).mean() * 100
    summary["short_run_lower_change_%"] = ((y_all[104:110]-bands[1][104:110]) / bands[1][104:110]).mean() * 100
    summary["short_run_upper_change_%"] = ((y_all[104:110]-bands[0][104:110]) / bands[0][104:110]).mean() * 100
    
    summary["medium_run_change_%"] = ((y_all[110:]-y_pred[110:]) / y_pred[110:]).mean() * 100
    summary["medium_run_lower_change_%"] = ((y_all[110:]-bands[1][110:]) / bands[1][110:]).mean() * 100
    summary["medium_run_upper_change_%"] = ((y_all[110:]-bands[0][110:]) / bands[0][110:]).mean() * 100
    """
    summary['cv_mape'] = cv_mape
    summary['model_type'] = model_type
    summary['prior_type'] = prior_type
    summary['init_method'] = selected_init_method
    post_periods=[x for x in range(y_train.shape[0],y_all.shape[0])]
    att = get_att(model_preds,post_periods ,y_all,y_train)
    summary.update(att)
    return post_periods,y_pred,bands,summary
    
    
   
    
def summarize(trace,model, model_preds, fname, model_type,prior_type, y_all, y_train,y_scaler,suffix,xname, covid_cases=None, estimator='bayes',selected_init_method='',cv_mape=10000, opath='../Model_Based/',time_labels =[], vlinepos = [], title="", y_label="Total Claims", national=False, transform = True,drug_name="hydroxychloroquine"):
    #print(az.summary(trace))
    #print(az.plot_trace(trace))
    # y_pred =y_scaler.inverse_transform(y_pred.astype('float64').flatten())
    #
    #
    if not transform:
        model_preds = y_scaler.inverse_transform(model_preds)
        y_all = y_scaler.inverse_transform(y_all.astype('float64').reshape(-1,1)).flatten()
        y_train =y_scaler.inverse_transform(y_train.astype('float64').reshape(-1,1)).flatten()
    
        
    post_periods, y_pred, bands,summary = mini_summarize(trace,model, model_preds,  model_type,prior_type, y_all, y_train,estimator,selected_init_method,cv_mape,fname)
    
    # if len(vlinepos) == 0:
    #     vlinepos=y_train.shape[0]
    
    #graph(y_pred,y_all,bands,vlinepos, title=fname+' Mape Train:{}'.format(np.round(summary['mape_train'],2)), xname=xname, opath='../Model_Based/',suffix=suffix)
   
   
    
    
    print("y_all","\n",y_all.shape)
    print("model_preds","\n",model_preds.shape)
    
    """
    percentages = np.divide( y_all-y_pred, y_pred )*100#((y_all-y_pred)/y_pred)*100
    #print(len(percentages), percentages)
    lower = np.divide( (y_all-bands[1]),bands[1] )*100
    #print(len(upper), upper)
    upper= np.divide( (y_all-bands[0]),bands[0] )*100 
    #print(len(lower), lower)
    bands_per = [ np.array(lower),np.array(upper)]
    efctsz = [0]*60+[ summary["att_efctsz_{}".format(str(ix))] for ix, x in enumerate(post_periods)]  # for x in range ]#np.divide( y_all-y_pred, y_pred )*100#((y_all-y_pred)/y_pred)*100
    lower_efctsz = [0]*60+[summary['lower_efctsz_'+str(ix)] for ix, x in enumerate(post_periods)]
    upper_efctsz = [0]*60+[summary['upper_efctsz_'+str(ix)] for ix, x in enumerate(post_periods)]
        
    bands_efctsz = [ np.array(lower_efctsz),np.array(upper_efctsz)]    
    """    
    if 'all'  in fname or 'trump' or 'DAKOTA' in fname:
        main_graph(y_pred,y_all,bands,vlinepos, covid_cases,title=title, y_label=y_label, xname=xname, opath=opath,suffix=suffix,time_labels=time_labels,drug_name=drug_name, treat_pos=len(y_train)-1)
        #graph_per(efctsz,bands_efctsz,vlinepos, title=fname,y_label="Effect Size (in Std. Units)", xname=xname, opath=opath,suffix=suffix,time_labels =time_labels )
    
    else:
        pass
        #graph(y_pred, y_all, bands, vlinepos, covid_cases, title=title, y_label=y_label, xname=xname, opath=opath,suffix=suffix,time_labels=time_labels,)
        #graph_per(percentages,bands_per,vlinepos, title=fname, xname=xname, opath=opath,suffix=suffix,time_labels =time_labels )
        
    """
    lower = np.divide( (y_all-bands[1]),bands[1] )*100
    #print(len(upper), upper)
    upper= np.divide( (y_all-bands[0]),bands[0] )*100 
    
    bands_effsize= []
    
    att = (y_all-y_pred)/ (model_preds.std(axis=0))
    lower_eff = np.divide( (y_all-bands[1]),(model_preds.std(axis=0)) )*100
    #print(len(upper), upper)
    upper_eff = np.divide( (y_all-bands[0]),(model_preds.std(axis=0)) )*100 
    bands_effsize= [np.array(lower_eff),np.array(upper_eff)]
    #graph_per(att ,bands_effsize,vlinepos, title=fname, xname=xname, opath=opath,suffix=suffix,time_labels =time_labels )
    
    if covid_cases is not None and national==False:
        diff_per_case = np.divide(y_all - y_pred, covid_cases)

        upper = np.divide(y_all - bands[1], covid_cases )

        lower = np.divide(y_all - bands[0], covid_cases) 

        bands_covid = [ np.array(lower),np.array(upper)]

        graph_covid(diff_per_case, bands_covid, vlinepos, title=fname, xname=xname, opath=opath,suffix=suffix,time_labels =time_labels, treat_pos =len(y_train) )
    """
    return summary





def graph_per(y_all,bands,vlinepos, title='Default', xname='Months', y_label="Percent (%)", opath='./Model_Based/',suffix='', time_labels =[]):
    #bands= np.stack(bands, axis=1)      
    colors= ['black','blue']
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import pandas as pa
    #from matplotlib.dates import DateFormatter
    import os
    font = {'family' : 'Times New Roman',
            'size'   : 16}
    plt.rc('font', **font)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots(figsize =(12,12))
    #plt.xticks(rotation = 90)
    ax.yaxis.labelpad = 5
    fig.subplots_adjust(bottom=0.1, right =1, left=0.1, top=0.7)
    
    time = [Week(t // 100, t % 100).sunday() for t in time_labels]
    
    ax.plot_date(time, y_all,lw =0.5, color=colors[0], label = 'Percentage', linestyle="-", marker='')
    
    #ax.plot(range(len(y_all)),y_pred,lw =0.5, color=colors[1],linestyle='-' ,label = 'Predicted')
    
    v1 = bands[0]
    v2 = bands[1]
    tick_spacing = 4
    end_time_index= len(y_all)-1
    ax.fill_between(time,v1,v2,alpha=0.2)
    ax.xaxis.set_ticks(np.array(time)[::4])
    # ax.yaxis.set_ticks(np.arange(0, 700, 100))
    # ax.set_yticklabels(np.arange(0, 700, 100).astype(str))
    #ax.set_ylim(-100, 100)
    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20
    ax.set_ylabel(y_label, font=font)
    ax.tick_params(axis='x', rotation=90)
    
    # ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)    
    #ax.set_ylim([25, 50])    
    
    ax.vlines(time[vlinepos[0]], ymin=np.min(y_all), ymax=np.max(y_all), color='r', linestyle='--', label="Raoult posts a video about HCQ", lw=2)
    ax.vlines(time[vlinepos[1]], ymin=np.min(y_all), ymax=np.max(y_all), color='purple', linestyle='--', label="Trump makes the first statement about HCQ", lw=2)
    # ax.vlines(time[124], ymin=np.min(y_all), ymax=np.max(y_all), color='r', linestyle='--', label="Texas Policy Expiration Date", lw=2)
    # ax.vlines(time[144], ymin=np.min(y_all), ymax=np.max(y_all), color='purple', linestyle='--', label="", lw=2)
    if len(vlinepos) > 0:
        ax.vlines(time[vlinepos[0]], ymin=np.min(y_all), ymax=np.max(y_all), color='blue', linestyle='--', label="HCQ Added to the FDA Short List", lw=2)
        ax.vlines(time[vlinepos[1]], ymin=np.min(y_all), ymax=np.max(y_all), color='g', linestyle='--', label="Shortages Resolved", lw=2)
    
    
    # ax =format_yaxis(ax, rounding= "float")
    
    ax.legend(loc='upper left', ncol=1)
  
    # ax.axvline(vlinepos, color='r', linestyle='--', lw=0.5)
    ax.axhline(y=0, color='blue', linestyle='-')
    # ax.axhline(y=75, color='blue', linestyle='-')
    
    time_labels=[str(x) for x in time_labels]
    suffix ="PER_"+suffix  
    ax.set_title('{}'.format(title))
    # ax.set_xlabel(xname)
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')
    fig.savefig('{}/{}'.format(opath,suffix )+'per_.png',bbox_inches='tight',dpi=100)
    df = pa.DataFrame(data={"time": time, "percentage" : y_all, "lower" : v1, "upper" : v2})
    df.to_csv('{}/{}'.format(opath,suffix )+'.csv', index=False)
    

def graph_covid(diff_per_case, bands_covid, vlinepos, title='Default', xname='Months', y_label="Prescription Claims per Covid Case", opath='./Model_Based/',suffix='', time_labels =[]):
    #bands= np.stack(bands, axis=1)      
    colors= ['black','blue']
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import pandas as pa
    #from matplotlib.dates import DateFormatter
    import os
    font = {'family' : 'Times New Roman',
            'size'   : 16}
    plt.rc('font', **font)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots(figsize =(12,12))
    #plt.xticks(rotation = 90)
    ax.yaxis.labelpad = 5
    fig.subplots_adjust(bottom=0.1, right =1, left=0.1, top=0.7)
    
    time = [Week(t // 100, t % 100).sunday() for t in time_labels][110:]
    
    ax.plot_date(time, diff_per_case[110:], lw =0.5, color=colors[0] ,label = 'Actual', linestyle="-", marker='')
    # ax.plot_date(time, claims_per_case_pred[110:], lw =0.5, color=colors[1] ,label = 'Predicted', linestyle="-", marker='')
    
    #ax.plot(range(len(y_all)),y_pred,lw =0.5, color=colors[1],linestyle='-' ,label = 'Predicted')
    
    v1 = bands_covid[0][110:]
    v2 = bands_covid[1][110:]
    tick_spacing = 4
    end_time_index= len(time)-1
    ax.fill_between(time,v1,v2,alpha=0.2)
    ax.xaxis.set_ticks(np.array(time)[::2])
    # ax.yaxis.set_ticks(np.arange(0, 700, 100))
    # ax.set_yticklabels(np.arange(0, 700, 100).astype(str))
    # ax.set_ylim(-50, 610)
    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20
    ax.set_ylabel(y_label, font=font)
    ax.tick_params(axis='x', rotation=90)
    
    # ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)    
    #ax.set_ylim([25, 50])    
    
    # ax.vlines(time[104], ymin=np.min(y_all), ymax=np.max(y_all), color='r', linestyle='--', label="Raoult posts a video about HCQ", lw=2)
    # ax.vlines(time[0], ymin=0, ymax=np.max(claims_per_case), color='purple', linestyle='--', label="Trump makes the first statement about HCQ", lw=2)
    # ax.vlines(time[124], ymin=np.min(y_all), ymax=np.max(y_all), color='r', linestyle='--', label="Texas Policy Expiration Date", lw=2)
    # ax.vlines(time[144], ymin=np.min(y_all), ymax=np.max(y_all), color='purple', linestyle='--', label="", lw=2)
    #if len(vlinepos) > 0:
    #    # ax.vlines(time[vlinepos[0] - 110], ymin=0, ymax=np.max(claims_per_case), color='blue', linestyle='--', label="HCQ Added to the FDA Short List", lw=2)
    #    ax.vlines(time[vlinepos[1] - 110], ymin=0, ymax=np.max(diff_per_case), color='g', linestyle='--', label="Short List Resolved", lw=2)
    
    
    # ax =format_yaxis(ax, rounding= "float")
    
    ax.legend(loc='upper left', ncol=1)
  
    # ax.axvline(vlinepos, color='r', linestyle='--', lw=0.5)
    ax.axhline(y=0, color='blue', linestyle='-')
    # ax.axhline(y=75, color='blue', linestyle='-')
    
    time_labels=[str(x) for x in time_labels]
    suffix ="Covid_"+suffix  
    ax.set_title('{}'.format(title))
    # ax.set_xlabel(xname)
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')
    fig.savefig('{}/{}'.format(opath,suffix )+'.png',bbox_inches='tight',dpi=100)
