import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pa
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
import os

def format_xaxis(ax,  labels, end_time_index, tick_spacing):
    """
    Handles x axis labels precisely
    ax is the axes from ax, plt
    positions - list of integers that directly map to each xtick. Play with this to define which ticks to show.
    labels  -  a list to index indices from positions. 
    returns ax
    """
    
    positions = [ x for x in range(0, end_time_index,tick_spacing)]+[end_time_index]
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

def get_trend(df, time_index):
    """
    Generates an integer index from 0 to number of time periods
    Sorts the data in ascending time order
    returns updated data
    """
    df[time_index] = df[time_index].astype(int)
    c=list(df[time_index].unique())
    c.sort()
    trend_dict={}
    i=0
    for x in c:
        trend_dict[x] = i
        i=i+1
    df['trend']   = df[time_index].map(lambda x: trend_dict[x])
    df.sort_values('trend', inplace=True)
    return df




def path_creator(path):
    """
    Creates a path of any depth.
    Note the path is created relative to code folder
    Do not start path with '../' - the function will handle that
    Example: 'Model Free/Chicago'
    returns nothing
    """
    paths = path.split('/')
    print(paths)
    new_path ='../{}/'
    for p in paths:
        if not os.path.exists(new_path.format(p)):
            new_path = new_path.format(p)
            print(new_path)
            os.makedirs(new_path.format(p))
            new_path = new_path+'/{}/'
        else:
            new_path = new_path.format(p)
            new_path = new_path+'/{}/'

            

def regression_plots(temp, title,path,ix, rounding='float'):
    import matplotlib.pyplot as plt
    font = {'family' : 'Times New Roman',
                    'size'   : 8}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize =(3,5))
    ax.yaxis.labelpad = 5
    fig.subplots_adjust(bottom=0.1, right =1, left=0.1, top=0.7)
    labels=temp.Period.to_list()
    #ax.boxplot(temp['Mean'])
    import numpy as np
    err= np.array([np.abs(temp['Mean']-temp['95% lower']).to_list(),np.abs(temp['95% upper']-temp['Mean']).to_list()])#list( list(item) for item in zip(temp['95% lower'].to_list(),temp['95% upper'].to_list()) )
    #print(err)
    plt.errorbar(temp['Period'],temp['Mean'], yerr=err, linestyle='None', marker='s',color='black',capthick=0.5,elinewidth=0.5,capsize=3)
    ax.xaxis.set_ticklabels([ labels[x] for x in range(0,len(labels))],rotation=0) 
    plt.plot(temp['Period'],temp['Mean'],'s',markersize=0.2,color='black')
    #plt.plot(temp['Period'],temp['95% upper'],'^', color='black')
    #plt.plot(temp['Period'],temp['95% lower'],'v', color='black')
    #ax.fill_between(temp['Mean'], temp['Mean'], fitted, facecolor=(0.4, 0.4, 0.9, 0.2))
    #plt.fill_between(temp['Period'],temp['95% lower'],temp['95% upper'],
    #             color='gray', alpha=0.2)
    #plt.plot(temp['Period'],temp['95% lower'],'.', color='gray')
    ax.set_title('{}'.format(title))
    ax =format_yaxis(ax, rounding)
    #y_min, y_max = ax.get_ylim()
    y_min = temp['95% lower'].min()
    y_max = temp['95% upper'].max()
    
    increment = 0.05
    if y_min>np.abs(1):
        increment=0.5
    for ix,xy in enumerate(zip(temp['Period'].to_list(),temp['Mean'].to_list())):                                       # <--
        xy_1 = (xy[0], temp['95% lower'].to_list()[ix]-increment)
        ax.annotate('%s' % np.round(xy[1],2), xy=xy_1, textcoords='data')
    
    if y_min<=0 and y_max>=0:
        ax.hlines(y=0,  colors='gray',xmin=0, xmax=2, linestyles='--', lw=0.5)
    #ax.set_ylim([y_min-5, y_max+5])
    new_path = path_creator(path)
    #print(new_path)
    fig.savefig('../'+path+'/{}'.format(str(ix)+title.replace(' ','_') )+'.png',bbox_inches='tight',dpi=200)
    
            
def twoaxis_plot(df,metric,grp_var,time_index, title, xname,path,tick_spacing = 4, trend='trend', vlines = ['201924'], rounding='float'):
    """
    Dual Axis Plot
    Data is assumed to be stacked so one group is tacked on top of another.
    df : data set
    metric: measure to plot
    grp_var: grouping variable
    time_index: the variable with year month day/ yearweek
    title:  Caption for the plot
    xname: name of x axis
    path : path to store the file , format is  'Model Free/Chicago'
    tick_spacing : number of tick spaces say every 4th tick,
    trend : list of integers sequenced by time - works with get trends
    vlines : list of String value of the label where vline should be. Note Th efunction automatically will convert the labels from the time index as string
    returns nothing
    """
    
    font = {'family' : 'Times New Roman',
            'size'   : 12}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize =(8,8))
    ax.yaxis.labelpad = 5
    fig.subplots_adjust(bottom=0.1, right =1, left=0.1, top=0.7)
    labels=[]  
    axis_count =0 
    
    if trend not in df.columns:
        df = get_trend(df, time_index) 
    
    for grp in sorted(list(df[grp_var].unique()), reverse=True): 
        if axis_count>1:
            print("Hey we have only two axes!!!")
            break
        
        df1 = df[df[grp_var]== grp]

        labels= df1.time_index.to_list()
        grp = ' '.join([ g.lower().capitalize() for g in grp.split(' ')])
        if axis_count ==0:
            ax.plot(df1['trend'],df1[metric],lw =0.5, color='red' ,label = grp)
            ax.set_ylabel(grp)
            ax.yaxis.labelpad = 15
            ax.xaxis.labelpad = 15
            ax =format_yaxis(ax, rounding)
            
        if axis_count ==1:
            ax2=ax.twinx()
            ax2.plot(df1['trend'],df1[metric],lw =0.5, color='blue' ,label = grp,linestyle='-.',)
            ax2.set_ylabel(grp, color="blue",fontsize=14 , rotation=270)
            ax2.legend(loc='upper right', ncol=1)
            ax2.yaxis.labelpad = 15
            ax2 = format_yaxis(ax2, rounding)
        axis_count=axis_count+1
    ax.legend(loc='upper left', ncol=1)
    
    end_time_index = df1['trend'].max()
    
    labels = [str(x) for x in labels]
    
    
    ax = format_xaxis(ax,  labels, end_time_index, tick_spacing)  
    ax.axhline(0, color='r', linestyle='--', lw=0.5)
    for vline in vlines:
        vline_position = labels.index(vline)
        ax.axvline(vline_position, color='r', linestyle='--', lw=0.5)
    
    ax.set_title('{}'.format(title))
    ax.set_xlabel(xname)
    
    new_path = path_creator(path)
    fig.savefig('../'+path+'/{}'.format(title.replace(' ','_') )+'.png',bbox_inches='tight',dpi=100)
    
    
def multi_plot(df,metric,grp_var,time_index, title, xname,path,tick_spacing = 4, trend='trend',xlabel="", event_lines = {}, rounding='float', y_pos=200,location='center left', y_minmax=False):
    """
    Single Axis Plot
    Data is assumed to be stacked so one group is tacked on top of another.
    df : data set
    metric: measure to plot
    grp_var: grouping variable
    time_index: the variable with year month day/ yearweek
    title:  Caption for the plot
    xname: name of x axis
    path : path to store the file , format is  'Model Free/Chicago'
    tick_spacing : number of tick spaces say every 4th tick,
    trend : list of integers sequenced by time - works with get trends
    vline : String value of the label where vline should be. Note Th efunction automatically will convert the labels from the time index as string
    returns nothing
    """
    
    
    font = {'family' : 'Arial',
            'size'   : 10}
    plt.rc('font', **font)
    #plt.rc('axes', labelsize=28)
    fig, ax = plt.subplots(figsize=(10,5))
    
    
    colors_= ['black',"green",'darkred','brown','midnightblue','orange','forestgreen','sandybrown','midnightblue']
    colors = ['black','blue','red','orange','lightgray','yellow',"pink"]+sorted(['black','blue','red','green','gray','yellow'],reverse=True)
    linestyles=['solid','dotted','dashed','dashdot','dashed',"dashdot"] + ['solid','dotted','dashed','dashdot','solid']
    #font = {'family' : 'Times New Roman',
    #        'size'   : 18}
    #plt.rc('font', **font)
    
    
    
    
    #fig, ax = plt.subplots(figsize =(8,12))
    ax.yaxis.labelpad = 5
    fig.subplots_adjust(bottom=0.1, right =1, left=0.1, top=0.7)
    labels=[]  

    
    
    lws = [x/10 for x in range(8,15)]
    
    
    week_correction = df.groupby([xlabel, time_index], as_index=False).agg({"week":"nunique"})
    week_correction.rename(columns={"state":"ignore"}, inplace=True)
    print("Week Correction")
    
    for ix, grp in enumerate(sorted(list(df[grp_var].unique()), reverse=True)): 
        df1 = df[df[grp_var]== grp]
        
        if "ignore" in df1.columns:
            df1.drop("ignore", axis=1, inplace=True)
        df1 = pa.merge(df1, week_correction, on= [xlabel, time_index], how='right')
        df1.fillna(0, inplace=True)
        print(df1.columns)
        if trend not in df.columns:
            df1 = get_trend(df1, time_index) 
        #df1.sort_values(["ignore"], ascending=True, inplace=True)
        #
        #print(df1.head())
        #for x in df1.columns:
        #    print("*",x,"*")
        #print(df1[df1.columns[-2]].mean())
        labels= df1[xlabel].to_list()
        print(grp, len(labels))
        if ' ' in grp:
            grp = ' '.join([ g.lower().capitalize() for g in grp.split(' ')])
        if '_' in grp:
            grp = ' '.join([ g.lower().capitalize() for g in grp.split('_')])
        ax.plot(df1['trend'],df1[metric],lw =2, color=colors[ix] ,label = grp, linestyle=linestyles[ix])
        print(df1.trend)
        
        #ax.set_ylabel(grp)
        ax.yaxis.labelpad = 15
        ax.xaxis.labelpad = 15
        ax = format_yaxis(ax, rounding)
            
    
    #ax.set_ylim(bottom=0)
    #vlines= [int(vline)]
    #labels=['20200329']
    #labels =[int(x) for x in  labels]
    
    y_pos = int(df[metric].max()*0.75 )
    if rounding=="float":
        y_pos = df[metric].max()*0.5 
        
    #print(event_lines, labels)
    #vline_position = labels.index(vline)
    #ax.axvline(vline_position, color='r', linestyle='--', lw=0.5)
    
    
    if y_minmax!=False:
        ax.set_ylim(0, 5)
    
    
    
    for ix,vl in enumerate(list(event_lines)):
        #print(vl)
        vline_position = labels.index(vl)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=1)
        ax.annotate(event_lines[vl].split(":")[0], xy =(vline_position, y_pos), xytext =(vline_position -2.5 , y_pos),rotation=90 )
    
    #ax.annotate('local max', xy=('2020-03-22', 60), #xytext=(3, 1.5),
    #        arrowprops=dict(facecolor='black', shrink=0.05),
    #        )        
    #ax.legend(loc='upper left', ncol=1)
    
    font2 = {'family' : 'Arial',
            'size'   : 8}
    #fig.legend(loc=location, ncol=1, prop=font2,  borderaxespad= 4)
    fig.legend(loc='upper center', ncol=2, prop=font2, bbox_to_anchor=(0.5, -0.15))
    
    # ... (other code) ...
    
    end_time_index = df1['trend'].max()
    
    labels = [str(x) for x in labels]
    #print(labels)
    #vline_position = labels.index(vline)
    
    ax = format_xaxis(ax,  labels, end_time_index, tick_spacing)  
    
    #ax.axvline(vline_position, color='r', linestyle='--', lw=0.5)
    
    ax.set_title('{}'.format(title))
    ax.set_xlabel(xname)
    ax.set_ylabel('Indexed IHC Per Capita')
    #new_path = path_creator(path)
    # fig.savefig('../'+path+'/{}'.format(title.replace(' ','_') )+'.png',bbox_inches='tight',dpi=100)
    fig.savefig('{}'.format(title.replace(' ','_') )+'.png',bbox_inches='tight',dpi=300)
    fig.show()
    