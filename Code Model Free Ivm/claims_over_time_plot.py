from matplotlib import pyplot as plt
import numpy as np
import pandas as pa
from isoweek import Week
np.random.seed(seed=42)
import os
from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter, StrMethodFormatter


def format_xaxis(ax,  labels, end_time_index, tick_spacing):
    """
    Handles x axis labels precisely
    ax is the axes from ax, plt
    positions - list of integers that directly map to each xtick. Play with this to define which ticks to show.
    labels  -  a list to index indices from positions. 
    returns ax
    """
    # print(labels, len(labels))
    positions = [ x for x in range(0, end_time_index,tick_spacing)]+[end_time_index]
    # print(positions)
    ax.xaxis.set_major_locator(MaxNLocator(len(positions)))
    ax.xaxis.set_ticks(positions)
    ax.xaxis.set_ticklabels([ labels[x] for x in positions],rotation=90)
    return ax

def format_yaxis(ax, rounding='integer'):
    if rounding=='integer':
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
        # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,}'))
        # ax.ticklabel_format(style='plain')
    else:
        ax.set_yticklabels(['{:,}'.format(pa.np.round(x,2)) for x in ax.get_yticks().tolist()])
    return ax





def main_graph(y_all, vlinepos, covid_cases, title="", y_label="", covid_label="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels=[],drug_name="hydroxychloroquine", treat_pos=0):
    
    
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
    
    from important_dates import get_time_indices
    
    event_lines = get_time_indices(drug=drug_name)
    
    y_max = max(y_all[33:])
    y_min = min(y_all[33:])
    
    
    """
    if drug_name=='hydroxychloroquine' or drug_name=='chloroquine phosphate':
        y_max=70
    elif drug_name=='ivermectin':
        y_max=400
    """
    labels_ = time_labels
    
    trend_ = [x for x in range(len(time_labels))]
    
    time = trend_
    end_time_index = max(trend_)


    #time = pa.to_datetime(time.values)
    if drug_name == "hydroxychlroquine":
        ax.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims', linestyle="-", marker='')
    
    elif drug_name == "ivermectin":
        ax.plot_date(time[33:], y_all[33:], lw =1.5, color=colors[2], label = 'Actual IVM Claims', linestyle="-", marker='')
    
    # y_max = 1205
    # y_min = 430
    
    y_pos = y_max * 0.75 + y_min * 0.25
    print("y pos", y_pos)
    
    
    
    for ix,vl in enumerate(list(event_lines)): 
        vline_position = labels_.index(vl)
        print(vline_position)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate("Event " + event_lines[vl].split(":")[0], xy =(vline_position-2, y_pos), rotation=90 , fontsize=14)

    if drug_name == "hydroxychloroquine":
        if "placebo" not in suffix:
            treatment_label = "Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)
        else:
            actual_treatment_ix = 59
            treatment_label = "Placebo Treatment Date"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='darkred', linestyle='--', label=treatment_label, lw=2)

            treatment_label = "Actual Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(actual_treatment_ix, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)

    #ax.xaxis.set_ticks(np.array(time)[::5])
    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20

    ax.tick_params(axis='x', rotation=90)
    
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.labelpad = 5
    # ax.yaxis.set_ticks([0, 50] + list(np.arange(100, 750, 100)))
    # ax.set_yticklabels(np.array([0, 50] + list(np.arange(100, 750, 100))).astype(str))
    ax.set_ylabel(y_label, font=font)
    
    ax.set_title(title, fontdict={"fontsize" : 18, "fontweight" : "bold", "fontfamily":"Times New Roman"})
    # ax.grid()
    
    ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)
    if y_all.mean() > 1:
        ax = format_yaxis(ax)
    # ax.set_ylim(-50, 610)
    # ax.yaxis.set_ticks(np.arange(min(y_all), max(y_all), 10))
    # ax.set_yticklabels(np.array([0, 50] + list(np.arange(100, 750, 100))).astype(str))
    font2 = {'family' : 'Times New Roman',
           'size'   : 14}
    fig.legend(loc='upper center', ncol=4, prop=font2, bbox_to_anchor=(0.5, -0.05))
    
    #fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),prop=
    #       fancybox=True, shadow=True, ncol=5 )
    
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')
    
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(covid_label, font=font)  # we already handled the x-label with ax1
    ax2.plot_date(time[33:], covid_cases[33:], color = colors[1], lw =1, linestyle='--',marker='', label="Weekly Covid Cases per 1,000,000 People")
    ax2.yaxis.labelpad = 5
    if covid_cases.mean() > 1:
        ax2 = format_yaxis(ax2)
    ax2.xaxis.set_ticks(np.array(time[33:]))
    ax2.set_xlabel('Week (Sunday)', font=font)
    ax2.xaxis.labelpad = 20
    ax2.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', rotation=90, labelsize=14)
    ax2.xaxis.set_ticks(np.array(time[33:]))
    fig.legends = []
    # fig.legend(loc="center left", ncol=1, prop=font2,  borderaxespad= 6)
    # ax.set_ylim(430, 1205)
    # ax2.set_ylim(430, 1205)
    fig.legend(loc='upper center', ncol=3, prop=font2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig("graphs/" + title.strip() + ".png",bbox_inches='tight',dpi=300)

    #plt.savefig("../Output/Pictures/main_national_indexed_final.png", bbox_inches='tight',dpi=100)
    
    return
from matplotlib.ticker import MaxNLocator, FuncFormatter, FormatStrFormatter, StrMethodFormatter 

def format_xaxis(ax,  labels, end_time_index, tick_spacing):
    """
    Handles x axis labels precisely
    ax is the axes from ax, plt
    positions - list of integers that directly map to each xtick. Play with this to define which ticks to show.
    labels  -  a list to index indices from positions. 
    returns ax
    """
    # print(labels, len(labels))
    positions = [ x for x in range(0, end_time_index,tick_spacing)]+[end_time_index]
    print(positions)
    ax.xaxis.set_major_locator(MaxNLocator(len(positions)))
    ax.xaxis.set_ticks(positions)
    ax.xaxis.set_ticklabels([ labels[x] for x in positions],rotation=90)
    return ax

def format_yaxis(ax, rounding='integer'):
    if rounding=='integer':
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
        # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,}'))
        # ax.ticklabel_format(style='plain')
    else:
        ax.set_yticklabels(['{:,}'.format(pa.np.round(x,2)) for x in ax.get_yticks().tolist()])
    return ax

def claims_by_trump_support(y_all_red, y_all_blue, vlinepos, title="", y_label_red="", y_label_blue="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels=[],drug_name="ivermectin", treat_pos=0):
    
    
    font = {'family' : 'Times New Roman',
            'size'   : 14}
    
    plt.rc('font', **font)
    #plt.rc('axes', labelsize=28)
    fig, ax = plt.subplots(figsize=(15,8))

    
    tick_spacing=4

    colors= ['black','blue',"red","green",'yellow','brown','aqua','darkgreen']
    
    colors_= ['black',"green",'purple','saddlebrown','midnightblue','orange','forestgreen','firebrick','midnightblue'] # sandybrown -> firebrick
    # colors_[2] - yellow -> purple
    # colors_[3] - brown -> purple
    
    from important_dates import get_time_indices
    
    event_lines = get_time_indices(drug=drug_name)
    
    y_max = max(y_all_red[33:])
    y_min = min(y_all_red[33:])
    
    labels_ = time_labels
    
    trend_ = [x for x in range(len(time_labels))]
    
    time = trend_
    end_time_index = max(trend_)

    if drug_name == "hydroxychlroquine":
        ax.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims (Red States)', linestyle="-", marker='')
    
    elif drug_name == "ivermectin":
        ax.plot_date(time[33:], y_all_red[33:], lw =1.5, color=colors[2], label = 'Actual IVM Claims (Red States)', linestyle="-", marker='')

    y_pos = y_max * 0.75 + y_min * 0.25
    # print("y pos", y_pos)
    
    for ix,vl in enumerate(list(event_lines)): 
        vline_position = labels_.index(vl)
        # print(vline_position)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate("Event " + event_lines[vl].split(":")[0], xy =(vline_position-1.5, y_pos), rotation=90 , fontsize=10)

    if drug_name == "hydroxychloroquine":
        if "placebo" not in suffix:
            treatment_label = "Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)
        else:
            actual_treatment_ix = 59
            treatment_label = "Placebo Treatment Date"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='darkred', linestyle='--', label=treatment_label, lw=2)

            treatment_label = "Actual Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(actual_treatment_ix, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)


    #ax.xaxis.set_ticks(np.array(time)[::5])
    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20

    ax.tick_params(axis='x', rotation=90, labelsize=14)
    
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.labelpad = 5
    # ax.yaxis.set_ticks([0, 50] + list(np.arange(100, 750, 100)))
    # ax.set_yticklabels(np.array([0, 50] + list(np.arange(100, 750, 100))).astype(str))
    ax.set_ylabel(y_label_red, font=font)
    
    ax.set_title(title, fontdict={"fontsize" : 18, "fontweight" : "bold", "fontfamily":"Times New Roman"})
    # ax.grid()
    
    ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)
    if y_all_red.mean() > 1:
        ax = format_yaxis(ax)

    font2 = {'family' : 'Times New Roman',
           'size'   : 14}
    fig.legend(loc='upper center', ncol=4, prop=font2, bbox_to_anchor=(0.5, -0.05))
    
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))

    print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')
    
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(y_label_blue, font=font)  # we already handled the x-label with ax1

    if drug_name == "hydroxychlroquine":
        ax2.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims (Red States)', linestyle="-", marker='')
    
    elif drug_name == "ivermectin":
        ax2.plot_date(time[33:], y_all_blue[33:], lw =1.5, color=colors[1], label = 'Actual IVM Claims (Blue States)', linestyle="-", marker='')

    ax2.yaxis.labelpad = 5
    if y_all_blue.mean() > 1:
        ax2 = format_yaxis(ax2)
    ax2.xaxis.set_ticks(np.array(time[33:]))
    ax2.set_xlabel('Week (Sunday)', font=font)
    ax2.xaxis.labelpad = 20
    ax2.tick_params(axis='y', labelsize=14)
    ax2.xaxis.set_ticks(np.array(time[33:]))
    fig.legends = []

    fig.legend(loc='upper center', ncol=3, prop=font2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig("graphs/" + title.strip() + ".png",bbox_inches='tight',dpi=300)
    
    return

def covid_by_trump_support(vlinepos, covid_cases_red, covid_cases_blue, title="", covid_label_red="", covid_label_blue="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels=[],drug_name="ivermectin", treat_pos=0):
    
    font = {'family' : 'Times New Roman',
            'size'   : 14}
    plt.rc('font', **font)
    #plt.rc('axes', labelsize=28)
    fig, ax = plt.subplots(figsize=(15,8))

    
    tick_spacing=4

    colors= ['black','blue',"red","green",'yellow','brown','aqua','darkgreen']
    
    colors_= ['black',"green",'purple','saddlebrown','midnightblue','orange','forestgreen','firebrick','midnightblue'] # sandybrown -> firebrick
    # colors_[2] - yellow -> purple
    # colors_[3] - brown -> purple
    
    from important_dates import get_time_indices
    
    event_lines = get_time_indices(drug=drug_name)
    
    y_max = max(covid_cases_red)
    y_min = 0
    
    
    """
    if drug_name=='hydroxychloroquine' or drug_name=='chloroquine phosphate':
        y_max=70
    elif drug_name=='ivermectin':
        y_max=400
    """
    labels_ = time_labels
    
    trend_ = [x for x in range(len(time_labels))]
    
    time = trend_
    end_time_index = max(trend_)


    #time = pa.to_datetime(time.values)
    if drug_name == "hydroxychlroquine":
        ax.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims', linestyle="-", marker='')
    
    elif drug_name == "ivermectin":
        covid_cases_red = [0] * (len(time[33:]) - len(covid_cases_red)) + list(covid_cases_red)
        # ax.plot_date(time[-covid_cases_red.shape[0]:], covid_cases_red, lw =1, color=colors[2], label = 'Weekly Covid Cases (Red States)', linestyle="--", marker='')
        ax.plot_date(time[33:], covid_cases_red, lw =1, color=colors[2], label = 'Weekly Covid Cases (Red States)', linestyle="--", marker='')
    
    # y_max = 1205
    # y_min = 430
    
    y_pos = y_max * 0.75 + y_min * 0.25
    # print("y pos", y_pos)
    
    
    
    for ix,vl in enumerate(list(event_lines)): 
        vline_position = labels_.index(vl)
        # print(vline_position)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate("Event " + event_lines[vl].split(":")[0], xy =(vline_position-1.5, y_pos), rotation=90 , fontsize=14)

    if drug_name == "hydroxychloroquine":
        if "placebo" not in suffix:
            treatment_label = "Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)
        else:
            actual_treatment_ix = 59
            treatment_label = "Placebo Treatment Date"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='darkred', linestyle='--', label=treatment_label, lw=2)

            treatment_label = "Actual Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(actual_treatment_ix, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)

    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20

    ax.tick_params(axis='x', rotation=90, labelsize=14)
    
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.labelpad = 5

    ax.set_ylabel(covid_label_red, font=font)
    
    ax.set_title(title, fontdict={"fontsize" : 18, "fontweight" : "bold", "fontfamily":"Times New Roman"})
    # ax.grid()
    
    ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)
    if np.mean(covid_cases_red) > 1:
        ax = format_yaxis(ax)

    font2 = {'family' : 'Times New Roman',
           'size'   : 14}
    fig.legend(loc='upper center', ncol=4, prop=font2, bbox_to_anchor=(0.5, -0.05))
    
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    # print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')
    
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(covid_label_blue, font=font)  # we already handled the x-label with ax1

    covid_cases_blue = [0] * (len(time[33:]) - len(covid_cases_blue)) + list(covid_cases_blue)
    ax2.plot_date(time[33:], covid_cases_blue, lw =1, color=colors[1], label = 'Weekly Covid Cases (Blue States)', linestyle="--", marker='')
    ax2.yaxis.labelpad = 5
    if np.mean(covid_cases_blue) > 1:
        ax2 = format_yaxis(ax2)
    ax2.xaxis.set_ticks(np.array(time[33:]))
    ax2.set_xlabel('Week (Sunday)', font=font)
    ax2.xaxis.labelpad = 20
    ax2.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', rotation=90, labelsize=14)
    ax2.xaxis.set_ticks(np.array(time[33:]))
    fig.legends = []
    plt.tight_layout()
    fig.legend(loc='upper center', ncol=3, prop=font2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig("graphs/" + title.strip() +"_covid.png",bbox_inches='tight',dpi=300)
    
    return
 
def claims_tx_ca(y_all_red, y_all_blue, vlinepos, title="", y_label_red="", y_label_blue="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels=[],drug_name="ivermectin", treat_pos=0):
    
    
    font = {'family' : 'Times New Roman',
            'size'   : 14}
    
    plt.rc('font', **font)
    #plt.rc('axes', labelsize=28)
    fig, ax = plt.subplots(figsize=(15,8))

    
    tick_spacing=4

    colors= ['black','blue',"red","green",'yellow','brown','aqua','darkgreen']
    
    colors_= ['black',"green",'purple','saddlebrown','midnightblue','orange','forestgreen','firebrick','midnightblue'] # sandybrown -> firebrick
    # colors_[2] - yellow -> purple
    # colors_[3] - brown -> purple
    
    from important_dates import get_time_indices
    
    event_lines = get_time_indices(drug=drug_name)
    
    y_max = max(y_all_red[33:])
    y_min = min(y_all_red[33:])
    
    labels_ = time_labels
    
    trend_ = [x for x in range(len(time_labels))]
    
    time = trend_
    end_time_index = max(trend_)

    if drug_name == "hydroxychlroquine":
        ax.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims (Red States)', linestyle="-", marker='')
    
    elif drug_name == "ivermectin":
        ax.plot_date(time[33:], y_all_red[33:], lw =1.5, color=colors[2], label = 'Actual IVM Claims (Texas)', linestyle="-", marker='')
        ax.plot_date(time[33:], y_all_blue[33:], lw =1.5, color=colors[1], label = 'Actual IVM Claims (California)', linestyle="-", marker='')

    y_pos = y_max * 0.75 + y_min * 0.25
    # print("y pos", y_pos)
    
    for ix,vl in enumerate(list(event_lines)): 
        vline_position = labels_.index(vl)
        # print(vline_position)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate("Event " + event_lines[vl].split(":")[0], xy =(vline_position-1.5, y_pos), rotation=90 , fontsize=10)

    if drug_name == "hydroxychloroquine":
        if "placebo" not in suffix:
            treatment_label = "Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)
        else:
            actual_treatment_ix = 59
            treatment_label = "Placebo Treatment Date"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='darkred', linestyle='--', label=treatment_label, lw=2)

            treatment_label = "Actual Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(actual_treatment_ix, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)


    #ax.xaxis.set_ticks(np.array(time)[::5])
    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20

    ax.tick_params(axis='x', rotation=90, labelsize=14)
    
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.labelpad = 5
    # ax.yaxis.set_ticks([0, 50] + list(np.arange(100, 750, 100)))
    # ax.set_yticklabels(np.array([0, 50] + list(np.arange(100, 750, 100))).astype(str))
    ax.set_ylabel(y_label_red, font=font)
    
    ax.set_title(title, fontdict={"fontsize" : 18, "fontweight" : "bold", "fontfamily":"Times New Roman"})
    # ax.grid()
    
    ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)
    if y_all_red.mean() > 1:
        ax = format_yaxis(ax)

    font2 = {'family' : 'Times New Roman',
           'size'   : 14}
    fig.legend(loc='upper center', ncol=4, prop=font2, bbox_to_anchor=(0.5, -0.05))
    
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))

    print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')

    fig.legends = []

    fig.legend(loc='upper center', ncol=3, prop=font2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig("graphs/" + title.strip() + ".png",bbox_inches='tight',dpi=300)
    
    return

def covid_by_trump_support(vlinepos, covid_cases_red, covid_cases_blue, title="", covid_label_red="", covid_label_blue="", y_covid_label_red="", y_covid_label_blue="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels=[],drug_name="ivermectin", treat_pos=0):
    
    font = {'family' : 'Times New Roman',
            'size'   : 14}
    plt.rc('font', **font)
    #plt.rc('axes', labelsize=28)
    fig, ax = plt.subplots(figsize=(15,8))

    
    tick_spacing=4

    colors= ['black','blue',"red","green",'yellow','brown','aqua','darkgreen']
    
    colors_= ['black',"green",'purple','saddlebrown','midnightblue','orange','forestgreen','firebrick','midnightblue'] # sandybrown -> firebrick
    # colors_[2] - yellow -> purple
    # colors_[3] - brown -> purple
    
    from important_dates import get_time_indices
    
    event_lines = get_time_indices(drug=drug_name)
    
    y_max = max(covid_cases_red)
    y_min = 0
    
    
    """
    if drug_name=='hydroxychloroquine' or drug_name=='chloroquine phosphate':
        y_max=70
    elif drug_name=='ivermectin':
        y_max=400
    """
    labels_ = time_labels
    
    trend_ = [x for x in range(len(time_labels))]
    
    time = trend_
    end_time_index = max(trend_)
    
    same_magnitude = (np.max(covid_cases_blue) * 0.5 <= np.max(covid_cases_red) <= np.max(covid_cases_blue) * 2)

    #time = pa.to_datetime(time.values)
    if drug_name == "hydroxychlroquine":
        ax.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims', linestyle="-", marker='')
    
    elif drug_name == "ivermectin":
        covid_cases_red = [0] * (len(time[33:]) - len(covid_cases_red)) + list(covid_cases_red)
        ax.plot_date(time[33:], covid_cases_red, lw =1, color=colors[2], label = covid_label_red, linestyle="--", marker='')
        
        if same_magnitude:
            covid_cases_blue = [0] * (len(time[33:]) - len(covid_cases_blue)) + list(covid_cases_blue)
            ax.plot_date(time[33:], covid_cases_blue, lw =1, color=colors[1], label = covid_label_blue, linestyle="--", marker='')
    
    # y_max = 1205
    # y_min = 430
    
    y_pos = y_max * 0.75 + y_min * 0.25
    # print("y pos", y_pos)
    
    
    
    for ix,vl in enumerate(list(event_lines)): 
        vline_position = labels_.index(vl)
        # print(vline_position)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate("Event " + event_lines[vl].split(":")[0], xy =(vline_position-1.5, y_pos), rotation=90 , fontsize=14)

    if drug_name == "hydroxychloroquine":
        if "placebo" not in suffix:
            treatment_label = "Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)
        else:
            actual_treatment_ix = 59
            treatment_label = "Placebo Treatment Date"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='darkred', linestyle='--', label=treatment_label, lw=2)

            treatment_label = "Actual Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(actual_treatment_ix, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)

    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20

    ax.tick_params(axis='x', rotation=90, labelsize=14)
    
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.labelpad = 5

    ax.set_ylabel(y_covid_label_red, font=font)
    
    ax.set_title(title, fontdict={"fontsize" : 18, "fontweight" : "bold", "fontfamily":"Times New Roman"})
    # ax.grid()
    
    ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)
    if np.mean(covid_cases_red) > 1:
        ax = format_yaxis(ax)

    font2 = {'family' : 'Times New Roman',
           'size'   : 14}
    fig.legend(loc='upper center', ncol=4, prop=font2, bbox_to_anchor=(0.5, -0.05))
    
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    # print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')
    if not same_magnitude:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(y_covid_label_blue, font=font)  # we already handled the x-label with ax1

        covid_cases_blue = [0] * (len(time[33:]) - len(covid_cases_blue)) + list(covid_cases_blue)
        ax2.plot_date(time[33:], covid_cases_blue, lw =1, color=colors[1], label = 'Weekly Covid Cases (Blue States)', linestyle="--", marker='')
        ax2.yaxis.labelpad = 5
        if np.mean(covid_cases_blue) > 1:
            ax2 = format_yaxis(ax2)
        ax2.xaxis.set_ticks(np.array(time[33:]))
        ax2.set_xlabel('Week (Sunday)', font=font)
        ax2.xaxis.labelpad = 20
        ax2.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', rotation=90, labelsize=14)
        ax2.xaxis.set_ticks(np.array(time[33:]))
    fig.legends = []
    plt.tight_layout()
    fig.legend(loc='upper center', ncol=3, prop=font2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig("graphs/" + title.strip() + "_covid.png",bbox_inches='tight',dpi=300)
    
    return

def covid_tx_ca(vlinepos, covid_cases_red, covid_cases_blue, title="", covid_label_red="", covid_label_blue="", xname='Weeks', opath='./Model_Based/',suffix='', time_labels=[],drug_name="ivermectin", treat_pos=0):
    
    font = {'family' : 'Times New Roman',
            'size'   : 14}
    plt.rc('font', **font)
    #plt.rc('axes', labelsize=28)
    fig, ax = plt.subplots(figsize=(15,8))

    
    tick_spacing=4

    colors= ['black','blue',"red","green",'yellow','brown','aqua','darkgreen']
    
    colors_= ['black',"green",'purple','saddlebrown','midnightblue','orange','forestgreen','firebrick','midnightblue'] # sandybrown -> firebrick
    # colors_[2] - yellow -> purple
    # colors_[3] - brown -> purple
    
    from important_dates import get_time_indices
    
    event_lines = get_time_indices(drug=drug_name)
    
    y_max = max(covid_cases_red)
    y_min = 0
    
    
    """
    if drug_name=='hydroxychloroquine' or drug_name=='chloroquine phosphate':
        y_max=70
    elif drug_name=='ivermectin':
        y_max=400
    """
    labels_ = time_labels
    
    trend_ = [x for x in range(len(time_labels))]
    
    time = trend_
    end_time_index = max(trend_)


    #time = pa.to_datetime(time.values)
    if drug_name == "hydroxychlroquine":
        ax.plot_date(time, y_all, lw =1.5, color=colors[2], label = 'Actual HCQ Claims', linestyle="-", marker='')
    
    elif drug_name == "ivermectin":
        covid_cases_red = [0] * (len(time[33:]) - len(covid_cases_red)) + list(covid_cases_red)
        covid_cases_blue = [0] * (len(time[33:]) - len(covid_cases_blue)) + list(covid_cases_blue)

        # ax.plot_date(time[-covid_cases_red.shape[0]:], covid_cases_red, lw =1, color=colors[2], label = 'Weekly Covid Cases (Red States)', linestyle="--", marker='')
        ax.plot_date(time[33:], covid_cases_red, lw =1, color=colors[2], label = 'Weekly Covid Cases (Texas)', linestyle="--", marker='')
        ax.plot_date(time[33:], covid_cases_blue, lw =1, color=colors[1], label = 'Weekly Covid Cases (California)', linestyle="--", marker='')
    
    # y_max = 1205
    # y_min = 430
    
    y_pos = y_max * 0.75 + y_min * 0.25
    # print("y pos", y_pos)
    
    
    
    for ix,vl in enumerate(list(event_lines)): 
        vline_position = labels_.index(vl)
        # print(vline_position)
        ax.axvline(vline_position, color=colors_[ix], linestyle='--',  label=event_lines[vl], lw=2)
        ax.annotate("Event " + event_lines[vl].split(":")[0], xy =(vline_position-1.5, y_pos), rotation=90 , fontsize=14)

    if drug_name == "hydroxychloroquine":
        if "placebo" not in suffix:
            treatment_label = "Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)
        else:
            actual_treatment_ix = 59
            treatment_label = "Placebo Treatment Date"    
            ax.vlines(treat_pos, ymin=np.min(y_all), ymax=y_max, color='darkred', linestyle='--', label=treatment_label, lw=2)

            treatment_label = "Actual Treatment Date:\nDidier Raoult posts his video about HCQ"    
            ax.vlines(actual_treatment_ix, ymin=np.min(y_all), ymax=y_max, color='black', linestyle='--', label=treatment_label, lw=2)

    ax.set_xlabel('Week (Sunday)', font=font)
    ax.xaxis.labelpad = 20

    ax.tick_params(axis='x', rotation=90, labelsize=14)
    
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.labelpad = 5

    ax.set_ylabel(covid_label_red, font=font)
    
    ax.set_title(title, fontdict={"fontsize" : 18, "fontweight" : "bold", "fontfamily":"Times New Roman"})
    # ax.grid()
    
    ax = format_xaxis(ax,  time_labels, end_time_index, tick_spacing)
    if np.mean(covid_cases_red) > 1:
        ax = format_yaxis(ax)

    font2 = {'family' : 'Times New Roman',
           'size'   : 14}
    fig.legend(loc='upper center', ncol=4, prop=font2, bbox_to_anchor=(0.5, -0.05))
    
    if not os.path.exists('{}'.format(opath)):
        os.makedirs('{}'.format(opath))
    # print("FILEPATH",'{}/{}'.format(opath,suffix )+'.png')

    ax.tick_params(axis='x', rotation=90, labelsize=14)
    fig.legends = []
    plt.tight_layout()
    fig.legend(loc='upper center', ncol=3, prop=font2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig("graphs/" + title.strip() + "_covid.png",bbox_inches='tight',dpi=300)
    
    return

    
def summarize(fname, y_all, y_train,y_scaler,suffix,xname, covid_cases=None, opath='../Model_Based/',time_labels =[], vlinepos = [], title="", y_label="Total Claims", covid_label="", national=False, transform = True,drug_name="hydroxychloroquine"):
    #print(az.summary(trace))
    #print(az.plot_trace(trace))
    # y_pred =y_scaler.inverse_transform(y_pred.astype('float64').flatten())
    #
    #
    if not transform:
        y_all = y_scaler.inverse_transform(y_all.astype('float64').reshape(-1,1)).flatten()
        y_train =y_scaler.inverse_transform(y_train.astype('float64').reshape(-1,1)).flatten()
    
    # if len(vlinepos) == 0:
    #     vlinepos=y_train.shape[0]
    
    #graph(y_pred,y_all,bands,vlinepos, title=fname+' Mape Train:{}'.format(np.round(summary['mape_train'],2)), xname=xname, opath='../Model_Based/',suffix=suffix)
   
   
    
    
    # print("y_all","\n",y_all.shape)
    
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
    if 'all' in fname or 'trump' or 'DAKOTA' in fname:
        main_graph(y_all, vlinepos, covid_cases,title=title, y_label=y_label, covid_label=covid_label, xname=xname, opath=opath,suffix=suffix,time_labels=time_labels,drug_name=drug_name, treat_pos=len(y_train)-1)
        #graph_per(efctsz,bands_efctsz,vlinepos, title=fname,y_label="Effect Size (in Std. Units)", xname=xname, opath=opath,suffix=suffix,time_labels =time_labels )
    
    else:
        pass
        #graph(y_pred, y_all, bands, vlinepos, covid_cases, title=title, y_label=y_label, xname=xname, opath=opath,suffix=suffix,time_labels=time_labels,)
        #graph_per(percentages,bands_per,vlinepos, title=fname, xname=xname, opath=opath,suffix=suffix,time_labels =time_labels )
    
    # if not national:
        
        
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
    return

