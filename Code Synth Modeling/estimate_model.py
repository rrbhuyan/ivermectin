import bayes_model_summary
import gc
import get_data
import glob
import numpy as np
import pandas as pd

from bayes_estimate import estimate_regression
from bayes_model_summary import summarize
from get_data import get_data
from itertools import product
from itertools import repeat
from random import shuffle

def estimate(kwargs):
    estimator='bayes'
    f, prior_type, model_type, estimator, vlinepos, title, y_label = kwargs
    
    filename_parts = f.split('/')
    clean_fname = filename_parts[-1].replace('.csv','') + "_" + filename_parts[-2]
    graphs_path, results_path = ('../Output_DP/' + filename_parts[3] + "/" + filename_parts[4] + '/Graphs', '../Output_DP/' + filename_parts[3] + "/" + filename_parts[4] +'/Results/')
    print(clean_fname +'\n')
    print('\n')
    suffix = clean_fname
    scale=False
    if model_type =='Poisson':
        scale=False
    
    X_all,y_all,X_train, y_train,y_scaler,time_labels, covid_cases, skip= get_data(f,scale=scale,remove={'trend', 'y','Unnamed: 0','pre_period','treat_post','year','week','trend', 'covid_cases'}, model_type=model_type)
    #trace,model_preds,model,posterior_predictive,selected_init_method,cv_mape= estimate_regression(X_train, y_train, X_all, y_all,prior_type,model_type,relaxed=False, estimator=estimator, y_scaler=None)
    try:
        trace,model_preds,model,posterior_predictive,selected_init_method,cv_mape= estimate_regression(X_train, y_train, X_all, y_all,prior_type,model_type,relaxed=False, estimator=estimator, y_scaler=None)
    
    except:
        return {'inestimable': clean_fname}
    
    summary= summarize(trace,model, model_preds, clean_fname, model_type,prior_type, y_all, y_train,y_scaler, suffix+'_{}'.format(prior_type),xname='Weeks',covid_cases=covid_cases, estimator=estimator,selected_init_method=selected_init_method,cv_mape = cv_mape,opath=graphs_path, time_labels=time_labels, vlinepos=vlinepos, title=title, y_label=y_label)
    df = pd.DataFrame.from_dict([summary])
    df.to_csv(results_path + suffix + ".csv")
    # gc.collect()
    return {'estimable': clean_fname, "model_preds": model_preds, "y_scaler" : y_scaler, }

def get_kwargs(filenames, prior_type, model_type, estimator, vlinepos, titles, y_labels):
    prior_types= [prior_type] #,'Hyperlasso',
    cartesian = list(product(filenames,prior_types))
    if len(titles) == 0:
        titles = [""] * len(cartesian)
    
    if len(y_labels) == 0:
        y_labels = ["Number of Prescription Claims"] * len(cartesian)
    
    files = [x[0] for x in cartesian]
    priors = [x[1] for x in cartesian]
    process_this= zip(files, priors, repeat(model_type), repeat(estimator), repeat(vlinepos), titles, y_labels)
    return process_this

def run(filenames = [], prior_type='Finnish', model_type='Linear', estimator='bayes', vlinepos=[], titles=[], y_labels=[]):
    res = []
    for prior_type in ['Spikeslab']: #'Finnish',,'Laplace','Spikeslab','Regularized','Spikeslab','Laplace'
    # for prior_type in ["HorseshoeNu"]:
        process_this = get_kwargs(filenames, prior_type, model_type, estimator, vlinepos=vlinepos, titles=titles, y_labels=y_labels)
        results = list(map(estimate,process_this ))
        res.extend(results)
        gc.collect()
        print('Done with: ',prior_type)