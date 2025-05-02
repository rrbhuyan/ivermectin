import numpy as np
from get_priors import *
import arviz as az
import pandas as pa
np.random.seed(seed=42)
from pymc.variational.callbacks import CheckParametersConvergence
import pymc.sampling_jax
import matplotlib.pyplot as plt
from jax import random
RANDOM_SEED = 58
rng_key = random.PRNGKey(0)


from bayes_model_summary import mean_absolute_percentage_error


from sklearn.model_selection import train_test_split, ShuffleSplit,KFold



def prior_predictive_check(model, X_train, model_type):
    with model:
        prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    for item in range(1):#len( prior_checks["β"])
        _, ax = plt.subplots()
        x = np.linspace(X_train.min(), X_train.max(), prior_checks["β"].shape[1])
        y= np.multiply(prior_checks["β"],x)
        print(y.shape)
        for a, b in zip(prior_checks["β0"], prior_checks["β"][item]):
            y = b* x
            if model_type=='Poisson':
                y = np.exp(y)
            ax.plot(x, y, c="k", alpha=0.4)

        ax.set_xlabel("Predictor (stdz)")
        ax.set_ylabel("Mean Outcome (stdz)")
        ax.set_title("Prior predictive checks -- Weakly regularizing priors")
        plt.show()


def predict(model,trace,  X_all):
    #predict
    with model:
        # update values of predictors:
        pm.set_data({"pred": X_all})
        # use the updated values and predict outcomes and probabilities:
        posterior_predictive = pm.sample_posterior_predictive(
            trace, var_names=["obs"], random_seed=42#, samples=2000
        )
        #print(posterior_predictive.keys())
        model_preds = posterior_predictive.posterior_predictive["obs"]  
        model_preds= az.extract_dataset(model_preds)['obs'].values
        #print(model_preds.shape)
        return np.transpose(model_preds),posterior_predictive.posterior_predictive

def predict1(model,trace,  X_all):
    #predict
    with model:
        # update values of predictors:
        pm.set_data({"pred": X_all})
        # use the updated values and predict outcomes and probabilities:
        posterior_predictive = pm.sample_posterior_predictive(
            trace, var_names=["mu"], random_seed=42, samples=1000, 
        )
        model_preds = posterior_predictive["mu"]  
        return model_preds,posterior_predictive

def get_model(X_train, y_train,prior_type, model_type):
    if prior_type=='Finnish':
        model = finnish_horseshoe(X_train,y_train,model_type)
    elif prior_type=='Regularized':
        model = regularized_horseshoe(X_train,y_train,model_type)
    elif prior_type=='Wide':
        model = wide(X_train,y_train,model_type)
    elif prior_type=='Laplace':
        model = laplace(X_train,y_train,model_type)
    elif prior_type=='Standard':
        model = standard_horseshoe(X_train, y_train, model_type)
    elif prior_type=='Spikeslab':
        model = spikeslab(X_train, y_train, model_type)
    elif prior_type=='HorseshoeNu':
        model = horseshoe_nu(X_train, y_train, model_type)
    elif prior_type=='Localstudent':
        model = localstudent(X_train, y_train, model_type)
    elif prior_type=='Hyperlasso':
        model = hyperlasso(X_train, y_train, model_type)
    elif prior_type=='Elasticnet':
        model = elasticnet(X_train, y_train, model_type)
    return model





def cross_val(X_train, y_train,y_scaler, prior_type, model_type,SAMPLE_KWARGS,try_others):
    
    mapes =[]
    kf= KFold(n_splits=5,random_state=42, shuffle=True)
    
    #print('reached')
    for j, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        #print('cv:{} '.format(j))
        X_train_cv = X_train[train_idx.tolist(),:]
        X_valid_cv = X_train[val_idx.tolist(),:]
        y_train_cv = y_train[train_idx.tolist()]
        y_valid_cv = y_train[val_idx.tolist()]
        
        model = get_model(X_train_cv, y_train_cv,prior_type, model_type)
        
        model, trace,selected_init_method = train_bayes(model,SAMPLE_KWARGS, try_others)
        #y_pred =y_scaler.inverse_transform(y_pred.astype('float64').flatten())
        y_valid_cv =y_scaler.inverse_transform(y_valid_cv.astype('float64')).flatten()
        #y_all =y_scaler.inverse_transform(y_all.astype('float64')).flatten()
        model_preds,posterior_predictive = predict(model, trace, X_valid_cv)
        model_preds = y_scaler.inverse_transform(model_preds)
        y_pred =model_preds.mean(axis=0)
        
    
        mape =  mean_absolute_percentage_error( y_valid_cv, y_pred)
        mapes.append(mape)
    print('mapes' , mapes)    
    return np.mean(mapes)    



def train_bayes1(model, SAMPLE_KWARGS, try_others=True):
    with model:
        switch= True
        ix=0
        init_methods = ['jitter+adapt_diag','advi_map','jitter+adapt_full','advi+adapt_diag','advi','adapt_full']
        while switch:
            try:
                SAMPLE_KWARGS['init'] = init_methods[ix]
                selected_init_method = init_methods[ix]
                print('Trying: ',SAMPLE_KWARGS['init'])
                trace = pm.sample(**SAMPLE_KWARGS)
                switch=False
                if trace.sample_stats['diverging'].sum().item()>10 and try_others==True:
                    print("diverging stats: ", trace.sample_stats['diverging'].sum().item(),len(trace))
                    switch=True
                    ix=ix+1
                    if ix>len(init_methods):
                        return model, trace,selected_init_method
            except Exception as e:
                    print(e)
                    ix=ix+1
                    if ix>len(init_methods):
                        return model, trace,selected_init_method
    return model, trace,selected_init_method


def train_bayes(model, SAMPLE_KWARGS, try_others=True):
    with model:
        SAMPLE_KWARGS['model'] =model

        switch= True
        ix=0
        try_others=True
        init_methods = [0.99]
        while switch:
            try:
                SAMPLE_KWARGS['target_accept'] = init_methods[ix]
                selected_init_method = init_methods[ix]
                print('Trying: ',SAMPLE_KWARGS['target_accept'])
                trace = pymc.sampling_jax.sample_numpyro_nuts(**SAMPLE_KWARGS)
                print("diverging stats: ",trace.sample_stats['diverging'].sum().item())
                switch=False
                if trace.sample_stats.diverging.sum().item()>10 and try_others==True:
                    print("diverging stats: ", trace.sample_stats['diverging'].sum().item(),len(trace))
                    switch=True
                    ix=ix+1
                    if ix>len(init_methods):
                        return model, trace,selected_init_method
            except Exception as e:
                    print(e)
                    ix=ix+1
                    if ix>len(init_methods):
                        return model, trace,selected_init_method
    return model, trace,'numpyro'




def train_bayesCV(prior_type,model_type, X_train, y_train,SAMPLE_KWARGS, try_others=True):
    #print('CV ing :  ', prior_type)
    curr_model =''
    min_loo = 100000
    proportions =[0.5,1,2]
    for proportion in proportions : 
        
        if prior_type=='Localstudent':
            model = localstudent(X_train,y_train,model_type, proportion)
        elif prior_type=='Hyperlasso':
            model = hyperlasso(X_train,y_train,model_type, proportion)    
        model, trace,selected_init_method = train_bayes(model,SAMPLE_KWARGS, try_others)
        print('here')
        
        curr_loo=-2*az.loo(trace,model)[0]
        #print('proportion: ',proportion,' min_loo: ', min_loo,' curr_loo: ', curr_loo )
        if curr_loo<=min_loo  and trace.sample_stats['diverging'].sum().item() <10:
            min_loo = curr_loo
            curr_model = model
            curr_trace = trace
            curr_selected_init_method = selected_init_method
        #print('proportion: ',proportion,' min_loo: ', min_loo,' curr_loo: ', curr_loo )
    return curr_model, curr_trace,curr_selected_init_method
        



def get_relaxed(trace):
    keep_betas = pa.DataFrame(az.summary(trace,var_names=[ 'β'] , filter_vars='like'))
    keep_betas=keep_betas.rename(columns={'hdi_3%':'hdi_3', 'hdi_97%':'hdi_97'})
    keep_betas['keep'] = keep_betas.apply(lambda row: 0 if row.hdi_3<0 and row.hdi_97>0 else 1,axis=1)
    keep_vars=[]
    for item in keep_betas[keep_betas['keep']==1].index.tolist():
        if '[' in item:
            keep_vars.append(int(item.replace('β[','').replace(']','')))
    return keep_vars


class MyCallback:
    def __init__(self, every=1000, max_rhat=1.05):
        self.every = every
        self.max_rhat = max_rhat
        self.traces = {}

    def __call__(self, trace, draw):
        if draw.tuning:
            return

        self.traces[draw.chain] = trace
        if len(trace) % self.every == 0:
            multitrace = pm.backends.base.MultiTrace(list(self.traces.values()))
            if pm.stats.rhat(multitrace).to_array().max() < self.max_rhat:
                raise KeyboardInterrupt


def divergence_callback(trace,draw):
    #print(trace)
    if trace.sample_stats['diverging'].sum().item()>5:
        raise KeyboardInterrupt()

def estimate_regression(X_train, y_train, X_all, y_all, prior_type,model_type,relaxed=False, estimator='bayes', try_others=True, y_scaler=None):
    SEED = 123456789 # for reproducibility

    rng = np.random.default_rng(SEED)   
    CHAINS = 2

    SAMPLE_KWARGS1 = {
        'cores': CHAINS,
        'target_accept': 0.99,
        'max_treedepth': 15,
        'random_seed': [SEED + i for i in range(CHAINS)],
        'return_inferencedata': True,
        'tune':1000,
        'draws':2000,
        'discard_tuned_samples': True,
        #'callback': MyCallback()
        
    }
    SAMPLE_KWARGS = {
        'chains': CHAINS,
        'random_seed': SEED,
        'tune':1000,
        'draws':2000,
        'chain_method':'parallel',
        'nuts_kwargs': {'max_tree_depth':15},
        #'callback': MyCallback()
        'progress_bar':False
    }
    
    model = get_model(X_train, y_train,prior_type, model_type)
    #prior_predictive_check(model, X_train, model_type)
    #print('y_min: ', y_train.min(), 'y_max: ', y_train.max())
    
    selected_init_method =''
    
    if estimator=='bayes' and prior_type not in ['Hyperlasso']:
        model, trace,selected_init_method = train_bayes(model,SAMPLE_KWARGS, try_others)
    
    if estimator=='bayes' and prior_type in ['Hyperlasso']:
        model, trace,selected_init_method = train_bayesCV(prior_type,model_type, X_train, y_train,SAMPLE_KWARGS, try_others)
    
    
    with model:
        if estimator=='advi':
            inference = pm.ADVI()
            approx = pm.fit(n=200000, method=inference,obj_optimizer=pm.adamax(learning_rate=0.1), callbacks=[CheckParametersConvergence(diff="absolute")])
            trace = approx.sample(draws=10000)
            #trace = pm.sample(**SAMPLE_KWARGS)
        if estimator=='svgd':
            approx = pm.fit(
                300,
                method="svgd",
                inf_kwargs=dict(n_particles=1000),
                obj_optimizer=pm.sgd(learning_rate=0.01),
            )
            trace = approx.sample(draws=1000)
   


    if relaxed:
        print('Trying to Relax! \n')
        keep_vars = get_relaxed(trace)
        #print(keep_vars)
        if len(keep_vars)<= 5:
            print('Cannot Relax!\n')
        else:    
            X_train = X_train[:,keep_vars]
            X_all = X_all[:,keep_vars]
            del model
            print(X_all.shape)
            model = laplace(X_train,y_train,model_type)
            with model:
                trace = pm.sample(**SAMPLE_KWARGS)
    #predict
    model_preds,posterior_predictive = predict(model,trace,  X_all)
    
    
    cv_mape =0
    if y_scaler is not None:
        #print('CVINGG\n')
        cv_mape = cross_val(X_train, y_train,y_scaler, prior_type, model_type,SAMPLE_KWARGS,try_others)
    return trace,model_preds,model,posterior_predictive,selected_init_method, cv_mape
    

   