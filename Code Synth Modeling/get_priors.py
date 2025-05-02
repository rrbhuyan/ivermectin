
import pymc as pm
import numpy as np
np.random.seed(seed=42)


def elasticnet(X_train, y_train,model_type):
    M = X_train.shape[1]
    with pm.Model() as model:
        σ1 = pm.HalfNormal("σ1", 2.5)
        λ1 = pm.HalfCauchy('λ1', 1.)
        λ2 = pm.HalfCauchy('λ2', 1.)
    
        scale = pm.Deterministic('scale', (8 *(λ2)*(σ1**2) )/(λ1**2) ) 
        
        BoundedGamma = pm.Bound(pm.Gamma, lower=1.0)
        
        τj = BoundedGamma('τj', mu=0.5, sigma=scale, shape=M)
        
        σ_ =pm.Deterministic('σ_', np.sqrt( 1/ (  λ2 * τj/ ((τj-1) *  σ1**2)       ) )  )
        #β = pm.Normal('beta', mu=0, sigma=sigma_, shape=M)
        β_ = pm.Normal('beta', mu=0, sigma=1, shape=M)
        
        
        β =pm.Deterministic('β', σ_*β_)
        σ = pm.HalfNormal("σ", 2.5)       
        β0 = pm.Normal("β0", 0, 10.)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model
    

def localstudent(X_train, y_train,model_type,ν =1.):
    M = X_train.shape[1]
    with pm.Model() as model:
        
        ν = 6
        λ = pm.HalfStudentT('lambda', 0.5)
        
        τ2 = pm.InverseGamma('tau_sq', alpha=0.5*ν, beta=0.5*ν/(λ), shape=M, testval=0.1)
        σ1 = pm.HalfNormal("σ1", 2.5)
        sigma_ =pm.Deterministic('sigma_', σ1 * np.sqrt(τ2))
        β_ = pm.StudentT('beta',nu = ν,mu=0, sigma=1, shape=M)
        β = pm.Deterministic('β', np.multiply(sigma_,β_) )
        
        
        σ = pm.HalfNormal("σ", 2.5)       
        β0 = pm.Normal("β0", 0, 10.)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model


def localstudent1(X_train, y_train,model_type,ν =1.):
    M = X_train.shape[1]
    with pm.Model() as model:
        λ = pm.HalfStudentT('lambda', 1)#pm.HalfCauchy('lambda', 1)
        σ1 = pm.HalfNormal("σ1", 2.5)
        sigma_ =pm.Deterministic('sigma_', (σ1) / λ)
        β = pm.StudentT('beta',nu = ν,mu=0, sigma=sigma_, shape=M)
        σ = pm.HalfNormal("σ", 2.5)       
        β0 = pm.Normal("β0", 0, 10.)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model

def hyperlasso(X_train, y_train,model_type,ν=0.5):
    M = X_train.shape[1]
    with pm.Model() as model:
        
        σ = pm.HalfNormal("σ", 2.5)
        #λ = pm.Gamma('lambda', 1,1)
        #λ = pm.HalfStudentT('lambda', 2)
        λ = pm.HalfCauchy('lambda', beta=1 )
        #λ = pm.Exponential('lambda', lam=0.5 )
        τj = pm.Gamma('tauj', alpha=ν, beta=1/(λ) , shape=M, testval=0.1)
        
        sigma_= pm.Deterministic('sigma_', np.sqrt(2*τj))
        
        β = pm.Laplace('beta', mu=0, b=sigma_, shape=M)
               
        β0 = pm.Normal("β0", 0, 1.)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        
        if model_type=='Linear':

            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model

def hyperlasso1(X_train, y_train,model_type,ν=0.5):
    M = X_train.shape[1]
    with pm.Model() as model:
        #ν=2.
        #λ = pm.HalfCauchy('lambda', beta=1.)
        #λ =  pm.Gamma('lambda', 1,1)
        λ = pm.HalfStudentT('lambda', 1,shape=M)
        #λ = pm.HalfCauchy('lambda', beta=1)
        τj = pm.Gamma('tauj', alpha=ν, beta=1/(λ) , shape=M, testval=0.1)
        phij_sq = pm.Exponential('phij_sq',τj , shape=M, testval=0.1)
       
        σ = pm.HalfNormal("σ", 2.5)
        sigma_ =pm.Deterministic('sigma_',  np.sqrt(phij_sq))
        
        β = pm.Normal('beta', mu=0, sigma=sigma_, shape=M)
               
        β0 = pm.Normal("β0", 0, 10.)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        
        if model_type=='Linear':
            
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model


def spikeslab_(X_train, y_train,model_type):
    M = X_train.shape[1]
    with pm.Model() as model:
        mu_indic = 0
        sigma_indic = 5
        tau = 2.5
        #lambda_hat = pm.Normal('lambda_hat', mu = mu_indic, sigma = sigma_indic, shape = M)
        indic_raw= pm.Normal('indic', mu=0, sigma=1,shape=M)
        indic = pm.invlogit(mu_indic+sigma_indic*indic_raw)
        
        σ = pm.HalfNormal("σ", 2.5)
        β = pm.Normal('β', mu=0, sigma=0.5, shape=M)
        #spike_raw = pm.Normal('spike_raw', mu = 0, sigma = 1, shape = M)
        spike = pm.Deterministic('spike',tau*β*indic)
        
        
        β0 = pm.Normal("β0", 0, 10.)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, spike))
        
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model





def spikeslab__(X_train, y_train,model_type):
    M = X_train.shape[1]
    with pm.Model() as model:
        mu_indic = 0
        sigma_indic = 10
        tau = 5
        #lambda_hat = pm.Normal('lambda_hat', mu = mu_indic, sigma = sigma_indic, shape = M)
        indic_raw= pm.Normal('indic', mu=0, sigma=1,shape=M)
        
        
        
        indic = pm.invlogit(mu_indic+sigma_indic*indic_raw)
        
        
        τ = pm.HalfCauchy('τ', 1)
        
        σ = pm.HalfNormal("σ", 2.5)
        β = pm.Normal('β', mu=0, sigma=0.5, shape=M)
        #spike_raw = pm.Normal('spike_raw', mu = 0, sigma = 1, shape = M)
        spike = pm.Deterministic('spike',tau*β*indic)
        
        
        β0 = pm.Normal("β0", 0, 10.)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, spike))
        
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model

def spikeslab(X_train, y_train,model_type):
    M = X_train.shape[1]
    with pm.Model() as model:
        mu_indic = 0
        sigma_indic = 0.5
        tau = 0.5
        #lambda_hat = pm.Normal('lambda_hat', mu = mu_indic, sigma = sigma_indic, shape = M)
        indic_raw= pm.Normal('indic', mu=0, sigma=1,shape=M)
        indic = pm.invlogit(mu_indic+sigma_indic*indic_raw)
        
        σ = pm.HalfNormal("σ", 10)
        β = pm.Normal('β', mu=0, sigma=0.25, shape=M)
        #spike_raw = pm.Normal('spike_raw', mu = 0, sigma = 1, shape = M)
        spike = pm.Deterministic('spike',tau*β*indic)
        
        
        β0 = pm.Normal("β0", 0, 2.5)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, spike))
        
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model
        
        
    
def horseshoe_nu(X_train, y_train, model_type):
    M = X_train.shape[1]
    ν = 1
    with pm.Model() as model:
        rₗ = pm.Normal('r_local', mu=0, sigma=1., shape=M)
        ρₗ = pm.InverseGamma('rho_local', alpha=0.5*ν, beta=0.5*ν, shape=M, testval=0.1)
        rᵧ = pm.Normal('r_global', mu=0, sigma=1)
        ρᵧ = pm.InverseGamma('rho_global', alpha=0.5, beta=0.5, testval=0.1)
        τ = rᵧ * pm.math.sqrt(ρᵧ)
        λ = rₗ * pm.math.sqrt(ρₗ)
        z = pm. Normal('z', mu=0, sigma=1, shape=M)
        β = pm.Deterministic('beta', z*λ*τ)
        β0 = pm.Normal('b', mu=0, sigma=10.)
        
        σ = pm.HalfNormal("σ", 2.5)
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu', pm.math.dot(pred, β))#β0 +
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model
    


def standard_horseshoe2(X_train, y_train, model_type):
    M = X_train.shape[1]
    m=min(10,int(M/2))
    with pm.Model() as model:
        
        σ = pm.HalfNormal("σ", 2.5)
        λ = pm.HalfCauchy('lambda', beta=0.05, shape=M) #, shape=M
        #τ = 0.3# pm.HalfCauchy('tau', beta=1)
        #τ_0 = m / (X_train.shape[1] - m) * σ / np.sqrt(X_train.shape[0])

        τ = pm.HalfCauchy('τ', .2)
        σ_ = pm.Deterministic('horseshoe',pm.math.sqrt(τ*λ) )
        β_ = pm.Normal('beta', mu=0, sigma=1, shape=M)
        
        
        β =pm.Deterministic('β',  np.multiply(σ_,β_) )
        
        β0 = pm.Normal("β0", 0, 0.5)
        
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            mu_1 = pm.Lognormal(mu,sigma=30)
            obs = pm.Poisson('obs', mu=mu_1, observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model




def standard_horseshoe(X_train, y_train, model_type):
    M = X_train.shape[1]
    
    with pm.Model() as model:
        
        
        λ = pm.Cauchy('lambda',alpha=0, beta=1, shape=M) #, shape=M
        #τ = 0.3# pm.HalfCauchy('tau', beta=1)
        #τ_0 = m / (X_train.shape[1] - m) * σ / np.sqrt(X_train.shape[0])

        τ = pm.Cauchy('τ', alpha=0,beta=1)
        σ_ = pm.Deterministic('horseshoe',τ*λ )
        β_ = pm.Normal('beta', mu=0, sigma=1, shape=M)
        
        
        β =pm.Deterministic('β', σ_*β_)
        
        β0 = pm.Normal("β0", 0, 1.)
        
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        σ = pm.HalfNormal("σ", 1)
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            mu_1 = pm.Lognormal(mu,sigma=30)
            obs = pm.Poisson('obs', mu=mu_1, observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model

def standard_horseshoe1(X_train, y_train, model_type):
    M = X_train.shape[1]
    with pm.Model() as model:
        ϵ = pm.InverseGamma('epsilon', alpha=0.5, beta=1., shape=M, testval=0.1)
        λ2 = pm.InverseGamma('lambda', alpha=0.5, beta=1./ϵ, shape=M, testval=0.1)
        ξ = pm.InverseGamma('xi', alpha=0.5, beta=1., testval=0.1)
        τ2 = pm.InverseGamma('tau', alpha=0.5, beta=1./ξ, testval=0.1)
        σ_ = pm.Deterministic('horseshoe', pm.math.sqrt(τ2*λ2))
        β = pm.Normal('beta', mu=0, sigma=σ_, shape=M)
        β0 = pm.Normal("β0", 0, 10.)
        
        σ = pm.HalfNormal("σ", 2.5)
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model
    




def wide11(X_train, y_train, model_type):
    M = X_train.shape[1]
    with pm.Model() as model:
        β = pm.Normal('β',  mu=0.0, sigma=10.0, shape=M)
        #β = pm.Cauchy('β', alpha=0, beta=cauchy_scale, shape=X.shape[1])
        β0 = pm.Normal("β0", 0, 10.)
        σ = pm.HalfNormal("σ", 2.5)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model



def wide(X_train, y_train, model_type):
    M = X_train.shape[1]
    
    with pm.Model() as model:
        cauchy_scale = pm.Uniform('cauchy_scale',0,0.4)
        #β = pm.Normal('β',  mu=0.0, sigma=10.0, shape=M)
        β = pm.Cauchy('β', alpha=0, beta=cauchy_scale, shape=M)
        β0 = pm.Normal("β0", 0, 1.)
        σ = pm.HalfNormal("σ", 2.5)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model
    

def laplace(X_train, y_train, model_type):
    #Laplace priors- put more weight near 0
    
    M = X_train.shape[1]
    with pm.Model() as model:
        λ = pm.Uniform('lambda',0,0.25)#Gamma('lambda', 1,1, shape=M) #, shape=M
        β_ =pm.Laplace('β_', mu=0.0, b=0.15 , shape=M)
        β =pm.Deterministic('β', λ *β_)
        β0 = pm.Normal("β0", 0, 0.85)
        
        σ = pm.HalfNormal("σ",1)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model



def laplace1(X_train, y_train, model_type):
    #Laplace priors- put more weight near 0
    
    M = X_train.shape[1]
    with pm.Model() as model:
        λ = pm.Uniform('lambda',0,5)#Gamma('lambda', 1,1, shape=M) #, shape=M
        β =pm.Laplace('β', mu=0.0, b=λ , shape=M)
        β0 = pm.Normal("β0", 0, 10.)
        σ = pm.HalfNormal("σ", 2.5)
        
        pred = pm.Data("pred", X_train, mutable=True)

        mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
        if model_type=='Linear':
            obs = pm.Normal("obs",mu , σ, observed=y_train)
        if model_type=='Poisson':
            obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
        if model_type=='NegativeBinomial':
            obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
        
    return model

def regularized_horseshoe1(X_train, y_train, model_type,proportion = 0.8):
    
    M = X_train.shape[1]
    N = X_train.shape[0]
    with pm.Model() as model:
            #priors
            proportion = pm.Uniform('proportion', 0,0.1)
            #print('fixed')
            #proportion=0.1
            #proportion = pm.Deterministic('proportion',pm.math.exp(proportion_))
            σ = pm.HalfNormal("σ", 0.5)
            #α = pm.Uniform('α', lower=0, upper=100)
            τ = pm.HalfStudentT("τ", 0.3, proportion * σ / np.sqrt(X_train.shape[1])) # replace the proportion- results dont vary much
            λ = pm.HalfStudentT("λ", 0.5, shape=X_train.shape[1])
            c2 = pm.InverseGamma("c2", alpha= 10,beta= 5)
            λ_ = λ * np.sqrt(c2 / (c2 + (τ**2) * (λ**2 ) ) )
            z = pm.Normal("z", 0., 1., shape=X_train.shape[1])
            β = pm.Deterministic("β", z * τ * λ_)
            β0 = pm.Normal("β0", 0, 1.)

            # Data
            pred = pm.Data("pred", X_train, mutable=True)
            y_obs = y_train

            mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
            
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ, observed=y_train)
            if model_type=='Poisson':
                obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
    return model


def regularized_horseshoe2(X_train, y_train, model_type,proportion = 0.8):
    
    M = X_train.shape[1]
    N = X_train.shape[0]
    with pm.Model() as model:
            #priors
            p0 = pm.DiscreteUniform('p0', 1,M-1)
            #print('fixed')
            #proportion=0.1
            proportion = pm.Deterministic('proportion',p0/(M-p0))
            σ = pm.HalfNormal("σ", 2.5)
            #α = pm.Uniform('α', lower=0, upper=100)
            τ = pm.HalfStudentT("τ", 2, proportion * σ / np.sqrt(X_train.shape[1])) # replace the proportion- results dont vary much
            λ = pm.HalfStudentT("λ", 5, shape=X_train.shape[1])
            c2 = pm.InverseGamma("c2", 1, 1)
            λ_ = λ * np.sqrt(c2 / (c2 + (τ**2) * (λ**2 ) ) )
            z = pm.Normal("z", 0., 0.5, shape=M)
            β = pm.Deterministic("β", z * τ * λ_)
            β0 = pm.Normal("β0", 0,2.5)

            # Data
            pred = pm.Data("pred", X_train, mutable=True)
            

            mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
            
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ_, observed=y_train)
            if model_type=='Poisson':
                obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
    return model


def regularized_horseshoe_x(X_train, y_train, model_type,proportion = 0.8):
    
    M = X_train.shape[1]
    N = X_train.shape[0]
    with pm.Model() as model:
            #priors
            proportion = pm.Uniform('proportion', 0,1)
            #print('fixed')
            #proportion=0.1
            #proportion = pm.Deterministic('proportion',pm.math.exp(proportion_))
            #σ = pm.HalfNormal("σ", 2.5)
            σ = pm.HalfNormal("σ", 10)
            #α = pm.Uniform('α', lower=0, upper=100)
            τ = pm.HalfStudentT("τ", 2, proportion * σ / np.sqrt(X_train.shape[1])) # replace the proportion- results dont vary much
            λ = pm.HalfStudentT("λ", 5, shape=X_train.shape[1])
            #τ = pm.HalfCauchy("τ",  proportion * σ / np.sqrt(X_train.shape[1])) # replace the proportion- results dont vary much
            #λ = pm.HalfCauchy("λ", 2, shape=X_train.shape[1])
            c2 = pm.InverseGamma("c2", 1, 1)
            λ_ = λ * np.sqrt(c2 / (c2 + (τ**2) * (λ**2 ) ) )
            z = pm.Normal("z", 0., 1., shape=X_train.shape[1])
            β = pm.Deterministic("β", z * τ * λ_)
            #β0= pm.Deterministic("β0", 0*z)
            #β0 = pm.Normal("β0", 0, 2.5)
            β0 = pm.Normal("β0", 0, 5)

            # Data
            pred = pm.Data("pred", X_train, mutable=True)
            
            #σ_ = pm.HalfNormal("σ_", 2.5)
            σ_ = pm.HalfNormal("σ_", 2.5)
            mu = pm.Deterministic('mu', β0+pm.math.dot(pred, β))
            
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ_, observed=y_train)
            if model_type=='Poisson':
                obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
    return model




def regularized_horseshoe(X_train, y_train, model_type,proportion = 0.8):
    
    M = X_train.shape[1]
    N = X_train.shape[0]
    
    prop =1
    if M>N:
        prop=((M-N)/M) *0.85
    with pm.Model() as model:
            #priors
            #proportion = pm.Uniform('proportion', 0,prop)
            #proportion = pm.Deterministic('proportion',pm.math.exp(proportion_))
            σ = pm.HalfNormal("σ", 5)
            #α = pm.Uniform('α', lower=0, upper=100)
            τ = pm.HalfStudentT("τ", 5, prop * σ / np.sqrt(X_train.shape[1])) # replace the proportion- results dont vary much
            λ = pm.HalfStudentT("λ", 5, shape=X_train.shape[1])
            c2 = pm.InverseGamma("c2", 1, 1)
            λ_ = λ * np.sqrt(c2 / (c2 + τ**2 * λ**2))
            z = pm.Normal("z", 0., 1., shape=X_train.shape[1])
            β = pm.Deterministic("β", z * τ * λ_)
            β0 = pm.Normal("β0", 0, 10)
            σ_ = pm.HalfNormal("σ_", 5)
            # Data
            pred = pm.Data("pred", X_train, mutable=True)
            #y_obs = y_train

            mu = pm.Deterministic('mu',β0+pm.math.dot(pred, β))
            
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ_, observed=y_train)
            if model_type=='Poisson':
                #lam =pm.Exponential('lam', lam=mu,shape=M-1)
                obs = pm.Poisson('obs', mu=pm.math.exp(pm.math.invlogit(mu) ), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
    return model


def regularized_horseshoe_actual(X_train, y_train, model_type,proportion = 0.8):
    
    M = X_train.shape[1]
    N = X_train.shape[0]
    with pm.Model() as model:
            #priors
            proportion = pm.Uniform('proportion', 0,1)
            #proportion = pm.Deterministic('proportion',pm.math.exp(proportion_))
            σ = pm.HalfNormal("σ", 2.5)
            #α = pm.Uniform('α', lower=0, upper=100)
            τ = pm.HalfStudentT("τ", 2, proportion * σ / np.sqrt(X_train.shape[1])) # replace the proportion- results dont vary much
            λ = pm.HalfStudentT("λ", 5, shape=X_train.shape[1])
            c2 = pm.InverseGamma("c2", 1, 1)
            λ_ = λ * np.sqrt(c2 / (c2 + τ**2 * λ**2))
            z = pm.Normal("z", 0., 1., shape=X_train.shape[1])
            β = pm.Deterministic("β", z * τ * λ_)
            β0 = pm.Normal("β0", 0, 10.)

            # Data
            pred = pm.Data("pred", X_train, mutable=True)
            #y_obs = y_train

            mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
            
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ, observed=y_train)
            if model_type=='Poisson':
                #lam =pm.Exponential('lam', lam=mu,shape=M-1)
                obs = pm.Poisson('obs', mu=pm.math.exp(pm.math.invlogit(mu) ), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
    return model

def regularized_horseshoe_new(X_train, y_train, model_type,proportion = 0.8):
    
    M = X_train.shape[1]
    N = X_train.shape[0]
    with pm.Model() as model:
            #priors
            proportion = pm.Uniform('proportion', 0,1)
            #proportion = pm.Deterministic('proportion',pm.math.exp(proportion_))
            σ = pm.HalfNormal("σ", 2.5)
            #α = pm.Uniform('α', lower=0, upper=100)
            τ = pm.HalfStudentT("τ", 2, proportion * σ / np.sqrt(X_train.shape[1])) # replace the proportion- results dont vary much
            λ = pm.HalfStudentT("λ", 5, shape=X_train.shape[1])
            
            c2 = pm.InverseGamma("c2", 1, 1)
            
            c = np.sqrt(c2)*2
            
            λ_ = λ * np.sqrt( (c**2) / ( (c**2) + (τ**2) * (λ**2)))
            z = pm.Normal("z", 0., 1., shape=X_train.shape[1])
            β = pm.Deterministic("β", z * τ * λ_)
            β0 = pm.Normal("β0", 0, 10.)

            # Data
            pred = pm.Data("pred", X_train, mutable=True)
            #y_obs = y_train

            mu = pm.Deterministic('mu',β0 + pm.math.dot(pred, β))
            
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ, observed=y_train)
            if model_type=='Poisson':
                #lam =pm.Exponential('lam', lam=mu,shape=M-1)
                obs = pm.Poisson('obs', mu=pm.math.exp(pm.math.invlogit(mu) ), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
    return model
def finnish_horseshoe1(X_train, y_train, model_type,proportion = 0.8):
        M= X_train.shape[1]
        N= X_train.shape[0]
        slab_scale = 3
        slab_scale_squared=slab_scale*slab_scale
        slab_degrees_of_freedom=25
        half_slab_df=slab_degrees_of_freedom*0.5
        alpha=3.0
        sigma=1.0

        prob_slope_is_meaninful=0.05

        with pm.Model() as model:
            proportion = pm.Uniform('proportion', 0,1)
            #proportion = pm.Deterministic('proportion',pm.math.exp(proportion_))
            
            σ = pm.HalfNormal("σ", 2.5)
            tau0 = (proportion) * (sigma / np.sqrt(1.0 * N))

            beta_tilde = pm.Normal('beta_tilde', mu=0, sigma=1, shape=M, testval=0.1)
            lamda = pm.HalfCauchy('lamda', beta=1, shape=M, testval=1.0)
            tau_tilde = pm.HalfCauchy('tau_tilde', beta=1, testval=0.1)
            c2_tilde = pm.InverseGamma('c2_tilde', alpha=half_slab_df, beta=half_slab_df, testval=0.5)


            tau=pm.Deterministic('tau', tau_tilde*tau0)
            c2=pm.Deterministic('c2',slab_scale_squared*c2_tilde)
            lamda_tilde =pm.Deterministic('lamda_tilde', pm.math.sqrt((c2 * pm.math.sqr(lamda) / (c2 + pm.math.sqr(tau) * pm.math.sqr(lamda)) ))) 

            β = pm.Deterministic('β', tau * lamda_tilde * beta_tilde)
            c=pm.Normal('c', mu=0.0, sigma=2.0, testval=1.0)


            sig=pm.Normal('sig', mu=0.0, sigma=2.0, testval=1.0)
            β0 = pm.Normal("β0", 0, 1.)

            pred = pm.Data("pred", X_train, mutable=True)

            mu= pm.Deterministic('mu', β0 +pm.math.dot(pred, β))
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ, observed=y_train)
            if model_type=='Poisson':
                obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
            return model
        


def finnish_horseshoe(X_train, y_train, model_type,proportion = 0.8):
        M= X_train.shape[1]
        N= X_train.shape[0]
        slab_scale = 3
        slab_scale_squared=slab_scale*slab_scale
        slab_degrees_of_freedom=25
        half_slab_df=slab_degrees_of_freedom*0.5
        alpha=3.0
        sigma=1.0

        prob_slope_is_meaninful=0.5

        with pm.Model() as model:
            proportion = pm.Uniform('proportion', 0,1)
            #proportion = pm.Deterministic('proportion',pm.math.exp(proportion_))
            
            σ = pm.HalfNormal("σ", 2.5)
            tau0 = (proportion) * (sigma / np.sqrt(1.0 * N))

            beta_tilde = pm.Normal('beta_tilde', mu=0, sigma=1, shape=M, testval=0.1)
            lamda = pm.HalfCauchy('lamda', beta=1, shape=M, testval=1.0)
            tau_tilde = pm.HalfCauchy('tau_tilde', beta=1, testval=0.1)
            c2_tilde = pm.InverseGamma('c2_tilde', alpha=half_slab_df, beta=half_slab_df, testval=0.5)


            tau=pm.Deterministic('tau', tau_tilde*tau0)
            c2=pm.Deterministic('c2',slab_scale_squared*c2_tilde)
            lamda_tilde =pm.Deterministic('lamda_tilde', pm.math.sqrt((c2 * pm.math.sqr(lamda) / (c2 + pm.math.sqr(tau) * pm.math.sqr(lamda)) ))) 

            β = pm.Deterministic('β', tau * lamda_tilde * beta_tilde)
            c=pm.Normal('c', mu=0.0, sigma=2.0, testval=1.0)


            sig=pm.Normal('sig', mu=0.0, sigma=2.0, testval=1.0)
            β0 = pm.Normal("β0", 0, 1.)

            pred = pm.Data("pred", X_train, mutable=True)

            mu= pm.Deterministic('mu', β0 +pm.math.dot(pred, β))
            if model_type=='Linear':
                obs = pm.Normal("obs",mu , σ, observed=y_train)
            if model_type=='Poisson':
                obs = pm.Poisson('obs', mu=pm.math.exp(mu), observed=y_train)
            if model_type=='NegativeBinomial':
                obs = pm.NegativeBinomial('obs', mu=pm.math.exp(mu),alpha=σ, observed=y_train)
            
            return model