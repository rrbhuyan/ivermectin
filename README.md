# Quantifying Regional Variation in Ivermectin Use During the Pandemic using Regularized Synthetic Controls

## Overview
This project analyzes the overconsumption of the antiparasitic drug Ivermectin (IVM) during the COVID-19 pandemic. Using regularized synthetic controls, we estimate state-wise elevated IVM claims and assess their persistence even after COVID-19 vaccines became widely available and multiple federal countermeasures were implemented.

## Key Features
- Analysis of weekly prescription claims data in the US over 36 months
- Use of continuous spike-and-slab shrinkage prior to regularized synthetic controls
- Estimation of state-wise elevated claims of IVM 
- Assessment of the persistence of these elevated levels 
- Analysis of regional variation across different states

## Data Sources
- Primary: GoodRx prescription claims data
- Secondary: COVID-19 case counts, 2016 US presidential election results, U.S. Census Data

## Methodology
- Synthetic control (SC) methodology
- Bayesian SC method with spike-and-slab shrinkage priors
- Regression analysis for susceptibility study

## Key Findings
- Modest, short-lived increase in IVM consumption when early interest started
- Peak surge in aggregate US consumption was 16 times above the counterfactual
- Political affiliation of states significantly explains variations in impact
- Persistence of these effects despite countermeasures

## Project Structure
The project is organized into the following directories:





1. `Processed Data/`: Stores processed and output data (Access to requested [here](https://www.dropbox.com/scl/fo/xpm82btzeucyvpzp0wnza/AOGuFzMf401Nbkgv2kWHiEo?rlkey=7dotyoromnf706lmw393xpmfq&st=3uh65oud&dl=0]) )
   - Aggregate data ('state_week/modeling_data/')
   - Preprocessed data for synthetic control modeling ('synth_data/ivermectin/')
   - Outputs of modeling: graphs and results ('synth_data/ivermectin/Graphs' and 'synth_data/ivermectin/Results', respectively)
   - Preprocessed data for regression modeling ('regression/')

2. `Code Data Prep/`: Contains data preparation scripts
   - Scripts for synthetic modeling preprocessing
   - Scripts for regression analysis preprocessing

3. `Code Synth Modeling/`: Contains synthetic modeling code

4. `Code Model Free Ivm/`: Contains code for generating model-free graphs

5. `Code Model Based Ivm/`: Contains code for generating model-based graphs

6. `Code Regression/`: Contains code for regression analysis

## Requirements

1. To Replicate Model Free Graphs and Model-Based Graphs (Figures 1-8):

Python 3.x
Libraries:

- isoweek
- Matplotlib
- Numpy
- os
- Pandas
- seaborn

2. For Placebo Graph (Figure 9):

All requirements from the previous section
Additional Python libraries:

- az
- gc
- glob
- itertools
- random

3. For Regression Tables (Tables 3 and 4):

R (latest version recommended)
R libraries:

- lmtest
- sandwich
- multiwayvcov
- stargazer
- fastDummies
- IRdisplay
- car

## Usage

1. Model Free Graphs:

- To replicate Figure 1, use notebook 'IVM Indexed National.ipynb' located at 'Code Model Free Ivm/'
- To replicate Figure 2, use notebook 'IVM Indexed Trump.ipynb' located at 'Code Model Free Ivm/'
- To replicate Figure 3, use notebook 'IVM Indexed Texas California.ipynb' located at 'Code Model Free Ivm/'
- To replicate Figure 4, use notebook 'IVM Map Graphs.ipynb' located at 'Code Model Free Ivm/'

2. Model-Based Graphs:

- To replicate Figure 5, use notebook 'IVM Map Graphs.ipynb' located at 'Code Model Based Ivm/'
- To replicate Figure 6 and Figure 10, use notebook 'IVM Indexed National.ipynb' located at 'Code Model Based Ivm/'
- To replicate Figure 7 and Figure 11, use notebook 'IVM Indexed Trump.ipynb' located at 'Code Model Based Ivm/'
- To replicate Figure 8, use notebook 'IVM Indexed Texas California.ipynb' located at 'Code Model Based Ivm/'

3. Placebo Graph:

- To replicate Figure 9, use notebook 'IVM Synthetic Controls Indexed National Placebo.ipynb' located at 'Code Synth Modeling/'

4. Regression Tables:

- To replicate Table 3 and Table 4, use notebook 'IVM Regression Tables.ipynb' located at 'Code Regression/'


## Contributors
- Dinesh Puranam, Ivan Belov, Rashmi Ranjan Bhuyan, Shantanu Dutta, Jeroen van Meijgaard, Reetabrata Mookherjee, and Gourab Mukherjee

## Contact
Dinesh Puranam, puranam@marshall.usc.edu
