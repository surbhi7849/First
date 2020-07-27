import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow_probability as tfp
import collections
import numpy as np
import pdb
import sys
import matplotlib.pyplot as plt
import arviz as az
from math import sqrt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.validation import check_array as check_arrays
import mlflow
import copy
import warnings
import BayesFramework.plot_utils as pl_ut
warnings.filterwarnings("ignore")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

thismodule = sys.modules[__name__]

tfd = tfp.distributions
tfb = tfp.bijectors

# @tf.function
def affine(x, kernel_diag, bias=tf.zeros([])):
    """`kernel_diag * x + bias` with broadcasting."""
    kernel_diag = tf.ones_like(x) * kernel_diag
    bias = tf.ones_like(x) * bias
    return x * kernel_diag + bias

def rem_specialchar(string):
    return(string.translate ({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=+_"}))
        
def rem_specialchar_array(string_array):
    '''Remaoving Special characters from the Column names'''
    return_array = []
    for string in string_array:
        if string == "global_intercept":
            return_array = return_array + [string]
        elif "intercept_" in string:
            string = string.replace('intercept_','')
            return_array = return_array + ["intercept_"+rem_specialchar(string.replace('intercept_',''))]
        else:
            return_array = return_array + [rem_specialchar(string)]
    return(return_array)

class BayesianEstimation():
    def __init__(self, data_df_original, model_config_df_original, framework_config_df,pickle_file=''):
        '''
    
    This class implements partially pooled heirarchical bayesian model for a given set of random and fixed effects
    ref: 
    
    inputs:
    
    data_df: Pandas DataFrame
    model_config_df:Pandas DataFrame made from config
                    Include_IDV	: 1/0 
                    RandomEffect: Random Effect=1, Fixed Effect=1
                    RandomFactor: Grouping Column
                    mu_d: distribution
                    mu_d_loc_alpha:<int/float>
                    mu_d_scale_beta	:<int/float>
                    sigma_d: distribution
                    sigma_d_loc_alpha: <int/float>
                    sigma_d_scale_beta: <int/float>	
                    mu_bijector	sigma_bijector: Identity/Exp	
                    fixed_d	fixed_d_loc_alpha:<int/float>	
                    fixed_d_scale_beta:<int/float>	
                    fixed_bijector:Identity/Exp

    framework_config_df:Pandas DataFrame made from config
                    objective:regression
                    sampler: hmc/nuts
                    num_chains: <int> number of mcmc chains (passed to hmc.sample_chains())
                    num_results: <int> number of results 
                    num_burnin_steps: <float> parameter to TFP Hmc sampler
                    num_leapfrog_steps: <float> parameter to TFP Hmc sampler
                    hmc_step_size: <float> parameter to TFP Hmc sampler
    output:
    
    Summary: 1) estimates of all random effect params with Rhat values group wise
             2) estimates of all fixed effect paramas with Rhat values on population 
    Traceplots: as per the arguments
    full sample trace: as per the arguments'''
    
        data_df=data_df_original.copy()
        model_config_df=model_config_df_original.copy()
        
        if('global_intercept' not in data_df.columns):
            data_df['global_intercept'] = 1
        
        self.join_dist_list = []
        t = time.localtime()
        self.dt= time.strftime('%b-%d-%Y_%H%M', t)
        
        
        model_config_df = model_config_df[model_config_df['Include_IDV']==1].copy()
        model_config_df.drop('Include_IDV',axis=1,inplace=True)
        model_config_df.loc[:,'DV'] = rem_specialchar_array(model_config_df['DV'])
        model_config_df.loc[:,'IDV'] = rem_specialchar_array(model_config_df['IDV'])
        #model_config_df.loc[:,'RandomFactor'][model_config_df['RandomFactor'].isna()] = ""
        model_config_df.loc[:,'RandomFactor'] = np.where(model_config_df['RandomFactor'].isna(), "", model_config_df['RandomFactor'])
        model_config_df.loc[:,'RandomFactor'] = rem_specialchar_array(model_config_df['RandomFactor'])
        self.model_config_df = model_config_df
        
        
        data_df.columns = rem_specialchar_array(data_df.columns)
        mapped_list=[]
        for column in [s for s in model_config_df['RandomFactor'].unique() if s not in [np.nan,'']]:
            column_name = sorted(data_df[column].unique())
            data_df[column+'_original'] = data_df[column]
            data_df[column] = data_df[column].astype(pd.api.types.CategoricalDtype(categories=column_name)).cat.codes
            data_df[column] = data_df[column].astype(int)
            mapped_list.append(column)
            mapped_list.append(column+'_original')
        self.data_df = data_df
        
        self.ModelParamsTuple = None
        self.SamplesTuple = None
        self.duplicate_data=None
        
        
        hyperparam_dict = {}
        for _,row in framework_config_df.iterrows():
            if(row['TagName'] not in ['objective','add_globalintercept']):
                hyperparam_dict[row['TagName']] = row['Value']
        spec_dict = {
                'dv':model_config_df['DV'].iloc[0],
                'idvs':[item.replace("intercept_","") for item in list(model_config_df['IDV'].drop_duplicates())],
                'group_cols':[s for s in model_config_df['RandomFactor'].unique() if s not in [np.nan,'']],
                'hyperparams':hyperparam_dict
                }
        self.spec_dict = spec_dict
        
        data_df = self.data_df
        self.mapped_df=data_df[mapped_list]
        data_df = data_df[list(set([self.spec_dict['dv']]+self.spec_dict['idvs']+self.spec_dict['group_cols']))]
        data_df = data_df.dropna()
        self.data_df = data_df
        
        dist_param = {}
        num_chains = spec_dict['hyperparams']['num_chains']
        for _,row in model_config_df.iterrows():
            if(row['IDV']=='global_intercept'):
                var = 'global_intercept'
                tmp_param = {
                        'fixed_d' : row['fixed_d'],
                        'fixed_d_loc' : row['fixed_d_loc_alpha'],
                        'fixed_d_scale' : row['fixed_d_scale_beta'],
                        'fixed_bijector' : row['fixed_bijector']
                        }
                dist_param[var] = tmp_param
                
            elif("intercept" in row['IDV']):
                var = row['RandomFactor']
                tmp_param = {
                        'mu_d' : row['mu_d'],
                        'mu_d_loc' : row['mu_d_loc_alpha'],
                        'mu_d_scale' : row['mu_d_scale_beta'],
                        'sigma_d' : row['sigma_d'],
                        'sigma_d_loc' : row['sigma_d_loc_alpha'],
                        'sigma_d_scale' : row['sigma_d_scale_beta'],
                        'mu_bijector' : row['mu_bijector'],
                        'sigma_bijector' : row['sigma_bijector']
                        }
                dist_param[var] = tmp_param
                
            else:
                var = row['IDV']
                if(row['RandomEffect']==1):
                    tmp_param = {
                            'mu_d' : row['mu_d'],
                            'mu_d_loc' : row['mu_d_loc_alpha'],
                            'mu_d_scale' : row['mu_d_scale_beta'],
                            'sigma_d' : row['sigma_d'],
                            'sigma_d_loc' : row['sigma_d_loc_alpha'],
                            'sigma_d_scale' : row['sigma_d_scale_beta'],
                            'mu_bijector' : row['mu_bijector'],
                            'sigma_bijector' : row['sigma_bijector']
                            }
                else:
                    tmp_param = {
                            'fixed_d' : row['fixed_d'],
                            'fixed_d_loc' : row['fixed_d_loc_alpha'],
                            'fixed_d_scale' : row['fixed_d_scale_beta'],
                            'fixed_bijector' : row['fixed_bijector']
                            }
                dist_param[var] = tmp_param
        dist_bij= copy.deepcopy(dist_param) 
        self.dist_bij= dist_bij
        for key1,value1 in dist_param.items():
            for key2,value2 in value1.items():
                if("bijector" in key2):
                    if(value2=='Identity'):
                        dist_param[key1][key2] = tfb.Identity()
                    elif(value2=='Exp'):
                        dist_param[key1][key2] = tfb.Exp()
        self.dist_param = dist_param
        
        random_effect=[]
        fixed_effect=[]
        
        for _, row in self.model_config_df.iterrows():
            if row['RandomEffect'] == 0:
                if(row['IDV'] != 'global_intercept'):
                    fixed_effect.append(row['IDV'])
            else:
                check="intercept_"+row['RandomFactor']
                l2=[]
                if(row['IDV']!=check):
                    l2.append(row['IDV'])
                    l2.append(row['RandomFactor'])
                    random_effect.append(l2)
        self.random_effect= random_effect
        self.fixed_effect= fixed_effect
        
        mlflow.start_run(run_name= "_"+self.dt)
        data={}
        data['random_effects']=self.random_effect
        data['fixed_effects']= self.fixed_effect
        data['group_column']= self.spec_dict['group_cols']
        
        mlflow.log_param('Hyperparams',self.spec_dict['hyperparams'])
        mlflow.log_param('Data', data)
        mlflow.log_param('distribution_and_bijectors',self.dist_bij)
        
        if pickle_file:

            pickle_off = open(pickle_file, 'rb')
            all_data={}
            all_data=pickle.load(pickle_off)
            samples =all_data['samples']
            acceptance_probs=all_data['acceptance_probs']
            
            self.join_dist_list=all_data['join_dist_list']
            self.ModelParamsTuple = collections.namedtuple('ModelParams',self.join_dist_list[:-1])
            print("got model param tuple")
            self.SamplesTuple = self.ModelParamsTuple._make(samples)
            print("got samples tuple")
             
            
            print('Acceptance Probabilities: ', acceptance_probs.numpy())
            try:
                
                for var in self.join_dist_list[:-1]:
                    if "mu" in var or "sigma" in var:
                        print('R-hat for ', var, '\t: ',tfp.mcmc.potential_scale_reduction(getattr(self.SamplesTuple, var)).numpy())
                        
            except Exception as e:
                print("------Error while calculating r-hat-----")
                print(e)
        

        
    
    def get_tensor_var(self,variable):
        return tf.convert_to_tensor(self.data_df[variable], dtype=tf.float32)
    
    def get_tensor_cat(self,variable):
        return tf.convert_to_tensor(self.data_df[variable])
    
    def create_Normal_dist(self,loc,scale):
        return("tfd.Normal(loc="+str(loc)+", scale="+str(scale)+")")
    
    def create_StudentT_dist(self,loc,scale):
        return("tfd.StudentT(loc="+str(loc)+", scale="+str(scale)+", df=3)")
                         
    
    def create_HalfCauchy_dist(self,loc,scale):
        return("tfd.HalfCauchy(loc="+str(loc)+", scale="+str(scale)+")")
        
    def create_LogNormal_dist(self,loc,scale):
        return("tfd.LogNormal(loc="+str(loc)+", scale="+str(scale)+")")
    
    def create_Gamma_dist(self,con,rate):
        return ("tfd.Gamma(concentration="+str(con)+", rate="+str(rate)+")")
        
    def select_distribution(self,dist_type,val1,val2):
        dist_list=["Normal","HalfCauchy","LogNormal",'StudentT',"Gamma"]
        try:
            if dist_type not in dist_list:
                raise ValueError(dist_type)
        except ValueError:
            print(dist_type, "distribution is not allowed.")
        if(dist_type=="Normal"):
            return self.create_Normal_dist(val1,val2)
        elif(dist_type=="HalfCauchy"):
            return self.create_HalfCauchy_dist(val1,val2)
        elif(dist_type=="LogNormal"):
            return self.create_LogNormal_dist(val1,val2)
        elif(dist_type=="StudentT"):
            return self.create_StudentT_dist(val1,val2)
        elif(dist_type=="Gamma"):
            return self.create_Gamma_dist(val1,val2)
    
    def preprocess(self):
        
        for key, value in self.spec_dict.items():
            if key == 'dv':
                try:
                    setattr( thismodule, value, self.get_tensor_var(value) )
                except:
                    print('An error occurred while converting dv into tensor.')
            elif key == 'idvs':
                for idv in value:
                    idv_1 = idv
                    try:
                        if idv_1 not in self.spec_dict['group_cols']:
                            setattr( thismodule, idv_1, self.get_tensor_var(idv_1) )
                        else:
                            setattr( thismodule, idv_1, self.get_tensor_cat(idv_1) )
                    except:
                        print('An error occurred while converting idvs into tensor.')

        pass


    def global_intercept_param(self):
        self.join_dist_list.append("fixed_slope_global_intercept")
        return [
                self.select_distribution(self.dist_param['global_intercept']['fixed_d'],self.dist_param['global_intercept']['fixed_d_loc'],self.dist_param['global_intercept']['fixed_d_scale'])
                ]

    def random_intercept_param(self, group_variable, group_count):
        self.join_dist_list.extend([
            "mu_intercept_"+group_variable,
            "sigma_intercept_"+group_variable,
            "intercept_"+group_variable])
        return [
               self.select_distribution(self.dist_param[group_variable]['mu_d'],self.dist_param[group_variable]['mu_d_loc'],self.dist_param[group_variable]['mu_d_scale']),    # mu_intercept : hyper-prior : TODO update it from user input
               self.select_distribution(self.dist_param[group_variable]['sigma_d'],self.dist_param[group_variable]['sigma_d_loc'],self.dist_param[group_variable]['sigma_d_scale']),  # sigma_intercept: hyper-prior : TODO update it from user input
               "lambda {0}, {1}: tfd.Independent(tfd.Normal(loc=affine(tf.ones([{2}]), {3}[..., tf.newaxis]),scale=tf.transpose({4}*[{5}])),reinterpreted_batch_ndims=1)".format(
                       "sigma_intercept_"+group_variable,
                       "mu_intercept_"+group_variable,
                       group_count,
                       "mu_intercept_"+group_variable,
                       "sigma_intercept_"+group_variable
                       )
               ]
        
    def fixed_slope_param(self, variable):
        self.join_dist_list.extend([
            "fixed_slope_"+variable        
        ])
        return [
                self.select_distribution(self.dist_param[variable]['fixed_d'],self.dist_param[variable]['fixed_d_loc'],self.dist_param[variable]['fixed_d_scale'])
                ]

    def random_slope_param(self, variable, group_variable, group_count):
        self.join_dist_list.extend([
            "mu_slope_"+variable+"_"+group_variable,
            "sigma_slope_"+variable+"_"+group_variable,
            "slope_"+variable+"_"+group_variable]
            )
        return [
               self.select_distribution(self.dist_param[variable]['mu_d'],self.dist_param[variable]['mu_d_loc'],self.dist_param[variable]['mu_d_scale']), # mu_slope : hyper-prior : TODO update it from user input
               self.select_distribution(self.dist_param[variable]['sigma_d'],self.dist_param[variable]['sigma_d_loc'],self.dist_param[group_variable]['sigma_d_scale']), # sigma_slope: hyper-prior : TODO update it from user input
               "lambda {0},{1}: tfd.Independent(tfd.Normal(loc=affine(tf.ones([{2}]), {3}[..., tf.newaxis]),scale=tf.transpose({4}*[{5}])),reinterpreted_batch_ndims=1)".format(
                       "sigma_slope_"+variable+"_"+group_variable,
                       "mu_slope_"+variable+"_"+group_variable,
                       group_count,
                       "mu_slope_"+variable+"_"+group_variable,
                       "sigma_slope_"+variable+"_"+group_variable
                       )                
               ]
    def random_gamma_intercept_param(self, group_variable, group_count):
        self.join_dist_list.extend([
            "mu_intercept_"+group_variable,
            "sigma_intercept_"+group_variable,
            "intercept_"+group_variable])
        return [
               self.select_distribution(self.dist_param[group_variable]['mu_d'],self.dist_param[group_variable]['mu_d_loc'],self.dist_param[group_variable]['mu_d_scale']),    # mu_intercept : hyper-prior : TODO update it from user input
               self.select_distribution(self.dist_param[group_variable]['sigma_d'],self.dist_param[group_variable]['sigma_d_loc'],self.dist_param[group_variable]['sigma_d_scale']),  # sigma_intercept: hyper-prior : TODO update it from user input
               "lambda {0}, {1}: tfd.Independent(tfd.Gamma(concentration=affine(tf.ones([{2}]), {3}[..., tf.newaxis]),rate=tf.transpose({4}*[{5}])),reinterpreted_batch_ndims=1)".format(
                       "sigma_intercept_"+group_variable,
                       "mu_intercept_"+group_variable,
                       group_count,
                       "mu_intercept_"+group_variable,
                       group_count,
                       "sigma_intercept_"+group_variable
                       )
               ]
    
    def random_gamma_slope_param(self, variable, group_variable, group_count):
        self.join_dist_list.extend([
            "mu_slope_"+variable+"_"+group_variable,
            "sigma_slope_"+variable+"_"+group_variable,
            "slope_"+variable+"_"+group_variable]
            )
        return [
               self.select_distribution(self.dist_param[variable]['mu_d'],self.dist_param[variable]['mu_d_loc'],self.dist_param[variable]['mu_d_scale']), # mu_slope : hyper-prior : TODO update it from user input
               self.select_distribution(self.dist_param[variable]['sigma_d'],self.dist_param[variable]['sigma_d_loc'],self.dist_param[group_variable]['sigma_d_scale']), # sigma_slope: hyper-prior : TODO update it from user input
               "lambda {0}, {1}: tfd.Independent(tfd.Gamma(concentration=affine(tf.ones([{2}]), {3}[..., tf.newaxis]),rate=tf.transpose({4}*[{5}])),reinterpreted_batch_ndims=1)".format(
                       "sigma_slope_"+variable+"_"+group_variable,
                       "mu_slope_"+variable+"_"+group_variable,
                       group_count,
                       "mu_slope_"+variable+"_"+group_variable,
                       group_count,
                       "sigma_slope_"+variable+"_"+group_variable
                       )                
               ]
        
    def error(self):
        self.join_dist_list.append("sigma_target")
        return [
                self.create_HalfCauchy_dist(0,5) #TODO input from user 
        ]

    def target(self, args, loc, scale):
        return [
            "lambda {}:  tfd.Independent(tfd.Normal(loc={},scale={}))".format(args,loc,scale)
        ]

    def create_joint_dist_seq(self):
        final_list = []
        target_likelihood_loc = []
        for _, row in self.model_config_df.iterrows():
            if 'intercept' in row['IDV']:
                if row['RandomEffect'] == 1:
                    # add random intercept term
                    random_factor = row['RandomFactor']
                    group_levels = tf.cast(self.data_df[random_factor].nunique(),tf.int32)
                    if row['mu_d']=='Gamma' or row['sigma_d']=='Gamma':
                        final_list.extend( self.random_gamma_intercept_param(random_factor, group_levels) )
                    else:
                        final_list.extend( self.random_intercept_param(random_factor, group_levels) )
                    # likelihood loc term
                    target_likelihood_loc.append("tf.gather({0}, {1}, axis=-1)".format(
                        "intercept_"+random_factor,
                        "{}".format(random_factor)
                        )
                        )

                else:
                    # add global/fixed intercept term
                    final_list.extend( self.global_intercept_param() )
                    target_likelihood_loc.append("affine({0}, {1}[..., tf.newaxis])".format(
                        "{}".format(row['IDV']),
                        "fixed_slope_{}".format(row['IDV'])
                        )
                        )

            else:
                if row['RandomEffect'] == 1:
                    # add random slope term
                    random_factor = row['RandomFactor']
                    group_levels = tf.cast(self.data_df[random_factor].nunique(),tf.int32)
                    if row['mu_d']=='Gamma' or row['sigma_d']=='Gamma':
                        final_list.extend( self.random_gamma_slope_param( row['IDV'], random_factor, group_levels ) )
                    else:
                        final_list.extend( self.random_slope_param( row['IDV'], random_factor, group_levels ) )
                    # likelihood loc term
                    target_likelihood_loc.append("affine({0}, tf.gather({1}, {2}, axis=-1))".format(
                        "{}".format(row['IDV']),
                        "slope_"+row['IDV']+"_"+random_factor,
                        "{}".format(random_factor)
                        )
                        )
                else:
                    # add fixed slope term
                    final_list.extend( self.fixed_slope_param(row['IDV']) )
                    target_likelihood_loc.append("affine({0}, {1}[..., tf.newaxis])".format(
                        "{}".format(row['IDV']),
                        "fixed_slope_{}".format(row['IDV'])
                        )
                        )
        
        # insert error term
        final_list.extend(self.error())

        # insert target likelihood
        args = ",".join(self.join_dist_list[::-1])
        loc = "(" + " + ".join(target_likelihood_loc) + ")"
        scale = str(self.join_dist_list[-1])

        final_list.extend(self.target(args, loc, scale))

        # finally add target variable in seq list
        self.join_dist_list.append(self.model_config_df['DV'].iloc[0])


        return final_list
    
    def create_joint_dist_seq_func(self):
        func_string = "def joint_dist_model({}):\n    return tfd.JointDistributionSequential([{}])".format(
            ",".join( self.spec_dict['idvs'] ),
            ",".join(self.create_joint_dist_seq())
            )
        return func_string
    
    def get_resolve_graph_of_JDS(self):
        func_string = "print(joint_dist_model.resolve_graph())"
        return func_string

    def create_log_prob_func(self):
        func_string = "@tf.function\ndef joint_dist_model_log_prob({}):\n    return joint_dist_model({}).log_prob([{}])".format(
            ",".join(self.join_dist_list[:-1]),
            ",".join( self.spec_dict['idvs'] ),
            ",".join(self.join_dist_list)
            )
        return func_string


    def select_sampling_technique(self,condition,initial_state,unconstraining_bijectors,seed):
        try:
            if(condition=="hmc"):
                hmc_step_size = self.spec_dict['hyperparams']['hmc_step_size']
                num_leapfrog_steps = self.spec_dict['hyperparams']['num_leapfrog_steps']
                
                sampler = tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=joint_dist_model_log_prob,
                        num_leapfrog_steps=num_leapfrog_steps,
                        step_size=hmc_step_size)
                kernel = tfp.mcmc.TransformedTransitionKernel(
                        inner_kernel=sampler,
                        bijector=unconstraining_bijectors)
                
            else:
                num_chains = self.spec_dict['hyperparams']['num_chains']
                hmc_step_size = self.spec_dict['hyperparams']['hmc_step_size']
                num_burnin_steps = self.spec_dict['hyperparams']['num_burnin_steps']
                
                target_accept_prob = .8
                num_adaptation_steps = int(0.8 * num_burnin_steps)
                
                step_size = [tf.fill([num_chains] + [1] * (len(s.shape) - 1), tf.constant(hmc_step_size, np.float32)) for s in initial_state]
                sampler = tfp.mcmc.NoUTurnSampler(
                        joint_dist_model_log_prob,
                        step_size=step_size,seed=seed)
                kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                        tfp.mcmc.TransformedTransitionKernel(
                                inner_kernel=sampler,
                                bijector=unconstraining_bijectors),
                        target_accept_prob=target_accept_prob,
                        num_adaptation_steps=num_adaptation_steps,
                        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(inner_results=pkr.inner_results._replace(step_size=new_step_size)),
                        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
                )
        except Exception as e:
            print("---Error occured while selecting sampler-----")
            print(e)
        
        return(kernel)
    
    def get_ready(self,seed):
        num_chains = tf.cast(self.spec_dict['hyperparams']['num_chains'],tf.int32)
        num_results = tf.cast(self.spec_dict['hyperparams']['num_results'],tf.int32)
        num_burnin_steps = tf.cast(self.spec_dict['hyperparams']['num_burnin_steps'],tf.int32)
        
        str_sample='(joint_dist_model({}).sample(seed=seed)[:-1])'.format(','.join(self.spec_dict['idvs']))
        initial_state_=   eval(str_sample)
                
        # TODO update this to read from config
        initial_state = []
        unconstraining_bijectors = []
        i=0
        for element in self.join_dist_list[:-1]:
            element_1 = element
            if(element_1 != "sigma_target"):
                for repl in ['mu_','sigma_','slope_','fixed_','intercept_']:
                    element_1 = element_1.replace(repl,'')
                if("intercept" not in element_1):
                    for repl in ['_'+s for s in self.spec_dict['group_cols']]:
                        element_1 = element_1.replace(repl,'')
            if element=="sigma_target":
                initial_state.append(tf.ones([num_chains], name='init_'+element)*initial_state_[i]); i += 1
                unconstraining_bijectors.append(tfb.Exp())
            elif element=="global_intercept":
                initial_state.append(tf.ones([num_chains], name='init_'+element)*initial_state_[i]); i += 1
                unconstraining_bijectors.append(self.dist_param[element_1]['fixed_bijector'])
            elif "sigma" in element:
                initial_state.append(tf.ones([num_chains], name='init_'+element)*initial_state_[i]); i += 1
                unconstraining_bijectors.append(self.dist_param[element_1]['sigma_bijector'])
            elif "mu" in element:
                initial_state.append(tf.ones([num_chains], name='init_'+element)*initial_state_[i]); i += 1
                unconstraining_bijectors.append(self.dist_param[element_1]['mu_bijector'])
            elif "fixed" in element:
                initial_state.append(tf.ones([num_chains], name='init_'+element)*initial_state_[i]); i += 1
                unconstraining_bijectors.append(self.dist_param[element_1]['fixed_bijector'])
            else:
                group_var_name = [s for s in element.split('_') if s in self.spec_dict['group_cols']][0]
                group_levels = tf.cast(self.data_df[group_var_name].nunique(),tf.int32) #EDIT
                initial_state.append(tf.ones([num_chains, group_levels ], name='init_'+element)*initial_state_[i]); i += 1
                unconstraining_bijectors.append(tfb.Identity()) # TODO update this to read from config - incomplete
        
        sampling_technique = self.spec_dict['hyperparams']['sampler']
        kernel = self.select_sampling_technique(sampling_technique,initial_state,unconstraining_bijectors,seed=seed)
        return sampling_technique, kernel, initial_state, unconstraining_bijectors, num_chains, num_results, num_burnin_steps

    @tf.function(experimental_compile=True)
    def sample_model(self, sampling_technique, kernel, initial_state, unconstraining_bijectors, num_chains, num_results, num_burnin_steps):
        """Samples from the model."""
        

        
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=kernel)

        
        if(sampling_technique=="hmc"):
            acceptance_probs = tf.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, tf.float32), axis=0)
            print("using hmc sampling to get acceptance prob")
        else:
            acceptance_probs = tf.reduce_mean(tf.cast(kernel_results.inner_results.inner_results.is_accepted, tf.float32), axis=0)
            print("using nuts sampling to get acceptance prob")
        
        return samples, acceptance_probs
    
    def train(self,fixed_seed=123):
        '''
        To create the Joint Distribution Sequence and Get the Model Metrics Results after Sampling 
        
        '''
        self.preprocess()
        print("preprocess done")
        try:
            exec(self.create_joint_dist_seq_func(), globals())
            print("creating joint distribution sequential")
        except Exception as e:
            print("---error occured while executing create_joint_dist_seq_func----")
            print(e)
        
        try:
            exec(self.create_log_prob_func(), globals())

        except Exception as e:
            print("----error occured while executing create_log_prob_func----")
            print(e)
        try:
            s='print(joint_dist_model({}).resolve_graph())'.format(','.join(self.spec_dict['idvs']))
            print('Printing Resolve Graph function')
            eval(s)
        except Exception as e:
            print("----error occured while executing resolve graph----")
            print(e)
        
       
        sampling_technique, kernel, initial_state, unconstraining_bijectors, num_chains, num_results, num_burnin_steps = self.get_ready(seed=fixed_seed)

        samples, acceptance_probs = self.sample_model(sampling_technique, kernel, initial_state, unconstraining_bijectors, num_chains, num_results, num_burnin_steps)

        
        
        
        self.ModelParamsTuple = collections.namedtuple('ModelParams',self.join_dist_list[:-1])

        self.SamplesTuple = self.ModelParamsTuple._make(samples)
        print("Creating samples tuple")
        all_data={}
        all_data['samples']=samples
        all_data['acceptance_probs']=acceptance_probs
        all_data['join_dist_list']=self.join_dist_list
        

        #pickling samples
        if not os.path.exists('output/Samples/'):
            os.makedirs('output/Samples/')
        pickling_on = open("output/Samples/Samples"+self.dt+".pickle","wb")
        pickle.dump(all_data, pickling_on)
        
        pickling_on.close()
        mlflow.log_artifact('output/Samples/Samples'+self.dt+'.pickle')
        
        # summary={}
        print('Acceptance Probabilities: ', acceptance_probs.numpy())
        try:
            #summary['Acceptance Probabilities']={0:acceptance_probs.numpy()}
            for var in self.join_dist_list[:-1]:
                if "mu" in var or "sigma" in var:
                    print('R-hat for ', var, '\t: ',tfp.mcmc.potential_scale_reduction(getattr(self.SamplesTuple, var)).numpy())
                    
        except Exception as e:
            print("------Error while calculating r-hat-----")
            print(e)
        
        pass

    def saving_model_metrics(self,config_file_name,output_folder_path="output/bayesian_model_train_summary/"):
        ''' 
    to save results and  trace
    
    out_folder_path: path where the outputs are to be saved
'''
        
        
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        print("="*80)
        print("Saving Model Metrics and TracePlots")
        print("="*80)
        #self.plot_alldimensions(output_folder_path=output_folder_path)
        #self.plot_posterior_allvars(output_folder_path=output_folder_path)
        pl=pl_ut.Plot(self.join_dist_list, self.SamplesTuple, self.mapped_df, self.dt )
        pl.plot_alldimensions(output_folder_path=output_folder_path)
        pl.plot_posterior_allvars(output_folder_path=output_folder_path)
        
        mlflow.log_artifact(config_file_name)
        mlflow.log_artifact('output/bayesian_model_train_summary/'+self.dt+'plot_trace_'+self.dt+'.pdf')
        mlflow.log_artifact('output/bayesian_model_train_summary/'+self.dt+'Group_Summary.xlsx')
        mlflow.log_artifact('output/bayesian_model_train_summary/'+"plot_posterior_"+self.dt+".pdf")
        mlflow.end_run()
        pass

    
    def reduce_samples(self,var_samples, reduce_fn):        
        """Reduces across leading two dims using reduce_fn. """
        try:
            if isinstance(var_samples, tf.Tensor):
                var_samples = var_samples.numpy() 
            var_samples = np.reshape(var_samples, (-1,) +  var_samples.shape[2:])
            
        except Exception as e:
            print("----Error while reducing sample----")
            print(e)
        
        return np.apply_along_axis(reduce_fn, axis=0, arr=var_samples)
    

    def sample_mean(self,samples):
        return self.reduce_samples(samples, np.mean)
     
    
    def predict(self,data_df,output_folder_path='output/bayesian_model_prediction/'): 
        ''' To Predict the values of the target variable 
            Saves the result summary for various groups and Predicted values with original dataset
        '''
        self.L_intercept=[]
        self.L_slope=[]
        print("="*80)
        print("Running prediction")
        print("="*80)        
        for var in self.join_dist_list[:-1]:
            if "mu" not in var and "sigma" not in var:
                if "intercept" in var:
                    self.L_intercept.append(var)
                elif "slope" in var:
                    self.L_slope.append(var)
        LinearEstimates = collections.namedtuple('LinearEstimates',self.L_intercept+self.L_slope)
        L=self.L_intercept+self.L_slope
        s=''
        for var in L:
            s=s+'''self.sample_mean(getattr(self.SamplesTuple,"'''+var+'''")),'''
        
        s=s[:len(s)-1]
        s='LinearEstimates('+s+')'
        tempdict = {}
        
        try:
            varying_intercepts_and_slopes_estimates =eval(s)
            for i in self.L_intercept:
                tempdict[i] = getattr(varying_intercepts_and_slopes_estimates,i)
            for i in self.L_slope:
                tempdict[i] = getattr(varying_intercepts_and_slopes_estimates,i)
        except Exception as e:
            print("---error while getting slope and intercept values----")
            print(e)
        
            
        parameters = tempdict
        duplicate_data=data_df.copy()
        duplicate_data.columns = rem_specialchar_array(duplicate_data.columns)

        
        for group in self.spec_dict['group_cols']:
            if not (group+'_original') in duplicate_data.columns:
                duplicate_data[group+'_original']= duplicate_data[group]
                duplicate_data[group][duplicate_data[group+'_original']==self.mapped_df[group+'_original']]= self.mapped_df[group]

        def value(row):
            val=0
            for slope in self.fixed_effect:
                val= val + (row[slope] * parameters["fixed_slope_"+slope])
            for group in self.spec_dict['group_cols']:
                val= val + (parameters["intercept_"+group][int(row[group])])
            for var_slope in self.random_effect:
                name="slope_"+var_slope[0]+"_"+var_slope[1]
                val= val + (row[var_slope[0]] * parameters[name][int(row[var_slope[1]])])
            if('fixed_slope_global_intercept' in parameters.keys()):
                val= val + parameters['fixed_slope_global_intercept']
            return val
        
        try:
            duplicate_data['y_pred'] = duplicate_data.apply(lambda row: value(row), axis = 1)
            
        except Exception as e:
            print("----error while predicting target variable----")
            print(e)
        
        self.duplicate_data=duplicate_data
        y_pred=duplicate_data['y_pred'].values
        
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        duplicate_data.to_csv(output_folder_path + self.dt + "_bayesian_predictions.csv",index=False)
        r2_value, RMSE, mape, mae, wmape=self.calculate_metrics()

        return y_pred, r2_value, RMSE, mape, mae, wmape
    
    def calculate_metrics(self):
        y_pred= self.duplicate_data['y_pred'].values
        y_true=self.duplicate_data[self.spec_dict['dv']].values
        
        #MAPE and WMAPE
        mask = y_true != 0
        mape=100*(np.fabs(y_true - y_pred)/y_true)[mask].mean()
        y_true_sum = y_true.sum()
        y_true_prod_mape = y_true[mask] * (100*(np.fabs(y_true - y_pred)/y_true)[mask])
        y_true_prod_mape_sum = y_true_prod_mape.sum()
        wmape = y_true_prod_mape_sum / y_true_sum
        
        #r2
        r2_scor= r2_score(y_true, y_pred)
        #mae
        mae= mean_absolute_error(y_true, y_pred)
        
        RMSE= sqrt(mean_squared_error(y_true, y_pred))
        print("MAPE  :", mape)
        print("WMAPE :", wmape)
        print("r2_score ", r2_scor)
        print("MAE :", mae)
        print("RMSE :", RMSE)
        mlflow.log_metrics({'r2_score': r2_scor, 'rmse':RMSE, 'MAE' :mae, 'MAPE': mape, 'WMAPE':wmape})

        
        return r2_scor, RMSE, mape, mae, wmape
