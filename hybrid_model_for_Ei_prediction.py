#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.simplefilter(action='ignore')
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import shap

import math
import keras
import keras.backend as K
from keras import regularizers
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras.constraints import NonNeg
# from keras_lr_multiplier import LRMultiplier
from scipy import optimize
from scipy.optimize import fsolve
from scipy import stats
from scipy.interpolate import interpn
from sklearn.model_selection import train_test_split
from matplotlib.colors import Normalize
from matplotlib import cm
import shap

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import os
import load_dataset_density_for_CI_hourly
import load_dataset_density_CI2_hourly

# keras.__version__
# tf.__version__


# In[ ]:


# Function of first order equation y= ax
def f_2(x, A):
    return A*x

# from sklearn.linear_model import LinearRegression
def diag_line(x,y,ax,color='black',xy=(.05,.76)):
    
    max_value = max(max(x),max(y))
    min_value = min(min(x),min(y))
    mean_value= np.mean(x)
    
        
    line = np.linspace(min_value, max_value, 100)
    ax.plot(line, line,'--',color=color)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    RMSE = (((y - x) ** 2).mean()) ** .5
    
    ax.annotate(f'$N$ = {len(x)} \n$R^{2}$ = {np.around(r_value**2,decimals=3)} \n$RMSE$ = {np.around(RMSE,decimals=3)}', 
                xy=xy, xycoords='axes fraction')

# for Rs prediction results, sometimes the equation has no solutions for H (NaN)
# and in fact, it should be b>0, H>0, so we can choose x1 = ((-b-sqrt(disc))/(2*a))
# exclude b<0, and H<0
# so when scatter the points, should drop NANs in LE_pre
def drop_outliers(b, H_pre, LE_true, LE_cal):        
    data = np.c_[b, H_pre, LE_true, LE_cal]
    names = ['b', 'H_pre', 'LE_true', 'LE_cal']
    df_LE = pd.DataFrame(data, columns=names)
    names1=['H_pre']
    for name in names1:
        df_LE[name][(df_LE[name]<-100)] = np.nan
    names2=['b', 'LE_true', 'LE_cal']
    for name in names2:
        df_LE[name][(df_LE[name]<0)] = np.nan
        
    df_LE = df_LE.dropna(axis=0, how='any')        
    return df_LE['LE_true'], df_LE['LE_cal']

def drop_outliers2(LE_true, b1, H_pre1, LE_cal1, b2, H_pre2, LE_cal2): 
#     LE_obr, LE_pred_rain, LE_pred_norain = drop_outliers2(bw, Hw, LE_pred_rain, bn, Hn, LE_pred_norain, LE_obr)
    data = np.c_[LE_true, b1, H_pre1, LE_cal1, b2, H_pre2, LE_cal2]
    names = ['LE_true', 'b1', 'H_pre1', 'LE_cal1', 'b2', 'H_pre2', 'LE_cal2']
    df_LE = pd.DataFrame(data, columns=names)
    
    names1=['H_pre1','H_pre2']
    for name in names1:
        df_LE[name][(df_LE[name]<-100)] = np.nan
    names2=['LE_true', 'b1', 'LE_cal1', 'b2', 'LE_cal2']
    for name in names2:
        df_LE[name][(df_LE[name]<0)] = np.nan
    
    df_LE = df_LE.dropna(axis=0, how='any')
    return df_LE['LE_true'], df_LE['LE_cal1'], df_LE['LE_cal2']

def drop_outliers3(y1, y2, y3, y4):        
    data = np.c_[y1, y2, y3, y4]
    names = ['y1', 'y2', 'y3','y4']
    df_y = pd.DataFrame(data, columns=names)
    df_y = df_y.dropna(axis=0, how='any') 
       
    return df_y['y1'], df_y['y2'], df_y['y3'], df_y['y4']

def quadratic(a,b,c):
    p=b**2-4*a*c
    solu = []
    for i in range(0,len(a)):       
        #quadratic equation with one unknown
        if p[i]>=0 and a[i]!=0:
#            x1=(-b[i]+math.sqrt(p[i]))/(2*a[i])
            x1=(-b[i]+np.sqrt(b[i]**2-4*a[i]*c[i]))/(2*a[i])
#            x1=np.divide(-b[i]+np.sqrt(p[i]),(2*a[i]))
#            x1=np.divide(-b[i]+(p[i])**(0.5),(2*a[i]))
#            x2=(-b[i]-math.sqrt(p[i]))/(2*a[i])
#            solu.append(max(x1,x2))
            # Acturally, b and H(sensible heat) should greater than 0
            # so only x1 is the true solution 
#             print('-b[i]+np.sqrt(p[i]):')
#             print(-b[i]+ np.sqrt(p[i]))
#             print('x1:')
#             print(x1)
            solu.append(x1)
        #linear equation with one unknown 
        elif a[i]==0:
            x1=x2=-c/b
            solu.append(x1)
        else:
            solu.append(np.nan) 
            
    # changed by Weiwei Zhan, final output is an array
    solu = np.array(solu)
    
    return solu

# PM_quad_plt: calculate the LE value using quadratic equation before ploting
def PM_quad_plt(auxi, Rs_pre):
    
    # LE_pre need these params to calculate,
    # changed by Weiwei Zhan, the first & 2nd dimensions of auxi are exchanged now 
    Rn_1 = auxi[:,0]
    G_1  = auxi[:,1]
    VPD_1 = auxi[:,2]
    Delta_1 = auxi[:,3]
    Density_1 = auxi[:,4]
    Ra_1 = auxi[:,5]
    r_1 = auxi[:,6]
    TA_1 = auxi[:,7]
    beta_1 = auxi[:,8]
    Cp = 1012
    
    a = (1/2.0)*beta_1*(Ra_1/(Density_1*Cp))**2
    b = (Delta_1*Ra_1)/(Density_1*Cp)+r_1*(Ra_1+Rs_pre)/(Density_1*Cp)        
    c = VPD_1-r_1*(Ra_1+Rs_pre)*(Rn_1-G_1)/(Density_1*Cp)
    
#     print('a:')
#     print(a)
#     print('b:')
#     print(b)
#     print('c:')
#     print(c)
    # disc = b**2-4*a*c
    H_pre = quadratic(a,b,c)
#     print('H_pre:')
#     print(H_pre)
    # the LE calculation results
    LE_pre = Rn_1 - G_1 - H_pre
#     print('LE_pre:')
#     print(LE_pre)
    return LE_pre, H_pre, b

            
def scatter_CIperformance(y_test, y_pred_test, target,
                          cmap='magma',test_only=False,save=False, site_name=None,fontsize=11,s=10,
                          xlabel='Observed', ylabel='Modelled', xlim=[0, 800], ylim=[0, 800],
                          output_dir='./model_and_code/pics_xulian/'):

#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir)
    y_pred_test = y_pred_test.astype('float32')
    y_test = y_test.astype('float32')
    
    plt.rcParams['font.size']   = fontsize
    plt.rcParams['savefig.dpi'] = 310
        
    # drop the outliers
    data = np.c_[y_pred_test, y_test]
    names = ['y1', 'y2']
    df_y = pd.DataFrame(data, columns=names)    
    df_y = df_y.dropna(axis=0, how='any') 
    y_pred_test = np.array(df_y['y1'])
    y_test = np.array(df_y['y2'])
    
    fig, ax = plt.subplots(figsize=(5,4))
    density_scatter(y_test,y_pred_test, cmap=cmap, s=s,bins=[30,30],ax=ax, fig=fig )
#     ax.set_title('Test Set ('+site_name+')',fontweight='bold')

#     ax.setabel(target[0]+' observations')
#     ax.set_ylabel(target[0]+' in hybrid model')
    ax.setim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.setabel(xlabel)
    ax.set_ylabel(ylabel)
    A1 = optimize.curve_fit(f_2, y_test, y_pred_test)[0]
    x1 = np.arange(-400, 800, 1)
    y1 = A1*x1
    plt.plot(x1, y1, "red")

    plt.tight_layout()  
    if save:
        plt.savefig(output_dir+site_name+'_'+target[0]+'_TestSet.pdf')

        

def hybrid_loss():
    
    # needed auxilary variables
#     Rn_1 = tf.gather(model.input[-1],indices=[0],axis=1)
#     G_1  = tf.gather(model.input[-1],indices=[1],axis=1)
#     VPD_1 = tf.gather(model.input[-1],indices=[2],axis=1)
#     Delta_1 = tf.gather(model.input[-1],indices=[3],axis=1)
#     Density_1 = tf.gather(model.input[-1],indices=[4],axis=1)
#     Ra_1 = tf.gather(model.input[-1],indices=[5],axis=1)
#     r_1 = tf.gather(model.input[-1],indices=[6],axis=1)
#     TA_1 = tf.gather(model.input[-1],indices=[7],axis=1)
#     beta_1 = tf.gather(model.input[-1],indices=[8],axis=1)
    aux_input = model.input[-1]
    
    Rn_1,  G_1, VPD_1, Delta_1, Density_1, Ra_1, r_1, TA_1, beta_1 =         aux_input[:, 0], aux_input[:, 1], aux_input[:, 2], aux_input[:, 3], aux_input[:, 4],             aux_input[:, 5], aux_input[:, 6], aux_input[:, 7], aux_input[:, 8]
    
    
#     G_1  = tf.gather(model.input[-1],indices=[1],axis=1)
#     VPD_1 = tf.gather(model.input[-1],indices=[2],axis=1)
#     Delta_1 = tf.gather(model.input[-1],indices=[3],axis=1)
#     Density_1 = tf.gather(model.input[-1],indices=[4],axis=1)
#     Ra_1 = tf.gather(model.input[-1],indices=[5],axis=1)
#     r_1 = tf.gather(model.input[-1],indices=[6],axis=1)
#     TA_1 = tf.gather(model.input[-1],indices=[7],axis=1)
#     beta_1 = tf.gather(model.input[-1],indices=[8],axis=1)
    Cp = 1012    
    
    
    def loss(y_true, y_pred):
        
        # Rs value
        Rs_pred = tf.exp(y_pred)
        Rs_true = tf.exp(y_true)
        
#         Rs_pred = y_pred
#         Rs_true = y_true
           
        # calculate b, c; a is removed as it is irrelevant with y_pred or y_true
        b_pred = tf.divide((Delta_1*Ra_1),(Density_1*Cp))+r_1*tf.divide((Ra_1+Rs_pred),(Density_1*Cp))
        b_true = tf.divide((Delta_1*Ra_1),(Density_1*Cp))+r_1*tf.divide((Ra_1+Rs_true),(Density_1*Cp))

        c_pred = VPD_1-r_1*(Ra_1+Rs_pred)*tf.divide((Rn_1-G_1),(Density_1*Cp))
        c_true = VPD_1-r_1*(Ra_1+Rs_true)*tf.divide((Rn_1-G_1),(Density_1*Cp))
        
        return K.mean(K.square(b_pred-b_true)) + K.mean(K.square(c_pred-c_true))
    
    return loss


def hybrid_loss_single():
    
    # needed auxilary variables
#     Rn_1 = tf.gather(model.input[-1],indices=[0],axis=1)
#     G_1  = tf.gather(model.input[-1],indices=[1],axis=1)
#     VPD_1 = tf.gather(model.input[-1],indices=[2],axis=1)
#     Delta_1 = tf.gather(model.input[-1],indices=[3],axis=1)
#     Density_1 = tf.gather(model.input[-1],indices=[4],axis=1)
#     Ra_1 = tf.gather(model.input[-1],indices=[5],axis=1)
#     r_1 = tf.gather(model.input[-1],indices=[6],axis=1)
#     TA_1 = tf.gather(model.input[-1],indices=[7],axis=1)
#     beta_1 = tf.gather(model.input[-1],indices=[8],axis=1)
    aux_input = X_aux_input[..., n_x:]
    
    Rn_1,  G_1, VPD_1, Delta_1, Density_1, Ra_1, r_1, TA_1, beta_1 =         aux_input[:, 0], aux_input[:, 1], aux_input[:, 2], aux_input[:, 3], aux_input[:, 4],             aux_input[:, 5], aux_input[:, 6], aux_input[:, 7], aux_input[:, 8]
    
    
#     G_1  = tf.gather(model.input[-1],indices=[1],axis=1)
#     VPD_1 = tf.gather(model.input[-1],indices=[2],axis=1)
#     Delta_1 = tf.gather(model.input[-1],indices=[3],axis=1)
#     Density_1 = tf.gather(model.input[-1],indices=[4],axis=1)
#     Ra_1 = tf.gather(model.input[-1],indices=[5],axis=1)
#     r_1 = tf.gather(model.input[-1],indices=[6],axis=1)
#     TA_1 = tf.gather(model.input[-1],indices=[7],axis=1)
#     beta_1 = tf.gather(model.input[-1],indices=[8],axis=1)
    Cp = 1012    
    
    
    def loss(y_true, y_pred):
        
        # Rs value
        Rs_pred = tf.exp(y_pred)
        Rs_true = tf.exp(y_true)
        
#         Rs_pred = y_pred
#         Rs_true = y_true
           
        # calculate b, c; a is removed as it is irrelevant with y_pred or y_true
        b_pred = tf.divide((Delta_1*Ra_1),(Density_1*Cp))+r_1*tf.divide((Ra_1+Rs_pred),(Density_1*Cp))
        b_true = tf.divide((Delta_1*Ra_1),(Density_1*Cp))+r_1*tf.divide((Ra_1+Rs_true),(Density_1*Cp))

        c_pred = VPD_1-r_1*(Ra_1+Rs_pred)*tf.divide((Rn_1-G_1),(Density_1*Cp))
        c_true = VPD_1-r_1*(Ra_1+Rs_true)*tf.divide((Rn_1-G_1),(Density_1*Cp))
        
        return K.mean(K.square(b_pred-b_true)) + K.mean(K.square(c_pred-c_true))
    
    return loss

def density_scatter(x, y, s, cmap='magma', xy=(.6,.1), ax=None, fig=None, sort=True, bins=20, fontsize=11, **kwargs)   :
    """
    Scatter plot colored by 2d histogram
    x: predicted values, 1-d numpy array 
    y: observed values, 1-d numpy array
    """
    if ax is None :
        fig , ax = plt.subplots()
        
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 1*(x_e[1:] + x_e[:-1]) , 1*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, s=s/10, cmap=cmap, **kwargs )
    
    # add the colorbar
    norm = Normalize(vmin = np.min(z), vmax = np.max(z)*0.3)
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm,cmap=cmap), ax=ax)
    cbar.ax.set_ylabel('Density')
    
    # add the values of metrics
    # regression & regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    N     = x.shape[0] # sample size
    RMSE  = (((x - y) ** 2).mean()) ** .5
    MBE   = (x - y).mean()
    
    ax.annotate(f'\n$r^{2} = %.2f$'% (r_value**2) + 
                f'\n$RMSE = %.2f$' % (RMSE),
                xy=xy, xycoords='axes fraction',fontsize=fontsize,color='red')
#                      f'\n$MBE = %.2f$'% MBE,  
#f'$N = %d$'% N +
    # add the diagonal line
    min_x, max_x = ax.getim()
    min_y, max_y = ax.get_ylim()

    max_value = max(max_x,max_y)
    min_value = min(min_x,min_y)
        
    line = np.linspace(min_value, max_value, 100)
    ax.plot(line, line,'--',color='black')
    
    
    return ax


def scatter_NNperformance(y_pred_train, y_pred_test,y_train,y_test,target,
                          cmap='magma',test_only=False,save=False, site_name=None,fontsize=11,s=10,
                          output_dir='./model_and_code/pics_xulian/', xylim=[-400, 1000]):

#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir)
    y_pred_train = y_pred_train.astype('float32')
    y_pred_test = y_pred_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    plt.rcParams['font.size']   = fontsize
    plt.rcParams['savefig.dpi'] = 310
 
    # drop the outliers
    data = np.c_[y_pred_train, y_train]
    names = ['y1', 'y2']
    df_y = pd.DataFrame(data, columns=names)    
    df_y.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_y = df_y.dropna(axis=0, how='any') 
    y_pred_train = np.array(df_y['y1'])
    y_train = np.array(df_y['y2'])

    data = np.c_[y_pred_test, y_test]
    names = ['y1', 'y2']
    df_y = pd.DataFrame(data, columns=names)
    df_y.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_y = df_y.dropna(axis=0, how='any') 
    y_pred_test = np.array(df_y['y1'])
    y_test = np.array(df_y['y2'])

    if test_only:
        
        fig, ax = plt.subplots(figsize=(5,4))
        density_scatter(y_test, y_pred_test, cmap=cmap, s=s,bins=[30,30],ax=ax,fig=fig )
        ax.set_title('Test Set ('+site_name+')',fontweight='bold')
        ax.setabel('Observed '+target[0]+' (W m$^{-2}$)')
        ax.set_ylabel('Predicted '+ target[0]+' (W m$^{-2}$)')
        
        plt.tight_layout()  
        if save:
            plt.savefig(output_dir+site_name+'_'+target[0]+'_TestSet.pdf')

    else:
        fig, axes = plt.subplots(figsize=(10,4),ncols=2)
        
#         print('y_pred_train:')
#         print(y_pred_train)
#         print('y_train:')
#         print(y_train)
        density_scatter(y_train,y_pred_train,cmap=cmap,s=s,bins=[30,30],ax=axes[0],fig=fig )
        density_scatter(y_test,y_pred_test, cmap=cmap, s=s,bins=[30,30],ax=axes[1],fig=fig )

        axes[0].set_title('Trainig set',fontweight='normal')
        axes[1].set_title('Test set',fontweight='normal')
        axes[0].setim(xylim[0], xylim[1])
        axes[0].set_ylim(xylim[0], xylim[1])
        axes[1].setim(xylim[0], xylim[1])
        axes[1].set_ylim(xylim[0], xylim[1])

        for ax in axes:
            ax.setabel('Observed '+target[0]+' (W m$^{-2}$)')
            ax.set_ylabel('Predicted '+ target[0]+' (W m$^{-2}$)')

        plt.tight_layout()  
#         plt.clim(0, 8e-5)
        if save:
            plt.savefig(output_dir+site_name+'_'+target[0]+'.eps')


# ### Load dry-hour dataset 
# 
# `Predictors`: X_train, X_val, X_test <br/>
# `Output`: Y_train, Y_val, Y_test (**Question:** ET or Rs??) <br/>
# `Auxillary variables`: Auxi_train, Auxi_val, Auxi_test <br/>
# 

# In[ ]:


CPindx = 2
index = 291
#   for rain-free dataset 
df_all,X_train,Y_train,X_val,Y_val,X_test,Y_test, Auxi_train, Auxi_val, Auxi_test,LE_train, LE_val, LE_test, Date_train, Date_val, Date_test, var_names, PFT_names, epochs, neurons_n, hidden_layer_n, num_steps, steps,= load_dataset_density_for_CI_hourly.input_data(index,['Rs'],CPindx)


mean = X_train.T.astype('float').mean(axis=0)
std  = X_train.T.astype('float').std(axis=0)

# keep the original value for PFT as input
if ('PFT' in var_names) == True:
    indx = var_names.index('PFT')
    mean[indx] = 0.0
    std[indx]  = 1.0

##!!!! to be changed later based on the updated data!!!!
X_train = ((X_train.T - mean) / std)
X_val = ((X_val.T - mean) / std)
X_test  = ((X_test.T - mean) / std)

# save the mean and atd values
savepath = './csvdata/meanv_hourly_RHcorr_CP' + str(CPindx) + '.csv'
meandata = np.column_stack((mean,std))
meandata = pd.DataFrame(meandata, columns = ['Mean','Std'])
meandata.to_csv(savepath,columns=None, header=True, index=True)

##!!!! to be changed later based on the updated data!!!!
m,n_x = X_train.shape            # (n_x: number of input variables, m:samples)
n_y    = Y_train.T.shape[1]      # n_y: number of output variables, 1
n_auxi = Auxi_train.T.shape[1]

# Y_train = Y_train[:,0]
# Y_test = Y_test[:,0]
Y_train = Y_train.T
Y_test = Y_test.T
Y_val = Y_val.T
Auxi_train = Auxi_train.T
Auxi_test = Auxi_test.T
Auxi_val = Auxi_val.T

Y_all = np.row_stack((Y_train,Y_val,Y_test))
X_all = np.row_stack((X_train,X_val,X_test))
Auxi_all = np.row_stack((Auxi_train,Auxi_val,Auxi_test))

print(m,n_x,n_y,n_auxi)


# ### Train the dry model （HMdry）

# In[ ]:


# set hyperparameters
num_epochs=2000
learning_rate = 0.001
minibatch_size = 128
print_cost = True
model_num = 1
n_neuron  = 256
activation = 'relu'

# define model structure
# inputs
X_input    = Input(shape=(n_x,), dtype='float32', name='X_input')
Auxi_input = Input(shape=(n_auxi,), dtype='float32', name='Auxi_input')

# 5 hidden layers
x         = Dense(n_neuron, activation=activation,name='hidden1')(X_input)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden2')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden3')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden4')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden5')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden6')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)

# output layer
# output layer
Z6        = Dense(1, activation=None,name='Z6')(x) # log value of Rs


model = Model(inputs=[X_input,Auxi_input], outputs=Z6)

model.compile(loss=hybrid_loss(),optimizer=tf.keras.optimizers.Adam(lr=learning_rate))


# In[ ]:


# model = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_norain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
#                                           compile=True)

hist = model.fit({'X_input':X_train,'Auxi_input': Auxi_train},
                 {'Z6': Y_train},
                 batch_size = minibatch_size,
                 epochs = num_epochs)


var_s = ['loss'] # ,'der_Reco'
fig, axes = plt.subplots(ncols=len(var_s),figsize=(6*len(var_s),4))


for (i,var) in enumerate(var_s):
    axes.plot(hist.history[var],label='Training')
#     axes.plot(history.history['val_'+var],label='Validation')
    axes.set_ylabel(var)
    axes.legend()

plt.tight_layout()


# save the model
model_norain = model
# save the trained best models
tf.keras.models.save_model(model_norain,'./model_and_code/bestmodel_xulian/model_norain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                            overwrite=True,save_format="h5")


# In[ ]:


model = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_norain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                          compile=False)

# log(Rs) -- y
Y_pred_train = model.predict({'X_input':X_train,'Auxi_input':Auxi_train})[:,0]
Y_pred_test  = model.predict({'X_input':X_test,'Auxi_input':Auxi_test})[:,0]
# Y_pred_norain  = model.predict({'X_input':X_all_data,'Auxi_input':Auxi_all_data})[:,0]

# calculate LE
LE_pred_train, H, b = PM_quad_plt(Auxi_train, np.exp(Y_pred_train))
LE_train, H, b = PM_quad_plt(Auxi_train, np.exp(Y_train[:,0]))
LE_train, LE_pred_train = drop_outliers(b, H, LE_train, LE_pred_train)

LE_pred_test, H, b = PM_quad_plt(Auxi_test, np.exp(Y_pred_test))
LE_test, H, b = PM_quad_plt(Auxi_test, np.exp(Y_test[:,0]))
LE_test, LE_pred_test = drop_outliers(b, H, LE_test, LE_pred_test)

# LE_pred_norain, H, b = PM_quad_plt(Auxi_all_data, np.exp(Y_pred_norain))
# LE_obr, H, b = PM_quad_plt(Auxi_all_data, np.exp(Y_all_data[:,0]))

# LE_pred_train, H, b = PM_quad_plt(Auxi_train, Y_pred_train)
# LE_train, H, b = PM_quad_plt(Auxi_train, Y_train)
# # LE_train, LE_pred_train = drop_outliers(b, H, LE_train, LE_pred_train)

# LE_pred_test, H, b = PM_quad_plt(Auxi_test, Y_pred_test)
# LE_test, H, b = PM_quad_plt(Auxi_test, Y_test)
# # LE_test, LE_pred_test = drop_outliers(b, H, LE_test, LE_pred_test)

# # scatterplot Y
# scatter_NNperformance(Y_pred_train, Y_pred_test,Y_train,Y_test,['log(Rs)'],
#                           cmap='magma',test_only=False,save=False, xylim=[-100,3000])

# # scatterplot Rs
# scatter_NNperformance(Rs_pred_train, Rs_pred_test,Rs_train,Rs_test,['Rs'],
#                           cmap='magma',test_only=False,save=False)

# scatterplot LE

scatter_NNperformance(LE_pred_train, LE_pred_test,LE_train,LE_test,['LE'],
                          cmap='magma',test_only=False,save=False, xylim=[0,800])

# scatterplot original LE and Hybrid LE
# scatter_NNperformance(LE_train,LE_test,LE_train_obr.T, LE_test_obr.T,['LE'],
#                           cmap='magma',test_only=False,save=False)


# In[ ]:


### Predicting wet-hour LE using HMdry 


# In[ ]:



CPindx = 2
index = 292
#   for rain-free dataset 
df_all,X_train,Y_train,X_val,Y_val,X_test,Y_test, Auxi_train, Auxi_val, Auxi_test,LE_train, LE_val, LE_test, Date_train, Date_val, Date_test, var_names, PFT_names, epochs, neurons_n, hidden_layer_n, num_steps, steps,= load_dataset_density_CI2_hourly.input_data(index,['Rs'],CPindx)

Y_all = np.column_stack((Y_train,Y_val,Y_test))
X_all = np.column_stack((X_train,X_val,X_test))
X_all = X_all[0:6,]
Auxi_all = np.column_stack((Auxi_train,Auxi_val,Auxi_test))

# load the no-rain mean values
mstd = pd.read_csv('./csvdata/meanv_hourly_RHcorr_CP' + str(CPindx) + '.csv')
mean_norain = np.array(mstd.loc[:,'Mean'])
std_norain = np.array(mstd.loc[:,'Std'])

X_all = ((X_all.T - mean_norain) / std_norain);

Y_all = Y_all.T
Auxi_all = Auxi_all.T


# In[ ]:


# load the trained best models
model_norain = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_norain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                          compile=False)

Y_pred_norain  = model_norain.predict({'X_input':X_all,'Auxi_input':Auxi_all})[:,0]
LE_pred_norain, Hw, bw = PM_quad_plt(Auxi_all, np.exp(Y_pred_norain))

LE_obr, H, b = PM_quad_plt(Auxi_all, np.exp(Y_all[:,0]))

# write the prediction to csv
savepath = './csvdata/LE_pred_norain_RHcorr_CP' + str(CPindx) + '.csv'
LE_all = np.column_stack((LE_obr, LE_pred_norain, Hw))
LE_all = pd.DataFrame(LE_all, columns = ['LE_obr', 'LE_pred_norain','H_pre'])
LE_all.to_csv(savepath,columns=None, header=True, index=True)


# ### Train the wet model (HMwet)

# In[ ]:


CPindx = 2
index = 292
#   for rain-free dataset 
df_all,X_train,Y_train,X_val,Y_val,X_test,Y_test, Auxi_train, Auxi_val, Auxi_test,LE_train, LE_val, LE_test, Date_train, Date_val, Date_test, var_names, PFT_names, epochs, neurons_n, hidden_layer_n, num_steps, steps,= load_dataset_density_CI2_hourly.input_data(index,['Rs'],CPindx)

Y_all_org = np.column_stack((Y_train,Y_val,Y_test))
X_all_org = np.column_stack((X_train,X_val,X_test))
Auxi_all_org = np.column_stack((Auxi_train,Auxi_val,Auxi_test))

mean = X_train.T.astype('float').mean(axis=0)
std  = X_train.T.astype('float').std(axis=0)

# keep the original value for PFT as input
if ('PFT' in var_names) == True:
    indx = var_names.index('PFT')
    mean[indx] = 0.0
    std[indx]  = 1.0

##!!!! to be changed later based on the updated data!!!!
X_train = ((X_train.T - mean) / std)
X_val = ((X_val.T - mean) / std)
X_test  = ((X_test.T - mean) / std)
X_all  = ((X_all_org.T - mean) / std)

# save the mean and atd values
savepath = './csvdata/meanv_rain_hourly_RHcorr_CP' + str(CPindx) + '.csv'
meandata = np.column_stack((mean,std))
meandata = pd.DataFrame(meandata, columns = ['Mean','Std'])
meandata.to_csv(savepath,columns=None, header=True, index=True)

##!!!! to be changed later based on the updated data!!!!
m,n_x = X_train.shape            # (n_x: number of input variables, m:samples)
n_y    = Y_train.T.shape[1]      # n_y: number of output variables, 1
n_auxi = Auxi_train.T.shape[1]

# Y_train = Y_train[:,0]
# Y_test = Y_test[:,0]
Y_train = Y_train.T
Y_test = Y_test.T
Y_all = Y_all_org.T
Auxi_train = Auxi_train.T
Auxi_test = Auxi_test.T
Auxi_all = Auxi_all_org.T

print(m,n_x,n_y,n_auxi)

# X_aux_train = 


# In[ ]:


# set hyperparameters
num_epochs=2000
learning_rate = 0.001
minibatch_size = 128
print_cost = True
model_num = 1
n_neuron  = 256
activation = 'relu'

# define model structure
# inputs
X_input    = Input(shape=(n_x,), dtype='float32', name='X_input')
Auxi_input = Input(shape=(n_auxi,), dtype='float32', name='Auxi_input')

# 5 hidden layers
x         = Dense(n_neuron, activation=activation,name='hidden1')(X_input)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden2')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden3')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden4')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden5')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)
x         = Dense(n_neuron, activation=activation,name='hidden6')(x)
x         = tfa.layers.GroupNormalization(32, 1)(x)

# output layer
# output layer
Z6        = Dense(1, activation=None,name='Z6')(x) # log value of Rs

model = Model(inputs=[X_input,Auxi_input], outputs=Z6)

model.compile(loss=hybrid_loss(),optimizer=tf.keras.optimizers.Adam(lr=learning_rate))


# In[ ]:


hist = model.fit({'X_input':X_train,'Auxi_input': Auxi_train},
                 {'Z6': Y_train},
                 batch_size = minibatch_size,
                 epochs = num_epochs)

var_s = ['loss'] # ,'der_Reco'
fig, axes = plt.subplots(ncols=len(var_s),figsize=(6*len(var_s),4))


for (i,var) in enumerate(var_s):
    axes.plot(hist.history[var],label='Training')
#     axes.plot(history.history['val_'+var],label='Validation')
    axes.set_ylabel(var)
    axes.legend()


plt.tight_layout()

# save the model
model_withrain = model
# save the trained best models
tf.keras.models.save_model(model_withrain,'./model_and_code/bestmodel_xulian/model_withrain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                            overwrite=True,save_format="h5")


# In[ ]:


model = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_withrain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                          compile=False)

# log(Rs) -- y
Y_pred_train = model.predict({'X_input':X_train,'Auxi_input':Auxi_train})[:,0]
Y_pred_test  = model.predict({'X_input':X_test,'Auxi_input':Auxi_test})[:,0]
# Y_pred_norain  = model.predict({'X_input':X_all_data,'Auxi_input':Auxi_all_data})[:,0]

# calculate LE
LE_pred_train, H, b = PM_quad_plt(Auxi_train, np.exp(Y_pred_train))
LE_train, H, b = PM_quad_plt(Auxi_train, np.exp(Y_train[:,0]))
LE_train, LE_pred_train = drop_outliers(b, H, LE_train, LE_pred_train)

LE_pred_test, H, b = PM_quad_plt(Auxi_test, np.exp(Y_pred_test))
LE_test, H, b = PM_quad_plt(Auxi_test, np.exp(Y_test[:,0]))
LE_test, LE_pred_test = drop_outliers(b, H, LE_test, LE_pred_test)

# LE_pred_norain, H, b = PM_quad_plt(Auxi_all_data, np.exp(Y_pred_norain))
# LE_obr, H, b = PM_quad_plt(Auxi_all_data, np.exp(Y_all_data[:,0]))

# LE_pred_train, H, b = PM_quad_plt(Auxi_train, Y_pred_train)
# LE_train, H, b = PM_quad_plt(Auxi_train, Y_train)
# # LE_train, LE_pred_train = drop_outliers(b, H, LE_train, LE_pred_train)

# LE_pred_test, H, b = PM_quad_plt(Auxi_test, Y_pred_test)
# LE_test, H, b = PM_quad_plt(Auxi_test, Y_test)
# # LE_test, LE_pred_test = drop_outliers(b, H, LE_test, LE_pred_test)

# # scatterplot Y
# scatter_NNperformance(Y_pred_train, Y_pred_test,Y_train,Y_test,['log(Rs)'],
#                           cmap='magma',test_only=False,save=False, xylim=[-100,3000])

# # scatterplot Rs
# scatter_NNperformance(Rs_pred_train, Rs_pred_test,Rs_train,Rs_test,['Rs'],
#                           cmap='magma',test_only=False,save=False)

# scatterplot LE

scatter_NNperformance(LE_pred_train, LE_pred_test,LE_train,LE_test,['LE'],
                          cmap='magma',test_only=False,save=False, xylim=[0,800])

# scatterplot original LE and Hybrid LE
# scatter_NNperformance(LE_train,LE_test,LE_train_obr.T, LE_test_obr.T,['LE'],
#                           cmap='magma',test_only=False,save=False)


# ### Predicting CI as HMwet minus HMdry

# In[ ]:


model_withrain = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_withrain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                            compile=False)
model_norain = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_norain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                          compile=False)

Y_pred_rain  = model_withrain.predict({'X_input':X_all,'Auxi_input':Auxi_all})[:,0]
LE_pred_rain, Hw, bw = PM_quad_plt(Auxi_all, np.exp(Y_pred_rain))

LE_obr, H, b = PM_quad_plt(Auxi_all, np.exp(Y_all[:,0]))

# prediction by the dry model
X_all_org = X_all_org[0:6,]
Auxi_all_org = Auxi_all_org

# load the no-rain mean values
mstd = pd.read_csv('./csvdata/meanv_hourly_RHcorr_CP' + str(CPindx) + '.csv')
mean_norain = np.array(mstd.loc[:,'Mean'])
std_norain = np.array(mstd.loc[:,'Std'])

X_all_org = ((X_all_org.T - mean_norain) / std_norain);
Auxi_all_org = Auxi_all_org.T
Y_pred_norain  = model_norain.predict({'X_input':X_all_org,'Auxi_input':Auxi_all_org})[:,0]
LE_pred_norain, Hd, bd = PM_quad_plt(Auxi_all_org, np.exp(Y_pred_norain))

# write the prediction to csv
savepath = './csvdata/LE_pred_rain_wsc_RHcorr_CP' + str(CPindx) + '.csv'
LE_all = np.column_stack((LE_obr, LE_pred_norain,LE_pred_rain, Hw, np.exp(Y_pred_norain), np.exp(Y_pred_rain)))
LE_all = pd.DataFrame(LE_all, columns = ['LE_obr', 'LE_pred_norain','LE_pred_rain','H_pre','Rs_pred_norain','Rs_pred_rain'])
LE_all.to_csv(savepath,columns=None, header=True, index=True)


# In[ ]:


CPindx=2;

# load the trained best models
model_withrain = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_withrain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                            compile=False)
model_norain = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_norain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                          compile=False)

# load the no-rain mean values
mstd = pd.read_csv('./csvdata/meanv_hourly_new_CP' + str(CPindx) + '.csv')
mean_norain = np.array(mstd.loc[:,'Mean'])
std_norain = np.array(mstd.loc[:,'Std'])

# load the rain mean values
mstd = pd.read_csv('./csvdata/meanv_rain_hourly_new_CP' + str(CPindx) + '.csv')
mean_rain = np.array(mstd.loc[:,'Mean'])
std_rain = np.array(mstd.loc[:,'Std'])

# adjust to have the same mean as observations
if CPindx==0:
    mean_rain[6]=72.7318;
    std_rain[6]=69.9272;
elif CPindx==1:
    mean_rain[6]=43.4162;
    std_rain[6]=44.4872;
else:
    mean_rain[6]=53.8489;
    std_rain[6]=49.9907;
    
mean_pft = [-0.107755,0.16035,0.208029, -0.005637, -0.507474, 0.2487279, -0.0088182, 0.03276777];

for yr in np.arange(2005,2006):
    for doy in np.arange(1,2): # calendar.isleap(yr)+366
        print('yr='+str(yr)+'; doy='+str(doy))
        filepath='./era5data/ERA5_predictors_' + str(yr)
        savepath='./era5data/Hybrid_ERA5_CP' +  str(CPindx) + '_' + str(yr)
        if doy<10:
            filepath = filepath + '00' + str(doy) + '.csv'
            savepath = savepath + '00' + str(doy) + '.csv'
        elif doy<100:
            filepath = filepath + '0' + str(doy) + '.csv'
            savepath = savepath + '0' + str(doy) + '.csv'
        else:
            filepath = filepath + str(doy) + '.csv'
            savepath = savepath + str(doy) + '.csv'
       
        if os.path.isfile(savepath):
            print('file already exists!')
        else:
#        xdata = pd.read_excel(filepath)
            xdata = pd.read_csv(filepath)
#         xdata[['WSc']] = xdata[['WSc']]+200 # for test only
            var_names = tuple(xdata.columns[1:len(xdata.columns)-2])
            x_names=['T2m','WS','VPD','Rn','PFT','LAI','WSc_P' + str(CPindx)]
            Auxi_names=['Rn','G','VPD','Delta','Density','Ra','r','T2m','beta','PFT','LAI']
#         print(var_names)
            Lat = xdata.loc[:,'Lat'].values.astype('float32')
            Lon = xdata.loc[:,'Lon'].values.astype('float32')
            Hour = xdata.loc[:,'Hour'].values.astype('float32')
            X_era5_org = xdata.loc[:,x_names].values.astype('float32')
        
            # X_era5_org.LAI[X_era5_org.LAI<0.1] = 0.1
            LAI = xdata.loc[:,'LAI'].values.astype('float32')
            LAI[np.where(LAI<0.1)] = 0.1
            X_era5_org[:,5] = LAI
            
            WSc = xdata.loc[:,'WSc_P' + str(CPindx)].values.astype('float32')
            PFT = xdata.loc[:,'PFT'].values.astype('float32')
            WSc[np.where(PFT==1)] = WSc[np.where(PFT==1)] + mean_pft[0]
            WSc[np.where(PFT==2)] = WSc[np.where(PFT==2)] + mean_pft[1]
            WSc[np.where(PFT==4)] = WSc[np.where(PFT==4)] + mean_pft[2]
            WSc[np.where(PFT==5)] = WSc[np.where(PFT==5)] + mean_pft[3]
            WSc[np.where(PFT==6)] = WSc[np.where(PFT==6)] + mean_pft[4]
            WSc[np.where(PFT==7)] = WSc[np.where(PFT==7)] + mean_pft[5]
            WSc[np.where(PFT==8)] = WSc[np.where(PFT==8)] + mean_pft[6]
            WSc[np.where(PFT==10)] = WSc[np.where(PFT==10)] + mean_pft[7]
            X_era5_org[:,6] = WSc
            
#             print(WSc)
            
            Auxi_era5 = xdata.loc[:,Auxi_names].values.astype('float32')
#         X_era5_org = X_era5_org.T;
#         Auxi_era5 = Auxi_era5.T;
        
            # LE_pred_rain
            X_era5 = X_era5_org.copy()
            X_era5 = ((X_era5 - mean_rain) / std_rain);
            Y_pred_rain  = model_withrain.predict({'X_input':X_era5,'Auxi_input':Auxi_era5})[:,0]
            LE_pred_rain, Hw, bw = PM_quad_plt(Auxi_era5, np.exp(Y_pred_rain))
            # LE_obr, LE_pred_rain = drop_outliers(b, H, LE_obr, LE_pred_rain)

            # LE_pred_norain
            X_era5 = X_era5_org[:,0:6].copy()
            X_era5 = ((X_era5 - mean_norain) / std_norain)
            Y_pred_norain  = model_norain.predict({'X_input':X_era5,'Auxi_input':Auxi_era5})[:,0]
            LE_pred_norain, Hn, bn = PM_quad_plt(Auxi_era5, np.exp(Y_pred_norain))
        
#         # scatterplot LE_hybrid (with rain) vs LE_hybrid (no rain)
#         scatter_CIperformance(LE_pred_norain, LE_pred_rain, ['LE'],
#                                   xlabel='LE_hybrid (no rain)', ylabel='LE_hybrid (with rain)',
#                                   cmap='magma',test_only=False,save=False,xlim=[0,800], ylim=[0,800])
        
#         # scatterplot LE_hybrid (with rain) vs LE_hybrid (no rain)
#         scatter_CIperformance(Y_pred_norain, Y_pred_rain, ['Y'],
#                                   xlabel='Rs_hybrid (no rain)', ylabel='Rs_hybrid (with rain)',
#                                   cmap='magma',test_only=False,save=False,
#                                   xlim=[0,10], ylim=[0,10])

            # write the prediction to csv
            LE_all = np.column_stack((Lat, Lon, Hour, LE_pred_rain, LE_pred_norain, Hw, Hn))
            LE_all = pd.DataFrame(LE_all, columns = ['Lat','Lon','Hour','LE_pred_rain', 'LE_pred_norain','Hw', 'Hn'])
            LE_all.to_csv(savepath,columns=None, header=True, index=True)


# In[ ]:


CPindx=2;

# load the trained best models
model_withrain = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_withrain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                            compile=False)
model_norain = tf.keras.models.load_model('./model_and_code/bestmodel_xulian/model_norain_hourly_RHcorr_CP' + str(CPindx) + '.h5py', 
                                          compile=False)

# load the no-rain mean values
mstd = pd.read_csv('./csvdata/meanv_hourly_new_CP' + str(CPindx) + '.csv')
mean_norain = np.array(mstd.loc[:,'Mean'])
std_norain = np.array(mstd.loc[:,'Std'])

# load the rain mean values
mstd = pd.read_csv('./csvdata/meanv_rain_hourly_new_CP' + str(CPindx) + '.csv')
mean_rain = np.array(mstd.loc[:,'Mean'])
std_rain = np.array(mstd.loc[:,'Std'])

# adjust to have the same mean as observations
if CPindx==0:
    mean_rain[6]=72.7318;
    std_rain[6]=69.9272;
elif CPindx==1:
    mean_rain[6]=43.4162;
    std_rain[6]=44.4872;
else:
    mean_rain[6]=53.8489;
    std_rain[6]=49.9907;
    
mean_pft = [-0.107755,0.16035,0.208029, -0.005637, -0.507474, 0.2487279, -0.0088182, 0.03276777];

