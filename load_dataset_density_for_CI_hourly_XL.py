# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:19:43 2019

@author: william
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
import argparse
import math
import dateutil


# set input data to 5%~95% percentile 
def filter_data(data,names,num1,num2):
    num_lower = num1
    num_upper = num2
    df = data
    for name in names:
        per_lower  = np.percentile(df[name],num_lower)
        per_upper = np.percentile(df[name],num_upper)
#        df = df[(df[name]>per_lower) & (df[name]<per_upper)]
        df[name][(df[name]<per_lower) | (df[name]>per_upper)] = np.nan
        
    return df

# set input data exclude 5%~95% percentile 
def filter_data_out(data,names,num1,num2):
    num_lower = num1
    num_upper = num2
    df = data
    for name in names:
        per_lower  = np.percentile(df[name],num_lower)
        per_upper = np.percentile(df[name],num_upper)
#        df = df[(df[name]>per_lower) & (df[name]<per_upper)]
        df[name][(df[name]>per_lower) & (df[name]<per_upper)] = np.nan
        
    return df

# set input data value between num1 and num2 
def filter_data2(data,names,num1,num2):
    num_lower = num1
    num_upper = num2
    df = data
    for name in names:
        df[name][(df[name]<num_lower) | (df[name]>num_upper)] = np.nan
        
    return df
    

# get the X_train, X_val, X_test dataset and Y_train, Y_val, Y_test, 
# LE_true_train, LE_true_val, LE_true_test
def input_data(num,target_names):
    # define target_names and filter_names (5~95%)    
    filter_names = ['Ca','Rn','G']  
    
    # file_path contains all the combinations between vars and PFTs
#    file_path = './statistics/PFT_8.csv'
    
    # file_path = './model_and_code/statistics/test_tensorflow_repeat.csv'
    file_path ='E:/workspace/2021.Canopy_Interception/model_and_code/statistics/test_tensorflow_repeat_hourly_XL.csv'
    dt = pd.read_csv(file_path)
    dt = dt.loc[:,['var','PFT','neurons_n','hidden_layer_n','epochs','num_steps','steps']]    
    
    # output file    
    print('This is:')
    print(num)
    var_inputs = ()
    PFT_inputs = ()
    var_arr =  dt['var'][num].split(',')
    PFT_arr =  dt['PFT'][num].split(',')
    epochs = int(dt['epochs'][num])
    neurons_n = int(dt['neurons_n'][num])
    hidden_layer_n = int(dt['hidden_layer_n'][num])
    num_steps = int(dt['num_steps'][num])
    steps = int(dt['steps'][num])

    for v in var_arr: 
        if len(v) != 0:
            var_inputs +=  tuple((v.strip(),))# tuple(("Ca",)) #
    
    for p in PFT_arr:
        if len(p) != 0:
            PFT_inputs +=  tuple((int(p.strip()),))# tuple((1,)) #
#        f.write(str(i)+'_'+str(var_inputs)+'_'+str(PFT_inputs)+'\n')
    for vt in var_inputs:
        print(vt)
    for pt in PFT_inputs:
        print(pt)
    
#    var_inputs = ['fpar','SM','PFT','TA','Ca','WS','PA','RH','Rn','G','VPD','h_canopy','PPFD_IN']
#    PFT_inputs = [1,2,3,4,5,6,7,8,9]
    print('var_inputs:')
    print(var_inputs)
    var_names = var_inputs
    PFT_names = PFT_inputs
#    target_names = ['LE']
    target_names = target_names
   
    
    # Inputva_21 is for latent heat flux
    # Inputva_23_G.nc is for ground heat flux
    # nc_obj=Dataset('./Dataset/Inputva_21.nc')
    nc_obj=Dataset('E:/workspace/2021.Canopy_Interception/Dataset/Inputva_norain_hourly_XL.nc')
    # nc_obj=Dataset('./Dataset/Inputva_P11_CP=1.nc')
    # fpar = nc_obj.variables['fpar'][:]
    # SM = nc_obj.variables['SM'][:]
    VPD = nc_obj.variables['VPD'][:]
    PFT = nc_obj.variables['PFT'][:]
    TA = nc_obj.variables['TA'][:]
    Ca = nc_obj.variables['Ca'][:]
    Rn = nc_obj.variables['Rn'][:]
    G = nc_obj.variables['G'][:]
    LE = nc_obj.variables['LEclose'][:]
    H = nc_obj.variables['H'][:]
    # force the Rn-G=LE+H
#    H = Rn-G-LE
    WS = nc_obj.variables['WS'][:]
    PA = nc_obj.variables['PA'][:]
    RH = nc_obj.variables['RH'][:]
#    GPP = nc_obj.variables['GPP'][:]
    Gs = nc_obj.variables['Gs'][:]
    Rs = nc_obj.variables['Rscorr'][:]    
#    LW_OUT = nc_obj.variables['LW_OUT'][:]
    
    Delta = nc_obj.variables['Delta'][:]
    Density = nc_obj.variables['Density'][:]
    Ra = nc_obj.variables['Ra'][:]
    r = nc_obj.variables['r'][:]
    h_canopy = nc_obj.variables['h_canopy'][:]
    # PPFD_IN = nc_obj.variables['PPFD_IN'][:]
#    print('nc_obj.variables:')
#    for v in nc_obj.variables:
#    # this is the name of the variable.
#        print(v)
#    print('nc_obj.variables:')
    LAI = nc_obj.variables['LAI'][:]
    
    # total Precipitation for a rain event
#    P_total = nc_obj.variables['P_id_n'][:]
    # rainfall duration for a rain event
#    P_dur = nc_obj.variables['P_id_a'][:]
    
    DateNumber = nc_obj.variables['DateNumber'][:]
    SiteNumber = nc_obj.variables['SiteNumber'][:]
    print('SiteNumber:')
    print(SiteNumber)
    print(SiteNumber.shape)
    
#    hours = nc_obj.variables['hours'][:]
#    minutes = nc_obj.variables['minutes'][:]
    # date3: 2012-01-02 00:00:00
    date1 = nc_obj.variables['date1'][:] 
    date2 = date1.data.astype('str')
    date3 = [] 
    year  = []    
    ds, dL = date2.shape
    for d1 in range(0, dL):
        dstr = ''
        for d2 in range(0, ds):
            dstr = dstr + date2[d2,d1]
        date3.append(dateutil.parser.parse(dstr))
        year.append(dateutil.parser.parse(dstr).year)

#    # time: 201201021030
#    time1 = nc_obj.variables['time'][:] 
#    time2 = time1.data.astype('str')
#    time3 = []    
#    ds, dL = time2.shape
#    for d1 in range(0, dL):
#        dstr = ''
#        for d2 in range(0, ds):
#            dstr = dstr + time2[d2,d1]
#        time3.append(dateutil.parser.parse(dstr))

        
    Cp = 1012
    # beta is a parameter in new PM equation      
    beta = -(5006.12*(TA - 1811.79)*np.exp((17.27*TA)/(237.3 + TA)))/(237.3 + TA)**4
    
    # re-calculate Rs by using new PM equation
#    Rs = (Density*1012/(r*LE))*((Delta*Ra*(Rn-G-LE)/(Density*1012))+(1/2)*beta
#          *(Ra/(Density*Cp))**2*(Rn-G-LE)**2+VPD)-Ra 

    Rs = (np.divide((Density*1012),(r*LE)))*(np.divide((Delta*Ra*(Rn-G-LE)),(Density*1012))+(1/2)*beta
      *(np.divide(Ra,(Density*Cp))**2*(Rn-G-LE)**2)+VPD)-Ra 
      
#    print('min>>>>>>>>>>>>>>>>>>>>>>>')
#    print
    # try log(Rs)
    Rs = np.log(Rs)
    
    # Rs_old is the Rs calculated by classic PM equation
#    Rs_old = (Density*1012/(r*LE))*((Delta*Ra*(Rn-G-LE)/(Density*1012))+VPD)-Ra

#    Rs_pd = pd.DataFrame(Rs, columns = ['logRs'])
#    Rs_pd = Rs_pd.dropna(axis=0,how='any')
#    Rs_pd.to_csv('./csvdata/Rs2.csv',columns=None, header=True, index=True)

    # try log(Rs_old)
#    print('test aa')
#    aa = np.divide((Density[12]*1012),(r[12]*LE[12]))
#    print(Density[12])
#    print(r[12])
#    print(LE[12])
#    print(aa)
    
    # the true H value (sensible heat)
#    H_filter = Rn-LE-G    
    
#    # only leave the data of disc >= 0 and a != 0
#    a = (1/2.0)*beta*(Ra/(Density*Cp))**2
#    b = (Delta*Ra)/(Density*Cp)+r*(Ra_1+target_pre)/(Density_1*Cp)        
#    c = VPD_1-r_1*(Ra_1+target_pre)*(Rn_1-G_1)/(Density_1*Cp)
#    disc = b**2-4*a*c
    
    # data_all
#    data_all = np.c_[fpar,SM,PFT,TA,Ca,WS,PA,RH,Rn,G,VPD,h_canopy,PPFD_IN,LE,H,
#                     Delta,Density,Ra,r,Rs,beta,Rs_old,DateNumber,SiteNumber,date3,
#                     LW_OUT,hours,minutes,time3,year,GPP]
    data_all = np.c_[PFT,LAI,TA,Ca,WS,PA,RH,Rn,G,VPD,h_canopy,LE,H,
                     Delta,Density,Ra,r,Rs,beta,DateNumber,SiteNumber,date3,
                     year]  # ,LAI, fpar,PPFD_IN,SM,GPP,Rs_old
    #    h = data_all.shape
    
    # delete the rows and columns which contains NAN
#    column_names = ['fpar','SM','PFT','TA','Ca','WS','PA','RH','Rn','G','VPD',
#                    'h_canopy','PPFD_IN','LE','H','Delta','Density','Ra','r',
#                    'Rs','beta','Rs_old','DateNumber','SiteNumber','date3',
#                    'LW_OUT','hours','minutes','time3','year','GPP']
    column_names = ['PFT','LAI','TA','Ca','WS','PA','RH','Rn','G','VPD',
                    'h_canopy','LE','H','Delta','Density','Ra','r',
                    'Rs','beta','DateNumber','SiteNumber','date3',
                    'year']  # ,'fpar','PPFD_IN','SM','GPP','Rs_old'
    df = pd.DataFrame(data_all,columns=column_names)
    
#    df['Rs'][df['Rs']==1] = np.nan

#    Rs_pd = pd.DataFrame(Rs, columns = ['logRs'])
#    Rs_pd = Rs_pd.dropna(axis=0,how='any')
#    Rs_pd.to_csv('./csvdata/Rs2.csv',columns=None, header=True, index=True)
#    df.to_csv('./csvdata/df.csv',columns=None, header=True, index=True)
    
    # set H and Rs greater than 0 and lower than 1000
#    df = filter_data2(df,['H_filter'],0,1000)
    print(df.isna().sum())
    df = filter_data2(df,['Rs'],-1000,6.907755278982137) # ,'Rs_old'，6.907755278982137
#    df = filter_data2(df,['LE'],0,1000) # ,'Rs_old'，6.907755278982137
    print(df.isna().sum())
    
    # make sure every variable used in training > 0
    df = filter_data2(df,var_names,-1000,10000000)
    print(df.isna().sum())
    
    # make sure every target variable used in training > 0
    # for log(Rs) as target, it should be -1000, 10000000
    df = filter_data2(df,target_names,-1000,10000000)
    print(df.isna().sum())
    
    # get 5%~95% percentile data
    names = filter_names #['Ca','Rn','G','Rs']
    # drop nan first to make sure the percentiles calculation right in filter_data function
    # filter_data can set the data outside 5%~95% as NAN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=0,how='any')
#    df = df.dropna(axis=0,subset=var_names)
    print(df.isna().sum())
    
    # filter_data can set the data outside 5%~95% as NAN
    df = filter_data(df,names,5,95)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=0,how='any')
#    df = df.dropna(axis=0,subset=var_names)
    #    print(df)
    print(df.isna().sum())
    
    # choose all PFT or per PFT 
    df = df[df['PFT'].isin(PFT_names)]
    print(df['PFT'].unique())

#    df.to_csv('./csvdata/df.csv',columns=None, header=True, index=True)

    # if per PFT std = 0, NAN, so drop PFT
    if len(PFT_names)==1:
        df = df.drop('PFT',axis=1)
    else:
        df = df
    
    print(df.isna().sum())
    
    # Shuffle all the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    # keep the rain-free dataset and rain events dataset has the same data size
#    df = df[0:67130]
#    df = df[0:56008]
#    df = df[0:66802]
#    df = df[0:25039]

#    df.to_csv('./csvdata/df.csv',columns=None, header=True, index=True)

    # seperate all the data into Train dataset and test dataset 
    df_all = df
    num = int(len(df)*0.8)
    # variable_names: variables used for predicting Rs in the training model
    # target_names  : target variables 
    variable_names = var_names #['fpar','SM','PFT','TA','Ca','WS','PA','RH','Rn','G','VPD']
#    variable_names = ['fpar','SM','PFT','TA','Ca','WS','PA','RH','Rn','G','VPD','LE']
    target_names  = target_names #['Rs']
    # variables used in calculating LE Prediction values 
#    var_use_pre     = ['Delta','Density','Ra','r','VPD','Rn','G','LE','H','TA','Rs','Rs_old','beta']
#    var_use_pre     = ['Rn','G','VPD','Delta','Density','Ra','r','TA','beta','H','GPP','PFT','PA']
    var_use_pre     = ['Rn','G','VPD','Delta','Density','Ra','r','TA','beta','PFT','PA']
    
    # var_pre: LE prediction need var_pre dataset when calculation
    var_pre      = df.loc[:,var_use_pre]
    
    # SiteNumber and DateNumber
    var_site_date     = ['SiteNumber','date3','year']
    
    # data: all data used in the study
    data = df.loc[:,variable_names]
    # variable_names2: variables added LE, Rs, and Rs_old
    variable_names2 = variable_names + (('LE'),('Rs'),)  # ('GPP'),('Rs_old'),
    # train_data: all data used in training model, 
    # partial_x_train, partial_y_train: train_subset, account for 0.8 of train_data
    # x_val, y_val: validate subset, account for 0.2 of train_data
    num_val      = int(num*0.8)
    
    np.random.seed(2021)
    rand_inds = np.random.permutation(len(df))
    train_inds = rand_inds[0:num_val]
    val_inds = rand_inds[num_val:num]
    test_inds = rand_inds[num:len(df)]

    train_data   = df.loc[:,variable_names2].iloc[train_inds,:].reset_index(drop=True)
    val_data     = df.loc[:,variable_names2].iloc[val_inds,:].reset_index(drop=True)
    test_data    = df.loc[:,variable_names2].iloc[test_inds,:].reset_index(drop=True)

#    train_data   = df.loc[:,variable_names2][0:num_val].reset_index(drop=True)
#    val_data     = df.loc[:,variable_names2][num_val:num].reset_index(drop=True)
#    test_data    = df.loc[:,variable_names2][num:len(df)].reset_index(drop=True)
#    train_data.to_csv('./csvdata/train_data.csv',columns=None, header=True, index=True)
    
#    train_data.to_csv('./csvdata/train_data.csv',columns=None, header=True, index=True)
#    test_data.to_csv('./csvdata/test_data.csv',columns=None, header=True, index=True)
#    auxi_train=var_pre[0:num_val].reset_index(drop=True)
#    auxi_train.to_csv('./csvdata/auxi_train.csv',columns=None, header=True, index=True)
#    auxi_test=var_pre[num:len(df)].reset_index(drop=True)
#    auxi_test.to_csv('./csvdata/auxi_test.csv',columns=None, header=True, index=True)

    # Auxi
    train_auxi_data   = np.array(var_pre.iloc[train_inds,:].reset_index(drop=True).T)
    val_auxi_data     = np.array(var_pre.iloc[val_inds,:].reset_index(drop=True).T)
    test_auxi_data    = np.array(var_pre.iloc[test_inds,:].reset_index(drop=True).T)
    
#    train_auxi_data   = np.array(var_pre[train_inds].reset_index(drop=True).T)
#    val_auxi_data     = np.array(var_pre[val_data].reset_index(drop=True).T)
#    test_auxi_data    = np.array(var_pre[test_inds].reset_index(drop=True).T)
    
    # SiteNumber and DateNumber
    Date_train   = np.array(df.loc[:,var_site_date].iloc[train_inds,:].reset_index(drop=True).T)
    Date_val     = np.array(df.loc[:,var_site_date].iloc[val_inds,:].reset_index(drop=True).T)
    Date_test    = np.array(df.loc[:,var_site_date].iloc[test_inds,:].reset_index(drop=True).T)
    
#    print('train_auxi_data>>>>>>>>>>>>>>>>>>>>>')
#    print(train_auxi_data)
    
#    train_set_y_orig = tf.transpose(train_data[target_names])
#    test_set_y_orig  = tf.transpose(test_data[target_names])
    train_set_y_orig = np.array(train_data[target_names].T)
    val_set_y_orig   = np.array(val_data[target_names].T) 
    test_set_y_orig  = np.array(test_data[target_names].T)
    
    # LE
    LE_train = np.array(train_data[['LE']].T)
    LE_val   = np.array(val_data[['LE']].T) 
    LE_test  = np.array(test_data[['LE']].T)
    
#    # LE
#    LE_train = np.array(train_data[['GPP']].T)
#    LE_val   = np.array(val_data[['GPP']].T) 
#    LE_test  = np.array(test_data[['GPP']].T)
    
#    # Normalized all the data
#    mean = train_data.mean(axis=0)
#    std = train_data.std(axis=0)
#    print('mean>>>')
#    print(mean)
#    print('std>>>')
#    print(std)
    # obtain the mean and std value of test value to see the if histogram of 
    # train and test dataset are similar after shuffling
#    test_mean = test_data.mean(axis=0)
#    test_std = test_data.std(axis=0)
    # when choose per PFT, the PFT std value will be 0, which results in a nan when normalized
    # drop column PFT when choose specific PFT value 
#    train_data = (train_data - mean) / std
#    val_data = (val_data - mean) / std
#    test_data  = (test_data - mean) / std
#    print(train_data)
#    print('----------------')
#    print(variable_names)
#    print('----------------')
#    print(train_data.loc[:,variable_names])
##    train_set_x_orig = tf.transpose(train_data[variable_names].reset_index(drop=True))    
##    test_set_x_orig = tf.transpose(test_data[variable_names].reset_index(drop=True))
    train_set_x_orig = np.array(train_data.loc[:,variable_names].reset_index(drop=True).T)
    val_set_x_orig = np.array(val_data.loc[:,variable_names].reset_index(drop=True).T)     
    test_set_x_orig  = np.array(test_data.loc[:,variable_names].reset_index(drop=True).T)        

    return df_all,train_set_x_orig, train_set_y_orig, val_set_x_orig, val_set_y_orig, \
test_set_x_orig, test_set_y_orig, train_auxi_data, val_auxi_data, test_auxi_data, \
LE_train, LE_val, LE_test, Date_train, Date_val, Date_test, var_names, PFT_names, \
epochs, neurons_n, hidden_layer_n, num_steps, steps