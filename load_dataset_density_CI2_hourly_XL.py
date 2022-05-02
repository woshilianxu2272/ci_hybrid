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
import datetime
import load_dataset_density_for_CI


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

    
# set input data value  'np.inf' 
def filter_data3(data,names,num_ign):
    
    df = data
    for name in names:
        df[name][df[name] == num_ign] = np.nan
        
    return df
   

# solve equation for numpy array
def quadratic(a,b,c):
    p=b*b-4*a*c
    solu = []
    for i in range(0,len(a)):       
        #quadratic equation with one unknown
        if p[i]>=0 and a[i]!=0:
            x1=(-b[i]+np.sqrt(b[i]**2-4*a[i]*c[i]))/(2*a[i])
#            x2=(-b[i]-math.sqrt(p[i]))/(2*a[i])
#            solu.append(max(x1,x2))
            # Acturally, b and H(sensible heat) should greater than 0
            # so only x1 is the true solution 
            solu.append(x1)
        #linear equation with one unknown 
        elif a[i]==0:
        	x1=x2=-c/b
        	solu.append(x1)
        else:
          solu.append(np.nan) 
    
    return solu

# PM_quad_plt2: calculate the LE value using quadratic equation before ploting
def PM_quad_plt2(auxi, Rs_pre):
    # LE_pre need these params to calculate
    Rn_1 = auxi['Rn']
    G_1  = auxi['G']
    VPD_1 = auxi['VPD']
    Delta_1 = auxi['Delta']
    Density_1 = auxi['Density']
    Ra_1 = auxi['Ra']
    r_1 = auxi['r']
    TA_1 = auxi['TA']
    beta_1 = auxi['beta']
    Cp = 1012
    
#    tf.clip_by_value(Rs_pre,1e-5,1000)
    
    a = (1/2.0)*beta_1*(Ra_1/(Density_1*Cp))**2
    b = (Delta_1*Ra_1)/(Density_1*Cp)+r_1*(Ra_1+Rs_pre)/(Density_1*Cp)        
    c = VPD_1-r_1*(Ra_1+Rs_pre)*(Rn_1-G_1)/(Density_1*Cp)
    disc = b**2-4*a*c
    H_pre = quadratic(a,b,c)
    
    # the LE calculation results
    LE_pre = Rn_1 - G_1 - H_pre
    
    return LE_pre, H_pre, b


# get the X_train, X_val, X_test dataset and Y_train, Y_val, Y_test, 
# LE_true_train, LE_true_val, LE_true_test
def input_data(num,target_names):
    # define target_names and filter_names (5~95%)    
    filter_names = ['Ca','Rn','G']  
#    filter_names = ['Rn','G'] 
    
    # file_path contains all the combinations between vars and PFTs
#    file_path = './statistics/PFT_8.csv'
    file_path ='E:/workspace/2021.Canopy_Interception/model_and_code/statistics/test_tensorflow_repeat_hourly_XL.csv'
    dt = pd.read_csv(file_path)
    dt = dt.loc[:,['var','PFT','neurons_n','hidden_layer_n','epochs',
                   'num_steps','steps','Rain']]    
    
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
    
    Rain = dt['Rain'][num]

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
    print(var_inputs)
    var_names = var_inputs
    PFT_names = PFT_inputs
#    target_names = ['LE']
    target_names = target_names

#    df_veg = pd.read_csv('./fluxdata/'+'Fluxnet_sites_Hysteresis.csv',
#                               encoding='gb2312',sep=',')
    
    # Inputva_21 is for latent heat flux
    # Inputva_23_G.nc is for ground heat flux
    # Inputva_P9.nc is close path rain events from  yeqiang's program
    # Inputva_P11_CP=0.nc is open path rain events from  yeqiang's program
    
#    nc_obj=Dataset('./fluxdata/Inputva_21.nc')
#    nc_obj=Dataset('./fluxdata/Inputva_G_11.nc')
#    nc_obj=Dataset('./fluxdata/Inputva_P9.nc')
    nc_obj=Dataset('E:/workspace/2021.Canopy_Interception/Dataset/Inputva_withrain_hourly_XL.nc')
#    nc_obj=Dataset('./fluxdata/Inputva_P11_CP=0.nc')
#    nc_obj=Dataset('./fluxdata/Inputva_P11_CP=1.nc')

#    nc_obj=Dataset('./fluxdata/Inputva_P13_CP=1.nc')
#    nc_obj=Dataset('./fluxdata/Inputva_P13_CP=0.nc')
#    nc_obj=Dataset('./fluxdata/Inputva_G_11_2.nc')
#    nc_obj=Dataset('./fluxdata/Inputva_G_11_3.nc')
#    fpar = nc_obj.variables['fpar'][:]
#    SM = nc_obj.variables['SM'][:]
    VPD = nc_obj.variables['VPD'][:]
    PFT = nc_obj.variables['PFT'][:]
    TA = nc_obj.variables['TA'][:]
    Ca = nc_obj.variables['Ca'][:]
    Rn = nc_obj.variables['Rn'][:]
    G = nc_obj.variables['G'][:]
#    LE = nc_obj.variables['LE'][:]
    LE = nc_obj.variables['LEclose'][:]
#    H = nc_obj.variables['H'][:]
    WS = nc_obj.variables['WS'][:]
    PA = nc_obj.variables['PA'][:]
    RH = nc_obj.variables['RH'][:]
#    GPP = nc_obj.variables['GPP'][:]
#    GPP = 1.0
#    Gs = nc_obj.variables['Gs'][:]
#    Rs = nc_obj.variables['Rscorr'][:]    
#    SW_OUT = nc_obj.variables['SW_OUT'][:]
#    SW_IN = nc_obj.variables['SW_IN'][:]
#    LW_OUT = nc_obj.variables['LW_OUT'][:]
#    LW_IN = nc_obj.variables['LW_IN'][:]
    LAI = nc_obj.variables['LAI'][:]
    Delta = nc_obj.variables['Delta'][:]
    r = nc_obj.variables['r'][:]
    
    
#     rain_events = 'Inputva_P9.nc'
#     if rain_events == 'Inputva_P9.nc':
#         # preciptation
#         P = nc_obj.variables['P'][:]
#         SN = nc_obj.variables['SN'][:]
#         Timescale = nc_obj.variables['Timescale'][:]
#     else:
#         P = np.array(range(0,len(RH)))*0.0
#         SN = np.array(range(0,len(RH)))*0.0
#         Timescale = np.array(range(0,len(RH)))*0.0

    # preciptation
    P = nc_obj.variables['P'][:]
    SN = nc_obj.variables['SN'][:]
    Timescale = nc_obj.variables['Timescale'][:]
    sid = np.array(range(0,len(P)))
    
    # W_at1 = nc_obj.variables['W_at1'][:]
    # W_at2 = nc_obj.variables['W_at2'][:]
    # W_at3 = nc_obj.variables['W_at3'][:]
    # W_at4 = nc_obj.variables['W_at4'][:]
    # W_at5 = nc_obj.variables['W_at5'][:]
    # W_at6 = nc_obj.variables['W_at6'][:]
    

    Delta = nc_obj.variables['Delta'][:]
    Density = nc_obj.variables['Density'][:]
#    Ra = nc_obj.variables['Ra'][:]
#    Ra = 1.0
    r = nc_obj.variables['r'][:]
    h_canopy = nc_obj.variables['h_canopy'][:]
    Ra = nc_obj.variables['Ra'][:]
    
#    PPFD_IN = nc_obj.variables['PPFD_IN'][:]
    
    DateNumber = nc_obj.variables['DateNumber'][:]
    
    # the number of rain events (from 1,2,3,4,5.....)
    P_id = nc_obj.variables['P_id'][:]
    # total Precipitation for a rain event
    P_id_n = nc_obj.variables['P_id_n'][:]
    # rainfall duration for a rain event
    P_id_a = nc_obj.variables['P_id_a'][:]
    
    if ('WSc' in list(nc_obj.variables)) == True:
        WSc = nc_obj.variables['WSc'][:]
    else:
        WSc = nc_obj.variables['P_at6'][:]
                                
    
    
    
    
    # P_total = total Precipitation for a complete rain event
    # P_dur = rainfall duration for a complete rain event
    P_total = np.zeros((len(P_id)))
    P_dur   = np.zeros((len(P_id)))
    uni_P_id = np.unique(P_id)
    for i in uni_P_id:
        if i>0:
            i = int(i)
            P_total[P_id==i]=P_id_a[i-1]
            P_dur[P_id==i]=P_id_n[i-1]
        
    
    Pidn = np.zeros((len(P_id_n)))
    Pida = np.zeros((len(P_id_n)))
    for i in range(0,len(P_id_n)):
        Pidn[i]=len(P_id[P_id==i+1])
        if Pidn[i]>0:
            Pida[i]=sum(P[P_id==i+1])
        else:
            Pida[i]=0
    
    PPn = Pidn/P_id_n
    PPa = Pida/P_id_a
    VPn = len(PPn[(PPn>0.8) | (PPa>0.8)])
    
    
#    P=np.log(P);
#    W_at1=np.log(W_at1);
#    W_at2=np.log(W_at2);
#    W_at3=np.log(W_at3);
#    W_at4=np.log(W_at4);
#    W_at5=np.log(W_at5);
#    W_at6=np.log(W_at6);
#    W_at7=np.log(W_at7);
#    W_at8=np.log(W_at8);
#    W_at9=np.log(W_at9);
#    W_at10=np.log(W_at10);
#    W_at11=np.log(W_at11);
#    W_at12=np.log(W_at12);
#    P_total_a=np.log(P_total_a);
    
    # P_total_a = actual total Precipitation for a rain event after filter
    # P_dur_a = actual rainfall duration for a rain event after filter (number of half hour)
    P_total_a = np.zeros((len(P_id)))
    P_dur_a   = np.zeros((len(P_id)))
    uni_P_id = np.unique(P_id)
    for i in uni_P_id:
        if i>0:
            i = int(i)
            P_total_a[P_id==i]=Pida[i-1]
            P_dur_a[P_id==i]=Pidn[i-1]
            
    
    data_all = np.c_[sid,PFT,LAI,TA,Ca,WS,PA,RH,Rn,G,
                     VPD,h_canopy,Ra,LE,
                     DateNumber,P,                     
                     Delta,Density,r,P_id,
                     P_total,P_dur,P_total_a,P_dur_a,
                     WSc]  #fpar,PPFD_IN,SM, W_at1, W_at2, W_at3, W_at4, W_at5, W_at6

    #    h = data_all.shape
    
    # delete the rows and columns which contains NAN

    column_names = ['SID','PFT','LAI','TA','Ca','WS','PA','RH','Rn','G',
                     'VPD','h_canopy','Ra','LE',
                     'DateNumber','P',
                     'Delta','Density','r','P_id',
                     'P_total','P_dur','P_total_a','P_dur_a',
                     'WSc'] # 'fpar','PPFD_IN','SM','W_at1', 'W_at2','W_at3','W_at4', 'W_at5', 'W_at6'
    
    df = pd.DataFrame(data_all,columns=column_names)
    
    # maybe yeqiang's code error? P_id has value with 0, just drop it here
    # P_id_n have 11604, and the P_id max is 11604, They are corresponding
    df[df['P_id']==0] = np.nan
    
    
    ######################### calculate Ra start ##############################
#    # calculate Ra
#    uni = np.unique(SiteNumber)
#    # Inputs:
#    # z: Velocity reference height in meters
#    # h: Characteristic roughness height in meters
#    # v: Horizontal wind velocity at reference height (z) in m/s
#    df['h_measure'] = 1.0
#    for uni_i in uni:
#        df['h_measure'][(df['SiteNumber']==uni_i)] = df_veg['Tower_h'][uni_i-1]
#
#    df_z   = df['h_measure'].astype('float')
#    df_h   = df['h_canopy'].astype('float')
#    df_d   = (2.0/3.0)*df_h
#    df_z_0 = 0.1*df_h
#    df_d   = df_d.astype('float')
#    df_z_0 = df_z_0.astype('float')
#    WS     = df['WS'].astype('float')
#    
#    df['Ra'] = np.log((df_z-df_d)/df_z_0)**2.0/(0.41**2*WS)
#    print(df['Ra'])
    ######################### calculate Ra over ##############################


    ######################### calculate Rs and Rs_old start ###################
    # calculate Rs and Rs_old 
    Ra      = np.array(df['Ra'].astype('float'))
    TA      = np.array(df['TA'].astype('float'))
    Density = np.array(df['Density'].astype('float'))
    r       = np.array(df['r'].astype('float'))
    LE      = np.array(df['LE'].astype('float'))
    Rn      = np.array(df['Rn'].astype('float'))
    G       = np.array(df['G'].astype('float'))
    VPD     = np.array(df['VPD'].astype('float'))
    
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
    # Rs_old = (Density*1012/(r*LE))*((Delta*Ra*(Rn-G-LE)/(Density*1012))+VPD)-Ra 
    
    # try log(Rs_old)
    # Rs_old = np.log(Rs_old)
    
    df['beta']   = beta
    df['Rs']     = Rs
    # df['Rs_old'] = Rs_old
    ######################### calculate Rs and Rs_old over ###################
    
    
    ##########################################################################
    ###### re-calculate LE to keep they are same to true value start #########
    
    LE_re, H_pre, b=PM_quad_plt2(df,np.exp(Rs))
#    LE_re, H_pre, b=PM_quad_plt2(df,Rs)
    df['LE_re'] = LE_re
    df['error_LE_re'] = abs(LE_re-LE)
    
    
    ###### re-calculate LE to keep they are same to true value over ##########
    ##########################################################################    
#    if Rain == 'without':
#        df = df[df['P']<=0.0]
#    if Rain == 'with':
#        df = df[df['P']>0.0]
#    if Rain == 'all':
#        df = df

    if (('PFT' in var_names) == True):
        var_need = var_names + (('Rn'),('G'),('LE'),
                                ('LE_re'),('error_LE_re'),('P'),
                                ('Rs'),('VPD'),('Delta'),
                                ('Density'),('Ra'),('r'),('beta'),('PA'),('SID'),('P_id'),)  # ('Rs_old'),
    else:
        var_need = var_names + (('Rn'),('G'),('LE'),
                                ('LE_re'),('error_LE_re'),('P'),
                                ('Rs'),('VPD'),('Delta'),
                                ('Density'),('Ra'),('r'),('beta'),('PA'),
                                ('PFT'),('SID'),('P_id'),)   # ,('Rs_old')
    
    var_need = {}.fromkeys(var_need).keys()
    # 
#    df = group_hourly.mean()
    df = df.loc[:,var_need]
    print('df 0:')
    df.to_csv('./csvdata/df0.csv',columns=None, header=True, index=True)
    
    sel_cols = var_names + (('Rn'),('G'),('LE'),('P'),
                                ('Rs'),('VPD'),('Delta'),
                                ('Density'),('Ra'),('r'),('beta'),('PA'),
                                ('PFT'),) # ('Rs_old'),
    sel_cols = {}.fromkeys(sel_cols).keys()
    
    
    # set H and Rs greater than 0 and lower than 1000
#    df = filter_data2(df,['H_filter'],0,1000)
#    print(df.isna().sum())
#    df = filter_data2(df,['Rs','Rs_old'],-1000,6.907755278982137)
    df = filter_data2(df,['Rs'],-1000,6.907755278982137) # ,'Rs_old'，6.907755278982137
#    df = filter_data2(df,['LE'],0,1000) # ,'Rs_old'，6.907755278982137
#    print(df.isna().sum())
    print('df 1:')
    df.to_csv('./csvdata/df1.csv',columns=None, header=True, index=True)
    
    # make sure every variable used in training > 0
    df = filter_data2(df,var_names,-1000,10000000)
#    print(df.isna().sum())
    print('df 2:')
    df.to_csv('./csvdata/df2.csv',columns=None, header=True, index=True)
    
    # make sure every target variable used in training > 0
    # for log(Rs) as target, it should be -1000, 10000000
    df = filter_data2(df,target_names,-1000,10000000)
#    print(df.isna().sum())
    print('df 3:')
    df.to_csv('./csvdata/df3.csv',columns=None, header=True, index=True)
    
    # change inf to NAN for Albedo
    df = filter_data3(df,sel_cols,np.inf)
    print('df 4:')
    df.to_csv('./csvdata/df4.csv',columns=None, header=True, index=True)
    df = filter_data3(df,sel_cols,-np.inf)
    print('df 5:')
    df.to_csv('./csvdata/df5.csv',columns=None, header=True, index=True)
    df = df.dropna(axis=0,how='any')
    print('df 6:')
    df.to_csv('./csvdata/df6.csv',columns=None, header=True, index=True)
    # keep Albedo in [0,1]
#    df = df.astype('float')
#    df = filter_data2(df,['Albedo'],0.0,1.0)
#    df = df.dropna(axis=0,how='any')
    # if LE < 0, your constrained should be changed
    df = filter_data2(df,['LE','LE_re'],0.0,2000.0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=0,how='any')
    df = filter_data2(df,['error_LE_re'],0.0,1.0)
    df = df.dropna(axis=0,how='any')
    print('df 8:')
    df.to_csv('./csvdata/df8.csv',columns=None, header=True, index=True)
    
    df = df.astype('float')
#    df = filter_data2(df,sel_cols,-7000,7000)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=0,how='any')
    
    # get 5%~95% percentile data
    names = filter_names #['Ca','Rn','G','Rs']
    # drop nan first to make sure the percentiles calculation right in filter_data function
    # filter_data can set the data outside 5%~95% as NAN
#    df = df.dropna(axis=0,how='any')
#    df = df.dropna(axis=0,subset=var_names)
#    print(df.isna().sum())
    
    # filter_data can set the data outside 5%~95% as NAN
#    df = filter_data(df,names,5,95)
    # 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(axis=0,how='any')
    print('df 10:')
    df.to_csv('./csvdata/df10.csv',columns=None, header=True, index=True)
#    df = df.dropna(axis=0,subset=var_names)
    #    print(df)
#    print(df.isna().sum())
    
    # choose all PFT or per PFT 
    df = df[df['PFT'].isin(PFT_names)]
    print(df['PFT'].unique())
    print('df 105:')
    df.to_csv('./csvdata/df105.csv',columns=None, header=True, index=True)
    
    # if per PFT std = 0, NAN, so drop PFT
    if len(PFT_names)==1:
        df = df.drop('PFT',axis=1)
    else:
        df = df

#    print(df.isna().sum())
    print('df 11:')
    df.to_csv('./csvdata/df11.csv',columns=None, header=True, index=True)

    ##########################################################################
    ####################### Shuffle all the dataset start ####################    

    # if you want to get a same size dataset of rain-free and rain events dataset
    # close path
    # 265 rain-free
    # 266 rain
    # 267 all
    
    # open path
    # 268 rain-free
    # 269 rain
    # 270 all
#    # for rain-free dataset 
#    df_all,X_train,Y_train,X_val,Y_val,X_test,Y_test, Auxi_train, Auxi_val, Auxi_test,\
#    LE_train, LE_val, LE_test, Date_train, Date_val, Date_test, var_names, \
#    PFT_names, epochs, neurons_n, hidden_layer_n, num_steps, steps,\
#    = load_dataset_density_for_CI.input_data(265,['Rs'])
#    df = pd.concat([df_all,df])
#    df = df.reset_index(drop=True)
    
    # Shuffle all the dataset
    df_all = df.reset_index(drop=True) #re-coding the samples (1-n)
#    df = df.sample(frac=1).reset_index(drop=True)
#    df = df[0:50000]
    
#    print(df)
    ####################### Shuffle all the dataset end ######################
    ##########################################################################     
    
    # seperate all the data into Train dataset and test dataset 
    num = int(len(df)*0.8)
    # variable_names: variables used in the training model
    # target_names  : target variables 
    variable_names = var_names #['fpar','SM','PFT','TA','Ca','WS','PA','RH','Rn','G','VPD']
#    variable_names = ['fpar','SM','PFT','TA','Ca','WS','PA','RH','Rn','G','VPD','LE']
    target_names  = target_names #['Rs']
    # variables used in calculating LE Prediction values 
#    var_use_pre     = ['Delta','Density','Ra','r','VPD','Rn','G','LE','H','TA','Rs','Rs_old','beta']
    var_use_pre     = ['Rn','G','VPD','Delta','Density','Ra','r','TA','beta','PFT','PA']
    
#    var_use_pre     = ['Rn','G','VPD','Delta','Density','Ra','r','TA','beta','H','GPP','PFT','PA']    
    
    # var_pre: LE prediction need var_pre dataset when calculation
    var_pre      = df.loc[:,var_use_pre]
    
    # SiteNumber and DateNumber
    var_site_date     = ['P']
    
    # data: all data used in the study
    data = df.loc[:,variable_names]
    # variable_names2: variables added LE, Rs, and Rs_old
    variable_names2 = variable_names + (('LE'),('Rs'),)  #,('Rs_old')
    # train_data: all data used in training model, 
    # partial_x_train, partial_y_train: train_subset, account for 0.8 of train_data
    # x_val, y_val: validate subset, account for 0.2 of train_data
    num_val      = int(num*0.8)
    
    np.random.seed(2021)
    rand_inds = np.random.permutation(len(df))
    train_inds = rand_inds[0:num_val]
    val_inds = rand_inds[num_val:num]
    test_inds = rand_inds[num:len(df)]

    
    df.to_csv('./csvdata/df_rain.csv',columns=None, header=True, index=True)
    pd.DataFrame(rand_inds, columns = ['index']).to_csv('./csvdata/random_inds.csv',columns=None, header=True, index=True)
    
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
#    var_pre[0:num_val].to_csv('./csvdata/train_auxi_data.csv',columns=None, header=True, index=True)

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

    return df_all, train_set_x_orig, train_set_y_orig, val_set_x_orig, val_set_y_orig, \
test_set_x_orig, test_set_y_orig, train_auxi_data, val_auxi_data, test_auxi_data, \
LE_train, LE_val, LE_test, Date_train, Date_val, Date_test, var_names, PFT_names, \
epochs, neurons_n, hidden_layer_n, num_steps, steps












