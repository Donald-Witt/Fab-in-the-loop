#Copyright ©Donald Witt 2023. All rights reserved.
#Code used to train the spectral predictor
from network import Network

import object_cache
import FabricationEnv as FabricationEnv

import numpy as np
import random

import torch as T
import torch.nn as nn

import math
import plotly.express as px

def split(data, number_of_points):
    divide=len(data)/number_of_points
    divide=int(divide)
    
    reduced_data=[]
    
    for i in range(0,len(data),divide):
        sublist=data[i:i+divide]
        avg=0
        for j in sublist:
            avg+=float(j)
        reduced_data.append(float(avg/len(sublist)))
    if len(reduced_data)!=30:
        print("error!!")
        reduced_data.append(float(avg/len(sublist)))
    return reduced_data
    
def set_resolution(wavelengths,data,resolution):
    average_resolution=0
    try:
        previous_point=wavelengths[0]
    except:
        print(wavelengths)
    for i in range(1,len(wavelengths)):
        average_resolution+=wavelengths[i]-previous_point
        previous_point=wavelengths[i]
    
    average_resolution=average_resolution/len(wavelengths)
    divide=resolution/average_resolution
    divide=int(divide)
    
    reduced_data=[]
    
    for i in range(0,len(data),divide):
        sublist=data[i:i+divide]
        avg=0
        for j in sublist:
            avg+=float(j)
        reduced_data.append(float(avg/len(sublist)))
   
    return reduced_data
    
database=object_cache.database()
env=FabricationEnv.FabricationEnv()

wavelength_resolution=1
spectral_predictors=[]
alpha=0.0001
count=0
for i in range(1490,1640,wavelength_resolution):
    try:
        spectral_predictors.append(Network(alpha=alpha,input_dims=[16],fc1_dims=30,fc2_dims=60,fc3_dims=120,fc4_dims=120,fc5_dims=60,fc6_dims=30,n_actions=1,name='spectral_predictor_'+str(i),usemodel=False))
        spectral_predictors[count].load_checkpoint()
    except:
        pass
    count+=1
def normalize_power(data):
    norm=[]
    for point in data:
        if math.isnan(point):
            point=-120#min power
        norm.append((point+120)/(0+120))
        
    return norm
    
def unnormalize_power(norm):
    data=[]
    for point in norm:
        data.append(point*(0+120)-120)
        
    return data
    

def normalize_scaling(data):
    norm=[]
    for point in data:
        norm.append((point+30)/(30+30))
        
    return norm
    
avg_loss=[]
total_training=1
current_avg_loss=0
previous_avg_loss=10000

for loop in range(0,10000):
    device=database.retrive_device(random.randrange(0,database.database_metadata.lastindex))
    deviceparam=device.parameters
    devicedata=device.data
    scaling_vector=normalize_scaling([device.bias[0][1],device.bias[1][1],device.bias[2][1],device.bias[3][1]])
    
    obs=deviceparam
    float_obs=[]
    for j in obs:
        float_obs.append(float(j))
    obs=float_obs
    obs=env.normalize_device(obs)
        
    #add scaling data
    for j in scaling_vector:
        obs.append(j)
            
    obs=T.FloatTensor(obs)
        
    act=set_resolution(devicedata[0],devicedata[1],wavelength_resolution)
  
    unormalized_act=act
    act=normalize_power(act)
    reshaped_act=[]
    for point in act:
        reshaped_act.append([point])
    act=T.FloatTensor(reshaped_act)
    count_predictor=0
    sub_loss=0
    sub_training=1
    for spectral_predictor in spectral_predictors:
        if unormalized_act[count_predictor]<-30:
            if random.uniform(0,1)<0.6:
                continue
        #zero the gradient
        spectral_predictor.optimizer.zero_grad()
        #predict the action
        act_pred=spectral_predictor.forward(obs)
        
        #compute the loss
        loss_fn = nn.MSELoss(reduction='sum')
        
        loss=loss_fn(act_pred,act[count_predictor])
        loss.backward()
        spectral_predictor.optimizer.step()

        sub_loss+=loss.item()
        sub_training+=1
        if loop%500==0:
            spectral_predictor.save_checkpoint()

        count_predictor+=1
    current_avg_loss+=(sub_loss)/sub_training
    avg_loss.append(current_avg_loss/total_training)
    total_training+=1
    print("avg loss: "+str(current_avg_loss/total_training))
    loss_indices=[]
    loss_count=1
        
    for point in avg_loss:
        loss_indices.append(loss_count)
        loss_count+=1
    if loop%50==0:
        previous_avg_loss=(sub_loss)/sub_training
        fig=px.line(x=loss_indices,y=avg_loss,title="Average loss",labels={"x":"Round","y":"Average loss"})
        fig.write_html("loss.html",auto_open=False)

print("Final save")
for spectral_predictor in spectral_predictors:
    spectral_predictor.save_checkpoint()

print("done")
