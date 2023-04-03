#Copyright ©Donald Witt 2023. All rights reserved.
#Code used to score the designs
import os
import csv
import math
import random
import pickle

from network import Network
import torch as T
import torch.nn as nn
import object_cache

class FabricationEnv():
    def __init__(self):
        self.count_episode=0#number of turns in a episode
        self.badscore_count=0
        self.best_score=-10000000
        self.new_designs=self.load_obj("new_devices")#list of the new designs
        self.database=object_cache.database()
        
        self.wavelength_resolution=1
        self.spectral_predictors=[]
        alpha=0.0001
        for i in range(1490,1640,self.wavelength_resolution):
            self.spectral_predictors.append(Network(alpha=alpha,input_dims=[16],fc1_dims=30,fc2_dims=60,fc3_dims=120,fc4_dims=120,fc5_dims=60,fc6_dims=30,n_actions=1,name='spectral_predictor_'+str(i),usemodel=True))
        for spectral_predictor in self.spectral_predictors:
            try:
                spectral_predictor.load_checkpoint()
            except:
                print("no data for wavelength")
        
    #takes device parameters "action" and returns score and done flag   
    def step(self,start,stop,action):
        action=list(action)
        action=self.unnormalize_device(action)
        self.count_episode=self.count_episode+1
        new_state=self.clip_device(action)
        new_state=self.normalize_device(new_state)
        
        #normalize the wavelength
        norm_start=(start-1490)/(1640-1490)
        norm_stop=(stop-1490)/(1640-1490)
    
        new_state.append(norm_start)
        new_state.append(norm_stop)
        [flag,reward]=self.score_device(start,stop,action)
        
        done=False
        if reward>1:
            print(action)#will record device parameters for future fab runs since it is far enough from previous devices
        
            self.new_designs.append([[float(action[0]),float(action[1]),int(action[2]),float(action[3]),float(action[4])
            ,int(action[5]),int(action[6]),int(action[7]),float(action[8]),int(action[9]),float(action[10]),float(action[11])],[start,stop],reward])
            
            self.save_obj(self.new_designs,"new_devices")

            #don't want to punish new designs
            self.badscore_count=0
            self.best_score=-10000000
        
        info=self.count_episode
        if self.count_episode>10:
            self.count_episode=0
            self.badscore_count=0
            self.best_score=-10000000
            done=True#end episode 
            return [new_state,reward,done,info]
            
        if reward<self.best_score:
            self.badscore_count=self.badscore_count+1
        else:
            self.best_score=reward
        
        if self.badscore_count>3:
            self.badscore_count=0
            self.best_score=-10000000
            done=True#end episode 
            return [new_state,reward,done,info]
            
        return [new_state,reward,done,info]
        
    
    #for compatibility with gym
    def reset(self):
        observation=self.normalize_device(self.database.retrive_device(random.randrange(0, self.database.database_metadata.lastindex).parameters))
        return observation 
        
    #for compatibility with gym
    def predict_device(self,normstart,normstop):
        obs=[]  
        obs.append(normstart)
        obs.append(normstop)
        obs=T.FloatTensor(obs)
        device=self.device_predictor.forward(obs)
        devicelist=[]
        for i in device:
            devicelist.append(i)
        return devicelist
    
    #normalize scaling
    def normalize_scaling(self,data):
        norm=[]
        for point in data:
            norm.append((point+30)/(30+30))
        
        return norm
        
    #normalize device
    def normalize_device(self,deviceparam):
        deviceclip=[]
        #r
        rmax=0.2
        rmin=0.035
        
        rn=(deviceparam[0]-rmin)/(rmax-rmin)
        
        #a    
        amax=0.5
        amin=0.035
        
        an=(deviceparam[1]-amin)/(amax-amin)
            
        #x  
        xmax=250
        xmin=80
        
        xn=(deviceparam[2]-xmin)/(xmax-xmin)
            
        #min feature 
        min_max=0.1
        min_min=0.01
        
        minn=(deviceparam[3]-min_min)/(min_max-min_min)
            
        #angle
        angle_max=45
        angle_min=10
        
        anglen=(deviceparam[4]-angle_min)/(angle_max-angle_min)
            
        #start
        start_max=70
        start_min=0
        
        startn=(deviceparam[5]-start_min)/(start_max-start_min)
          
        #ap start
        apstart_max=70
        apstart_min=0
        
        apstartn=(deviceparam[6]-apstart_min)/(apstart_max-apstart_min)  
        
        #ap end
        apend_max=250
        apend_min=20
        
        apendn=(deviceparam[7]-apend_min)/(apend_max-apend_min)  
            
        #lattice feature
        lattice_max=0.15
        lattice_min=0.05
        
        latticen=(deviceparam[8]-lattice_min)/(lattice_max-lattice_min) 
        
            
        #divide
        divide_max=150
        divide_min=10
        
        dividen=(deviceparam[9]-divide_min)/(divide_max-divide_min) 
            
        #front
        front_max=0.1
        front_min=-0.1
        
        frontn=(deviceparam[10]-front_min)/(front_max-front_min) 
            
        #back
        back_max=0.1
        back_min=-0.1
        
        backn=(deviceparam[11]-back_min)/(back_max-back_min)
        
        return_data=[rn,an,xn,minn,anglen,startn,apstartn,apendn,latticen,dividen,frontn,backn]
        for i in deviceparam[12:]:
            return_data.append(i)
        return return_data
        
    #reverse normalization
    def unnormalize_device(self,deviceparam):
        deviceclip=[]
        #r
        rmax=0.2
        rmin=0.035
        
        r=deviceparam[0]*(rmax-rmin)+rmin
        
        #a    
        amax=0.5
        amin=0.035
        
        a=deviceparam[1]*(amax-amin)+amin
        
        #x  
        xmax=250
        xmin=80
        
        x=deviceparam[2]*(xmax-xmin)+xmin
            
        #min feature 
        min_max=0.1
        min_min=0.01
        
        min=deviceparam[3]*(min_max-min_min)+min_min
            
        #angle
        angle_max=45
        angle_min=10
        
        angle=deviceparam[4]*(angle_max-angle_min)+angle_min
            
        #start
        start_max=70
        start_min=0
        
        startv=deviceparam[5]*(start_max-start_min)+start_min
          
        #ap start
        apstart_max=70
        apstart_min=0
        
        apstart=deviceparam[6]*(apstart_max-apstart_min)+apstart_min 
        
        #ap end
        apend_max=250
        apend_min=20
        
        apend=deviceparam[7]*(apend_max-apend_min)+apend_min   
            
        #lattice feature
        lattice_max=0.15
        lattice_min=0.05
        
        lattice=deviceparam[8]*(lattice_max-lattice_min)+lattice_min 
        
            
        #divide
        divide_max=150
        divide_min=10
        
        divide=deviceparam[9]*(divide_max-divide_min)+divide_min 
            
        #front
        front_max=0.1
        front_min=-0.1
        
        front=deviceparam[10]*(front_max-front_min)+front_min 
            
        #back
        back_max=0.1
        back_min=-0.1
        
        backv=deviceparam[11]*(back_max-back_min)+back_min
        
        #bias 80nm
        bias80_max=0.02
        bias80_min=-0.02
        
        backv=deviceparam[11]*(back_max-back_min)+back_min
        return [r,a,x,min,angle,startv,apstart,apend,lattice,divide,front,backv]
    
    #internal functions  
    def clip_device(self,deviceparam):
        deviceclip=[]
        #r
        if deviceparam[0]>0.2:
            deviceclip.append(0.2)
        elif deviceparam[0]<0.035:
            deviceclip.append(0.035)
        else:
            deviceclip.append(deviceparam[0])
        
        #a    
        if deviceparam[1]>0.5:
            deviceclip.append(0.5)
        elif deviceparam[1]<0.035:
            deviceclip.append(0.035)
        else:
            deviceclip.append(deviceparam[1])
            
        #x    
        if deviceparam[2]>250:
            deviceclip.append(250)
        elif deviceparam[2]<80:
            deviceclip.append(80)
        else:
            deviceclip.append(deviceparam[2])
            
        #min feature 
        if deviceparam[3]>0.1:
            deviceclip.append(0.1)
        elif deviceparam[3]<0.01:
            deviceclip.append(0.01)
        else:
            deviceclip.append(deviceparam[3])
            
        #angle
        if deviceparam[4]>45:
            deviceclip.append(45)
        elif deviceparam[4]<10:
            deviceclip.append(10)
        else:
            deviceclip.append(deviceparam[4])
            
        #start
        if deviceparam[5]>70:
            deviceclip.append(70)
        elif deviceparam[5]<0:
            deviceclip.append(0)
        else:
            deviceclip.append(deviceparam[5])
            
        #ap start 
        if deviceparam[6]>70:
            deviceclip.append(70)
        elif deviceparam[6]<0:
            deviceclip.append(0)
        else:
            deviceclip.append(deviceparam[6])
            
        #ap end
        if deviceparam[7]>250:
            deviceclip.append(70)
        elif deviceparam[7]<20:
            deviceclip.append(0)
        else:
            deviceclip.append(deviceparam[7])
        
        #lattice feature
        if deviceparam[8]>0.15:
            deviceclip.append(0.15)
        elif deviceparam[8]<0.05:
            deviceclip.append(0.05)
        else:
            deviceclip.append(deviceparam[8])
            
        #divide
        if deviceparam[9]>150:
            deviceclip.append(150)
        elif deviceparam[9]<10:
            deviceclip.append(10)
        else:
            deviceclip.append(deviceparam[9])
            
        #front
        if deviceparam[10]>0.1:
            deviceclip.append(0.1)
        elif deviceparam[10]<-0.1:
            deviceclip.append(-0.1)
        else:
            deviceclip.append(deviceparam[10])
            
        #back
        if deviceparam[11]>0.1:
            deviceclip.append(0.1)
        elif deviceparam[11]<-0.1:
            deviceclip.append(-0.1)
        else:
            deviceclip.append(deviceparam[11])
        
        return deviceclip
    
    #take into account actions outside allowed range and penalize 
    def score_device(self,start,stop,deviceparam):
        
        score=0
        #r
        if deviceparam[0]>0.2:
            score=score-1-abs(deviceparam[0]/0.2)*2    
        elif deviceparam[0]<0.035:
            score=score-1-abs(0.035/deviceparam[0])*2
        elif math.isnan(deviceparam[0])==True:
            print("nan  error")
            score=score-1
            
        #a
        if deviceparam[1]>0.5:
            score=score-1-abs(deviceparam[1]/0.5)*2
        elif deviceparam[1]<0.035:
            score=score-1-abs(0.035/deviceparam[1])*2
        elif math.isnan(deviceparam[1])==True:
            print("nan error")
            score=score-1
           
        #x 
        if deviceparam[2]>250:
            score=score-1-abs(deviceparam[2]/250)*2
        elif deviceparam[2]<80:
            score=score-1-abs(80/deviceparam[2])*2
        elif math.isnan(deviceparam[2])==True:
            print("nan error")
            score=score-1
        
        #min feature
        if deviceparam[3]>0.1:
            score=score-1-abs(deviceparam[3]/0.1)*2
        elif deviceparam[3]<0.01:
            score=score-1-abs(0.01/deviceparam[3])*2
        elif math.isnan(deviceparam[3])==True:
            print("nan error")
            score=score-1
            
        #angle 
        if deviceparam[4]>45:
            score=score-1-abs(deviceparam[4]/45)*2
        elif deviceparam[4]<10:
            score=score-1-abs(10/deviceparam[4])*2
        elif math.isnan(deviceparam[4])==True:
            print("nan error")
            score=score-1
            
            
        #start 
        if deviceparam[5]>70:
            score=score-1-abs(deviceparam[5]/70)*2
        elif deviceparam[5]<0:
            score=score-1-abs(0-deviceparam[5])*2
        elif math.isnan(deviceparam[5])==True:
            print("nan error")
            score=score-1
               
        #ap start 
        if deviceparam[6]>70:
            score=score-1-abs(deviceparam[6]/70)*2
        elif deviceparam[6]<0:
            score=score-1-abs(0-deviceparam[6])*2
        elif math.isnan(deviceparam[6])==True:
            print("nan error")
            score=score-1
            
        #ap end 
        if deviceparam[7]>250:
            score=score-1-abs(deviceparam[7]/250)*2
        elif deviceparam[7]<0:
            score=score-1-abs(20/deviceparam[7])*2
        elif math.isnan(deviceparam[7])==True:
            print("nan error")
            score=score-1
            
        #lattice feature 
        if deviceparam[8]>0.15:
            score=score-1-abs(deviceparam[8]/0.15)*2
        elif deviceparam[8]<0.05:
            score=score-1-abs(0.05/deviceparam[8])*2
        elif math.isnan(deviceparam[8])==True:
            print("nan error")
            score=score-1
            
        #divide 
        if deviceparam[9]>150:
            score=score-1-abs(deviceparam[9]/150)*2
        elif deviceparam[9]<10:
            score=score-1-abs(10/deviceparam[9])*2
        elif math.isnan(deviceparam[9])==True:
            print("nan error")
            score=score-1
            
        #front 
        if deviceparam[10]>0.1:
            score=score-1-(deviceparam[10]/0.1)*2
        elif deviceparam[10]<-0.1:
            score=score-1-abs(0.1/deviceparam[10])*2
        elif math.isnan(deviceparam[10])==True:
            print("nan error")
            score=score-1
            
            
        #back 
        if deviceparam[11]>0.1:
            score=score-1-abs(deviceparam[11]/0.1)*2
        elif deviceparam[11]<-0.1:
            score=score-1-abs(0.1/deviceparam[11])*2
        elif math.isnan(deviceparam[11])==True:
            print("nan error")
            score=score-1
            
        if score<0:
            print("out of range")
            print(10**(score/22))
            return [0,10**(score/22)]
        
        score=1
        
        normalized_deviceparam=self.normalize_device(deviceparam)
        [wavelength,power]=self.predict(normalized_deviceparam)

        try:
            average=self.Average(power[wavelength.index(self.takeClosest(start,wavelength)):wavelength.index(self.takeClosest(stop,wavelength))])
        except:
            if stop<1630:
                average=self.Average(power[wavelength.index(self.takeClosest(start,wavelength)):wavelength.index(self.takeClosest(stop+5,wavelength))])
            else:
                average=self.Average(power[wavelength.index(self.takeClosest(start-5,wavelength)):wavelength.index(self.takeClosest(stop,wavelength))])
        score+=((average+120)/(0+120))
        if deviceparam[1]>deviceparam[0]:
            return [1,10**score]
        return [2,10**(score-(deviceparam[0]/deviceparam[1]))]#not preferred but ok depending on how bad
        
    
    #predict the spectrum
    def predict(self,inputdata):
        wavelength=[]
        wavelength_count=0
        normpower=[]
        scaling_vector=self.normalize_scaling([4, 4, 8, 8])#current from the process hard code for now!!!!
        input_parameters=inputdata.copy()
        
        for i in scaling_vector:
            input_parameters.append(i)
        
        for point in range(1490,1640,self.wavelength_resolution):
            wavelength.append(point)
            spectral_predictor=self.spectral_predictors[wavelength_count]
            normpower.append(float(spectral_predictor.forward(T.FloatTensor(input_parameters))))
            wavelength_count+=1
        return [wavelength,self.unnormalize_power(normpower)]
        
    def unnormalize_power(self,norm):
        data=[]
        for point in norm:
            data.append(point*(0+120)-120)
        
        return data
      
    #pick a divice at random and normalize it's parameters
    def random_action(self):
        deviceparam=self.database.retrive_device(random.randrange(0, self.database.database_metadata.lastindex).parameters)
        return self.normalize_device(deviceparam)
        
    def device_features(self,start,stop,deviceparam):
        devicedata=self.devicedb.retrive_device_data(deviceparam)
        
        if devicedata==-1:
            return -1
            
        wavelength=devicedata[0]
        power=devicedata[1]
        
        try:
            average_power=self.Average(power[wavelength.index(self.takeClosest(start,wavelength)):wavelength.index(self.takeClosest(stop,wavelength))])
        except:
            average_power=-200
            
        peak=max(power)
        
        peak_location=power.index(peak)
        peak_wavelength=wavelength[peak_location]
        
        return [average_power,peak,peak_wavelength]
         
    #find index of closest
    def takeClosest(self,num,collection):
        return min(collection,key=lambda x:abs(x-num))
   
    # get average of a list 
    def Average(self,lst): 
        return sum(lst) / len(lst) 
    
    #save an object 
    def save_obj(self,obj,name):
        with open(name + '.pkl', 'wb+') as f:
            pickle.dump(obj, f, 2)
        f.close()
        
    #load an object
    def load_obj(self,name):
        while True:
            data=[]
            try:
                with open(name+'.pkl','rb') as f:
                    data=pickle.load(f)
                f.close()
                break
            except FileNotFoundError:
                with open(name+'.pkl','wb+') as f:
                    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
                f.close()
                break
            except EOFError:
                print("EOFError!!!!!!!")
                break
        return data
