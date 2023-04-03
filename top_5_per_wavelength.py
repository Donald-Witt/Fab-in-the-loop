#Copyright ©Donald Witt 2023. All rights reserved.
#find the top 5 devices per wavelength
import object_cache
import time
import random
import pickle
# get average of a list
def Average(lst):
        return sum(lst) / len(lst)
#find index of closest
def takeClosest(num,collection):
        return min(collection,key=lambda x:abs(x-num))
        
def score_spectrum(start,stop,wavelength,power):
    try:
        average=Average(power[wavelength.index(takeClosest(start,wavelength)):wavelength.index(takeClosest(stop,wavelength))])
    except:
        if stop<1630:
            average=Average(power[wavelength.index(takeClosest(start,wavelength)):wavelength.index(takeClosest(stop+5,wavelength))])
        else:
            average=Average(power[wavelength.index(takeClosest(start-5,wavelength)):wavelength.index(takeClosest(stop,wavelength))])
                    
    return average

#save
def save_obj(obj,name):
    with open(name + '.pkl', 'wb+') as f:
            pickle.dump(obj, f, 2)
    f.close()
    
database=object_cache.database()
wavelengths=[[1490,1500],[1530,1560],[1530,1560],[1600,1640],[1490,1640],[1545,1555],[1490,1610],[1580,1590],[1520,1570],[1510,1580]]
timetorun=0
count=0
best_devices=[]
best_per_range=[[],[],[],[],[],[],[],[],[],[]]
scores_per_range=[[],[],[],[],[],[],[],[],[],[]]

for point in range(0,5):
    for wavelength_range in wavelengths:
        best_score=-100
        best_device=-1
        start=time.time()
        for device_index in range(0,database.database_metadata.lastindex):
            #remove biased devices
            #if (device_index>example_1) and (device_index<example_2):
            #    continue
            device=database.retrive_device(device_index)
            data=device.data
            score=score_spectrum(wavelength_range[0],wavelength_range[1],data[0],data[1])
            if score>=best_score:
                already=False
                for index in best_per_range[wavelengths.index(wavelength_range)]:
                    if index == device_index:
                        already=True
                if already==False:
                    best_score=score
                    best_device=device_index
        best_per_range[wavelengths.index(wavelength_range)].append(best_device)
        scores_per_range[wavelengths.index(wavelength_range)].append(best_score)
        timetorun+=time.time()-start
        count+=1

for wavelength_range in wavelengths:
    best_devices.append([wavelength_range,best_per_range[wavelengths.index(wavelength_range)],scores_per_range[wavelengths.index(wavelength_range)]])
    
    
print(best_devices)
print(timetorun/count)
save_obj(best_devices,"top_5")
