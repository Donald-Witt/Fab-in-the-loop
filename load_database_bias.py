#Copyright ©Donald Witt 2023. All rights reserved.
#add the biased designs to the database
import object_cache
import os
import glob
import csv
import math

def filename_to_parameters(devicename):
    
    splitname=devicename.split("_")
   
    
    r=splitname[3]
    r=float(r.replace("r",""))
   
    a=float(splitname[5])
    
    x=int(splitname[7])
   
    minfeature=float(splitname[11])
    
    taperangle=int(splitname[14])
    
    hole_start=int(splitname[17])
    
    apstart=int(splitname[20])
    
    apend=int(splitname[23])
    
    laticefeature=float(splitname[26])
    
    divide=int(splitname[31])
    
    frontap=float(splitname[34])
    
    backap=float(splitname[37])

    #fix this in the future to include more bias parameters
    #if "bias" in devicename:
    #    biases=float(splitname[34])
    #else:
    #    biases=0
    biases=float(splitname[39])*2*1000#diameter

    datetime=splitname[-2]+"_"+splitname[-1].replace(".csv","")
    
    
    return [r,a,x,minfeature,taperangle,hole_start,apstart,apend,laticefeature,divide,frontap,backap],biases,datetime

def extract_datafromfile(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        csvdata = list(reader)
    wavlength=[]
    power1=[]
    power2=[]
    for point in csvdata:
        #drop any off scale powers
        if math.isnan(float(point[0])):
            continue
        if math.isnan(float(point[1])):
            power1.append(-100)
        if math.isnan(float(point[2])):
            power2.append(-100)
        wavlength.append(float(point[0])*1e9)
        power1.append(float(point[1]))
        power2.append(float(point[2]))
        
    return [wavlength,power1]


database=object_cache.database()
datalocations=glob.glob("./input data bias/*/")
for location in datalocations:
    notesfile=open(location+"/notes.txt","r")
    notes=notesfile.read()
    notesfile.close()
    
    biasfile=open(location+"/biases.txt","r")
    biasdata=biasfile.readlines()
    biasfile.close()
    chipbiases=[]
    for data in biasdata:
        data=data.split(",")
        chipbiases.append([float(data[0]),float(data[1])])
    
    datafiles=glob.glob(location+"*.csv")
    for file in datafiles:
        parameters,biases,datetime=filename_to_parameters(file.split("/")[-1])
        r=parameters[0]
        diameter=r*2
        bias=[]
        for i in chipbiases:
            bias.append(list(i))
            
        if diameter<0.100:
            bias[0][1]=bias[0][1]+biases
        elif diameter<0.120:
            bias[1][1]=bias[1][1]+biases
        elif diameter<0.140:
            bias[2][1]=bias[2][1]+biases
        else:
            bias[3][1]=bias[3][1]+biases
            
        data=extract_datafromfile(file)
        if len(data[0])==0:
            print("no data for device")
            continue
        data=extract_datafromfile(file)
        if max(data[1])>0:
            print("bad data saturated")
            continue
        elif max(data[1])<-40:
            print("bad data low power")
            continue
        database.add_device_to_db(parameters=parameters,bias=bias,data=extract_datafromfile(file))
