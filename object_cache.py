#Copyright ©Donald Witt 2023. All rights reserved.
#Code used for device database
from pymemcache.client import base
from pymemcache import serde
import pickle
import time

class database_metadataobj():
    def __init__(self):
        self.datapath="./database/"
        self.lastindex=-1
        self.database_updatetime=time.time()
        
class database_table():
    def __init__(self):
        self.data=[]
        
class deviceobj():
    def __init__(self,deviceid,parameters,bias,data,timestamp,notes):
          self.deviceid=deviceid
          self.parameters=parameters
          self.bias=bias
          self.data=data
          self.timestamp=timestamp
          self.notes=notes
        
class database():
    def __init__(self):
        self.client = base.Client(('localhost', 11211), serde=serde.pickle_serde)
        try:
            self.database_metadata=self.load_obj("database_metadata")
        except FileNotFoundError:
            self.database_metadata=database_metadataobj()
            self.save_obj(self.database_metadata,"database_metadata")
        try:
            self.maintable=self.load_obj("database_maintable")
        except:
            self.maintable=database_table()
            self.save_obj(self.maintable,"database_maintable")
        
        #load all the devices
        if self.database_metadata.lastindex==-1:
            #no data in database
            pass
        else:
            current_device=0
            while current_device<=self.database_metadata.lastindex:
                #load the device from file
                device=self.load_obj(self.database_metadata.datapath+str(current_device))
                
                while True:
                    result, cas=self.client.gets(str(device.deviceid))
                    #store the device in memory cache
                    if self.client.set(str(device.deviceid),device):
                        break
                    #cache colision retry
                
                current_device+=1
                
    def add_device_to_db(self,parameters,bias,data,timestamp="",notes=""):
        self.database_metadata.lastindex+=1
        deviceid=self.database_metadata.lastindex
        device=deviceobj(deviceid=self.database_metadata.lastindex,parameters=parameters,bias=bias,data=data,timestamp=timestamp,notes=notes)
        
        #save the device
        self.save_obj(device,self.database_metadata.datapath+str(device.deviceid))
        
        #update the modified time
        self.database_metadata.database_updatetime=time.time()
        
        #save the metadata
        self.save_obj(self.database_metadata,"database_metadata")
        
        #add device to table
        self.add_to_table(device.deviceid,device.parameters)
        
        #add device to memory cache
        self.client.set(str(device.deviceid),device)
    
    def add_to_table(self,id,parameters):
        current_table=self.maintable.data
        updated_table=[]
        
        try:
            id_list=current_table[0]
            empty_table=False
        except IndexError:
            empty_table=True
            id_list=[]
        id_list.append(id)
        updated_table.append(id_list)
        
        element=1
        for parameter in parameters:
            if empty_table==False:
                elementlist=current_table[element]
            else:
                elementlist=[]
            elementlist.append(parameter)
            updated_table.append(elementlist)
            element+=1
        
        self.maintable.data=updated_table
        self.save_obj(self.maintable,"database_maintable")
    
    def retrive_device(self,deviceid):
        device=self.client.get(str(deviceid))
        
        if device is None:
            #device not in cache
            device=self.load_obj(self.database_metadata.datapath+str(deviceid))
                
            while True:
                result, cas=self.client.gets(str(device.deviceid))
                #store the device in memory cache
                if self.client.set(str(device.deviceid),device):
                    break
                #cache colision retry
        return device
        
    def save_obj(self,obj,name):
        with open(name+'.pkl','wb+') as f:
            pickle.dump(obj,f,4)
            f.close()
    
    def load_obj(self,name):
        while True:
            data={}
            try:
                with open(name+'.pkl','rb') as f:
                    data=pickle.load(f)
                f.close()
                break
            except EOFError:
                print("EOFError!!!!!!!")
        return data
