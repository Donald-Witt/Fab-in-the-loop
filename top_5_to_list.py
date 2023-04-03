#Copyright ©Donald Witt 2023. All rights reserved.
#create a list version of the top 5 devices for fabrication

import pickle
from operator import itemgetter

#max per file
max_file=625

fab_list=[]

def load_obj(name):
    while True:
        data=[]
        try:
            with open(name+'.pkl','rb') as f:
                data=pickle.load(f)
            f.close()
            break
        except EOFError:
            print("EOFError!!!!!!!")
            break
    return data

def save_obj(obj,name):
    with open(name+'.pkl','wb+') as f:
        pickle.dump(obj,f,4)
        f.close()

new_devices=load_obj("top_5")
import object_cache
database=object_cache.database()
for devices in new_devices:
    for index in devices[1]:
        parameters=database.retrive_device(index).parameters
        fab_list.append([parameters,devices[0]])
        
#split list into max beamer size
count=1
for i in range(0, len(fab_list), max_file):
    save_obj(fab_list[i:i + max_file],"klayout_top5_devices"+str(count))
    count=count+1
