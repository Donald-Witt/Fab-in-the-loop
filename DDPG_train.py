#Copyright ©Donald Witt 2023. All rights reserved.
#Code used to train the DDPG portion of the fab-in-the-loop algorithm
from DDPG import Agent
import FabricationEnv as FabricationEnv
import numpy as np
import random
import object_cache
database=object_cache.database()
env=FabricationEnv.FabricationEnv()

agent=Agent(alpha=0.000001,beta=0.0005,input_dims=[14],tau=0.0001,env=env,batch_size=32,layer1_size=600,layer2_size=400,n_actions=12)

score_history=env.load_obj("score_history")

current_best=env.load_obj("current_best")

for i in range(10000):
    print(current_best)
    for bestdevice in current_best:
        print(bestdevice)
        done=False
        score=0
        start=bestdevice[0][0]
        stop=bestdevice[0][1]
        
        norm_start=(start-1490)/(1640-1490)
        norm_stop=(stop-1490)/(1640-1490)
        print(database.retrive_device(bestdevice[1]).parameters)
        obs=env.normalize_device(database.retrive_device(current_best[1][1]).parameters)
        obs.append(norm_start)
        obs.append(norm_stop)
        print(obs)
        while not done:
            act=list(agent.choose_action(obs))
            
            new_state,reward,done,info=env.step(start,stop,act)
            agent.remember(obs,act,reward,new_state,int(done))
            agent.learn()
            score+=reward
            obs=new_state
        score_history.append(float(score))
        if i%1000==0:
            env.save_obj(score_history,"score_history")
        print("episode ",i,"score %.f"%score,'100 episode average %.2f'%np.mean(score_history[-100:]))
        
        if i%1000==0:
            agent.save_models()

print("final save")
env.save_obj(score_history,"score_history")
agent.save_models()
print("done")
