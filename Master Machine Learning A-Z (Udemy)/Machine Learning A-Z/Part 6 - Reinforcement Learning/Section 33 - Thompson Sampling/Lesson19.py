import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
titles = list(dataset)

#Implement UCB
from random import betavariate
N = 10000
d = 10
ads_selected = {}
ads_reward = {}
ads_punish = {}
ad_per_round = []

for i in range(len(titles)):
    ads_selected[titles[i]] = 0
    ads_reward[titles[i]] = 0
    ads_punish[titles[i]] = 0

total_reward = 0
for i in range(N):
    ad_selected = titles[0]
    max_random = 0
    for key, value in ads_reward.items():
        random_beta =  betavariate(value + 1, ads_punish[key] + 1)
        if (random_beta > max_random):
            max_random = random_beta
            ad_selected = key
            
    ads_selected[ad_selected] += 1
    reward = dataset[ad_selected][i]
    if (reward > 0):
        ads_reward[ad_selected] += 1
    else: 
        ads_punish[ad_selected] += 1
    total_reward += reward
    ad_per_round.append(ad_selected)
    
#Visualize the results
plt.bar(range(len(ads_selected.keys())), height = ads_selected.values()) 
plt.xticks(range(len(ads_selected.keys())), labels = ads_selected.keys())
plt.title('Histogram')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
