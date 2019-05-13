import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
titles = list(dataset)

#Implement UCB
import math
from random import randint
N = 10000
d = 10
ads_selected = {}
ads_reward = {}
initial_ads = []
ad_per_round = []

for i in range(len(titles)):
    ads_selected[titles[i]] = 0
    ads_reward[titles[i]] = 0
    initial_ads.append(titles[i])

total_reward = 0
for i in range(N):
    ad_selected = 0
    if (i < d):
        random_index = randint(0,len(initial_ads) - 1)
        ad_selected = initial_ads[random_index]
        initial_ads.remove(initial_ads[random_index])
    else:
        max_ucb = -1
        max_key = 0
        for key, value in ads_selected.items():
            avg_reward = ads_reward[key]/value
            confidence_interval = math.sqrt((3 * math.log(i+1))/(2 * value))
            ucb = avg_reward + confidence_interval
            if (ucb > max_ucb):
                max_ucb = ucb
                ad_selected = key
               
    ads_selected[ad_selected] += 1
    reward = dataset[ad_selected][i]
    ads_reward[ad_selected] += reward
    total_reward += reward
    ad_per_round.append(ad_selected)
    
#Visualize the results
plt.bar(range(len(ads_selected.keys())), height = ads_selected.values()) 
plt.xticks(range(len(ads_selected.keys())), labels = ads_selected.keys())
plt.title('Histogram')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()