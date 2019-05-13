import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append(np.array(dataset.values[i, :], dtype = 'str').tolist())
    
#Train Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2) 

#Visualize the results
results = list(rules)
the_rules = []
for result in results:
    the_rules.append({'rule': ','.join(result.items),
                      'support':result.support,
                      'confidence':result.ordered_statistics[0].confidence,
                      'lift':result.ordered_statistics[0].lift})
results = pd.DataFrame(the_rules, columns = ['rule', 'support', 'confidence', 'lift'])