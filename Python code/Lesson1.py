import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
plt.switch_backend('agg')

df_data = pd.read_csv("cracow_apartments.csv", sep=",")
""" df_data.head()
print(df_data) """

def init(n):
    return  {
            "w": np.zeros(n), 
            "b": 0.0
            }

def predict(x, parameters):
    # Prediction initial value
    prediction = 0
    
    # Adding multiplication of each feature with it's weight
    for weight, feature in zip(parameters["w"], x):
        prediction += weight * feature
        
    # Adding bias
    prediction += parameters["b"]
        
    return prediction

def mae(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
    
    # Summing absolute differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += np.abs(prediction - target)
        
    # Calculating mean
    mae_error = (1.0 / samples_num) * accumulated_error
    
    return mae_error


def mse(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
    
    # Summing square differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += (prediction - target)**2
        
    # Calculating mean and dividing by 2
    mae_error = (1.0 / (2*samples_num)) * accumulated_error
    
    return mae_error

def trainBF(X, y, model_parameters, step=0.1, iterations=100):    
    # Make prediction for every data sample
    predictions = [predict(x, model_parameters) for x in X]

    # Calculate cost for model - MSE
    lowest_error = mse(predictions, y)
    
    print("\nInitial state:")
    print(" - error: {}".format(lowest_error))
    print(" - parameters: {}".format(model_parameters))
 
    for i in range(iterations):
        candidates, errors = list(), list()

        # w increased, b increased
        param_candidate = deepcopy(model_parameters)
        param_candidate["b"] += step
        param_candidate["w"] += step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w increased, b unchanged
        param_candidate = deepcopy(model_parameters)
        param_candidate["w"] += step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w increased, b decreased
        param_candidate = deepcopy(model_parameters)
        param_candidate["b"] -= step
        param_candidate["w"][0] += step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w unchanged, b increased
        param_candidate = deepcopy(model_parameters)
        param_candidate["b"] += step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w unchanged, b unchanged
        param_candidate = deepcopy(model_parameters)
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w unchanged, b decreased
        param_candidate = deepcopy(model_parameters)
        param_candidate["b"] -= step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w decreased, b increased
        param_candidate = deepcopy(model_parameters)
        param_candidate["b"] += step
        param_candidate["w"] -= step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w decreased, b unchanged
        param_candidate = deepcopy(model_parameters)
        param_candidate["w"] -= step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # w decreased, b decreased
        param_candidate = deepcopy(model_parameters)
        param_candidate["b"] -= step
        param_candidate["w"] -= step
        candidate_pred = [predict(x, param_candidate) for x in X]
        candidate_error = mse(candidate_pred, y)

        candidates.append(param_candidate)
        errors.append(candidate_error)

        # Update with parameters for which loss is smallest
        for candidate, candidate_error in zip(candidates, errors):
            if candidate_error < lowest_error:
                lowest_error = candidate_error
                model_parameters["w"], model_parameters["b"] = candidate["w"], candidate["b"]
        
        # Display training progress every 20th iteration
        if i % 20 == 0:
            print("\nIteration {}:".format(i))
            print(" - error: {}".format(lowest_error))
            print(" - parameters: {}".format(model_parameters))
    
    print("\nFinal state:")
    print(" - error: {}".format(lowest_error))
    print(" - parameters: {}".format(model_parameters))

def trainGD(X, y, model_parameters, learning_rate=0.0005, iterations=20000):
    # Make prediction for every data sample
    predictions = [predict(x, model_parameters) for x in X]

    # Calculate initial cost for model - MSE
    initial_error = mse(predictions, y)
    
    print("Initial state:")
    print(" - error: {}".format(initial_error))
    print(" - parameters: {}".format(model_parameters))
    
    for i in range(iterations):
        # Sum up partial gradients for every data sample, for every parameter in model
        accumulated_grad_w0 = 0
        accumulated_grad_b = 0   
        for x, y_target in zip(X, y):
            accumulated_grad_w0 += (predict(x, model_parameters) - y_target)*x[0]
            accumulated_grad_b += (predict(x, model_parameters) - y_target)
            
        # Calculate mean of gradient
        w_grad = (1.0/len(X)) * accumulated_grad_w0
        b_grad = (1.0/len(X)) * accumulated_grad_b
        
        # Update parameters by small part of averaged gradient
        model_parameters["w"] -= (learning_rate * w_grad)
        model_parameters["b"] -= (learning_rate * b_grad)
        
        if i % 4000 == 0:
            print("\nIteration {}:".format(i))
            print(" - error: {}".format(mse([predict(x, model_parameters) for x in X], y)))
            print(" - parameters: {}".format(model_parameters))
            
    print("\nFinal state:")
    print(" - error: {}".format(mse([predict(x, model_parameters) for x in X], y)))
    print(" - parameters: {}".format(model_parameters))


# Used features and target value
features = ["size"]
target = ["price"]

# Slice Dataframe to separate feature vectors and target value
X, y = df_data[features].values, df_data[target].values

# Initialize model parameters
n = len(features)
model_parameters = init(n)

""" # Make prediction for every data sample
predictions = [predict(x, model_parameters) for x in X]

orange_parameters = {'b': 200, 'w': np.array([3.0])}
lime_parameters = {'b': -160, 'w': np.array([12.0])}

# Make prediction for every data sample
orange_pred = [predict(x, orange_parameters) for x in X]
lime_pred = [predict(x, lime_parameters) for x in X]

# Model error
mse_orange_error = mse(orange_pred, y)
mse_lime_error = mse(lime_pred, y)

print(mse_orange_error, mse_lime_error) """

# trainBF(X, y, model_parameters)
trainGD(X, y, model_parameters)

