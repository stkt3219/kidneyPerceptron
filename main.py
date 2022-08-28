import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv

threshold = 1

# initializes random weights
weights = []
for i in range(0, 4):
    n = round(random.uniform(0, 1), 2)
    weights.append(n)
print("Initial Weights: ", weights)
print("-------------------------------------------------")


def convert(output):
    if output == 1:
        str = "SUCCESS"
        return str
    else:
        str = "NO SUCCESS"
        return str


# activation function
def step_function(weighted_sum):
    if weighted_sum >= threshold:
        print("Prediction: SUCCESS")
        return 1
    else:
        print("Prediction: NO SUCCESS")
        return 0


# predicts output
def predict(x):
    sum = 0
    for i in range(len(x)):
        sum = np.dot(x, weights)
    output = step_function(sum)
    return output


# updates weights
def update_weights(data, target, epochs):
    learning_rate = 0.1
    for e in range(epochs):
        global_error = 0
        #print("EPOCH: ", e)
        #print(" ")
        for i in range(len(data)):
            output = target[i]
            input = data[i]

            #print("Inputs: ", input)
            #print("Expected: ", convert(output))

            prediction = predict(input)
            error = output - prediction

            global_error = global_error + abs(error)
            #print("Error: ", error)

            for j in range(len(weights)):
                weights[j] = weights[j] + learning_rate * error * input[j]
            #print("-------------")
        #print("**********************************")
        if global_error == 0:
            break

    print("Adjusted Weights: ", weights)

def main():

    epoch = 150

    data = pd.read_csv("kidney_stone_data.csv")
    # print(data.head())
    # print(data.isna().sum())

    # print(data.shape)

    x = data[['treatment', 'stone_size']]
    y = data['success']
    xd = pd.get_dummies(x)

    xtrain, xtest, ytrain, ytest = train_test_split(xd, y, test_size=0.10, random_state=0)

    xtrain = xtrain.to_numpy()
    xtest = xtest.to_numpy()
    ytrain = ytrain.to_numpy()
    ytest = ytest.to_numpy()

    input = xtrain
    target = ytrain

    update_weights(input, target, epoch)

    results = []
    print(" ")
    print("For Test Data: ")
    print(" ")
    for t in range(len(xtest)):
        inputs = xtest[t]
        target_t = ytest[t]
        result = predict(inputs)
        results.append(result)
        #print("Predicted: ", convert(result))
        print("Expected: ", convert(target_t))
        print("----------------")

    print("Perceptron Accuracy = ", accuracy_score(ytest, results))

if __name__ == '__main__':

    main()