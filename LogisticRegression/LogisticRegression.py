# def getCostMatrix():
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from NaiveBayes import NaiveBayes

trainingPercent = 0.7
EPOC = 2000


class LogisticRegression:
    # Get the data from inputFile and transform it to list of list
    def loadFile(self, fileName):
        file = open(fileName)
        lines = file.readlines()
        inputdata = []

        for line in lines:
            inputdata.append([float(x) for x in line.strip().split(',')])
        # print(inputdata)
        return inputdata

    # Append 0 at start and remove the target class
    def convertIntoFeatures(self, inputData):
        featureData = []
        for data in inputData:
            featureData.append([1] + data)
        return featureData

    # Compute 1/1+e^-z
    def sigmoid(self, z):
        deno = 1.0 + math.exp(-1.0 * z)
        return float(1.0 / deno)

    # Compute sum of(weight*X values)
    def computeHypothesis(self, weights, features):
        # print()
        a = np.array(weights)
        b = np.array(features)
        z = a.dot(b)
        return self.sigmoid(z)

    def costFunctionDerivative(self, featureData, targetdata, weight, j):
        errorSum = 0
        alpha = 0.01
        length = len(weight)
        const = alpha / length

        for i in range(length):
            hi = self.computeHypothesis(weight, featureData[i])
            xij = featureData[i][j]
            error = (hi - targetdata[i]) * xij
            errorSum += error

        return const * errorSum

    def gradientDecent(self, train_data, weight):

        targetData = []
        t_data = []
        for data in train_data:
            targetData.append(data[-1])
            t_data.append(data[:-1])
            # data.pop()

        for i in range(EPOC):
            for j in range(len(weight)):
                cost_function_derivative = self.costFunctionDerivative(t_data, targetData, weight, j)
                weight[j] = weight[j] - cost_function_derivative

        return weight

    def predict(self, weight, testData):

        targetData = []
        t_data = []
        for data in testData:
            targetData.append(data[-1])
            t_data.append(data[:-1])

        correctResult = 0

        for i in range(len(targetData)):
            predictValue = np.round(self.computeHypothesis(weight, t_data[i]))
            if predictValue == targetData[i]:
                correctResult += 1
        accuracy = (correctResult / len(targetData)) * 100
        return accuracy

    # combine 2 equation for y=0 and y=1
    def computCost(self, featureData, y, weight):
        total = 0
        length = len(y)
        for i in range(len(y)):
            hypothesis = int(self.computeHypothesis(weight, featureData[i]))
            total += (y[i] * math.log(hypothesis)) + ((1 - y[i]) * (math.log(1 - hypothesis[i])))
        return (-1) * total / length

        # def plotCurve(self, train_data, percentageArray):
        #     for p in percentageArray:


logi = LogisticRegression()
inputData = logi.loadFile("banknote.txt")
random.shuffle(inputData)
featureData = logi.convertIntoFeatures(inputData)

# print(featureData)
train = int(trainingPercent * len(featureData))
weight = [0] * (len(featureData[0]) - 1)
train_data = featureData[0:train]
train_data_naive = inputData[0:train]
test_data_naive = inputData[train:]

percentageArray = [0.01, 0.02, 0.05, 0.1, 0.625, 1]

logi_accuracy = []
naive_accuracy = []

nb = NaiveBayes.NaiveBayes()

for p in percentageArray:
    l_accuracy = 0
    n_accuracy = 0
    for i in range(5):
        random.shuffle(train_data)
        train = int(p * len(train_data))
        newTrain_data = train_data[0:train]

        # Logistic regression
        weight = logi.gradientDecent(newTrain_data, weight)
        l_accuracy += logi.predict(weight, featureData[train:])

        # Naive

        random.shuffle(train_data_naive)
        new_data = train_data_naive[0:train]
        # print("naive bayes data", new_data)
        data_by_class = nb.divideInputDataByClass(new_data)
        parameters = nb.findFeaureParameters(data_by_class)
        n_accuracy += nb.predict(test_data_naive, parameters)

    logi_accuracy.append(l_accuracy / 5)
    naive_accuracy.append(n_accuracy / 5)

print("Part 2: Plot the curve: ")
print("logistic regression accuracy: ", logi_accuracy)
print("naive bayes accuracy: ", naive_accuracy)

plt.plot(percentageArray, logi_accuracy)
plt.plot(percentageArray, naive_accuracy)

plt.show()
