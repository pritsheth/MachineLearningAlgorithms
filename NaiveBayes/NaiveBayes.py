import math
import random
from collections import defaultdict

import numpy as np


class NaiveBayes:
    accuracy = 0

    def loadFile(self, fileName):
        file = open(fileName)
        lines = file.readlines()
        inputdata = []

        for line in lines:
            inputdata.append([float(x) for x in line.strip().split(',')])
        return inputdata

    def getMean(self, data):
        return sum(data) / len(data)

    def getStandardDeviation(self, data):
        total = 0
        mean = self.getMean(data)
        # new_total = sum([[] for i in data])
        for i in data:
            total += (mean - i) ** 2
        var = total / float(len(data))
        if var == 0.0:
            var = 1.0
        return math.sqrt(var)

    def getProbabilityDensityValue(self, x, u, sigma):
        expo = math.exp(-((x - u) ** 2) / (2 * sigma * sigma))
        deno = 1 / math.sqrt(2 * math.pi * sigma * sigma)
        return deno * expo

    def divideInputDataByClass(self, data):
        dict = defaultdict(list)
        for i in data:
            dict[i[-1]].append(i[:-1])
        return dict

    def findFeaureParameters(self, data):
        dict = defaultdict(list)
        for class_label, data in data.items():
            for x in zip(*data):
                dict[int(class_label)].append([self.getMean(list(x)), self.getStandardDeviation(list(x))])

        return dict

    def decideTargetClass(self, prob0, prob1):
        return (0.0, 1.0)[prob0 < prob1]

    def predict(self, testData, parameters):
        correct = 0
        for data in testData:
            probabilityOfClass0 = 1.0
            probabilityOfClass1 = 1.0
            for i in range(len(data) - 1):
                probabilityOfClass0 *= self.getProbabilityDensityValue(data[i], parameters[0][i][0],
                                                                       parameters[0][i][1])
            for i in range(len(data) - 1):
                probabilityOfClass1 *= self.getProbabilityDensityValue(data[i], parameters[1][i][0],
                                                                       parameters[1][i][1])

            if data[-1] == self.decideTargetClass(probabilityOfClass0, probabilityOfClass1):
                correct += 1

        return correct / len(testData) * 100

    def createSamples(self, data):
        train = int(0.333 * len(data))
        A = data[0:train]
        B = data[train:2 * train]
        C = data[2 * train:]

        list = []
        list.append(A + B)
        list.append(B + C)
        list.append(A + C)

        print("Part 3  Show the power of generative model:")

        for dataSamples in list:
            data_by_class = self.divideInputDataByClass(dataSamples)
            parameters = self.findFeaureParameters(data_by_class)
            print("Given training samples mean and standard deviation parameters", parameters[1])
            random_samples_para = self.createRandomSamples(parameters[1])
            print("Generated random samples paramteres", random_samples_para)
            print(" ")

    def createRandomSamples(self, paramters):
        result = []
        mean = []
        std = []
        for para in paramters:
            mean.append(para[0])
            std.append(para[1])
        data = np.random.normal(mean, std, (400, 4))
        for x in zip(*data):
            result.append([self.getMean(list(x)), self.getStandardDeviation(list(x))])

        return result


nb = NaiveBayes()
data = nb.loadFile("banknote.txt")
random.shuffle(data)
# Part 3 assignment:
nb.createSamples(data)

# train = int(0.70 * len(data))
# train_data = data[0:train]
# test_data = data[train:]
#
# data_by_class = nb.divideInputDataByClass(data)
# parameters = nb.findFeaureParameters(data_by_class)


#
#
#
# percentageArray = [0.01, 0.02, 0.05, 0.1, 0.625, 1]
#
# naive_accuracy = []
#
# for p in percentageArray:
#     l_accuracy = 0
#     n_accuracy = 0
#     for i in range(5):
#         # Naive
#
#         # random.shuffle(train_data)
#         train = int(p * len(train_data))
#         new_data = train_data[0:train]
#
#         data_by_class = nb.divideInputDataByClass(new_data)
#         parameters = nb.findFeaureParameters(data_by_class)
#         n_accuracy += nb.predict(test_data, parameters)
#
#     naive_accuracy.append(n_accuracy / 5)
#
# print("naive ", naive_accuracy)
