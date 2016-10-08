'''
Made some changes based on the tree code in Machine Learning in Action,including:
1. class a DecisionTree instead of functions
2. add sample weight
3. use numpy.array instead of list
4. fit into
'''
__author__ = 'Xin'

from math import log
import collections
import numpy as np


class DecisionTree(object):
    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''DecisionTree can only be called with keyword
                   arguments for the following keywords: training_datafile,
                     max_depth_desired, min_sample_split,min_leaf,criterion''')
        allowed_keys = ['training_datafile', 'criterion', 'max_depth_desired', 'min_sample_split', 'min_leaf']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")
        training_datafile = criterion = max_depth_desired = min_sample_split = None

        if kwargs and not args:
            if 'training_datafile' in kwargs: training_datafile = kwargs.pop('training_datafile')
            if 'criterion' in kwargs: criterion = kwargs.pop('criterion')
            if 'min_sample_split' in kwargs: min_sample_split = kwargs.pop('min_sample_split')
            if 'max_depth_desired' in kwargs: max_depth_desired = kwargs.pop('max_depth_desired')
            if 'min_leaf' in kwargs: min_leaf = kwargs.pop('min_leaf')

        if not args and training_datafile:
            self._training_datafile = training_datafile
        if max_depth_desired:
            self._max_depth_desired = max_depth_desired
        else:
            self._max_depth_desired = 10
        if min_sample_split:
            self._min_sample_split = min_sample_split
        else:
            self._min_sample_split = 0
        if criterion:
            self._criterion = criterion
        else:
            self._criterion = 'Gini'
        if min_leaf:
            self._min_leaf = min_leaf
        else:
            self._min_leaf = 0
        else:
            self._prune=False

    def readDataSet(self):


    def startCreateTree(self):
        return self.createTree(self._dataSet, self._labels, 0)

    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:  # the the number of unique elements and their occurance
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)  # log base 2
        return shannonEnt

    def calcGiniIndex(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:  # the the number of unique elements and their occurance
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        giniIndex = 1
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            giniIndex -= prob * prob  # gini index
        return giniIndex


    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self,dataSet):  # list of lists,and the last element of list is the class label
        if self._criterion == 'entropy':
            numFeatures = len(dataSet[0]) - 1
            baseEntropy = self.calcShannonEnt(dataSet)
            bestInfoGain = 0
            bestFeature = -1
            for i in range(numFeatures):  # iterate over all the features
                featList = [example[i] for example in dataSet]  #create a list of all the examples of this feature
                uniqueVals = set(featList)  #get a set of unique values
                newEntropy = 0.0
                for value in uniqueVals:
                    subDataSet = self.splitDataSet(dataSet, i, value)
                    if len(subDataSet) < self._min_leaf:  #if the subdataset less than 30,cannot be a leaf node
                        break
                    prob = len(subDataSet) / float(len(dataSet))
                    newEntropy += prob * self.calcShannonEnt(subDataSet)
                infoGain = baseEntropy - newEntropy  #calculate the info gain; ie reduction in entropy
                if (infoGain > bestInfoGain):  #compare this to the best gain so far
                    bestInfoGain = infoGain  #if better than current best, set to best
                    bestFeature = i
            return bestFeature  # returns an integer
        elif self._criterion == 'Gini':
            numFeatures = len(dataSet[0]) - 1
            bestImpurity = 10
            bestFeature = -1
            for i in range(numFeatures):  # iterate over all the features
                featList = [example[i] for example in dataSet]  #create a list of all the examples of this feature
                uniqueVals = set(featList)  #get a set of unique values
                newEntropy = 0.0
                for value in uniqueVals:
                    subDataSet = self.splitDataSet(dataSet, i, value)
                    if len(subDataSet) < self._min_leaf:
                        newEntropy=10  #which means this feature can not be used as the requirement of minleaf
                        break
                    prob = len(subDataSet) / float(len(dataSet))
                    newEntropy += prob * self.calcGiniIndex(subDataSet)
                if (bestImpurity > newEntropy):  #compare this to the  so far
                    bestImpurity = newEntropy  #if better than current best, set to best
                    bestFeature = i
            print(bestImpurity)
            return bestFeature  # returns an integer

    def distrCnt(self, classList):  # return the distribution
        classCount = collections.Counter(classList)
        totalNum = len(classList)
        ans = sorted(classCount.keys(), key=lambda d: classCount[d], reverse=True)[0]
        return ans + ':' + str(round(classCount[ans] / totalNum, 3))


    def majority(self,classList): #return the major class's probability
        classCount = collections.Counter(classList)
        totalNum = len(classList)
        ans = sorted(classCount.keys(), key=lambda d: classCount[d], reverse=True)[0]
        return ans,round(classCount[ans] / totalNum, 3)

    def createTree(self, dataSet, labels, depth):
        maxDepth = self._max_depth_desired
        minSplit = self._min_sample_split
        classList = [example[-1] for example in dataSet]
        if depth >= maxDepth:
            return self.distrCnt(classList)
        if classList.count(classList[0]) == len(classList):
            return self.distrCnt(classList)  # stop splitting when all of the classes are equal
        if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
            return self.distrCnt(classList)
        if len(classList) <= minSplit:  # stop splitting when there are less than min_sample_split samples
            return self.distrCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)  # feature index
        if bestFeat == -1:  # stop splitting when the leaf nodes have less than min_leaf samples
            return self.distrCnt(classList)
        bestFeatLabel = labels[bestFeat]  # feature name
        print(bestFeatLabel)
        myTree = {bestFeatLabel: {}}
        new_labels = labels.copy()  # avoid changing the original lables
        del (new_labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = new_labels[:]  # copy all of labels, so trees don't mess up existing labels,like list.copy()
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels,
                                                           depth + 1)
        return myTree


    def classify(self, inputTree, featLabels, testVec):
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict):
            classLabel = self.classify(valueOfFeat, featLabels, testVec)
        else:
            classLabel = valueOfFeat
        return classLabel
