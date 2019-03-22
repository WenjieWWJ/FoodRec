'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")

        self.num_train_users = self.trainMatrix.__len__()

        num_instances = 0
        for i in self.trainMatrix:
            num_instances = num_instances + self.trainMatrix[i].__len__()
        self.num_instances = num_instances

        num_test = 0
        for i in self.testRatings:
            num_test = num_test + self.testRatings[i].__len__()
        self.num_test = num_test

    def load_rating_file_as_list(self, filename):
        """
        根据用户查找用来test的正例
        """
        ratingList = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList[str(user)] = []
                line = f.readline()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList[str(user)].append(item)
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                arr = [i.strip('\n') for i in arr]
                temp = arr[0]
                temp = temp[1:]
                temp = temp[:-1]
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                    # negatives列表中保存了user id(第一个元素)和这个用户对应的负例
                negativeList[temp] = negatives
                line = f.readline()
                # print(negativeList)
                # 一个dict.可以实现按用户查找
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        """
        返回一个字典，可以实现按用户查找正例
        """
        user_positive_items = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                user_positive_items[str(u)] = []
                line = f.readline()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                # user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                user,item = int(arr[0]), int(arr[1])
                user_positive_items[str(user)].append(item)
                line = f.readline()
        return user_positive_items

# dataset = Dataset('Data/' + 'ml-1m')
