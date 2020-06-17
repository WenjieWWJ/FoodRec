
class Dataset(object):
    def __init__(self, path):
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
                negativeList[temp] = negatives
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
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
                user,item = int(arr[0]), int(arr[1])
                user_positive_items[str(user)].append(item)
                line = f.readline()
        return user_positive_items
