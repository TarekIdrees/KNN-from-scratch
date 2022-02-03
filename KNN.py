##########################################
# Mohamed Mahmoud Ali           20188043 #
# Mohamed Abd Almajed idrees    20188061 #
# Tarek Abd Almajed idrees      20188062 #
##########################################


from math import *
from csv import reader


# Load data As List
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            row = ''.join(row)  # convert List to string
            row = row.split()  # split the columns
            floatRow = [float(element) for element in row]  # reconvert string to list of row with float value
            floatRow[-1] = int(floatRow[-1])  # make label class  integer
            dataset.append(floatRow)
    return dataset


# calculate the Euclidean distance between two rows (train row & test row)
def euclidean_distance(train_row, test_row):
    distance = 0.0
    for feature in range(len(train_row) - 1):
        distance += (train_row[feature] - test_row[feature]) ** 2
    return sqrt(distance)


# Get the most nearest neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda rowDistant: rowDistant[1])  # sort the each train row according to each euclidean distance
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])  # discard the distance and append only the row of feature
    return neighbors


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)  # calculate the nearest neighbors
    output_values = [row[-1] for row in neighbors]  # get the value of label class
    prediction = max(set(output_values), key=output_values.count)  # get the max frequency of predicted value
    return prediction


# Calculate accuracy percentage and number of correct predictions
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0, correct


def KNN(data_train, data_test):
    for k in range(1, 10):
        actual = []
        predicted = []
        print("K value : ", k)
        for i in range(len(data_test)):
            label = predict_classification(data_train, data_test[i], k)
            print('Data=%s, Predicted: %s ' % (data_test[i][-1], label))
            actual.append(data_test[i][-1])
            predicted.append(label)
        accuracyPercentage, numberOfCorrectPrediction = accuracy(actual, predicted)
        print("Number of correctly classified instances : %s Total number of instances : %d" % (
            numberOfCorrectPrediction, len(actual)))
        print("Accuracy :", accuracyPercentage)


if __name__ == '__main__':
    data_train = load_csv("yeast_training.txt")
    data_test = load_csv("yeast_test.txt")
    KNN(data_train, data_test)
