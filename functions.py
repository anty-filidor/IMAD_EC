import csv


def import_file(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def break_class_wine(dataset): #do win
    target = []
    for i in range(len(dataset)):
        target.append(dataset[i][0])
        del dataset[i][0]
    return dataset, target


def break_class_other(dataset): #dla innych
    target = []
    for i in range(len(dataset)):
        target.append(dataset[i][-1])
        del dataset[i][-1]
    return dataset, target


def squared_distance(weights):
    return weights*weights