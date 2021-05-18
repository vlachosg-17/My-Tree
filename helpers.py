import numpy as np
from sklearn.utils import shuffle
from time import time
from tqdm import tqdm

def load_abalone(filePath):
    Data, Labels = [], []
    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0:
                break
            line = line.split(",")
            line = [float(l.strip()) if k not in [0,len(line)-1]\
                                     else l.strip() \
                                     for k, l in enumerate(line)]
            Data.append(line[:-1])
            Labels.append(line[-1])
    data, labels = np.array(Data, dtype=object), np.array(Labels, dtype=object)
    return data, labels

def load_mushroom(filePath):
    Data, Labels = [], []
    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0:
                break
            line = line.split(",")
            line = [l.strip() for l in line]
            Data.append(line[1:])
            Labels.append(line[0])
    data, labels = np.array(Data), np.array(Labels)
    return data, labels

def load_iris(filePath, shuffle_data=True, seed=None):
    labels=[]
    data=[]
    with open(filePath, "r") as f:
        for line in f:
            line=line.strip()
            if len(line)==0:
                break
            line=line.split(",")
            data.append(line[:-1])
            labels.append(line[-1])
    data, labels = np.array(data, dtype="float32"), np.array(labels)
    if shuffle_data:
        if seed!=None:
            return shuffle(data, labels, random_state=seed)
        else:
            return shuffle(data, labels)
    else:
        return data, labels

def numerize_class(y):
    classes = np.sort(np.unique(y))
    n_classes=np.array([i for i in range(len(classes))])
    ys=np.array([n_classes[i] for el in y for i in range(len(classes)) if el==classes[i]])
    return ys

def one_hot(classes):
    u=[[1 if h==c else 0 for h in range(len(classes))] for c in classes]
    return u

def one_hot_classes(y):
    ys=np.copy(y)
    if ys.ndim<2:
        ys=np.expand_dims(ys, axis=1)
    ys, classes = numerize_class(ys)
    one_hot_classes = one_hot(classes)
    return np.array([one_hot_classes[c] for el in ys for c in range(len(classes)) if el==classes[c]], dtype="int")  

def split_sets(*args, test_ratio):
    train, test = [], []
    n = round(args[0].shape[0] * (1 - test_ratio))
    for arg in args:
        trainArg, testArg = np.split(arg, [n])
        train.append(trainArg)
        test.append(testArg)
    return train, test

def vote(labels):
    classes, counts = np.unique(labels, return_counts=True)
    return classes[np.argmax(counts)]

def bootstrap_sample(data, labels, number, with_raws=True):
    if with_raws:
        sampleItems = np.random.choice(range(data.shape[0]), size = number, replace = True)
        return data[sampleItems], labels[sampleItems]
    else:
        sampleItems = np.random.choice(range(data.shape[1]), size = number, replace = True)                
        return data[:,sampleItems], labels 

def overSample(data, labels, k, l):
    new_data = np.hstack((data, labels.reshape(-1,1)))
    newData = new_data.copy()
    labelPos = []

    # First find all indices with label l
    for rowPos in range(len(new_data)):
        if(new_data[rowPos,-1] == l):
            labelPos.append(rowPos)

    # Select ( with replacement ) k indices to duplicate
    for _ in range(k):
        j=np.random.randint(0, len(labelPos) -1)
        selectedItemIndex = labelPos[int(j)]
        selectedItem = new_data[selectedItemIndex,:]

        newData = np.vstack((newData,selectedItem))
    newLabels = newData[:,-1]
    newData = newData[:,0:newData.shape[1]-1]
    return newData, newLabels

def in_data_set(data,vector):
    f = False
    for raw in data:
        if all(vector==raw):
            f = True
            break
        else:
            continue
    return f 

def order(nodes):
    def fun(e):
        return e["depth"]
    S = sorted(nodes, key=fun)
    return [S[i]["node"] for i in range(len(S))]

def order_tree(tree, nodes=[]):
    node = tree["parent"]
    nodes.append({"depth": node.depth, "node": node})
    if not node.leaf:
        order_tree(tree["left"], nodes = nodes)
        order_tree(tree["right"], nodes = nodes)

    return order(nodes)


if __name__ == "__main__":
    data, labels = load_mushroom("data/agaricus-lepiota.data")
    print(data.shape, labels.shape)
    print(np.unique(labels))
    print(np.unique(numerize_class(labels)))
    
    data, labels = load_iris("data/iris.data", shuffle_data=True)
    print(data.shape, labels.shape)
    print(np.unique(labels))
    print(np.unique(numerize_class(labels)))
    
    