from matplotlib import pyplot as plt

from utils.helpers import load_iris, split_sets
from graph_tree.plot_tree import dt_graph
from src.tree import DTree
from src.random_forest import RForest
from utils.metric import *

data, labels = load_iris("data/iris.data", shuffle_data=True, seed=0)
(trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.5)
tree = DTree(min_size=20)
tree = tree.grow(trainX, trainY)
predY = tree.predict(teX)
m = metrics(predY, teY)
print_metrics(m)
# dt_graph(tree)
# plt.show()

rf = RForest(ntrees=200, min_size=1, max_depth=10, seed=0)
rf = rf.fit(trainX, trainY)
predY = rf.predict(teX)
m = metrics(predY, teY)
print_metrics(m)

