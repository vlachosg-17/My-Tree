from helpers import load_iris, split_sets, load_mushroom, load_abalone, overSample
# from my_tree import MyDecisionTreeClassifier
# from layouts import TreeLayout
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from metric import *
import tkinter

root = tkinter.Tk()
dims = root.winfo_screenwidth(), root.winfo_screenheight()
    
def set_limits(ax, x_range=None, y_range=None):
    if ax is None: ax = plt.gca()
    if x_range is not None: ax.set_xlim(x_range[0], x_range[1])
    if y_range is not None: ax.set_ylim(y_range[0], y_range[1])
    return ax

def set_screen(ax, dims):
    mng = plt.get_current_fig_manager()
    mng.resize(*dims)
    return ax

def axes_off(ax):
    if ax is None: ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax
    
def get_xrange(tree, width, x_range=[0, 0]):
    if x_range[1] < tree.x:
        x_range[1] = tree.x
    if x_range[0] > tree.x:
        x_range[0] = tree.x
    for child in tree.children:
        get_xrange(child, width, x_range=x_range)
    
    return x_range[0]-width/2.0, x_range[1]+width/2.0 

def get_yrange(tree, height, y_range=[0, 0]):
    if y_range[1] > -tree.y:
        y_range[1] = -tree.y
    if y_range[0] < -tree.y:
        y_range[0] = -tree.y
    for child in tree.children:
        get_yrange(child, height, y_range=y_range)
    
    return y_range[1]-height/2.0, y_range[0]+height/2.0

def make_shape(ax, tree, w, h):
    if len(tree.children) != 0:
        S = mpatch.Ellipse((tree.x, -tree.y), w, h, color="green")
    else:
        S = mpatch.Ellipse((tree.x, -tree.y), w, h, color="red")
    ax.add_patch(S)
    ax.annotate(make_ann(tree), (tree.x, -tree.y), 
                color='black', weight='bold', 
                fontsize=10*w, ha='center', 
                va='center')
    return ax

def make_ann(node):
    if len(node.children) != 0:
        if isinstance(node.node.thres, list):
            return "X$_%d$ in %s" % (node.node.var, str(node.node.thres))\
                #     +"\n"+\
                #    "size: %d" % node.node.size + "\n"\
                #    "gini: %.3f" % node.node.gini
                   
        else:
            return "X$_%d$ < %.3f" % (node.node.var, node.node.thres)\
            # +"\n"+\
            #        "size: %d" % node.node.size +"\n"+\
            #        "gini: %.3f" % node.node.gini
                   
    else:
        return str(node.node.label)

def set_layout(tree, ax=None, w=dims[0]/2000, h=dims[0]/2000):
    if ax is None: ax = plt.gca()
    ax = make_shape(ax, tree, w, h)
    if len(tree.children) != 0:
        for child in tree.children:
            ax.scatter([tree.x, child.x], [-tree.y-w/2.0, -child.y+w/2.0], color="white")
            s = mpatch.ConnectionPatch([tree.x, -tree.y-w/2.0], [child.x, -child.y+w/2.0], "data")
            ax.add_patch(s)
            set_layout(child, ax=ax)
    return set_limits(ax, get_xrange(tree, w), get_yrange(tree, h))
    