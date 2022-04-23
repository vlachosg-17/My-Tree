import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from utils.metric import *
import tkinter

class TreeLayout:
    def layout(self, tree):
        return self.addmods(self.setup(tree))

    def addmods(self, tree, mod=0):
        tree.x += mod
        for c in tree.children:
            self.addmods(c, mod+tree.mod)
        return tree

    def setup(self, tree, depth=0):
        if len(tree.children) == 0:
            tree.x = 0
            tree.y = depth
            return tree

        if len(tree.children) == 1:
            tree.x = self.setup(tree.children[0], depth+1).x
            return tree

        left = self.setup(tree.children[0], depth+1)
        right = self.setup(tree.children[1], depth+1)

        tree.x = self.fix_subtrees(left, right)
        return tree

    def fix_subtrees(self, left, right):
        li, ri, diff, loffset, roffset, lo, ro = self.contour(left, right)
        diff += 1
        diff += (right.x + diff + left.x) % 2

        right.mod = diff
        right.x += diff

        if right.children:
            roffset += diff

        if ri and not li:
            lo.thread = ri
            lo.mod = roffset - loffset
        elif li and not ri:
            ro.thread = li
            ro.mod = loffset - roffset

        return (left.x + right.x) / 2

    def nextright(self, tree):
        if tree.thread:   
            return tree.thread
        if tree.children: 
            return tree.children[-1]
        else:             
            return None

    def nextleft(self, tree):
        if tree.thread:   
            return tree.thread
        if tree.children: 
            return tree.children[0]
        else:            
            return None

    def contour(self, left, right, max_offset=None, loffset=0, roffset=0, left_outer=None, right_outer=None):
        delta = left.x + loffset - (right.x + roffset)
        if not max_offset or delta > max_offset:
            max_offset = delta

        if not left_outer:
            left_outer = left
        if not right_outer:
            right_outer = right

        lo = self.nextleft(left_outer)
        li = self.nextright(left)
        ri = self.nextleft(right)
        ro = self.nextright(right_outer)

        if li and ri:
            loffset += left.mod
            roffset += right.mod
            return self.contour(li, ri, max_offset,
                                loffset, roffset, lo, ro)

        return (li, ri, max_offset, loffset, roffset, left_outer, right_outer)

    def knuth_layout(self, tree, depth=0, i=0):
        """
        set layout from the left-->parent-->right
        and only intergers
        """
        if tree.left_child:
            self.knuth_layout(tree.left_child, depth+1, i)
        tree.x = i
        tree.y = depth
        i+=1
        print(str(tree)+" --> pos:"+str((tree.x, tree.y)))    
        if tree.right_child:
            self.knuth_layout(tree.right_child, depth+1, i)
        
        return tree


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
        print(node)
        if isinstance(node.thres, list):
            return "X$_{%d}$ in %s" % (node.var, str(node.thres))\
                #     +"\n"+\
                #    "size: %d" % node.node.size + "\n"\
                #    "gini: %.3f" % node.node.gini
                   
        else:
            return "X$_{%d}$ < %.3f" % (node.var, node.thres)\
            # +"\n"+\
            #        "size: %d" % node.node.size +"\n"+\
            #        "gini: %.3f" % node.node.gini
                   
    else:
        return str(node.label)

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

def dt_graph(tree):
    tr = TreeLayout().layout(tree)
    ax = axes_off(set_layout(tr))
    return ax