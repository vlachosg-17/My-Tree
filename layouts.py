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