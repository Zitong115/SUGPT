# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:17:17 2023

@author: ztli
"""

import copy
import collections
import math
import numpy as np
import time
import tqdm
import utils

class ArrayTreeForest(object):
    
    def __init__(self, H, D, graphfile, gen_delete = False, delete_vertex = False, reverse = False, delete_p = -1):
        self.height = H
        self.match_depth = D
        
        self.reverse = reverse
        self.graphfile = graphfile
        self.delete_p = delete_p
        
        # build graph
        self.G, self.A, self.deleted_vertices, self.deleted_edges = utils.ReadDatafile(self.graphfile,
                                                                                       gen_delete = gen_delete, 
                                                                                       delete_vertex = delete_vertex, 
                                                                                       delete_p = delete_p)
        # self.nodelist = list(self.G.nodes)
        self.vertices_num = self.A.shape[0]
        self.edges_num = int(self.A.count_nonzero()/2)
        print("Graph contains %d vertices, %d edges." % (self.vertices_num, self.edges_num))
        
        self.K = math.ceil(self.vertices_num/(self.height-1))
        self.trees = self.InitializeForest()
        self.grouppertree = self.trees[0].group_num
        self.normal_defaultplace = self.trees[0].defaultplace
        self.last_defaultplace = self.trees[self.K-1].defaultplace
        print("%d trees initialized." % (len(self.trees)))
        
    def InitializeForest(self):
        t1 = time.time()
        
        trees = {}
        #intervals = []
        
        low = 0
        high = 1
        
        for i in range(self.K):
            low = i * (self.height - 1)
            high = min((i+1) * (self.height - 1), self.vertices_num)
            height = high - low + 1
            
            #intervals.append([low,high])
            
            if( i ):
                if(height == self.height):
                    trees[i] = ArrayTree(height, self.match_depth)
                else:
                    # the last tree
                    print("the %d-th (last) tree height:%d" % (i+1, height))
                    assert i == self.K - 1
                    trees[i] = ArrayTree(height, self.match_depth)
            else:
                trees[i] = ArrayTree(self.height, self.match_depth)
            
            
        if(self.reverse):
            for i in range(self.K):
                trees[i+self.K] = copy.deepcopy(trees[0])
        
        t2 = time.time()
        
        print("Initialize forest timing: %.3f" % (t2-t1))
        
        return trees # intervals
    
    # get i-th interval
    def GetIntervals(self,i):
        if( i!=self.K-1 ):
            return [i * (self.height - 1), (i+1) * (self.height - 1)]
        else:
            return [i * (self.height - 1), min((i+1) * (self.height - 1), self.vertices_num)]
    
    def PlaceVertex2Tree(self, vertex, interval, tree):
        
        for j in range(interval[0],interval[1]):
            
            if(j == interval[0]):
                place = 1 if self.A[vertex, j] == 0 else 2
                continue
            
            if(self.A[vertex,j]):
                place = tree.GetRightChild(place) # 1 -> right
            else:
                place = tree.GetLeftChild(place)
        
        return place
    
    def GetDefaultLeaf(self, interval, tree):
        
        for j in range(interval[0],interval[1]):
            
            if(j == interval[0]):
                place = 1
                continue
            
            place = tree.GetLeftChild(place)
        
        return place
    
    def PlaceVertex2TreeByRow(self, vertex, A_nonzero_row, A_nonzero_col):
        
        row_idx = self.node2AmatrixOrder(vertex)
        col_idx = np.where(A_nonzero_row == row_idx)
        nonzero_col = A_nonzero_col[col_idx]
        
        # Find intervals that contrains nonzero columnns and calculate precisely
        influenced_interval_idx = set([math.floor(col/(self.height-1)) for col in nonzero_col])
        
        for k in range(self.K):
            
            interval = self.GetIntervals(k)
            
            if(k not in influenced_interval_idx):
                try:
                    self.trees[k].AddVertex2Leaf(vertex, self.normal_defaultplace, self.match_depth)
                except:
                    self.trees[k].AddVertex2Leaf(vertex, self.last_defaultplace, self.match_depth)
            else:
                place = self.PlaceVertex2Tree(row_idx, interval, self.trees[k])
                self.trees[k].AddVertex2Leaf(vertex, place, self.match_depth)

    def PlaceVertex2TreeByCol(self, treeno, A_nonzero_row, A_nonzero_col, nodeset):
        
        # note that the adjacency matrix is a symmetrix matrix.
        
        interval = self.GetIntervals(treeno)
        rows_idx = np.where((A_nonzero_row < interval[1]) &
                            (A_nonzero_row >= interval[0]))
        cols_idx = A_nonzero_col[rows_idx]
        cols_idx_unique = np.unique(cols_idx) # vertices needs to be processed
        
        vertices_to_be_processed = set(cols_idx_unique)
        vertices_zero = nodeset - vertices_to_be_processed
        
        # For each tree, process all the vertices in batch
        self.trees[treeno].nonzeros |= vertices_to_be_processed
        
        for vertex in vertices_to_be_processed:
            place = self.PlaceVertex2Tree(vertex, interval, self.trees[treeno])
            self.trees[treeno].AddVertex2Leaf(vertex, place, self.match_depth)
                
    def node2AmatrixOrder(self, node_val):
        return node_val #self.nodelist.index(node_val)
        
    def VerticesAlignment(self):
        
        t1 = time.time()
        
        for i in tqdm.tqdm(self.G.nodes):#range(self.vertices_num)):
            for k in range(self.K):
                
                interval = self.GetIntervals(k)
                place = -1
                ii = self.node2AmatrixOrder(i)
                place = self.PlaceVertex2Tree(ii, interval,  self.trees[k])
                
                # add vertex to trees
                try:
                    self.trees[k].AddVertex2Leaf(i, place, self.match_depth)
                except:
                    print(i, interval, k, place)
                    exit(1)
                        
                # iterate from opposite direction
                if(self.reverse):
                    place = -1
                    
                    # change the direction
                    for j in range(interval[1],interval[0], -1):
                        
                        if(j == interval[0]):
                            place = 1 if self.A[i,j] == 0 else 2
                            continue
                        
                        if(self.A[i,j]):
                            place = self.trees[k].GetRightChild(place) # 1 -> right
                        else:
                            place = self.trees[k].GetLeftChild(place) # 0 -> left
                    
                        # add vertex to trees
                        try:
                            self.trees[k].AddVertex2Leaf(i, place, self.match_depth)
                        except:
                            print(i, interval, k, j, place)
                            exit(1)
                        # self.trees[k + self.K].leaf[place].append(i)
        
        t2 = time.time()
        
        print("Vertices alignment timing: %.3f" % (t2-t1))
        
        return
    
    def VerticesAlignmentByRow(self):
        
        A_nonzero_row, A_nonzero_col = self.A.nonzero()[0],  self.A.nonzero()[1]
        
        t1 = time.time()
        
        for i in tqdm.tqdm(self.G.nodes):
            self.PlaceVertex2TreeByRow(i, A_nonzero_row, A_nonzero_col)
            
        if(self.reverse):
            # TO DO
            pass
        
        t2 = time.time()
        
        print("Vertices alignment timing: %.3f" % (t2-t1))
        
        return
    
    def VerticesAlignmentByCol(self):
        
        A_nonzero_row, A_nonzero_col = self.A.nonzero()[0],  self.A.nonzero()[1]
        nodeset = set(self.G.nodes)
        
        t1 = time.time()
        
        for i in tqdm.tqdm(range(self.K)):
            self.PlaceVertex2TreeByCol(i, A_nonzero_row, A_nonzero_col, nodeset)
            
        if(self.reverse):
            # TO DO
            pass
        
        t2 = time.time()
        
        print("Vertices alignment timing: %.3f" % (t2-t1))
        
        return
    
    def genGroups(self):
        groupzero_sum = 0
        for i in range(self.K):
            self.trees[i].genGroup(self.match_depth, self.vertices_num)
            groupzero_sum += self.trees[i].groupzero_pct
            # print(self.trees[i].group) demo print
            
        print("avg groupzero_pct:", groupzero_sum/self.K)
        
class Group(object):
    
    def __init__(self):
        self.vertices = collections.defaultdict(set)
        self.supernodes = collections.defaultdict(set)
        
class ArrayTree(object):
    
    def __init__(self, height, dist):
        self.height = height
        self.D = dist
        self.par_size = self.height - 1
        self.leaf = self.InitializeTree(self.par_size)
        self.vertex2leaf = {}
        self.vertex2group = {}
        self.group = collections.defaultdict(set)
        self.nonzeros_copy = None
        self.group_num = 2**(self.par_size)/2**dist
        self.defaultplace = 2**self.par_size - 1
        self.nonzeros = set()
        self.groupzero_pct = 0
        self.after_zero = set()
        
    # initialize leafnode dict
    # par_size: size of partition
    def InitializeTree(self, par_size):
        tree = collections.defaultdict(set)
        return tree
    
    def GetRightChild(self, root:int):
        return 2 * (root + 1)
    
    def GetLeftChild(self, root:int):
        return 2 * root + 1
    
    def GetRoot(self, node:int):
        return math.floor((node - 1)/2)
    
    def AddVertex2Leaf(self, vertex, place, dist):
        if(place != self.defaultplace):
            self.leaf[place].add(vertex)
            self.vertex2leaf[vertex] = place
            self.nonzeros.add(vertex)
            
            new_groupno = self.getGroupno(place, dist)
            self.group[new_groupno].add(vertex)
        else:
            if(vertex in self.nonzeros):
                self.nonzeros.remove(vertex)
            else:
                pass
    
    def UpdateVertexPlace(self, vertex, dist, oldplace, newplace):
        self.RemoveVertexFromLeaf(vertex, oldplace, dist=self.D)
        self.AddVertex2Leaf(vertex, newplace, dist=self.D)
        
    def RemoveVertexFromLeaf(self, vertex, place, dist):
        if(place != self.defaultplace):
            self.leaf[place].remove(vertex)
            self.vertex2leaf.pop(vertex)
            self.nonzeros.remove(vertex)
            
            old_groupno = self.getGroupno(place, dist)
            
            if(old_groupno):
                self.group[old_groupno].remove(vertex)
            elif(vertex in self.group[old_groupno]):
                # the place is not default place but the vertex is in group 0, like its place is 1026
                self.group[old_groupno].remove(vertex)
        else:
            pass
        
    def AddBatchVertex2Leaf(self, vertex_batch, place):
        # self.leaf[place] |= vertex_batch
        
        for vertex in vertex_batch:
            self.AddVertex2Leaf(vertex, place, self.D)
        
    def getGroupno(self, place, dist):
        # D=1,len(G)=2; D=2,len(G)=4; D=3,len(G)=8... len(G)=2**D
        ret = math.floor((place - (2**self.par_size - 1)) / (2**dist))
        
        assert ret >= 0
        return ret
    
    def genGroup(self, dist, vertices_num):
        
        #self.nonzeros_copy = self.nonzeros.copy()
        
        for i in self.leaf:
            groupno = self.getGroupno(i, dist)
            self.group[groupno] |= self.leaf[i]
        
        #self.group[0] |= (set(range(vertices_num)) - self.nonzeros)
        # to accelerate further matching
        self.groupzero_pct = (vertices_num - len(self.nonzeros)) / vertices_num
        
    def getGroupbyGroupno(self, groupno, vertices_num):
        if(groupno == 0):
            return self.getFirstGroup(vertices_num)
        else:
           return self.group[groupno]
        
        return self.group[groupno]
    
    def getFirstGroup(self, vertices_num):
        # add zero elements to group 0
        return self.group[0]
    
    def GetPlaceByVertex(self, vertex):
        if(vertex in self.nonzeros):
            place = self.vertex2leaf[vertex]
        else:
            place = self.defaultplace
            
        return place
    
    def FindBrother(self, node:int, dist:int):
        
        ret = set()
        
        father = self.GetPlaceByVertex(node)
        
        for i in range(dist):
            father = self.GetRoot(father)
        
        q = collections.deque([father])
        
        while(q):
            node = q.popleft()
            
            if(node in self.leaf):
                ret |= set(self.leaf[node])
            else:
                q.append(self.GetLeftChild(node))
                q.append(self.GetRightChild(node))
        
        return ret