# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:17:17 2023

@author: ztli
"""

import copy
import collections
import evaluation
import math
import numpy as np
import os
import pickle
import time
import tqdm
import utils
from dask import delayed

class Graph(object):
    
    def __init__(self, config, gen_delete, dp = -1, eval_mosso = False, dataset = None):
        self.partition_size = config["H"]
        self.match_depth = config["D"]
        self.dataset_name = config["dataset"] if not dataset else dataset
        
        self.reverse = config["reverse"]
        self.graphfile = config["graph_file"]

        self.delete_p = dp if dp > 0 else config["delete_p"]
        self.delete_vertex = config["vertex_deletion"]
        self.zeros_file_prefix = config["zeros_file_prefix"]
        self.load_zero = config["load_zero"]
        self.gen_delete = gen_delete

        relabel = False #if eval_mosso == True else True
        
        # build graph
        self.G, self.A, self.deleted_vertices, self.deleted_edges = utils.ReadDatafile(dataset = dataset if dataset else config['dataset'],
                                                                                       subgraph = config["subgraph"],
                                                                                       dataset_folder = config["dataset_folder"],
                                                                                       delete_vertex = self.delete_vertex,
                                                                                       sample_frac = config["sample_frac"],
                                                                                       delete_p = self.delete_p,
                                                                                       gen_delete = gen_delete,
                                                                                       relabel = relabel,
                                                                                       genMossoInput = config["genMossoInput"], 
                                                                                       mossoinputfolder = config["mossoinputfolder"])
        
        self.vertices_num = self.G.number_of_nodes()
        self.edges_num = self.G.number_of_edges()
        self.avg_degree = self.GetAvgDegree()
        print("Graph contains %d vertices, %d edges." % (self.vertices_num, self.edges_num))
        
        self.K = math.ceil(self.vertices_num/(self.partition_size))
        self.trees = self.InitializeForest()
        self.normal_defaultplace = self.trees[0].defaultplace
        self.last_defaultplace = self.trees[self.K-1].defaultplace
        print("%d trees initialized." % (len(self.trees)))
    
    def UpdateGraph(self, dp, config):
        self.G, self.A, self.deleted_vertices, self.deleted_edges = utils.ReadDatafile(dataset = config["dataset"],
                                                                                       subgraph = config["subgraph"],
                                                                                       dataset_folder = config["dataset_folder"],
                                                                                       delete_vertex = self.delete_vertex,
                                                                                       sample_frac = config["sample_frac"],
                                                                                       delete_p = dp,
                                                                                       gen_delete = self.gen_delete,
                                                                                       relabel = True,
                                                                                       genMossoInput = config["genMossoInput"], 
                                                                                       mossoinputfolder = config["mossoinputfolder"])
        return self.G, self.A, self.deleted_vertices, self.deleted_edges
        
    def GetAvgDegree(self):

        sum_degree = 0

        for v in self.G.nodes:
            sum_degree += self.G.degree[v]

        avg_degree = sum_degree / self.vertices_num

        if(self.partition_size < 0):
            # set partition_size automatically
            # self.partition_size = int(math.sqrt(self.vertices_num / avg_degree))
            # self.partition_size = int(self.vertices_num / math.sqrt(self.edges_num * avg_degree))
            
            C = 1044 / (1.3 * 0.1) 
            self.partition_size = int(math.sqrt(self.vertices_num * C / self.edges_num))

            # estimated_vertices_cpr = 1.8
            # self.partition_size = int(self.vertices_num / math.sqrt(self.edges_num * estimated_vertices_cpr))
            # if(self.partition_size < 0):
            #     self.partition_size = int(math.sqrt(self.vertices_num / avg_degree))
            
            if(self.partition_size < 0):
                print("self.partition_size < 0 : ", self.partition_size)
                self.partition_size = 30
            
        print("average degree: %.3f, self.partition_size is %d" % (sum_degree / self.vertices_num, self.partition_size))

    def InitializeForest(self):
        t1 = time.time()
        
        trees = {}
        #intervals = []
        
        low = 0
        high = 1
        
        for i in range(self.K):
            low = i * (self.partition_size)
            high = min((i+1) * (self.partition_size), self.vertices_num)
            height = high - low
            
            #intervals.append([low,high])
            
            if( i ):
                if(height == self.partition_size):
                    trees[i] = Tree(height, self.match_depth)
                else:
                    # the last tree
                    print("the %d-th (last) tree height: %d" % (i+1, height))
                    assert i == self.K - 1
                    trees[i] = Tree(height, self.match_depth)
            else:
                trees[i] = Tree(self.partition_size, self.match_depth)
            
            
        if(self.reverse):
            for i in range(self.K):
                trees[i+self.K] = copy.deepcopy(trees[0])
        
        t2 = time.time()
        
        print("Initialize forest timing: %.3f" % (t2-t1))
        
        return trees
    
    # get i-th interval
    def GetIntervals(self, i):
        if( i!=self.K-1 ):
            return [i * (self.partition_size), (i+1) * (self.partition_size)]
        else:
            return [i * (self.partition_size), min((i+1) * (self.partition_size), self.vertices_num)]
    
    # the vertex in interval should be put to which place in according tree
    def PlaceVertex2Tree(self, vertex, interval, tree):
        
        for j in range(interval[0], interval[1]):
            
            if(j == interval[0]):
                place = 1 if self.A[vertex, j] == 0 else 2
                continue
            
            if(self.A[vertex,j]):
                place = tree.GetRightChild(place) # 1 -> right
            else:
                place = tree.GetLeftChild(place)
        
        return place

    def PlaceVertex2TreeByCol(self, treeno, A_nonzero_row, A_nonzero_col, nodeset):
        
        # note that the adjacency matrix is a symmetrix matrix.
        
        interval = self.GetIntervals(treeno)
        rows_idx = np.where((A_nonzero_row < interval[1]) &
                            (A_nonzero_row >= interval[0]))
        cols_idx = A_nonzero_col[rows_idx]
        cols_idx_unique = np.unique(cols_idx) # vertices needs to be processed
        
        vertices_to_be_processed = set(cols_idx_unique)
        # vertices_zero = nodeset - vertices_to_be_processed
        
        # For each tree, process all the vertices in batch
        self.trees[treeno].nonzeros |= vertices_to_be_processed
        
        for vertex in vertices_to_be_processed:
            place = self.PlaceVertex2Tree(vertex, interval, self.trees[treeno])
            self.trees[treeno].AddVertex2Leaf(vertex, place)
        
        return 0
    
    def GetZeroGroupOnIthTree(self, i):
        if(0 in self.trees[i].group.keys()):
            return self.trees[i].group[0].vertices | self.GetZeroVerticesOnIthTree(i)
        else:
            return self.GetZeroVerticesOnIthTree(i)
    
    def getZeroGroupfilename(self, treeno):
        
        return os.path.join(self.zeros_file_prefix, str(treeno))
    
    def GetZeroVerticesOnIthTree(self, i):
        
        if(self.load_zero):
            zeros_file_name = self.getZeroGroupfilename(i)
            if(os.path.exists(zeros_file_name)):
                with open(zeros_file_name, "rb") as handle:
                    zeros = pickle.load(handle)
                    return zeros
            else:
                zeros = set(range(self.vertices_num)) - self.trees[i].nonzeros
                
                with open(zeros_file_name, "wb") as handle:
                    pickle.dump(zeros, handle)
                
                return zeros
        else:
            return set(range(self.vertices_num)) - self.trees[i].nonzeros
    
    def UpdateZeroFile(self, i):
        if(self.load_zero):
            zeros_file_name = self.getZeroGroupfilename(i)
            zeros = set(range(self.vertices_num)) - self.trees[i].nonzeros
            len_zeros_prev = -1
            len_zeros = len(zeros)
            
            if(os.path.exists(zeros_file_name)):
                with open(zeros_file_name, "rb") as handle:
                    zeros_prev = pickle.load(handle)
                    len_zeros_prev = len(zeros_prev)
                    print("%d zero elements in prev" % len_zeros_prev)
                    
            with open(zeros_file_name, "wb") as handle:
                pickle.dump(zeros, handle)
                print("%d zero elements currently" % len_zeros)
    
    def PlaceVertex2TreeByRow(self, vertex, A_nonzero_row, A_nonzero_col):
        
        row_idx = vertex
        col_idx = np.where(A_nonzero_row == row_idx)
        nonzero_col = A_nonzero_col[col_idx]
        
        # Find intervals that contrains nonzero columnns and calculate precisely
        influenced_interval_idx = set([math.floor(col/(self.partition_size)) for col in nonzero_col])
        
        for k in range(self.K):
            
            interval = self.GetIntervals(k)
            
            if(k not in influenced_interval_idx):
                try:
                    self.trees[k].AddVertex2Leaf(vertex, self.normal_defaultplace)
                except:
                    self.trees[k].AddVertex2Leaf(vertex, self.last_defaultplace)
            else:
                place = self.PlaceVertex2Tree(row_idx, interval, self.trees[k])
                self.trees[k].AddVertex2Leaf(vertex, place)
    
    def VerticesAlignmentByRow(self):
        
        A_nonzero_row, A_nonzero_col = self.A.nonzero()[0],  self.A.nonzero()[1]
        
        t1 = time.time()
        
        for i in self.G.nodes:
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
        
        for i in range(self.K):
            self.PlaceVertex2TreeByCol(i, A_nonzero_row, A_nonzero_col, nodeset)
            
        if(self.reverse):
            # TO DO
            pass
        
        t2 = time.time()
        
        print("Vertices alignment timing: %.3f" % (t2-t1))
        
        return
    
    def VerticesAlignmentByColPara(self):
        
        A_nonzero_row, A_nonzero_col = self.A.nonzero()[0],  self.A.nonzero()[1]
        nodeset = set(self.G.nodes)
        ttmp = []
        
        t1 = time.time()
        
        for i in (range(self.K)):
            tmp = delayed(self.PlaceVertex2TreeByCol)(i, A_nonzero_row, A_nonzero_col, nodeset)
            ttmp.append(tmp)
        
        ttmp.compute()
        
        if(self.reverse):
            # TO DO
            pass
        
        t2 = time.time()
        
        print("Vertices alignment timing: %.3f" % (t2-t1))
        
        return
    
    def genGroups(self):
        groupzero_sum = 0
        for i in range(self.K):
            self.trees[i].genGroup(self.vertices_num)
            groupzero_sum += self.trees[i].groupzero_pct
            
        print("avg groupzero_pct:", groupzero_sum/self.K)
        
    def UpdateA(self):
        self.A = utils.nx.adjacency_matrix(self.G)
        
class Group(object):
    
    def __init__(self):
        self.vertices = set()
        self.supernodes = set()
    
    def AddVertex(self, vertex):
        self.vertices.add(vertex)
    
    def RemoveVertex(self, vertex):
        self.vertices.remove(vertex)
        
class Tree(object):
    
    def __init__(self, partition_size, match_depth):
        self.D = match_depth
        self.par_size = partition_size
        self.leaf = self.InitializeTree(self.par_size)
        self.vertex2leaf = {}
        # self.vertex2group = {}
        self.group = {}
        # self.nonzeros_copy = None
        self.group_num = 2**(self.par_size)/2**self.D
        self.defaultplace = 2**self.par_size - 1
        self.nonzeros = set()
        self.groupzero_pct = 0
        # self.after_zero = set()
        
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
    
    def getGroupno(self, place):
        # D=1,len(G)=2; D=2,len(G)=4; D=3,len(G)=8... len(G)=2**D
        ret = math.floor((place - (2**self.par_size - 1)) / (2**self.D))
       
        assert ret >= 0
        return ret
    
    def AddVertex2Group(self, groupno, vertex):
        
        if(groupno not in self.group.keys()):
            self.group[groupno] = Group()
        
        self.group[groupno].AddVertex(vertex)
    
    def AddVertex2Leaf(self, vertex, place):
        if(place != self.defaultplace):
            self.leaf[place].add(vertex)
            self.vertex2leaf[vertex] = place
            self.nonzeros.add(vertex)
            
            new_groupno = self.getGroupno(place)
            self.AddVertex2Group(new_groupno, vertex)
        else:
            if(vertex in self.nonzeros):
                self.nonzeros.remove(vertex)
            else:
                pass

    def RemoveVertexFromLeaf(self, vertex, place):
        if(place != self.defaultplace):
            try:
                self.leaf[place].remove(vertex)
            except:
                return
            self.vertex2leaf.pop(vertex)
            self.nonzeros.remove(vertex)
            
            old_groupno = self.getGroupno(place)
            
            # the place is not default place but the vertex is in group 0, like its place is 1026
            if(old_groupno or vertex in self.group[old_groupno].vertices):
                self.group[old_groupno].RemoveVertex(vertex)
        else:
            pass
        
    def UpdateVertexPlace(self, vertex, oldplace, newplace):
        self.RemoveVertexFromLeaf(vertex, oldplace)
        self.AddVertex2Leaf(vertex, newplace)
        
    def genGroup(self, vertices_num):
        
        for i in self.leaf:
            groupno = self.getGroupno(i)
            try:
                self.group[groupno].vertices |= self.leaf[i]
            except:
                print(groupno, i)
        # to accelerate further matching
        self.groupzero_pct = (vertices_num - len(self.nonzeros)) / vertices_num
    
    def AddSv2Group(self, groupno, sv):
        self.group[groupno].supernodes.add(sv)
        return
    
    def getGroupbyGroupno(self, groupno):
        if(groupno in self.group.keys()):
            return self.group[groupno].vertices
        else:
            return set()
        
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