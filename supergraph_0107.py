# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:23:08 2023

@author: ztli

#same as 1116 ver. but added the part of GSS
"""

import GSS
import collections
import copy
from dask import delayed
import numpy as np
import math
import pandas as pd
import random
from sys import exit
import time
from tqdm import tqdm
import utils
import pandas as pd

SUM_AGG = "sum"
OR_AGG = "or"
AVG_AGG = "avg"
MIN_EDGE_GENERATE_SN_SIZE = 10
ECHO_PERIOD = 10000
missing_nodes = []


class Supernode(object):
    
    def __init__(self, candidate, aggmethod, graph, se_gen_method = "EDG"):
        self.vertices = candidate
        self.rep = None
        self.cnt = None
        # self.ele_cnt = np.zeros(graph.vertices_num) 
        self.ele_cnt = collections.defaultdict(lambda: 0)
        self.stop_round = -1
        self.used_tree = set()
        if(se_gen_method != "EDG"):
            self.GetRepresentation(graph, aggmethod)
    
    def addUsedTree(self, treeno):
        self.used_tree.add(treeno)
        return
    
    def GetRepresentation(self, graph, aggmethod):
        if(aggmethod == SUM_AGG):
            self.Sum(graph)
        elif(aggmethod == OR_AGG):
            self.Or(graph)
        elif(aggmethod == AVG_AGG):
            self.Average(graph)
            
    def Average(self, graph):
        n = len(self.vertices)
        ret = np.zeros([1, graph.vertices_num])
        
        for i in self.vertices:
            cand_vec = graph.A.getrow(i).toarray()
            ret = ret + cand_vec
        
        ret = ret/n
        ret = np.squeeze(ret)
        
        self.rep = ret
        
        return
    
    def Or(self, graph, store_cnt = False):
        
        cnt = np.zeros([1, graph.vertices_num])
        
        for i in self.vertices:
            cand_vec = graph.A.getrow(i).toarray()
            cnt = cnt + cand_vec
            
        self.rep = cnt.astype(bool).astype(int)
        self.cnt = cnt
        
        return 
    
    def Sum(self, graph):
        
        rep = np.zeros([1, graph.vertices_num])
        
        for i in self.vertices:
            cand_vec = graph.A.getrow(i).toarray()
            rep = rep + cand_vec
            
        self.rep = rep
        self.cnt = rep
    
    def UpdateRepUsingCnt(self, aggmethod):
        if(aggmethod == SUM_AGG):
            self.rep = self.cnt
        elif(aggmethod == OR_AGG):
            self.rep = self.cnt.astype(bool).astype(int)
        elif(aggmethod == AVG_AGG):
            self.rep = self.cnt / len(self.vertices)
            self.rep = np.squeeze(self.rep)
    
    def RemoveVfromSn(self, v, graph, aggmethod):
        if(isinstance(v, int)):
            if( v in self.vertices):
                
                self.vertices.remove(v)
                
                if(self.cnt):
                    v_vec = graph.A.getrow(v).toarray()
                    self.cnt -= v_vec
                    self.UpdateRepUsingCnt(aggmethod)
        elif(isinstance(v, set)):
            self.vertices -= v
            
            if(self.cnt):
                for vi in v:
                    v_vec = graph.A.getrow(vi).toarray()
                    self.cnt -= v_vec
                    self.UpdateRepUsingCnt(aggmethod)
        
        self.EleCntRemove(v)
        
        return
    
    def AddVtoSn(self, v, graph, aggmethod):
        self.vertices.add(v)
        if(self.cnt):
            cand_vec = graph.A.getrow(v).toarray()
            self.cnt += cand_vec
            self.UpdateRepUsingCnt(aggmethod)
        return
    
    def EleCntAdd(self, v):
        if(isinstance(v, int)):
            self.ele_cnt[v] += 1
        elif(isinstance(v, set)):
            # self.ele_cnt[list(v)] += 1
            for vv in v:
                self.ele_cnt[vv] += 1
    
    def EleCntRemove(self, v):
        if(isinstance(v, int) and self.ele_cnt[v] >= 1):
            self.ele_cnt[v] -= 1
        elif(isinstance(v, set)):
            # self.ele_cnt[list(v)] -= 1
            # self.ele_cnt[self.ele_cnt<0] = 0
            for vv in v:
                self.ele_cnt[vv] -= 1
                if(self.ele_cnt[vv] < 0):
                    self.ele_cnt.pop(vv)
            
    def CountCandidate(self, t):
        # return utils.np.count_nonzero(self.ele_cnt > t)
        ret = 0
        for v in self.ele_cnt:
            if(self.ele_cnt[v] > t):
                ret += 1
        return ret
    
    def UpdateCandidate(self, t, dismissed_vertices = None, cand2set = False):
        # self.vertices = set([i for i in self.ele_cnt.keys() if (self.ele_cnt[i] > t)])
        
        # self.vertices = np.where(self.ele_cnt > t)[0]
        
        self.vertices = []
        for v in self.ele_cnt:
            if(self.ele_cnt[v] > t):
                self.vertices.append(v)

        if(cand2set):
            self.MakeVertices2Set()
        if(dismissed_vertices):
            self.vertices &= dismissed_vertices
            
        return self.vertices
    
    def MakeVertices2Set(self):
        self.vertices = set(self.vertices)
    
class Supergraph(object):
    
    def __init__(self, graph, config, dp = -1, GSS = False):
        self.dataset = config["dataset"]
        self.delete_p = dp if dp > 0 else config["delete_p"]
        self.graph = graph
        self.edge_threshold = config["T"]
        self.zero_match_escape = config["zero_match_escape"]
        self.other_match_escape_pct = config["other_match_escape_pct"]
        self.other_match_escape = config["other_match_escape"] #int(self.graph.vertices_num * self.other_match_escape_pct)
        self.use_ele_cnt = config["use_ele_cnt"]
        
        # self.group2supernode key encoding
        self.encoding_n = 10 ** math.ceil(math.log10(self.graph.trees[0].group_num))
        self.group2supernode = collections.defaultdict(set)
        self.vertex2supernode = {}
        self.supernode = {} #collections.defaultdict(set)
        
        self.superedge = []
        self.genedge_selection_percentile = config["genedge_selection_percentile"]
        
        self.candidate_earlystop_thres = config["candidate_earlystop_thres"]
        if(config["candidate_earlystop_thres"] < 0 ):

            self.candidate_earlystop_thres = int(self.graph.partition_size / 10)

            if(self.candidate_earlystop_thres < 0):
                print("error self.candidate_earlystop_thres: %d, reset to 4" % self.candidate_earlystop_thres)
                self.candidate_earlystop_thres = 4
        else:
            # set earlystop_threshold by manual
            self.candidate_earlystop_thres = config["candidate_earlystop_thres"]
        print("candidate_earlystop_thres is set to :", self.candidate_earlystop_thres)

        self.max_candidate = config["max_candidate"]
        self.sn_agg_method = config["agg_node_method"]
        self.se_gen_method = config["cal_edge_method"]
        self.gen_zero_sn_first = config["gen_zero_sn_first"]
        self.GSS = GSS
        if(GSS):
            self.f = 4
            self.w = int(self.graph.vertices_num / 2)
            self.superedge_weight = collections.defaultdict(lambda: 0)
            self.edge_threshold = 1

    def ResetDeletep(self, dp):
        self.delete_p = dp
    
    def GenEdgeDeleteIdx(self, dp):
        deleted_edges_idx, deleted_nodes_idx = utils.GenerateDeletedIdx(self.graph.G, delete_p = dp, delete_vertex = self.graph.delete_vertex)
        return deleted_edges_idx, deleted_nodes_idx

    def AddCandidate2Supernode(self, sn, nodepool, candidate, remove_from_groups = False, unsummary = False):

        # 20230607 modified
        if(self.use_ele_cnt):
            filtered_candidate = self.supernode[sn].vertices
        else:
            filtered_candidate = candidate & ( nodepool | set([sn]) )
            
            if(sn in self.supernode.keys()):
                print("conflicting supernode: %d" % sn)
            
            self.supernode[sn] = Supernode(candidate = filtered_candidate, 
                                        aggmethod = self.sn_agg_method, 
                                        graph = self.graph)
        
        if( unsummary ==False and len(filtered_candidate) == 0):
            print("sn %d len(filtered_candidate) == 0" % (sn))
            exit(1)
        for cand in filtered_candidate:
            self.vertex2supernode[cand] = sn
        return nodepool - set(filtered_candidate), len(filtered_candidate)
        
    def GetPlaceByVertex(self, treeno, vertex):
        if(vertex in self.graph.trees[treeno].nonzeros):
            place = self.graph.trees[treeno].vertex2leaf[vertex]
        else:
            if(treeno < self.graph.K-1):
                place = self.graph.normal_defaultplace
            else:
                place = self.graph.last_defaultplace
            
        return place
    
    # if the vertex has been visited in treeno-th tree,
    # then return pre-stored candidate directly.
    # Otherwise, store the candidate in self.bros for future use.
    def FindCandidateByTree(self, treeno, vertex, sv = None, ret_groupno = False, vertices_pool = set([])):
        place = self.GetPlaceByVertex(treeno, vertex)
        groupno = self.graph.trees[treeno].getGroupno(place)
        
        if(groupno == 0):
            # candidate = ( vertices_pool & self.graph.GetZeroGroupOnIthTree(treeno)) | set([vertex]) 
            # candidate = ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno) |  set([vertex])
            candidate = set([vertex]) 
        else:
            candidate = vertices_pool & self.graph.trees[treeno].getGroupbyGroupno(groupno) | set([vertex])
        
        assert vertex in candidate

        if(sv):
            self.group2supernode[self.GetGroupid(treeno,groupno)].add(sv)
                    
        if(ret_groupno == False):
            return candidate
        else:
            return candidate, groupno
        

    def GetDistanceBtwSvs(self, i, j, aggmethod="MHD", edgelist = None):
        
        d = -1
        
        assert i in self.supernode.keys()
        assert j in self.supernode.keys()
               
        if(aggmethod == "COS"):
            d = utils.GetCosineDistance(self.supernode[i].rep, 
                                        self.supernode[j].rep)
        elif(aggmethod == "MHD"):
            if(abs(len(self.supernode[i].vertices) - len(self.supernode[j].vertices)) > self.edge_threshold):
                return d
            d = utils.GetManhattanDistance(self.supernode[i].rep, 
                                         self.supernode[j].rep)
        elif(aggmethod == "EDG"):
            
            d = utils.GetEdgeCountV0(self.supernode[i].vertices,
                                   self.supernode[j].vertices,
                                   self.graph.G,
                                   self.edge_threshold)
        else:
            print("Invalid distance calculation method.")
            
        return d
    
    # process trees which contains non-zero elements first
    def GetOrderedTree(self, vertex):
        
        A_nonzero_row, A_nonzero_col = self.graph.A.nonzero()[0], self.graph.A.nonzero()[1]
        
        rows_idx = np.where(A_nonzero_row==vertex)
        cols_idx = A_nonzero_col[rows_idx]
        nonzero_tree_idx = list(set([math.floor(col/(self.graph.partition_size)) for col in cols_idx]))
        ordered_tree = nonzero_tree_idx + [i for i in range(self.graph.K) if i not in nonzero_tree_idx]
        assert len(ordered_tree) == self.graph.K
        
        return ordered_tree
        
    def FindCandidateOfOneVertices(self, vertex, escape = 0, vertices_pool = None, influenced_tree2sv = None):
        
        candidate = set([vertex])

        if(influenced_tree2sv):
            find_node = self.supernode[vertex].stop_round + 1
        else:
            find_node = 0

        # process trees which contains non-zero elements first
        ordered_tree = self.GetOrderedTree(vertex)

        for k,i in enumerate(ordered_tree):

            if(influenced_tree2sv and i not in influenced_tree2sv):
                continue
            
            # print("for sv %d we have tree %d " % (vertex, i))
            
            candidate, tmpcandidate = None, None
            if(k == 0):
                candidate = self.FindCandidateByTree(i, vertex, sv = vertex, vertices_pool = vertices_pool)

                # 20230607 modified
                if(self.use_ele_cnt):
                    self.supernode[vertex].EleCntAdd(candidate)
                    lenc = self.supernode[vertex].CountCandidate(t = find_node)
                else:
                    lenc = len(candidate)
            else:
                tmpcandidate = self.FindCandidateByTree(i, vertex, sv = vertex, vertices_pool = vertices_pool) 

                # 20230607 modified
                if(self.use_ele_cnt):
                    self.supernode[vertex].EleCntAdd(tmpcandidate)
                    lenc = self.supernode[vertex].CountCandidate(t = find_node)
                else:
                    candidate = candidate & tmpcandidate
                    lenc = len(candidate)
            
            # record the tree
            
            if(lenc <= self.candidate_earlystop_thres):
                if(self.use_ele_cnt):
                    self.supernode[vertex].UpdateCandidate(t = find_node, cand2set = True, dismissed_vertices = vertices_pool)
                    self.supernode[vertex].stop_round = find_node
                return self.supernode[vertex].vertices, find_node+1
            
            find_node += 1

        if(self.use_ele_cnt):
            self.supernode[vertex].UpdateCandidate(t = find_node - 1, cand2set = True, dismissed_vertices = vertices_pool)
            self.supernode[vertex].stop_round = find_node
            
        if((self.max_candidate > 0 and lenc > self.max_candidate)):
            candidate = set(random.sample(candidate, k = self.max_candidate)) | set([vertex])

        #  print("%d full loop with %d candidate." % (vertex, lenc))
        return self.supernode[vertex].vertices, find_node + 1 

    def genSupernodeForZeros(self, match_dist = 0):
        
        if(match_dist):
            for i in tqdm(range(self.graph.K)):
                dist = 0
                if(i == 0):
                    candidate = self.graph.GetZeroGroupOnIthTree(i)
                else:
                    if(dist <= match_dist):
                        flag = random.randint(0,1)
                        if(not flag):# skip max match_dist times
                            dist += 1
                            continue
                    tmpcandidate = self.graph.GetZeroGroupOnIthTree(i)
                    candidate = candidate & tmpcandidate
                
        else:
            for i in tqdm(range(self.graph.K)):
                if(i == 0):
                    candidate = self.graph.GetZeroGroupOnIthTree(i)
                else:
                    tmpcandidate = self.graph.GetZeroGroupOnIthTree(i)
                    candidate = candidate & tmpcandidate
                # prune
                if(len(candidate)==1):
                    continue
            
        sv = candidate.pop()
        candidate.add(sv)
        self.AddCandidate2Supernode(sv, set(range(self.graph.vertices_num)), candidate)
        
        return sv
    
    def Addvertex2supernode(self, src, sn):
        src = int(src)

        self.vertex2supernode[src] = sn
        
        if(sn not in self.supernode.keys()):
            self.supernode[sn] = Supernode(candidate = set([src]), aggmethod = None, graph = None, se_gen_method = "EDG")
        else:
            self.supernode[sn].vertices.add(src)

    def genSupergraphForGSS(self):

        edge_connector = '-'

        t1 = utils.time()
        for edge in self.graph.G.edges():

            src = str(edge[0])
            dst = str(edge[1])

            h_src = (GSS.BOB1(src, len(src))>> self.f) % self.w
            h_dst = (GSS.BOB1(dst, len(dst))>> self.f) % self.w

            self.Addvertex2supernode(src, h_src)
            self.Addvertex2supernode(dst, h_dst)

            sekey = edge_connector.join([str(h_src), str(h_dst)])
            self.superedge_weight[sekey] += 1
        t2 = utils.time()
        
        for sekey in self.superedge_weight.keys():
            h_list = sekey.split(edge_connector)
            assert len(h_list) == 2
            self.superedge.append([int(h_list[0]), int(h_list[1])])

        t3 = utils.time()
        for node in self.graph.G.nodes():
            h = (GSS.BOB1(str(node), len(str(node)))>> self.f) % self.w
            if(self.graph.G.degree[node] == 0):    
                self.Addvertex2supernode(node, h)
            else:
                assert node in self.supernode[h].vertices

        t4 = utils.time()

        print("timing for loops 1, 2, 3: %.3f, %.3f, %.3f" % (t2- t1, t3-t2, t4-t3))

    def genSupernodeForAll(self, resummary = False):
        
        if(self.GSS):
            self.genSupergraphForGSS()
            return

        # shuffle vertices to process
        vertices_pool = list(self.graph.G.nodes) #range(vertices_num))
        random.shuffle(vertices_pool)
        vertices_pool = set(vertices_pool)
        
        # process zero group at first
        if(self.gen_zero_sn_first):
            t1 = time.time()
            zero_sv = self.genSupernodeForZeros(match_dist = self.zero_match_escape)
            t2 = time.time()
            print("zero match dist: %d. zero supernode has %d vertices. Generate zero sv timing: %.3f " \
                  % (self.zero_match_escape, len(self.supernode[zero_sv].vertices),t2-t1))
            
            vertices_pool -= self.supernode[zero_sv].vertices
        
        self.genSupernodeFromVpool(vertices_pool=vertices_pool, resummary=resummary)

    def genSupernodeFromVpool(self, vertices_pool, ret = False, resummary = False):
        t1 = time.time()
        
        sv = []
        
        find_step_sum = 0
        
        generated_supernode_cnt = 0
        generated_supernode_cnt_period = 0
        
        while(len(vertices_pool)):
            
            if(generated_supernode_cnt and generated_supernode_cnt % ECHO_PERIOD == 0 ):
                print("having generated %d supernodes, %d vertices left, avg find step: %.3f" % \
                      (generated_supernode_cnt, len(vertices_pool), (find_step_sum)/generated_supernode_cnt_period))
                # find_step_sum  = 0
                # generated_supernode_cnt_period = 0
                
            # pick a vertex at random
            vertex = vertices_pool.pop()
            
            if(vertex in self.supernode.keys()):
                print("conflict supernode in line 407:", vertex)

            if(ret):
                sv.append(vertex)
            
            if(self.use_ele_cnt):
                self.supernode[vertex] = Supernode(candidate = set(), 
                                           aggmethod = self.sn_agg_method, 
                                           graph = self.graph)

            # find candidate for one vertex
            candidate, find_step = self.FindCandidateOfOneVertices(vertex, 
                                                                   escape = self.other_match_escape,
                                                                   vertices_pool = vertices_pool | set([vertex]))
            find_step_sum += find_step

            # group all candidates as a supernode, remove candidate from vertices pool
            vertices_pool, removed_vertices_cnt = self.AddCandidate2Supernode(vertex, vertices_pool, candidate)

            generated_supernode_cnt += 1
            generated_supernode_cnt_period += 1
        
        t2 = time.time()


        print("Generate superNODE timing: %.3f" % (t2-t1))
        
        if(ret):
            return sv
        
        else:
            return None
    
    def PrintMinMaxDegreeinCand(self, candidate, prefix, minmax = True):
        
        degree_list = []
        
        min_degree = 1e10
        max_degree = -1
        for cand in candidate:
            if(minmax):
                min_degree = min(min_degree, self.graph.G.degree[cand])
                max_degree = max(max_degree, self.graph.G.degree[cand])
            else:
                degree_list.append(self.graph.G.degree[cand])
                
        c = collections.Counter(degree_list)
        
        if(minmax):
            print("%s mindegree: %d maxdegree:%d" %(prefix, min_degree, max_degree))
        else:
            print("%s degree counter:" % (prefix))
            print(c)
        
    
    def ShowInfo(self):

        self.n_supernode = len(self.supernode)
        self.n_superedge = len(self.superedge)
        print("Supergraph contains %d supernodes and %d superedges" % 
              (self.n_supernode, self.n_superedge))
        
        self.cpr_supernode = self.graph.G.number_of_nodes()/self.n_supernode
        self.cpr_superedge = self.graph.G.number_of_edges()/self.n_superedge if self.n_superedge else -1
        
        print("Vertex compression ratio: %.3f, edge compression ratio: %.3f" %
              (self.cpr_supernode, self.cpr_superedge))
    
    def UpdateSuperedge(self, newsv, allsv):
        
        self.superedge = []
        self.genSuperedge(supernode = self.supernode)
        
        return
    
    def ProcessSeByDask(self, i, supernode_t, edgelist):
        
        n = len(supernode_t)
        ret = []

        for j in range(i+1,n):
            
            if(self.se_gen_method != "EDG"):
                if(len(self.supernode[supernode_t[j]].vertices) == 1 and len(self.supernode[supernode_t[i]].vertices) == 1):
                    continue
                elif(abs(len(self.supernode[supernode_t[j]].vertices) - len(self.supernode[supernode_t[i]].vertices)) > self.edge_threshold):
                    continue
            else:
                if(len(self.supernode[supernode_t[j]].vertices) == 1 and len(self.supernode[supernode_t[i]].vertices) == 1):
                    d = self.graph.G.has_edge(supernode_t[j],supernode_t[i])
                else:
                    d = self.GetDistanceBtwSvs(supernode_t[i], supernode_t[j], edgelist = edgelist, aggmethod=self.se_gen_method)
                
            if(self.se_gen_method == "EDG"):
                if( d ):
                    ret.append([supernode_t[i], supernode_t[j]])
            elif( d > 0 and d < self.edge_threshold):
                ret.append([supernode_t[i], supernode_t[j]])
        
        return ret


    def genSuperedge(self, supernode, aggmethod = "MHD", decay = 1):
        
        """
        if(isinstance(supernode, dict)):
            supernode_t = tuple(supernode.keys())
        elif(isinstance(supernode, list) or isinstance(supernode, set)):
            supernode_t = tuple(supernode)

        n = len(supernode_t)
        
        # TODO: CHANGE TO ASSERT
        for sv in supernode_t:
            if(sv not in self.supernode.keys()):
                print("ERROR!!! sv:", sv, "not in self.supernode.keys" )
                exit(1)
        """

        supernodes_size_list = np.array([len(self.supernode[i].vertices) for i in self.supernode.keys()])
        thres = np.percentile(supernodes_size_list, self.genedge_selection_percentile)
        print("Superedge generation: %d percentile of len(supernode.vertices) is %.3f " % (self.genedge_selection_percentile,thres))
        
        t1 = time.time()

        def MakeSeKeys(v1, v2, supernode_num):
            return str(v1) + '.' + str(v2)
        
        def ExtractSeKeys(key, supernode_num):
            keypart = key.split('.')
            return int(keypart[0]), int(keypart[1])

        supernode_num = len(self.supernode)
        secounter = collections.defaultdict(lambda: 0)
        
        for e in self.graph.G.edges():
            sv1 = self.vertex2supernode[e[0]]
            sv2 = self.vertex2supernode[e[1]]

            if(sv1 == sv2):
                continue

            if(sv1 not in supernode or sv2 not in supernode):
                continue

            tmpkey = MakeSeKeys(sv1, sv2, supernode_num)
            secounter[tmpkey] += 1
        
        for tmpkey in secounter:
            sv1, sv2 = ExtractSeKeys(tmpkey, supernode_num)
            thres = (len(self.supernode[sv1].vertices) * self.edge_threshold) *  (len(self.supernode[sv2].vertices) * self.edge_threshold) * decay
            if(secounter[tmpkey] > thres):
                self.superedge.append([sv1, sv2])
        
        t2 = time.time()
        
        print("Generate superEDGE timing: %.3f" % (t2-t1))

    def genSuperedgeForAll(self):

        if(self.GSS):
            return

        self.genSuperedge(supernode = self.supernode)
    
    # find the groups vertex di and vertex dj are in
    def DeleteOneEdge(self, di, dj, directed = False):
        
        influenced_set = set()
        
        # A[di,dj] is in di-th row, ki-th partition(tree)

        # A[di,dj] is in di-th row, ki-th partition
        old_treeno_i, old_treeno_j = math.floor(di/(self.graph.partition_size)), math.floor(dj/(self.graph.partition_size))
        
        # C'
        di_place = self.graph.PlaceVertex2Tree(di, self.graph.GetIntervals(old_treeno_j), self.graph.trees[old_treeno_j])
        di_oldplace = self.GetPlaceByVertex(old_treeno_j, di)
        
        if(directed == False):
            dj_place = self.graph.PlaceVertex2Tree(dj, self.graph.GetIntervals(old_treeno_i), self.graph.trees[old_treeno_i])
            dj_oldplace = self.GetPlaceByVertex(old_treeno_i, dj)
    
        # using place and ki to build particular gid, and add it to influenced groups set.
        di_old_groupid = self.graph.trees[old_treeno_j].getGroupno(di_oldplace)
        di_new_groupid = self.graph.trees[old_treeno_j].getGroupno(di_place)

        dj_old_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_oldplace)
        dj_new_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_place)

        if(di_old_groupid != di_new_groupid):
            influenced_set.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_old_groupid)) 
            influenced_set.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_new_groupid))
        
        if(directed == False):
            if(dj_old_groupid != dj_new_groupid): 
                influenced_set.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_old_groupid))
                influenced_set.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_new_groupid))        

        return influenced_set
    
    # find the groups vertex di and vertex dj are in
    def AddOneEdge(self, di, dj, directed = False):
        
        influenced_set = set()
        
        # A[di,dj] is in di-th row, ki-th partition(tree)

        # A[di,dj] is in di-th row, ki-th partition
        old_treeno_i, old_treeno_j = math.floor(di/(self.graph.partition_size)), math.floor(dj/(self.graph.partition_size))
        
        # C'
        di_place = self.graph.PlaceVertex2Tree(di, self.graph.GetIntervals(old_treeno_j), self.graph.trees[old_treeno_j])
        di_oldplace = self.GetPlaceByVertex(old_treeno_j, di)
        
        if(directed == False):
            dj_place = self.graph.PlaceVertex2Tree(dj, self.graph.GetIntervals(old_treeno_i), self.graph.trees[old_treeno_i])
            dj_oldplace = self.GetPlaceByVertex(old_treeno_i, dj)
    
        # using place and ki to build particular gid, and add it to influenced groups set.
        di_old_groupid = self.graph.trees[old_treeno_j].getGroupno(di_oldplace)
        di_new_groupid = self.graph.trees[old_treeno_j].getGroupno(di_place)

        dj_old_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_oldplace)
        dj_new_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_place)

        if(di_old_groupid != di_new_groupid):
            influenced_set.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_old_groupid)) 
            influenced_set.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_new_groupid))
        
        if(directed == False):
            if(dj_old_groupid != dj_new_groupid): 
                influenced_set.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_old_groupid))
                influenced_set.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_new_groupid))

        return influenced_set

        
    def GetGroupid(self, treeno, groupno):
        return treeno * (self.encoding_n) + groupno
    
    def ParseTreenoGroupnoFromId(self, gid):
        treeno = math.floor(gid / self.encoding_n)
        groupno = gid - treeno * self.encoding_n
        return treeno, groupno
    
    def RemoveSupernode(self, sv, groupid):
        
        # this supernode has already been deleted
        assert sv in self.supernode.keys()
        
        self.group2supernode[groupid].remove(sv)
        
        # self.superedge = list(filter(lambda x: x[0]!=sv and x[1]!=sv, self.superedge))

        # vertices to be dismissed
        # sv_contained_vertices = copy.deepcopy(self.supernode[sv].vertices) 
        
        del self.supernode[sv]
        # return sv_contained_vertices
    
    def ReviseSns(self, dismissed_vertices, influenced_sv_set = None, uninfluenced_sv_set = None):
        removed_sv = set()
        
        tmpkey = -1

        changed_sv_set = set()
        replaced_sv_tmp1, replaced_sv_tmp2 = {}, {}
        
        # change supernodes whose key is not contained in regarding vertices
        for sv in list(influenced_sv_set) :
            
            if(sv not in self.supernode[sv].vertices and len(self.supernode[sv].vertices)):
                
                removed_sv.add(sv)
                self.supernode[tmpkey] = self.supernode.pop(sv)

                replaced_sv_tmp1[sv] = tmpkey
                    
                changed_sv_set.add(tmpkey)
                tmpkey -= 1

        influenced_sv_set |= changed_sv_set
        influenced_sv_set -= removed_sv
        
        removed_sv = set()
        changed_sv_set = set()
        
        removed_sv_step1 = 0

        def RemoveSvFromGroup2Supernode(sv):
            for g in self.group2supernode:
                self.group2supernode[g].discard(sv)

        for sv in list(influenced_sv_set) :
                
            if(sv < 0):

                newkey = list(self.supernode[sv].vertices)[0]
                
                if(newkey in self.supernode): # not added
                    assert not len(self.supernode[newkey].vertices)
                    removed_sv_step1 += 1
                    RemoveSvFromGroup2Supernode(newkey)
                    removed_sv.add(newkey)

                # assert newkey not in self.supernode.keys() or (newkey in self.supernode.keys() and not len(self.supernode[newkey].vertices))
                
                influenced_sv_set.add(newkey)
                self.supernode[newkey] = self.supernode.pop(sv) # here involve removed
                
                # if(tmp_stop_round >= 0):
                #     self.supernode[newkey].stop_round = tmp_stop_round
                #     tmp_stop_round = -100

                removed_sv.add(sv)
                changed_sv_set.add(newkey)
                replaced_sv_tmp2[sv] = newkey
                
                # update vertex2supernode mapping
                for v in self.supernode[newkey].vertices:
                    self.vertex2supernode[v] = newkey
                
            else:
                if( sv in dismissed_vertices ):
                    print(sv, "in dismissed_vertices, vertices:" , self.supernode[sv].vertices)
                    exit(1)

        print("%d supernodes removed while changing keys:" % removed_sv_step1)
        
        influenced_sv_set -= removed_sv
        influenced_sv_set |= changed_sv_set
        removed_sv = set()

        for sv in list(influenced_sv_set) :
            dismissed_vertices -= self.supernode[sv].vertices

            # remove empty supernodes
            if(len(self.supernode[sv].vertices) <= 0):

                removed_sv.add(sv)

                del self.supernode[sv]

        # update group2supernode mapping
        
        replaced_old_sv = set(list(replaced_sv_tmp1.keys()))
        for g in self.group2supernode:
            self.group2supernode[g] -= replaced_old_sv
            self.group2supernode[g] -= removed_sv
            
            if(len(replaced_old_sv & self.group2supernode[g])):
                for old_sv in (replaced_old_sv & self.group2supernode[g]):
                    self.group2supernode[g].remove(old_sv)
                    self.group2supernode[g].add(replaced_sv_tmp2[replaced_sv_tmp1[old_sv]])

        print("len(self.supernode):", len(self.supernode))
        
        # influenced_sv_set -= changed_sv_set

        if(influenced_sv_set):
            influenced_sv_set -= removed_sv
        
        removed_sv_step2 = len(removed_sv)
        print("%d empty supernodes have been removed" % len(removed_sv))

        return influenced_sv_set, dismissed_vertices, removed_sv_step1, removed_sv_step2
        
    # for sv's candidate in group
    # remove the candidate's representation from sv, mainly cnt
    # later on these candidates will be reassigned
    def RemoveGroupFromOneSvV0(self, sv, treeno, groupno):
        
        prev_vertices = self.supernode[sv].vertices.copy()
        
        vertices_pool = set(list(range(self.graph.vertices_num)))

        # for the v-th vertex
        if(groupno == 0):
            # candidate = ( vertices_pool & self.graph.GetZeroGroupOnIthTree(treeno)) | set([vertex]) 
            # _group = ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno)
            _group = self.graph.trees[treeno].getGroupbyGroupno(groupno)
        else:
            _group = self.graph.trees[treeno].getGroupbyGroupno(groupno)

        self.supernode[sv].RemoveVfromSn( _group, self.graph, self.sn_agg_method )
        self.supernode[sv].stop_round -= 1

        assert self.supernode[sv].stop_round >= -1

        d_vertices = prev_vertices - self.supernode[sv].vertices
        
        for v in d_vertices:
            del self.vertex2supernode[v]
        
        return d_vertices
    
    def RemoveGroupFromOneSv(self, sv, treeno, groupno):

        # for the v-th vertex

        if(groupno == 0):
            # candidate = ( vertices_pool & self.graph.GetZeroGroupOnIthTree(treeno)) | set([vertex]) 
            # vertices_pool = set(list(range(self.graph.vertices_num)))
            # _group = ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno) 
            _group = self.graph.trees[treeno].getGroupbyGroupno(groupno)
        else:
            _group = self.graph.trees[treeno].getGroupbyGroupno(groupno)

        if(len(_group & self.supernode[sv].vertices) == 0):
            return
        
        self.supernode[sv].EleCntRemove( _group )
        self.supernode[sv].stop_round -= 1

        if(self.supernode[sv].stop_round < -1):
            print("debug 916 self.supernode[sv].stop_round< -1, sv:", sv, treeno, groupno)
        assert self.supernode[sv].stop_round >= -1
        return
    
    # Remove group influence from Supernodes
    def RemoveZerogroup(self, influenced_svs, treeno, groupno):
        assert groupno == 0

        vertices_pool = set(list(range(self.graph.vertices_num)))
        _group = ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno) 
        dismissed_vertices = set()

        for sv in influenced_svs:
            prev_vertices = self.supernode[sv].vertices.copy()
            self.supernode[sv].EleCntRemove( _group )
            self.supernode[sv].stop_round -= 1

            assert self.supernode[sv].stop_round >= -1

            d_vertices = prev_vertices - self.supernode[sv].vertices
        
            for v in d_vertices:
                del self.vertex2supernode[v]

            dismissed_vertices |= d_vertices 
        
        return dismissed_vertices

    def RemoveGroupInfFromSvsSubfunc1(self, g, ):
        return

    # simply remove influenced groups from existing supernodes counters and record influenced trees
    def RemoveGroupInfFromSvs(self, influenced_groups, dismissed_vertices):

        influenced_tree2sv = collections.defaultdict(set)
        influenced_sv_set = set()

        t1 = time.time()
        cnt_influenced_groups = 0
        debug_cnt = 0
        
        for g in influenced_groups:
            influenced_svs = self.group2supernode[g].copy()
            
            if(len(influenced_svs) == 0):
                continue
            
            treeno, groupno = self.ParseTreenoGroupnoFromId(g)

            influenced_tree2sv[treeno] |= influenced_svs
            
            # count incluenced supernode
            influenced_sv_set |= influenced_svs
            self.group2supernode[g] -= influenced_svs

            cnt_influenced_groups += 1    
            for sv in influenced_svs:
                dismissed_vertices |= self.supernode[sv].vertices
                debug_cnt += 1
                self.RemoveGroupFromOneSv(sv, treeno, groupno)

        t2 = time.time()

        print("loop 1 in RemoveGroupInfFromSvs timing: %.3f, cnt_influenced_groups: %d" % (t2 - t1, cnt_influenced_groups))
        
        return influenced_tree2sv, influenced_sv_set, dismissed_vertices
    
    # move vertices in deleted_edges from one group to another if necessary - v0
    def UpdateGroupV0(self, edgelist, deleted_edge_idx, directed = False, influenced_groups = None):
        
        cnt = 0

        for deleted_edge_id in deleted_edge_idx:
            
            deleted_edge = edgelist[deleted_edge_id]
            di,dj = deleted_edge[0], deleted_edge[1]
            
            # remove node and edit adjecancy matrix accordingly
            self.graph.G.remove_edge(di,dj)
            
            # self.graph.A[di,dj] = 0
            # if(directed == False):
            #     self.graph.A[dj,di] = 0
        self.graph.UpdateA()

        for i, deleted_edge_id in enumerate(deleted_edge_idx):
            
            deleted_edge = edgelist[deleted_edge_id]
            di,dj = deleted_edge[0], deleted_edge[1]
            
            if(i<10):
                print("deleted edge %d : %d - %d " % (i, di, dj))
                
            # A[di,dj] is in di-th row, ki-th partition
            old_treeno_i, old_treeno_j = math.floor(di/(self.graph.partition_size)), math.floor(dj/(self.graph.partition_size))
            
            # C'
            di_place = self.graph.PlaceVertex2Tree(di, self.graph.GetIntervals(old_treeno_j), self.graph.trees[old_treeno_j])
            if(directed == False):
                dj_place = self.graph.PlaceVertex2Tree(dj, self.graph.GetIntervals(old_treeno_i), self.graph.trees[old_treeno_i])
            
            
            di_oldplace = self.GetPlaceByVertex(old_treeno_j, di)
            if(directed == False):
                dj_oldplace = self.GetPlaceByVertex(old_treeno_i, dj)
            
            #di_place = self.GetPlaceByVertex(kj, di)
        
            #if(directed == False):
            #    dj_place = self.GetPlaceByVertex(ki, dj)
        
            # using place and ki to build particular gid, and add it to influenced groups set.
            influenced_groups.add(self.GetGroupid(treeno = old_treeno_j, groupno=self.graph.trees[old_treeno_j].getGroupno(di_place)))
            if(directed == False):
                influenced_groups.add(self.GetGroupid(treeno = old_treeno_i, groupno=self.graph.trees[old_treeno_i].getGroupno(dj_place)))

            if(di_place != di_oldplace):
                self.graph.trees[old_treeno_j].UpdateVertexPlace(di, 
                                                    oldplace = di_oldplace,
                                                    newplace = di_place)
                cnt += 1
            if(directed == False and dj_place!=dj_oldplace):
                self.graph.trees[old_treeno_i].UpdateVertexPlace(dj,
                                                        oldplace = dj_oldplace,
                                                        newplace = dj_place)
                cnt += 1
        
        print("Vertices have changed groups %d times "%cnt)
        return influenced_groups
    
    # move vertices in deleted_edges from one group to another if necessary
    def UpdateGroup(self, edgelist, deleted_edge_idx, directed = False, influenced_groups = None, implement = False):
        
        cnt = 0

        if(implement == False):
            for deleted_edge_id in deleted_edge_idx:
                
                deleted_edge = edgelist[deleted_edge_id]
                di,dj = deleted_edge[0], deleted_edge[1]
                
                # remove node and edit adjecancy matrix accordingly
                self.graph.G.remove_edge(di,dj)
                
            self.graph.UpdateA()

        for i, deleted_edge_id in enumerate(deleted_edge_idx):
            
            deleted_edge = edgelist[deleted_edge_id]
            di,dj = deleted_edge[0], deleted_edge[1]
            
            # A[di,dj] is in di-th row, ki-th partition
            old_treeno_i, old_treeno_j = math.floor(di/(self.graph.partition_size)), math.floor(dj/(self.graph.partition_size))
            
            # C'
            di_place = self.graph.PlaceVertex2Tree(di, self.graph.GetIntervals(old_treeno_j), self.graph.trees[old_treeno_j])
            if(directed == False):
                dj_place = self.graph.PlaceVertex2Tree(dj, self.graph.GetIntervals(old_treeno_i), self.graph.trees[old_treeno_i])
            
            
            di_oldplace = self.GetPlaceByVertex(old_treeno_j, di)
            if(directed == False):
                dj_oldplace = self.GetPlaceByVertex(old_treeno_i, dj)
        
            # using place and ki to build particular gid, and add it to influenced groups set.
            di_old_groupid = self.graph.trees[old_treeno_j].getGroupno(di_oldplace)
            di_new_groupid = self.graph.trees[old_treeno_j].getGroupno(di_place)

            dj_old_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_oldplace)
            dj_new_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_place)

            if(di_old_groupid != di_new_groupid):
                # print(old_treeno_j, di_old_groupid , di_new_groupid)
                influenced_groups.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_old_groupid)) #self.GetGroupid(treeno = old_treeno_j, groupno=di_old_groupid))
                influenced_groups.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_new_groupid))
                cnt += 2
            if(directed == False):
                if(dj_old_groupid != dj_new_groupid): 
                    # print(old_treeno_i, dj_old_groupid , dj_new_groupid)
                    influenced_groups.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_old_groupid))
                    influenced_groups.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_new_groupid))
                    cnt += 2
                # influenced_groups.add(self.GetGroupid(treeno = old_treeno_i, groupno=self.graph.trees[old_treeno_i].getGroupno(dj_place)))

            if(implement):
                if(di_place != di_oldplace):
                    self.graph.trees[old_treeno_j].UpdateVertexPlace(di, 
                                                        oldplace = di_oldplace,
                                                        newplace = di_place)
                    # print(di, old_treeno_j, di_oldplace, di_place, self.graph.trees[old_treeno_j].getGroupno(di_oldplace), self.graph.trees[old_treeno_j].getGroupno(di_place))
                    
                if(directed == False and dj_place!=dj_oldplace):
                    self.graph.trees[old_treeno_i].UpdateVertexPlace(dj,
                                                            oldplace = dj_oldplace,
                                                            newplace = dj_place)
                    # print(dj, old_treeno_i, dj_oldplace, dj_place, self.graph.trees[old_treeno_i].getGroupno(dj_oldplace), self.graph.trees[old_treeno_i].getGroupno(dj_place))
            
        return influenced_groups
    
    def UpdateGroupAddDel(self, insert_edges, removed_edges, directed, influenced_groups, implement):

        cnt = 0

        if(implement == False):
            for cur_edge in insert_edges:
                
                di,dj = cur_edge[0], cur_edge[1]
                
                # remove node and edit adjecancy matrix accordingly
                self.graph.G.add_edge(di,dj)
                
            for cur_edge in removed_edges:
                
                di,dj = cur_edge[0], cur_edge[1]
                
                # remove node and edit adjecancy matrix accordingly
                try:
                    self.graph.G.remove_edge(di,dj)
                except:
                    pass

            self.graph.UpdateA()

        for i, cur_edge in enumerate(insert_edges + removed_edges):
            
            di,dj = cur_edge[0], cur_edge[1]
            
            # A[di,dj] is in di-th row, ki-th partition
            old_treeno_i, old_treeno_j = math.floor(di/(self.graph.partition_size)), math.floor(dj/(self.graph.partition_size))
            
            # C'
            di_place = self.graph.PlaceVertex2Tree(di, self.graph.GetIntervals(old_treeno_j), self.graph.trees[old_treeno_j])
            if(directed == False):
                dj_place = self.graph.PlaceVertex2Tree(dj, self.graph.GetIntervals(old_treeno_i), self.graph.trees[old_treeno_i])
            
            
            di_oldplace = self.GetPlaceByVertex(old_treeno_j, di)
            if(directed == False):
                dj_oldplace = self.GetPlaceByVertex(old_treeno_i, dj)
        
            # using place and ki to build particular gid, and add it to influenced groups set.
            di_old_groupid = self.graph.trees[old_treeno_j].getGroupno(di_oldplace)
            di_new_groupid = self.graph.trees[old_treeno_j].getGroupno(di_place)

            dj_old_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_oldplace)
            dj_new_groupid = self.graph.trees[old_treeno_i].getGroupno(dj_place)

            if(di_old_groupid != di_new_groupid):
                # print(old_treeno_j, di_old_groupid , di_new_groupid)
                influenced_groups.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_old_groupid)) #self.GetGroupid(treeno = old_treeno_j, groupno=di_old_groupid))
                influenced_groups.add(self.GetGroupid(treeno = old_treeno_j, groupno=di_new_groupid))
                
                # if(implement):
                #    sv_i = self.vertex2supernode[di]
                #    self.group2supernode[self.GetGroupid(old_treeno_j,di_new_groupid)].add(sv_i)
                cnt += 2

            if(directed == False):
                if(dj_old_groupid != dj_new_groupid): 
                    
                    influenced_groups.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_old_groupid))
                    influenced_groups.add(self.GetGroupid(treeno = old_treeno_i, groupno=dj_new_groupid))
                    
                #    if(implement):
                #        sv_j = self.vertex2supernode[dj]
                #        self.group2supernode[self.GetGroupid(old_treeno_i,dj_new_groupid)].add(sv_j)
                    cnt += 2

            if(implement):
                if(di_place != di_oldplace):
                    self.graph.trees[old_treeno_j].UpdateVertexPlace(di, 
                                                        oldplace = di_oldplace,
                                                        newplace = di_place)
                                        
                if(directed == False and dj_place!=dj_oldplace):
                    self.graph.trees[old_treeno_i].UpdateVertexPlace(dj,
                                                            oldplace = dj_oldplace,
                                                            newplace = dj_place)
        
        return influenced_groups

    # add candidates on treeno-th tree to supernode sv with an escape rate 0.8
    # remove the added candidate from dismissed_vertices and the number of vertices assigned
    def UpdateGroupInfOnSvsByTree(self, influenced_tree2sv, influenced_sv_set, dismissed_vertices, updateSvOnTree = True):
        
        def RemoveCandidateFromExistingSN(cur_sv, candidate):
            cnt = 0
            for cand in candidate:
                oldsv = self.vertex2supernode[cand]
                if(oldsv != cur_sv): # candidate has changed supernodes
                    self.supernode[oldsv].RemoveVfromSn(cand, graph = self.graph, aggmethod = self.sn_agg_method)
                    cnt += 1
            return cnt

        # the whole vertices pool
        d_vertices = dismissed_vertices
        removed_candidate = 0
        early_stop_cnt = 0

        t1 = time.time()
        print("len(influenced_tree2sv):%d, len(influenced_sv_set): %d, len(d_vertices):%d" % (len(influenced_tree2sv),len(influenced_sv_set),len(d_vertices)))
        debug = 0
        for sv in influenced_sv_set:
            
            if(len(d_vertices) <= 0 ):
                break
            
            # find candidate for one vertex only in influenced trees
            candidate, find_step = self.FindCandidateOfOneVertices(sv, 
                                                                   escape = self.other_match_escape,
                                                                   vertices_pool = d_vertices,
                                                                   influenced_tree2sv = influenced_tree2sv)
            
            # if(debug < 5):
            #     print("debug:", sv, candidate, find_step)
            debug += 1
            if(find_step < self.graph.K):
                early_stop_cnt += 1

            # remove candidate from other supernode
            removed_candidate += RemoveCandidateFromExistingSN(sv, candidate)
            
            # group all candidates as a supernode, remove candidate from vertices pool
            d_vertices, removed_vertices_cnt = self.AddCandidate2Supernode(sv, d_vertices, candidate, unsummary=True)

        t2 = time.time()
        
        print("Generate superNODE timing: %.3f, %d candidates have been moved from existing supernodes to new ones, %d early stops" % (t2-t1, removed_candidate, early_stop_cnt))

        return d_vertices, influenced_sv_set
    
    
    def printDebugInfo(self):
        return

    def AlignSupernode(self, uninfluenced_sv_set):
        allkeys = list(self.supernode.keys())
        vertices_cnt = 0
        left_vertices = set(list(self.graph.G.nodes))
        
        for sv in allkeys:
            tmp_vertices = self.supernode[sv].vertices
            left_vertices -= tmp_vertices

            for v in tmp_vertices:
                if(v not in self.vertex2supernode):
                    self.vertex2supernode[v] = sv
                elif(self.vertex2supernode[v] != sv): # no such outputs
                    print("Not aligned vertex: %d, sv: %d,vertex2supernode[v]: %d, in uninfluenced_sv_set: %d" % (v, sv, self.vertex2supernode[v], sv in uninfluenced_sv_set))
                    # self.supernode[sv].RemoveVfromSn(int(v), graph = self.graph, aggmethod = self.sn_agg_method)

            vertices_cnt += len(self.supernode[sv].vertices)

    def EdgeInsDelGSS(self, deleted_edge_idx, deleted_edges = [], insert_edges = []):
        
        if(not deleted_edges):
            assert len(deleted_edge_idx)

            deleted_edges = []
            edgelist= list(self.graph.G.edges)
            for edge_idx in deleted_edge_idx:
                deleted_edges.append([edgelist[edge_idx][0], edgelist[edge_idx][1]])
        
        print("Remove %d edges in total." % len(deleted_edges))

        cnt = 0

        edge_connector = '-'
        
        t1 = time.time()
        for deleted_edge in deleted_edges:
            
            src, dst = str(deleted_edge[0]), str(deleted_edge[1])

            h_src = (GSS.BOB1(src, len(src))>> self.f) % self.w
            h_dst = (GSS.BOB1(dst, len(dst))>> self.f) % self.w

            sekey = edge_connector.join([str(h_src), str(h_dst)])
            self.superedge_weight[sekey] -= 1

        for insert_edge in insert_edges:
            
            src, dst = str(insert_edge[0]), str(insert_edge[1])

            h_src = (GSS.BOB1(src, len(src))>> self.f) % self.w
            h_dst = (GSS.BOB1(dst, len(dst))>> self.f) % self.w

            sekey = edge_connector.join([str(h_src), str(h_dst)])
            self.superedge_weight[sekey] += 1

        t2 = time.time()

        def FindKeysByPositiveValues(seweight:dict):
            value_array = np.array(list(seweight.values()))
            pos_pos = list(np.where(value_array > 0)[0])
            key_list = list(seweight.keys())
            return [key_list[i] for i in pos_pos]
        
        tmpsekeys = FindKeysByPositiveValues(self.superedge_weight)
        print("len(self.superedge_weight): %d, len(tmpsekeys): %d" % (len(self.superedge_weight),len(tmpsekeys) ))
        t5 = time.time()

        self.superedge = []
        for sekey in tmpsekeys:
            h_list = sekey.split(edge_connector)
            assert len(h_list) == 2
            self.superedge.append([int(h_list[0]), int(h_list[1])])
     
        t3 = time.time()
        print("Timing in GSS for loops 1, 2: %.3f, %.3f, %.3f " % (t2 - t1, t5 - t2, t3 - t5))
        """
        for deleted_edge in deleted_edges:
            self.graph.G.remove_edge(deleted_edge[0], deleted_edge[1])
        
        for insert_edge in insert_edges:
            self.graph.G.add_edge(insert_edge[0], insert_edge[1])
        """

        print("Unsummarize by GSS timing: %.3f " % (t3 - t1))

        return

    def EdgeDelete(self, deleted_edge_idx, removed_vertices = []):
        
        if(self.GSS):
            self.EdgeDeleteGSS(deleted_edge_idx)
            return self.supernode.keys()

        t1 = time.time()
        
        influenced_groups = set()
        edgelist= list(self.graph.G.edges)
        
        # find groups which deleted_edges are in
        if(len(removed_vertices)):
            assert self.graph.delete_vertex
            
            # the edges are already stored in deleted_edge_idx
            for deleted_edge in deleted_edge_idx:
                
                di,dj = deleted_edge[0], deleted_edge[1]
                
                influenced_groups |= self.DeleteOneEdge(di, dj, directed = utils.dataset_info[self.graph.dataset_name]["directed"])
        else:
            assert not self.graph.delete_vertex
            
            influenced_groups = self.UpdateGroup(edgelist, deleted_edge_idx, directed = utils.dataset_info[self.graph.dataset_name]["directed"], influenced_groups = set([]), implement = False)

        t2 = time.time()
        assert len(influenced_groups)
        print("find influenced_groups timing: %.3f, %d influenced_groups" % (t2-t1, len(influenced_groups)))
        dismissed_vertices = set()
        
        # remove the influenced group from the supernode that uses this group of vertice.
        # We first calculate influenced groups to avoid remove repeatedly
        
        influenced_tree2sv, influenced_sv_set, dismissed_vertices = self.RemoveGroupInfFromSvs(influenced_groups, dismissed_vertices)
        
        t3 = time.time()

        uninfluenced_sv_set = set(list(self.supernode.keys())) - influenced_sv_set
        print("RemoveGroupInfFromSvs timing: %.3f, len(uninfluenced_sv_set): %d, len(self.supernode): %d" % (t3-t2, len(uninfluenced_sv_set), len(self.supernode)))
        print("uninfluenced_sv_set top3:", list(uninfluenced_sv_set)[:3], "influenced_sv_set:", list(influenced_sv_set)[:3])
        
        # Update Group state
        influenced_groups = self.UpdateGroup(edgelist, deleted_edge_idx, directed = utils.dataset_info[self.graph.dataset_name]["directed"], influenced_groups = set([]), implement = True)
        #print(self.graph.A.todense())
        t4 = time.time()
        print("UpdateGroup timing: %.3f, 8214 in influenced_sv_set: %d" % (t4-t3, 8214 in influenced_sv_set))
        
        if(self.graph.load_zero):
            for i in range(self.graph.K):
                self.graph.UpdateZeroFile(i)
        
        # some vertices in the influenced groups has been changed to other groups
        # the dismissed_vertices can be assigned to current supernodes
        if(self.use_ele_cnt):
            
            assert len(uninfluenced_sv_set & influenced_sv_set) == 0
            dismissed_vertices, influenced_sv_set = self.UpdateGroupInfOnSvsByTree(influenced_tree2sv, influenced_sv_set, dismissed_vertices)
            influenced_sv_set_len = len(influenced_sv_set)
            t5 = time.time()
            print("UpdateGroupInfOnSvs timing: %.3f, len(dismissed_vertices): %d, len(self.supernode): %d" % (t5-t4, len(dismissed_vertices), len(self.supernode)))
            
            influenced_sv_set, dismissed_vertices, removed_sv_step1, removed_sv_step2 = self.ReviseSns(dismissed_vertices, influenced_sv_set, uninfluenced_sv_set)

            t6 = time.time()
            print("ReviseSns timing: %.3f, len(dismissed_vertices): %d, len(self.supernode): %d" % (t6-t5, len(dismissed_vertices), len(self.supernode)))
            
        else:
            influenced_sv_set = set([])
        
        newsv =  self.genSupernodeFromVpool(dismissed_vertices, ret = True)
        print("newsv[:3]: ", newsv[:3])
        assert len(set(newsv) & uninfluenced_sv_set) == 0
        t7 = time.time()

        if(influenced_sv_set_len + len(uninfluenced_sv_set) + len(newsv) - removed_sv_step1 - removed_sv_step2 != len(self.supernode) ):
            print("figures not matched", influenced_sv_set_len , len(uninfluenced_sv_set) , len(newsv) , removed_sv_step1 , removed_sv_step2 , len(self.supernode))
            exit(1)
        print("genSupernodeFromVpool timing: %.3f, %d new supernodes generated, len(self.supernode):%d" % (t7-t6, len(newsv), len(self.supernode)))
        
        for i in [0,1,2,4,6,8,8204,8207,4111]:
            if(i in self.supernode.keys()):
                print(self.supernode[i].vertices, self.supernode[i].stop_round, self.supernode[i].ele_cnt)
        # self.AlignSupernode(uninfluenced_sv_set)

        # shuffle dismissed_vertices
        # vertices_pool = list(newsv) + list(influenced_sv_set)
        # random.shuffle(vertices_pool)
        # vertices_pool = set(vertices_pool)
        
        # Update superedge
        self.UpdateSuperedge(None, None)
        
        t8 = time.time()
        print("UpdateSuperedge timing: %.3f"  % (t8-t7))
        
        if(len(removed_vertices)):
            print("Process NODE deletion timing: %.3f" % (t8-t1))
        else:
            print("Process EDGE deletion timing: %.3f" % (t8-t1))
        
        return uninfluenced_sv_set
        # print("Having removed %d superNODEs and added %d superNODEs" % (cnt_deletedsv, len(newsv)))
        # print("Having removed %d superEDGEs and added %d superEDGEs" % (cnt_deletedse, cnt_addedse))

    def EdgeInsertGSS(self, insert_edges):
        
        print("Insert %d edges in total." % len(insert_edges))

        cnt = 0

        edge_connector = '-'
        
        t1 = time.time()
        for insert_edge in insert_edges:
            # self.graph.G.remove_edge(edgelist[deleted_edge][0], edgelist[deleted_edge][1])
            src, dst = str(insert_edge[0]), str(insert_edge[1])

            h_src = (GSS.BOB1(src, len(src))>> self.f) % self.w
            h_dst = (GSS.BOB1(dst, len(dst))>> self.f) % self.w

            sekey = edge_connector.join([str(h_src), str(h_dst)])
            self.superedge_weight[sekey] += 1

        t2 = time.time()

        def FindKeysByPositiveValues(seweight:dict):
            value_array = np.array(list(seweight.values()))
            pos_pos = list(np.where(value_array > 0)[0])
            key_list = list(seweight.keys())
            return [key_list[i] for i in pos_pos]

        tmpsekeys = FindKeysByPositiveValues(self.superedge_weight)
        
        t5 = time.time()

        self.superedge = []
        for sekey in tmpsekeys:
            h_list = sekey.split(edge_connector)
            assert len(h_list) == 2
            self.superedge.append([int(h_list[0]), int(h_list[1])])
     
        t3 = time.time()
        
        for insert_edge in insert_edges:
            self.graph.G.add_edge(insert_edge[0], insert_edge[1])

        print("Unsummarize by GSS timing: %.3f, remove %d superedges " % (t3 - t1, cnt))

        return

    def EdgeAddDel(self, insert_edges, removed_edges, GSS = False):

        t1 = time.time()

        if(GSS):
            #self.EdgeInsertGSS(insert_edges = insert_edges)
            self.EdgeInsDelGSS(deleted_edge_idx = [], deleted_edges = removed_edges, insert_edges = insert_edges)
            return self.supernode.keys()
        
        influenced_groups = set()
        
        # find groups which deleted_edges are in
        assert not self.graph.delete_vertex
        
        influenced_groups = self.UpdateGroupAddDel(insert_edges, removed_edges, directed = utils.dataset_info[self.graph.dataset_name]["directed"], 
                                                   influenced_groups = set([]), implement = False)

        t2 = time.time()
        assert len(influenced_groups)
        print("find influenced_groups timing: %.3f, %d influenced_groups" % (t2-t1, len(influenced_groups)))
        dismissed_vertices = set()
        
        # remove the influenced group from the supernode that uses this group of vertice.
        # We first calculate influenced groups to avoid remove repeatedly
        
        influenced_tree2sv, influenced_sv_set, dismissed_vertices = self.RemoveGroupInfFromSvs(influenced_groups, dismissed_vertices)
        
        t3 = time.time()

        uninfluenced_sv_set = set(list(self.supernode.keys())) - influenced_sv_set
        print("RemoveGroupInfFromSvs timing: %.3f, len(uninfluenced_sv_set): %d, len(self.supernode): %d, len(dismissed_vertices): %d" % (t3-t2, len(uninfluenced_sv_set), len(self.supernode), len(dismissed_vertices)))
        # print("uninfluenced_sv_set top3:", list(uninfluenced_sv_set)[:3], "influenced_sv_set:", list(influenced_sv_set)[:3])
        
        # Update Group state
        influenced_groups = self.UpdateGroupAddDel(insert_edges, removed_edges, directed = utils.dataset_info[self.graph.dataset_name]["directed"], 
                                             influenced_groups = set([]), implement = True)
        #print(self.graph.A.todense())
        t4 = time.time()
        print("UpdateGroup timing: %.3f" % (t4-t3))
        
        # some vertices in the influenced groups has been changed to other groups
        # the dismissed_vertices can be assigned to current supernodes
        if(self.use_ele_cnt):
            
            assert len(uninfluenced_sv_set & influenced_sv_set) == 0
            dismissed_vertices, influenced_sv_set = self.UpdateGroupInfOnSvsByTree(influenced_tree2sv, influenced_sv_set, dismissed_vertices)
            influenced_sv_set_len = len(influenced_sv_set)
            t5 = time.time()
            print("UpdateGroupInfOnSvs timing: %.3f, len(dismissed_vertices): %d, len(self.supernode): %d" % (t5-t4, len(dismissed_vertices), len(self.supernode)))
            
            influenced_sv_set, dismissed_vertices, removed_sv_step1, removed_sv_step2 = self.ReviseSns(dismissed_vertices, influenced_sv_set, uninfluenced_sv_set)

            t6 = time.time()
            print("ReviseSns timing: %.3f, len(dismissed_vertices): %d, len(self.supernode): %d" % (t6-t5, len(dismissed_vertices), len(self.supernode)))
            
        else:
            influenced_sv_set = set([])
        
        newsv =  self.genSupernodeFromVpool(dismissed_vertices, ret = True)

        assert len(set(newsv) & uninfluenced_sv_set) == 0
        t7 = time.time()

        if(influenced_sv_set_len + len(uninfluenced_sv_set) + len(newsv) - removed_sv_step1 - removed_sv_step2 != len(self.supernode) ):
            print("figures not matched", influenced_sv_set_len , len(uninfluenced_sv_set) , len(newsv) , removed_sv_step1 , removed_sv_step2 , len(self.supernode))
            exit(1)
        print("genSupernodeFromVpool timing: %.3f, %d new supernodes generated, len(self.supernode):%d" % (t7-t6, len(newsv), len(self.supernode)))

       # self.AlignSupernode(uninfluenced_sv_set)

        # Update superedge
        self.UpdateSuperedge(None, None)
        
        t8 = time.time()
        print("UpdateSuperedge timing: %.3f"  % (t8-t7))
        print("Process EDGE insertion & deletion timing: %.3f" % (t8-t1))
        
        return uninfluenced_sv_set
    
    # Vertex deletion can be regarded as deleting multiple edges related to targeted vertice
    # the extra move is to removed the targetd vertices from the vertices pool
    def VertexDelete(self, deleted_vertices):
        
        deleted_edges = filter(lambda x:x[0] in deleted_vertices or x[1] in deleted_vertices,
                               self.graph.G.edges)
        
        self.EdgeDelete(deleted_edges, deleted_vertices)
        
    def EdgeAddition(self):
        pass
    
    def VertexAddition(self):
        pass
        