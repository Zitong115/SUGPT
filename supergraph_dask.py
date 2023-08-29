# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:23:08 2023

@author: ztli
"""

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
        if(se_gen_method != "EDG"):
            self.GetRepresentation(graph, aggmethod)
        
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
    
    def __init__(self, graph, config, dp = -1):
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
            # choose earlystop_threshold automatically
            # degree_list = np.array([d for n, d in self.graph.G.degree()])
            # self.candidate_earlystop_thres = np.percentile(degree_list, config["earlystop_thres_selection_percentile"])

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

    def ResetDeletep(self, dp):
        self.delete_p = dp
    
    def GenEdgeDeleteIdx(self, dp):
        deleted_edges_idx, deleted_nodes_idx = utils.GenerateDeletedIdx(self.graph.G, delete_p = dp, delete_vertex = self.graph.delete_vertex)
        return deleted_edges_idx, deleted_nodes_idx

    def AddCandidate2Supernode(self, sn, nodepool, candidate, remove_from_groups = False):
        
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
        
        if( len(filtered_candidate) == 0):
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
            # print("treeno:%d, groupno:%d, gid:%d" %(treeno, groupno, self.GetGroupid(treeno,groupno)))
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
            """
            d = utils.GetEdgeCount(self.supernode[i].vertices,
                                   self.supernode[j].vertices,
                                   self.graph.A.nonzero()[0],
                                   self.graph.A.nonzero()[1],
                                   self.edge_threshold)
            """

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
        
    def FindCandidateOfOneVertices(self, vertex, escape = 0, vertices_pool = None):
        
        candidate = set([vertex])
        find_node = 0
        
        # process trees which contains non-zero elements first
        ordered_tree = self.GetOrderedTree(vertex)

        for k,i in enumerate(ordered_tree):

            p = random.random()
                
            if(escape and p > escape):
                continue
            
            if(k == 0):
                candidate = self.FindCandidateByTree(i, vertex, sv = vertex, vertices_pool = vertices_pool)
                
                # 20230607 modified
                if(self.use_ele_cnt):
                    self.supernode[vertex].EleCntAdd(candidate)
                    # lenc = len(self.supernode[vertex].U
                    # 
                    # pdateCandidate(t = find_node * self.other_match_escape))
                    # lenc = len(self.supernode[vertex].UpdateCandidate(t = find_node))
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
            
            if(lenc <= self.candidate_earlystop_thres):
                if(self.use_ele_cnt):
                    self.supernode[vertex].UpdateCandidate(t = find_node, cand2set = True, dismissed_vertices = vertices_pool)
                    self.supernode[vertex].stop_round = find_node
                return candidate, find_node+1
            
            find_node += 1

        if(self.use_ele_cnt):
            self.supernode[vertex].UpdateCandidate(t = find_node - 1, cand2set = True, dismissed_vertices = vertices_pool)
            self.supernode[vertex].stop_round = find_node
            
        if((self.max_candidate > 0 and lenc > self.max_candidate)):
            candidate = set(random.sample(candidate, k = self.max_candidate)) | set([vertex])

        #  print("%d full loop with %d candidate." % (vertex, lenc))
        return candidate, find_node + 1 
    
    def genSupernodeFromVpool(self, vertices_pool, ret = False):
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
    
    def genSupernodeForAll(self):
        
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
        
        #self.PrintMinMaxDegreeinCand(vertices_pool, "vertices left", minmax=False)
        
        self.genSupernodeFromVpool(vertices_pool)
        
    
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
    
    # generate superedge decrementally
    def UpdateSuperedgeV0(self, newsv, allsv):
        
        # for those old superedges involves newsv, remove them (already been done in RemoveSupernode)
        print("generating new superedges amoung %d influenced and regenrated supernodes..." % len(newsv))
        old_se_len = len(self.superedge)
        
        # self.edge_threshold += self.delete_p/3

        # generate superedge within new supernodes
        self.genSuperedge(newsv, decay = 1) #newsv type: set
        new_se_len = len(self.superedge)
        print("%d superedges have been generated among regenerated supernodes." % (new_se_len-old_se_len))
        
        oldsv = set(allsv) - set(newsv)
        
        cnt_newse = 0
        
        print("%d old and %d new supernodes" % (len(oldsv), len(newsv)))
        
        c = 0
        flag = False
        edgelist = list(self.graph.G.edges)

        # generate superedge between new supernodes and old supernodes
        for svi in newsv:
            if(np.count_nonzero(self.supernode[svi].rep)==0):
                    c+=1
            for svj in oldsv:
                
                if(self.se_gen_method != "EDG"):
                    if(len(self.supernode[svi].vertices) == 1 and len(self.supernode[svj].vertices) == 1):
                        continue
                    elif(abs(len(self.supernode[svi].vertices) - len(self.supernode[svj].vertices)) > self.edge_threshold):
                        continue
                else:
                    if(len(self.supernode[svi].vertices) == 1 and len(self.supernode[svi].vertices) == 1):
                        d = self.graph.G.has_edge(svi, svj)
                    else:
                        d = self.GetDistanceBtwSvs(svi, svj, edgelist = edgelist, aggmethod=self.se_gen_method)
                
                if(self.se_gen_method == "EDG"):
                    if( d ):
                        self.superedge.append([svi, svj])
                        cnt_newse += 1
                elif( d > 0 and d < self.edge_threshold):
                    self.superedge.append([svi, svj])
                    cnt_newse += 1
                
                """
                if(len(self.supernode[svi].vertices) == 1 and len(self.supernode[svj].vertices) == 1):
                    continue
                elif(abs(len(self.supernode[svi].vertices) - len(self.supernode[svj].vertices)) > self.edge_threshold):
                    continue
                else:
                    d = self.GetDistanceBtwSvs(svi, svj, aggmethod=self.se_gen_method)
                
                if(self.se_gen_method == "EDG"):
                    if( d ):
                        self.superedge.append([svi, svj])
                elif( d > 0 and d < self.edge_threshold):
                    self.superedge.append([svi, svj])
                    cnt_newse += 1
                """

        print("%d superedges generated bwt old and new supernodes" % cnt_newse)
        return cnt_newse
    
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
        
        edge_early_stop = 0

        def MakeSeKeys(v1, v2, supernode_num):
            return str(v1) + '.' + str(v2)
            if(v1 < 0 or v2 < 0):
                print("v1, v2, supernode_num:", v1, v2,supernode_num)
                exit(1)
            return v1 * (10**math.ceil(math.log10(supernode_num))) + v2
        
        def ExtractSeKeys(key, supernode_num):
            keypart = key.split('.')
            return int(keypart[0]), int(keypart[1])
            v2 = key % 10**math.ceil(math.log10(supernode_num))
            v1 = (key - v2) / (10**math.ceil(math.log10(supernode_num)))
            if(v1 < 0 or v2 < 0):
                print("v1, v2, key, supernode_num:", v1, v2, key, supernode_num)
                exit(1)
            return int(v1), int(v2)

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
        
        """
        for i in tqdm(range(n)):

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
                        self.superedge.append([supernode_t[i], supernode_t[j]])
                    
                    if( d == 2):
                        edge_early_stop += 1
                elif( d > 0 and d < self.edge_threshold):
                    self.superedge.append([supernode_t[i], supernode_t[j]])
        """    
        
        t2 = time.time()
        
        print("Generate superEDGE timing: %.3f, edge_early_stop: %d " % (t2-t1, edge_early_stop))
    
    def genSuperedgeForAll(self):
        self.genSuperedge(supernode = self.supernode)
    
    # find the groups vertex di and vertex dj are in
    def DeleteOneEdge(self, di, dj, directed = False):
        
        influenced_set = set([])
        
        # A[di,dj] is in di-th row, ki-th partition(tree)
        ki = math.floor(di/(self.graph.partition_size))
        # A[dj,di] is in di-th row, ki-th partition(tree)
        if(directed == False):
            kj = math.floor(dj/(self.graph.partition_size))
        
        # locate place according to vertex
        di_place = self.GetPlaceByVertex(kj, di)
        
        if(directed == False):
            dj_place = self.GetPlaceByVertex(ki, dj)
        
        # using place and ki to build particular gid, and add it to influenced groups set.
        influenced_set.add(self.GetGroupid(treeno=kj, groupno=self.graph.trees[kj].getGroupno(di_place)))
        if(directed == False):
            influenced_set.add(self.GetGroupid(treeno=ki, groupno=self.graph.trees[ki].getGroupno(dj_place)))

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
    
    def ReviseSns(self, dismissed_vertices, influenced_sv_set = None):
        removed_sv = set()
        
        allkeys = list(self.supernode.keys())
        
        tmpkey = -1

        for sv in influenced_sv_set:
            dismissed_vertices -= self.supernode[sv].vertices
        
        # changed here
        for sv in allkeys:
            tmp_vertices = list(self.supernode[sv].vertices)
            for v in tmp_vertices:
                if(v not in self.vertex2supernode):
                    self.vertex2supernode[v] = sv
                elif(self.vertex2supernode[v] != sv):
                    self.supernode[sv].RemoveVfromSn(int(v), graph = self.graph, aggmethod = self.sn_agg_method)

        for sv in allkeys:
            
            # remove empty supernodes
            if(len(self.supernode[sv].vertices) == 0):
                # continue
                removed_sv.add(sv)
                
                del self.supernode[sv]
                
                if( sv in self.vertex2supernode and self.vertex2supernode[sv] == sv):
                    del self.vertex2supernode[sv]
            else:
                # adjust supernode keys
                if(sv not in self.supernode[sv].vertices):
                    
                    removed_sv.add(sv)
                    self.supernode[tmpkey] = self.supernode.pop(sv)
                    tmpkey -= 1
                    
                else:
                    if( sv in dismissed_vertices ):
                        print(sv, "in dismissed_vertices, vertices:" , self.supernode[sv].vertices)
                        exit(1)
        
        self.supernode_copy = self.supernode.copy()
        for sv in self.supernode:
            if(sv < 0):
                newkey = list(self.supernode[sv].vertices)[0]
                if(newkey < 0):
                    print("newkey, self.supernode[sv].vertices", newkey, self.supernode[sv].vertices)
                self.supernode_copy[newkey] = self.supernode_copy.pop(sv)
                for v in self.supernode_copy[newkey].vertices:
                    self.vertex2supernode[v] = newkey


        self.supernode = self.supernode_copy

        self.superedge = list(filter(lambda x:x[0] in self.supernode.keys() and x[1] in self.supernode.keys(),
                        self.superedge))

        if(influenced_sv_set):
            influenced_sv_set -= removed_sv
        
        print("%d empty supernodes have been removed" % len(removed_sv))

        return influenced_sv_set
        
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

        self.supernode[sv].EleCntRemove( _group )
        self.supernode[sv].stop_round -= 1

        assert self.supernode[sv].stop_round >= -1

        return

    def RemoveZerogroupV0(self, influenced_svs, treeno, groupno):
        assert groupno == 0

        vertices_pool = set(list(range(self.graph.vertices_num)))
        _group = ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno) 
        dismissed_vertices = set()

        for sv in influenced_svs:
            prev_vertices = self.supernode[sv].vertices.copy()
            self.supernode[sv].RemoveVfromSn( _group, self.graph, self.sn_agg_method )
            self.supernode[sv].stop_round -= 1

            assert self.supernode[sv].stop_round >= -1

            d_vertices = prev_vertices - self.supernode[sv].vertices
        
            for v in d_vertices:
                del self.vertex2supernode[v]

            dismissed_vertices |= d_vertices 
        
        return dismissed_vertices
    
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

    # Remove group influence from Supernodes
    def RemoveGroupInfFromSvsV0(self, influenced_groups, dismissed_vertices):
        
        for g in influenced_groups:
            
            influenced_svs = self.group2supernode[g]
            treeno, groupno = self.ParseTreenoGroupnoFromId(g)
            
            if(groupno == 0):
                dismissed_vertices |= self.RemoveZerogroupV0(influenced_svs, treeno, groupno)
            else:
                for sv in influenced_svs:
                    dismissed_vertices |= self.RemoveGroupFromOneSvV0(sv, treeno, groupno)

    def RemoveGroupInfFromSvsSubfunc1(self, g, ):
        return

    def RemoveGroupInfFromSvs(self, influenced_groups, dismissed_vertices):

        influenced_tree2sv = collections.defaultdict(set)
        influenced_sv_set = set()
        
        zero_group_sv = collections.defaultdict(set)

        t1 = time.time()
        for g in influenced_groups:
            influenced_svs = self.group2supernode[g]
            
            if(len(influenced_svs) == 0):
                continue
            
            treeno, groupno = self.ParseTreenoGroupnoFromId(g)

            influenced_tree2sv[treeno] |= influenced_svs
            # count incluenced supernode
            influenced_sv_set |= influenced_svs

            for sv in influenced_svs:
                dismissed_vertices |= self.supernode[sv].vertices

            """
            if(groupno == 0):
                zero_group_sv[treeno] |= influenced_svs
                continue
            """
            for sv in influenced_svs:
                self.RemoveGroupFromOneSv(sv, treeno, groupno)
        t2 = time.time()

        print("loop 1 in RemoveGroupInfFromSvs timing: %.3f" % (t2 - t1))
        
        return influenced_tree2sv, influenced_sv_set, dismissed_vertices

        def MakeZerogroupSvsMatrix(zero_group_svs):
            zero_group_svs_t = tuple(zero_group_svs)
            svs_matrix = np.zeros((len(zero_group_svs_t), self.graph.vertices_num))

            for i, sv in enumerate(zero_group_svs_t):
                svs_matrix[i] = self.supernode[sv].ele_cnt

            return zero_group_svs_t, svs_matrix

        def MinusCandidate2Matrix(zero_group_svs_matrix, zero_group_candidate):
            zero_candidate_matrix_onerow = np.zeros(self.graph.vertices_num)
            zero_candidate_matrix_onerow[list(zero_group_candidate)] = 1
            zero_group_candidate_matrix = np.tile(zero_candidate_matrix_onerow, len(zero_group_svs_matrix))
            zero_group_candidate_matrix = zero_group_candidate_matrix.reshape([len(zero_group_svs_matrix), self.graph.vertices_num])
            assert len(zero_group_candidate_matrix) < 2 or len((np.nonzero(zero_group_candidate_matrix[0] - zero_group_candidate_matrix[1]))[0]) == 0
                              
            zero_group_svs_matrix = zero_group_svs_matrix - zero_group_candidate_matrix
            zero_group_svs_matrix[zero_group_svs_matrix < 0] = 0
            return zero_group_svs_matrix
        
        def ExtractValFromMatrix(zero_group_svs_t, zero_group_svs_matrix):
            for i, sv in enumerate(zero_group_svs_t):
                self.supernode[sv].ele_cnt = zero_group_svs_matrix[i]
            return
        
        t1 = time.time()
        for treeno in zero_group_sv:

            vertices_pool = set(list(range(self.graph.vertices_num)))
            # _group = ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno) 

            zero_group_candidate =  ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno)  #self.graph.GetZeroGroupOnIthTree(treeno)
            zero_group_svs_t, zero_group_svs_matrix = MakeZerogroupSvsMatrix(zero_group_sv[treeno])
            zero_group_svs_matrix = MinusCandidate2Matrix(zero_group_svs_matrix, zero_group_candidate)
            ExtractValFromMatrix(zero_group_svs_t, zero_group_svs_matrix)
            
            for sv in zero_group_sv[treeno]:
                # self.supernode[sv].EleCntRemove( _group )
                self.supernode[sv].stop_round -= 1

                assert self.supernode[sv].stop_round >= -1
        
        t2 = time.time()

        print("loop 2 in RemoveGroupInfFromSvs timing: %.3f" % (t2 - t1))

        return influenced_tree2sv, influenced_sv_set, dismissed_vertices
        
    # move vertices in deleted_edges from one group to another if necessary
    def UpdateGroup(self, edgelist, deleted_edge_idx, directed = False, influenced_groups = None):
        
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
        
        for deleted_edge_id in deleted_edge_idx:
            
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
    
    # add candidates on treeno-th tree to supernode sv with an escape rate 0.8
    # remove the added candidate from dismissed_vertices and the number of vertices assigned
    def UpdateSvOnTree(self, sv, treeno, dismissed_vertices):
        
        # e = random.random()
        
        # if the supernode is empty, then we do not process it. / randomly skip.
        # if(len(self.supernode[sv].vertices) == 0):# or e > self.other_match_escape):
        #     return dismissed_vertices, 0
        
        # here the parameter sv is not None in FindCandidateByTree Function, 
        # thus the new sv would be added
        tmpcandidate, groupno = self.FindCandidateByTree(treeno, sv, sv = sv, vertices_pool = dismissed_vertices,ret_groupno=True)
        candidate = set()

        # 20230607 modified
        if(self.use_ele_cnt):
            self.supernode[sv].EleCntAdd(tmpcandidate)
            self.supernode[sv].stop_round += 1
        else:
            # the candidate has to be released before
            candidate = tmpcandidate & dismissed_vertices & self.supernode[sv].vertices

            # assign candidate to supernodes
            for v in candidate:
                self.vertex2supernode[v] = sv
    
        return dismissed_vertices - candidate, len(candidate)

    def UpdateGroupInfOnSvsByTreeV0(self, influenced_groups, dismissed_vertices, updateSvOnTree = True):
        
        # the whole vertices pool
        d_vertices = dismissed_vertices
        updatedsv_cnt = 0
        influenced_sv_set = set()
        influenced_tree2sv = {}

        for g in influenced_groups:
            
            # the changed groups have related to which supernodes
            influenced_svs = self.group2supernode[g]
            if(len(influenced_svs) == 0):
                continue
            
            treeno, groupno = self.ParseTreenoGroupnoFromId(g)

            # group supernodes  to trees
            if(treeno not in influenced_tree2sv):
                influenced_tree2sv[treeno] = influenced_svs
            else:
                influenced_tree2sv[treeno] |= influenced_svs
            
            # count incluenced supernode
            influenced_sv_set |= influenced_svs
        
        def UpdateSvInfoByGroup(sv, candidate, groupid):
            assert sv in candidate
            self.group2supernode[groupid].add(sv)
            self.supernode[sv].EleCntAdd(candidate)
            self.supernode[sv].stop_round += 1


        if(updateSvOnTree):
            vertices_pool = dismissed_vertices
            # the supernode on this tree should reconsider its candidate
            for treeno in influenced_tree2sv:
                zero_group_svs = set()
                for sv in influenced_tree2sv[treeno]:
                    place = self.GetPlaceByVertex(treeno, sv)
                    groupno = self.graph.trees[treeno].getGroupno(place)

                    if(groupno == 0):
                        zero_group_svs.add(sv)

                zero_group_candidate = ( vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno) 
                
                for sv in zero_group_svs:
                    UpdateSvInfoByGroup(sv, zero_group_candidate | set([sv]) , self.GetGroupid(treeno,groupno))

                for sv in influenced_tree2sv[treeno] - zero_group_svs:
                    candidate = vertices_pool & self.graph.trees[treeno].getGroupbyGroupno(groupno) | set([sv])
                    UpdateSvInfoByGroup(sv, candidate, self.GetGroupid(treeno,groupno))
        
        def UpdateSv(sv, candidate, d_vertices, prev_candidate):
            for v in candidate:
                self.vertex2supernode[v] = sv
                    
            d_vertices -= candidate
            d_vertices |= (set(prev_candidate) - candidate)
            return d_vertices
        
        for sv in influenced_sv_set:

            if(len(d_vertices) <= 1000):
                break
            prev_candidate = self.supernode[sv].vertices.copy()
            candidate = self.supernode[sv].UpdateCandidate(t = self.supernode[sv].stop_round, 
                                                           dismissed_vertices = d_vertices,
                                                           cand2set = True)
           
            if(self.supernode[sv].stop_round < self.graph.K-1):
                if(len(candidate) <= self.candidate_earlystop_thres):
                    # accept
                    # assign candidate to supernodes
                    d_vertices = UpdateSv(sv, candidate, d_vertices, prev_candidate)
                    updatedsv_cnt += 1
                    
                else:
                    # withdraw
                    self.supernode[sv].vertices = prev_candidate
            else:
                # accept
                d_vertices = UpdateSv(sv, candidate, d_vertices, prev_candidate)
                updatedsv_cnt += 1
                
        # remove old related superedges
        self.superedge = list(filter(lambda x:x[0] not in influenced_sv_set and x[1] not in influenced_sv_set,
                        self.superedge))
        
        print("supernodes have been updated %d times" % (updatedsv_cnt))
                
        return d_vertices, influenced_sv_set
    
    def UpdateGroupInfOnSvsByTree(self, influenced_groups, influenced_tree2sv, influenced_sv_set, dismissed_vertices, updateSvOnTree = True):
        
        # the whole vertices pool
        d_vertices = dismissed_vertices
        updatedsv_cnt = 0

        def UpdateSvInfoByGroup(sv, candidate, groupid, non_zero = False):
            if(non_zero):
                assert sv in candidate
            self.group2supernode[groupid].add(sv)
            if(non_zero):
                self.supernode[sv].EleCntAdd(candidate)
            self.supernode[sv].stop_round += 1

        def MakeZerogroupSvsMatrix(zero_group_svs):
            zero_group_svs_t = tuple(zero_group_svs)
            svs_matrix = np.zeros((len(zero_group_svs_t), self.graph.vertices_num))

            for i, sv in enumerate(zero_group_svs_t):
                svs_matrix[i] = self.supernode[sv].ele_cnt

            return zero_group_svs_t, svs_matrix

        def AddCandidate2Matrix(zero_group_svs_matrix, zero_group_candidate):
            zero_candidate_matrix_onerow = np.zeros(self.graph.vertices_num)
            zero_candidate_matrix_onerow[list(zero_group_candidate)] = 1
            zero_group_candidate_matrix = np.tile(zero_candidate_matrix_onerow, len(zero_group_svs_matrix))
            zero_group_candidate_matrix = zero_group_candidate_matrix.reshape([len(zero_group_svs_matrix), self.graph.vertices_num])
            if(len(zero_group_candidate_matrix) >= 2 and len(np.nonzero(zero_group_candidate_matrix[0] - zero_group_candidate_matrix[1])[0]) != 0):
                print(len(zero_group_candidate_matrix))
                print(np.nonzero(zero_group_candidate_matrix[0] - zero_group_candidate_matrix[1]))
                print(np.nonzero(zero_group_candidate_matrix[0] - zero_group_candidate_matrix[1])[0])
                exit(1)

            zero_group_svs_matrix = np.add( zero_group_candidate_matrix, zero_group_svs_matrix )
            return zero_group_svs_matrix
        
        def ExtractValFromMatrix(zero_group_svs_t, zero_group_svs_matrix):

            #print("ExtractValFromMatrix")
            for i, sv in enumerate(zero_group_svs_t):
                self.supernode[sv].ele_cnt = zero_group_svs_matrix[i]
            return

        t1 = time.time()

        if(updateSvOnTree):
            vertices_pool = d_vertices
            # the supernode on this tree should reconsider its candidate
            for treeno in influenced_tree2sv:
                
                for sv in influenced_tree2sv[treeno]:
                    place = self.GetPlaceByVertex(treeno, sv)
                    groupno = self.graph.trees[treeno].getGroupno(place)

                    # if(groupno == 0):
                    #    zero_group_svs.add(sv)

                    candidate = vertices_pool & self.graph.trees[treeno].getGroupbyGroupno(groupno) | set([sv])
                    UpdateSvInfoByGroup(sv, candidate, self.GetGroupid(treeno,groupno), non_zero = True)

                """
                zero_group_candidate =  (vertices_pool - self.graph.trees[treeno].nonzeros) | self.graph.trees[treeno].getGroupbyGroupno(groupno) #self.graph.GetZeroGroupOnIthTree(treeno)
                zero_group_svs_t, zero_group_svs_matrix = MakeZerogroupSvsMatrix(zero_group_svs)
                zero_group_svs_matrix = AddCandidate2Matrix(zero_group_svs_matrix, zero_group_candidate)
                ExtractValFromMatrix(zero_group_svs_t, zero_group_svs_matrix)

                for sv in zero_group_svs:
                    UpdateSvInfoByGroup(sv, None , self.GetGroupid(treeno,groupno), non_zero = False)

                for sv in influenced_tree2sv[treeno] - zero_group_svs:
                    candidate = vertices_pool & self.graph.trees[treeno].getGroupbyGroupno(groupno) | set([sv])
                    UpdateSvInfoByGroup(sv, candidate, self.GetGroupid(treeno,groupno), non_zero = True)
                """
        t2 = time.time()

        print("loop 1 in UpdateGroupInfOnSvsByTree timing: %.3f" % (t2 - t1))

        def UpdateSv(sv, candidate, d_vertices, prev_candidate):
            assert len(candidate & d_vertices) == len(candidate)
            
            """
            for v in candidate_list:
                # if this node has been assigned to another supernode, then remove it from its original supernode
                if(v in self.vertex2supernode):
                    self.supernode[self.vertex2supernode[v]].RemoveVfromSn(int(v), graph = self.graph, aggmethod = self.sn_agg_method)
                # otherwise, this supernode is in its original
                else:
                    assert v in d_vertices
            """
            for v in candidate:
                self.vertex2supernode[v] = sv
            
            for v in prev_candidate - candidate:
                del self.vertex2supernode[v]
            
            d_vertices -= candidate
            d_vertices |= (set(prev_candidate) - candidate)
            
            assert len((set(prev_candidate) - candidate) & d_vertices) == len((set(prev_candidate) - candidate))
            
            return d_vertices
        
        t1 = time.time()

        for i, sv in enumerate(influenced_sv_set):

            if(len(d_vertices) <= 1000):
                break
            prev_candidate = self.supernode[sv].vertices.copy()


            candidate = self.supernode[sv].UpdateCandidate(t = self.supernode[sv].stop_round, 
                                                           dismissed_vertices = d_vertices,
                                                           cand2set = True)

            if(self.supernode[sv].stop_round < self.graph.K-1):
                if(len(candidate) <= self.candidate_earlystop_thres):
                    # accept
                    # assign candidate to supernodes
                    d_vertices = UpdateSv(sv, candidate, d_vertices, prev_candidate)
                    updatedsv_cnt += 1
                else:
                    # withdraw
                    self.supernode[sv].vertices = prev_candidate
            else:
                # accept
                d_vertices = UpdateSv(sv, candidate, d_vertices, prev_candidate)
                updatedsv_cnt += 1
        
        t2 = time.time()

        print("loop 2 in UpdateGroupInfOnSvsByTree timing: %.3f" % (t2 - t1))

        # remove old related superedges
        self.superedge = list(filter(lambda x:x[0] not in influenced_sv_set and x[1] not in influenced_sv_set,
                        self.superedge))
        
        print("supernodes have been updated %d times" % (updatedsv_cnt))
                
        return d_vertices, influenced_sv_set
    
    def UpdateGroupInfOnSvs(self, influenced_groups, dismissed_vertices, updateSvOnTree = True):
        
        # the whole vertices pool
        d_vertices = dismissed_vertices
        updatedsv_cnt = 0
        influenced_sv_set = set()

        for g in influenced_groups:
            
            # the changed groups have related to which supernodes
            influenced_svs = self.group2supernode[g]
            
            if(len(influenced_svs) == 0):
                continue
            
            # locate the group (in which tree and which part)
            treeno, groupno = self.ParseTreenoGroupnoFromId(g)
            
            # count incluenced supernode
            influenced_sv_set |= influenced_svs
            
            # the supernode on this tree should reconsider its candidate
            if(updateSvOnTree):
                for sv in influenced_svs:
                    
                    d_vertices, extra_candidate = self.UpdateSvOnTree(sv, treeno, d_vertices)
        

        for sv in influenced_sv_set:

            if(len(d_vertices) <= 1000):
                break
            prev_candidate = self.supernode[sv].vertices.copy()
            candidate = self.supernode[sv].UpdateCandidate(t = self.supernode[sv].stop_round, 
                                                           dismissed_vertices = d_vertices,
                                                           cand2set = True)
           
            if(self.supernode[sv].stop_round < self.graph.K-1):
                if(len(candidate) <= self.candidate_earlystop_thres):
                    # accept
                    # assign candidate to supernodes
                    for v in candidate:
                        self.vertex2supernode[v] = sv

                    updatedsv_cnt += 1
                    d_vertices -= candidate
                    d_vertices |= (set(prev_candidate) - candidate) # add those redismissed candidate
                else:
                    # withdraw
                    self.supernode[sv].vertices = prev_candidate
            else:
                # accept
                
                for v in candidate:
                    self.vertex2supernode[v] = sv

                updatedsv_cnt += 1
                d_vertices -= candidate
                d_vertices |= (set(prev_candidate) - candidate) # add those redismissed candidate

        # remove old related superedges
        self.superedge = list(filter(lambda x:x[0] not in influenced_sv_set and x[1] not in influenced_sv_set,
                        self.superedge))
        
        print("supernodes have been updated %d times" % (updatedsv_cnt))
                
        return d_vertices, influenced_sv_set
        
    def EdgeDelete(self, deleted_edge_idx, removed_vertices = []):
        
        t1 = time.time()
        
        influenced_groups = set()
        edgelist= list(self.graph.G.edges)

        # dask
        # find groups which deleted_edges are in
        if(len(removed_vertices)):
            assert self.graph.delete_vertex
            
            # the edges are already stored in deleted_edge_idx
            for deleted_edge in deleted_edge_idx:
                
                di,dj = deleted_edge[0], deleted_edge[1]
                
                influenced_groups |= self.DeleteOneEdge(di, dj, directed = utils.dataset_info[self.graph.dataset_name]["directed"])
        else:
            assert not self.graph.delete_vertex
            for deleted_edge_id in deleted_edge_idx:
                
                deleted_edge = edgelist[deleted_edge_id]
                
                di,dj = deleted_edge[0], deleted_edge[1]
            
                influenced_groups |= self.DeleteOneEdge(di, dj, directed = utils.dataset_info[self.graph.dataset_name]["directed"])

        t2 = time.time()
        print("find influenced_groups timing: %.3f, %d influenced_groups" % (t2-t1, len(influenced_groups)))
        dismissed_vertices = set()
        
        # remove the influenced group from the supernode that uses this group of vertice.
        # We first calculate influenced groups to avoid remove repeatedly
        # dask
        
        influenced_tree2sv, influenced_sv_set, dismissed_vertices = self.RemoveGroupInfFromSvs(influenced_groups, dismissed_vertices)
        t3 = time.time()
        print("RemoveGroupInfFromSvs timing: %.3f" % (t3-t2))

        # dask
        if(self.use_ele_cnt == False):
            self.ReviseSns(dismissed_vertices)
        
        # Update Group state
        influenced_groups = self.UpdateGroup(edgelist, deleted_edge_idx, directed = utils.dataset_info[self.graph.dataset_name]["directed"], influenced_groups = influenced_groups)
        t4 = time.time()
        print("UpdateGroup timing: %.3f" % (t4-t3))
        
        if(self.graph.load_zero):
            for i in range(self.graph.K):
                self.graph.UpdateZeroFile(i)
        
        # some vertices in the influenced groups has been changed to other groups
        # the dismissed_vertices can be assigned to current supernodes
        if(self.use_ele_cnt):
            dismissed_vertices, influenced_sv_set = self.UpdateGroupInfOnSvsByTree(influenced_groups, influenced_tree2sv, influenced_sv_set, dismissed_vertices)
            
            t5 = time.time()
            print("UpdateGroupInfOnSvs timing: %.3f, len(dismissed_vertices): %d" % (t5-t4, len(dismissed_vertices)))
            
            influenced_sv_set = self.ReviseSns(dismissed_vertices, influenced_sv_set)
            
            t6 = time.time()
            print("ReviseSns timing: %.3f" % (t6-t5))
            
        else:
            influenced_sv_set = set([])
        newsv =  self.genSupernodeFromVpool(dismissed_vertices, ret = True)
        t7 = time.time()
        print("genSupernodeFromVpool timing: %.3f, %d old supernodes influenced." % (t7-t6, len(influenced_sv_set)))
        
        # shuffle dismissed_vertices
        vertices_pool = list(newsv) + list(influenced_sv_set)
        random.shuffle(vertices_pool)
        vertices_pool = set(vertices_pool)
        
        # Update superedge
        self.UpdateSuperedge(vertices_pool, self.supernode)
        
        t8 = time.time()
        print("UpdateSuperedge timing: %.3f"  % (t8-t7))
        
        if(len(removed_vertices)):
            print("Process NODE deletion timing: %.3f" % (t8-t1))
        else:
            print("Process EDGE deletion timing: %.3f" % (t8-t1))
        
        # print("Having removed %d superNODEs and added %d superNODEs" % (cnt_deletedsv, len(newsv)))
        # print("Having removed %d superEDGEs and added %d superEDGEs" % (cnt_deletedse, cnt_addedse))

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
        