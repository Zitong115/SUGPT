# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:18:00 2023

@author: ztli
"""

import collections
from itertools import combinations
import networkx as nx
import numpy as np
import pandas as pd
import random
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
from tqdm import tqdm
import utils

MOSSO_OUTPUT_FILE_TYPE = "txt"
MY_OUTPUT_FILE_TYPE = "csv"
JARCCARD_NEIGHBOR = "jarccard_neighbor"
COMMUNITY = "community"
PATH_CONNECTIVITY = "path_connectivity"
CORE_NUMBER = "core_number"

class Evaluator(object):
    
    def __init__(self, supergraph, supernode, superedge, evaluated_file, 
                 evaluated_file_type, original_graph, dataset, evaluation_type):
        
        assert evaluated_file_type in [MOSSO_OUTPUT_FILE_TYPE, MY_OUTPUT_FILE_TYPE]
        
        if(supergraph):
            self.supergraph = supergraph
            self.supernode = supergraph.supernode
            self.superedge = supergraph.superedge
            self.vertex2supernode = supergraph.vertex2supernode
        else:
            self.supergraph = None
            self.supernode = collections.defaultdict(set)
            self.superedge = []
            self.vertex2supernode = {}
        
        self.evaluated_file = evaluated_file
        self.evaluated_file_type = evaluated_file_type
        self.original_graph = original_graph
        self.evaluation_type = evaluation_type
        self.dataset = dataset

    def EvaluateMosso(self):
        return

    def ReadOutputFile(self):
        result_df = None
        
        if(self.evaluated_file_type == MOSSO_OUTPUT_FILE_TYPE):
            # this is targeted at reading output file by Mosso
            result_df = pd.read_csv(self.evaluated_file, delimiter="\t", 
                                    header = None)
            result_df.columns = ["Vertex","Supernode"]
        elif(self.evaluated_file_type == MY_OUTPUT_FILE_TYPE):
            if(len(self.supernode)):
                return result_df
            result_df = pd.read_csv(self.evaluated_file)
            if(result_df.columns[0] == "Supernode" and result_df.columns[1] == "Vertex"):
                print("Swap our csv output columns.")
                result_df = result_df[["Vertex","Supernode"]]
        else:
            print("Invalid evaluated_file_type for evaluator")
        
        return result_df
    
    def ReadSupernodeFromDf(self, result_df):
        
        mosso_existed_nodes = set()
        baseline_nodes = set(self.original_graph.nodes)

        if(self.evaluated_file_type == MOSSO_OUTPUT_FILE_TYPE):
            V, P = result_df.iloc[0, 0], result_df.iloc[0, 1]
            cnt = 1
            while(1):
                vertex, sn = result_df.iloc[cnt, 0], result_df.iloc[cnt, 1]
                if(vertex not in baseline_nodes):
                    print("vertex %d not in baseline_nodes, cnt: %d" % (vertex, cnt))
                mosso_existed_nodes.add(vertex)
                if(cnt > V):
                    cnt += 1
                    break
                                
                if(sn == -1):
                    self.supernode[vertex].add(vertex)
                    self.vertex2supernode[vertex] = vertex
                else:
                    self.supernode[sn].add(vertex)
                    self.vertex2supernode[vertex] = sn
                    
                cnt += 1
            
            while(1):
                edge_st, edge_ed = result_df.iloc[cnt, 0], result_df.iloc[cnt, 1]
                if(edge_ed == -1 and edge_st == -1):
                    cnt += 1
                    break
                self.superedge.append([edge_st, edge_ed])
                cnt += 1
            
            print("%d nodes are not assigned to supernodes in mosso." % len(baseline_nodes - mosso_existed_nodes))

            left_nodes = baseline_nodes - mosso_existed_nodes
            for v in left_nodes:
                self.supernode[v] = {v}

            print("vertex compression ratio: %.3f" % (self.original_graph.number_of_nodes() / len(self.supernode)))
            print("edge compression ratio: %.3f" % (self.original_graph.number_of_edges() / len(self.superedge)))

        elif(self.evaluated_file_type == MY_OUTPUT_FILE_TYPE):
            if(len(self.supernode)==0):
                for cnt in range(len(result_df)):
                    vertex, sn = result_df.iloc[cnt, 0], result_df.iloc[cnt, 1]
                    self.supernode[sn].vertices.add(vertex)
            else:
                return
        return
    
    def NeighborSimilarity(self):
        
        sum_jarccard = 0.0
        
        missing_vertices_cnt = 0
        valid_cnt = 0
        
        for sn in self.supernode.keys():
            if(self.evaluated_file_type == MOSSO_OUTPUT_FILE_TYPE):
                vertices = self.supernode[sn]
            elif(self.evaluated_file_type == MY_OUTPUT_FILE_TYPE):
                vertices = self.supernode[sn].vertices
            neighbor_intersection = set()
            neighbor_union = set()
            
            for i,vertex in enumerate(vertices):
                
                try:
                    cur_neighbors = set(self.original_graph.neighbors(vertex))
                except:
                    missing_vertices_cnt += 1
                    
                if(i == 0):
                    neighbor_intersection |= cur_neighbors
                    neighbor_union |= cur_neighbors
                else:
                    neighbor_intersection &= cur_neighbors
                    neighbor_union |= cur_neighbors
                
            if(len(neighbor_union)==0):
                continue
            else:
                sum_jarccard += len(neighbor_intersection) / len(neighbor_union)
                valid_cnt += 1
        
        if(missing_vertices_cnt):
            print("%d missing vertices" % missing_vertices_cnt)
        return sum_jarccard / valid_cnt
    
    def SampleGraph(self, G, sample = 0.25, supergraph = False, sampled_nodes = None):
        
        random.seed(0)

        if( supergraph ):
            assert not sampled_nodes

            if(sample > 0):
                k = int( len(self.supernode) * sample )
            else:
                k = 15000 # yt:10000
            sampled_supernodes = random.sample(self.supernode.keys(), k)
            self.supernode = {key: self.supernode[key] for key in sampled_supernodes}
            self.superedge = list(filter(lambda x:x[0] in sampled_supernodes and x[1]  in sampled_supernodes, self.superedge))

            sampled_nodes = []

            if(self.evaluated_file_type == MOSSO_OUTPUT_FILE_TYPE):
                for i, sampled_supernode in enumerate(sampled_supernodes):
                    sampled_nodes.extend(self.supernode[sampled_supernode])
            elif(self.evaluated_file_type == MY_OUTPUT_FILE_TYPE):
                for i, sampled_supernode in enumerate(sampled_supernodes):
                    sampled_nodes.extend(list(self.supernode[sampled_supernode].vertices))
            
            return sampled_nodes
        
        else:
            assert len(sampled_nodes)
            sampled_graph = G.subgraph(sampled_nodes)
            return sampled_graph

    def CoreNumber(self):

        t1 = time.time()
        summary = self.BuildExtendedGraph()
        summary.remove_edges_from(nx.selfloop_edges(summary))

        original_core_number = nx.core_number(self.original_graph)
        summary_core_number = nx.core_number(summary)

        dist = []

        for v in original_core_number:
            if(v in summary_core_number):
                dist.append(abs(original_core_number[v] - summary_core_number[v]))
        t2 = time.time()

        print("core number evaluation timing: %.3f" % (t2 - t1))
        return np.average(dist)

    def BuildExtendedGraph(self):
        g = nx.Graph()
        p = 0.6

        for sv in self.supernode:
            if(self.evaluated_file_type == MOSSO_OUTPUT_FILE_TYPE):
                tmpgraph = nx.complete_graph(list(self.supernode[sv]))
            elif(self.evaluated_file_type == MY_OUTPUT_FILE_TYPE):
                tmpgraph = nx.complete_graph(list(self.supernode[sv].vertices))
            g = nx.compose(g, tmpgraph)

        np.random.seed(1)

        if(self.evaluated_file_type == MY_OUTPUT_FILE_TYPE):
            for se in self.superedge:
                svi, svj = se[0], se[1]
                for vi in self.supernode[svi].vertices:
                    for vj in self.supernode[svj].vertices:
                        r = random.random()
                        if(r < p * p):
                            g.add_edge(vi,vj)
        elif(self.evaluated_file_type == MOSSO_OUTPUT_FILE_TYPE):
            for se in self.superedge:
                svi, svj = se[0], se[1]
                for vi in self.supernode[svi]:
                    for vj in self.supernode[svj]:
                        r = random.random()
                        if(r < p * p):
                            g.add_edge(vi,vj)
        return g

    def BuildSupergraph(self):
        
        sg = nx.Graph()
        
        sg.add_nodes_from(list(self.supernode.keys()))
        
        sg.add_edges_from(self.superedge)
        
        return sg
    
    def ExtendSgCommunity(self, sg_community, vertex2supernode = None):
        
        sg_extended_community = []
        if(self.supergraph):
            missing_nodes = set(range(self.supergraph.graph.vertices_num))
        else:
            missing_nodes = set(range(self.original_graph.number_of_nodes()))

        if(vertex2supernode):
            tmpdict = collections.defaultdict(list)
            for v in vertex2supernode:
                tmpdict[vertex2supernode[v]].append(v)

            ret = [tmpdict[sv] for sv in tmpdict]
            return ret

        cnt = 0
        
        for c in sg_community:
            
            tmp_community = []
            
            for sn in c:
                
                if(self.evaluated_file_type == MOSSO_OUTPUT_FILE_TYPE):
                    tmp_community.extend(list(self.supernode[sn]))
                else:
                    tmp_community.extend(list(self.supernode[sn].vertices))
                    cnt += len(list(self.supernode[sn].vertices))
                    missing_nodes -= self.supernode[sn].vertices

            sg_extended_community.append(tmp_community)

        return sg_extended_community
    
    def ChecCommunityConsistance(self, original_c, supergraph_c):
        
        def GetNodesInCommunity(communities):
            ret = set()
            tmp = 0
            for i,c in enumerate(communities):
                if(len(set(c) & ret)):
                    print("ERROR: community crossed!")
                    print("i:",i, "sec(c):",c,",set(c) & ret:",set(c) & ret)
                    exit(1)
                
                ret |= set(c)
            return ret
        
        original_nodes = GetNodesInCommunity(original_c)
        print("begin GetNodesInCommunity(supergraph_c)")
        supergraph_nodes = GetNodesInCommunity(supergraph_c)
        
        print("168 len(original_nodes): %d, len(supergraph_nodes): %d" %(len(original_nodes), len(supergraph_nodes)))
        #print("len(original_nodes - supergraph_nodes):", len(original_nodes - supergraph_nodes))
        #print("supergraph_nodes - original_nodes:", supergraph_nodes - original_nodes)
    
    def CommunityDetection(self, vertex2supernode = None):
        
        # ours
        """
        if(self.supergraph):
            original_graph_community = utils.CommunityDetection(self.supergraph.graph.G, "Original")
        # mosso
        else:
            original_graph_community = utils.CommunityDetection(self.original_graph, "Original")
        """
        original_graph_community = utils.CommunityDetection(self.original_graph, "Original")

        # print("original graph has %d nodes after deletion" % self.original_graph.number_of_nodes())

        supergraph = self.BuildSupergraph()
        
        supergraph_community = utils.CommunityDetection(supergraph, "Super")
        supergraph_community_extended = self.ExtendSgCommunity(supergraph_community, vertex2supernode = vertex2supernode)

        # assign nodes to clusters
        """
        if(self.supergraph):
            label_true = np.zeros(self.supergraph.graph.vertices_num) # original graph
            label_pred = np.zeros(self.supergraph.graph.vertices_num) # supergraph
        else:
            label_true = np.zeros(self.original_graph.number_of_nodes()) # original graph
            label_pred = np.zeros(self.original_graph.number_of_nodes()) # supergraph
        """
        
        label_true = np.zeros(self.original_graph.number_of_nodes()) # original graph
        label_pred = np.zeros(self.original_graph.number_of_nodes()) # supergraph
        
        def MakeLabelFromCommunity(community, label, vertex2supernode = None, num2node_map = None):

            if(vertex2supernode):
                for i in vertex2supernode:
                    label[i] = vertex2supernode[i]
                return label

            for i, c in enumerate(community):
                
                for node in c:
                    
                    if(num2node_map):
                        label[num2node_map[node]] = i
                    else:
                        label[node] = i
            
            return label
        
        print("%d communities in original_graph, %d in supergraph" % (len(original_graph_community), len(supergraph_community_extended)))

        """
        if(self.supergraph):
            modularity_original = nx.community.modularity(self.supergraph.graph.G, original_graph_community)
        else:
            modularity_original = nx.community.modularity(self.original_graph, original_graph_community)
        """
        modularity_original = nx.community.modularity(self.original_graph, original_graph_community)

        try:
            """
            if(self.supergraph):
                modularity_supergraph = nx.community.modularity(self.supergraph.graph.G, supergraph_community_extended)
            """
            modularity_supergraph = nx.community.modularity(self.original_graph, supergraph_community_extended)
        except:
            self.ChecCommunityConsistance(original_graph_community, supergraph_community_extended)
            utils.exit(1)

        print("modularity of original graph: %.3f, of supergraph: %.3f" % (modularity_original, modularity_supergraph))
        
        if(self.supergraph):
            label_true = MakeLabelFromCommunity(original_graph_community, label_true)
            label_pred = MakeLabelFromCommunity(supergraph_community_extended, label_pred)
        else:
            nodemap = {}
            for i,v in enumerate(self.original_graph.nodes):
                nodemap[v] = i

            label_true = MakeLabelFromCommunity(original_graph_community, label_true, num2node_map = nodemap)
            label_pred = MakeLabelFromCommunity(supergraph_community_extended, label_pred, num2node_map = nodemap)

        return normalized_mutual_info_score(label_true, label_pred)
    

    def genQueryPairs(self, sample_frac = 0.5):
        node_range = list(self.original_graph.nodes())
        random.seed(1)

        if( self.dataset == "gd" or self.dataset == "db"): # or self.dataset == "db"
            node_range_idx = random.sample(range(len(node_range)), int(0.1 * len(node_range)))
        elif(self.dataset == "yt"):
            node_range_idx = random.sample(range(len(node_range)), int(0.025 * len(node_range)))
        else:
            node_range_idx = range(len(node_range))

        if(sample_frac == 1):
            return [c for c in combinations(node_range, 2)]
        
        all_pairs_idx = [c for c in combinations(node_range_idx, 2)]

        # fixed seeds
        np.random.seed(0)
        
        ret_pairs_idx = np.random.choice(range(len(all_pairs_idx)), int(sample_frac * len(node_range)), replace = False)
        ret_pairs = []

        for pair_idx in ret_pairs_idx:
            ret_pairs.append([node_range[all_pairs_idx[pair_idx][0]], node_range[all_pairs_idx[pair_idx][1]]])
        return ret_pairs

    def PathQueryInSg(self, v1, v2, newsg):

        if(v1 not in self.vertex2supernode):
            print(v1, "not found in self.vertex2supernode (evaluation)")
            self.vertex2supernode[v1] = v1

        if(v2 not in self.vertex2supernode):
            print(v2, "not found in self.vertex2supernode (evaluation)")
            self.vertex2supernode[v2] = v2

        if(self.vertex2supernode[v1] == self.vertex2supernode[v2]):
            return True
        else:
            if(nx.has_path(newsg, self.vertex2supernode[v1], self.vertex2supernode[v2])):
                return True
            else:
                return False
    
    def PathQueryInOriginalGraph(self, v1, v2):
        if(nx.has_path(self.original_graph, v1, v2)):
            return True
        else:
            return False
    
    """
    For supernodes in supergraph, query whether there is a path among nodes v1 and v2.
    If v1 and v2 are in the same supernode, return True.
    Otherwise, find the supernode sv1 that v1 belongs to and sv2 with the same meaning.
    If there is a path between sv1 and sv2, return True, otherwise, return False. 
    """
    def Querypath(self):

        t1 = time.time()
        querypairs = self.genQueryPairs()
        t2 = time.time()
        print("Generate query pairs timing: %.3f, len(querypairs): %d" % (t2 - t1, len(querypairs)))

        total = len(querypairs)
        hit_count = 0
        newsg = self.BuildSupergraph()

        t1 = time.time()
        for v1,v2 in querypairs:
            connected_in_supergraph = self.PathQueryInSg(v1, v2, newsg)
            connected_in_original_graph = self.PathQueryInOriginalGraph(v1,v2)
            if(connected_in_supergraph == connected_in_original_graph):
                hit_count += 1
        t2 = time.time()
        print("Query all pairs timing: %.3f" % (t2 - t1))

        return hit_count / total

    def Evaluate(self, vertex2supernode = None):
        result_df = self.ReadOutputFile()
        self.ReadSupernodeFromDf(result_df)
        
        if(JARCCARD_NEIGHBOR in self.evaluation_type):
            print("Avg Jaccard similarity of neighbors: ", self.NeighborSimilarity())
        
        if(COMMUNITY in self.evaluation_type):
            print("NMI of community detection comparison: ", self.CommunityDetection(vertex2supernode = vertex2supernode))

        if(PATH_CONNECTIVITY in self.evaluation_type):
            print("Path query hit ratio: ", self.Querypath())
        
        if(self.dataset == "db" or self.dataset == "yt" or self.dataset == "gd"):
            print("len(self.supernode): %d, len(self.superedge): %d, begin sampling nodes..." % (len(self.supernode), len(self.superedge)))
            sampled_nodes = self.SampleGraph(G = None, supergraph = True, sample = -1, sampled_nodes = None)
            self.original_graph = self.SampleGraph(G = self.original_graph, supergraph = False, sample = -1, sampled_nodes = sampled_nodes)
            print("sampling done. len(self.supernode): %d , len(self.superedge): %d, len(sampled_nodes): %d, self.original_graph.number_of_nodes(): %d." % (len(self.supernode), len(self.superedge), len(sampled_nodes), self.original_graph.number_of_nodes()))
        
        if(CORE_NUMBER in self.evaluation_type):
            print("Core number avg difference: ", self.CoreNumber())