# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:18:18 2023

@author: ztli
"""

import os
import collections
import math
import networkx as nx
import numpy as np
import pandas as pd
import random
from sys import exit
import shutil
from time import time
import yaml
#from supergraph_0103 import Supergraph, Supernode

SEED = 1
random.seed(SEED)

dataset_info = {"gd":{"subgraph_num":3, "labeled_node":True, "subgraph2label":True,"featured_node":True, "directed": False},
                "gf":{"subgraph_num":8, "labeled_node":True, "subgraph2label":True,"featured_node":False, "directed": False},
                "mt":{"subgraph_num":6, "labeled_node":True, "subgraph2label":True,"featured_node":True, "directed": False},
                "ef":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mf":{"subgraph_num":1, "labeled_node":True, "subgraph2label":False,"featured_node":True, "directed": False},
                "mo-T0":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T1":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T2":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T3":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T4":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T5":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T6":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T7":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T8":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "mo-T9":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "db":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "yt":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "lj":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "sk":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "ea":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": True},
                "ss":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": True},
                "se":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": True},
                "cc":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "ch":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False},
                "pg":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": True},
                "test":{"subgraph_num":1, "labeled_node":False, "subgraph2label":False,"featured_node":False, "directed": False}}

dataset_dfcolumns = ["FromNodeId", "ToNodeId"]

def ReadYmlConfig(filename):
    
    if(not os.path.exists(filename)):
        print("Can not find config file!")
        exit(1)
        
    with open(filename, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
        return config

def WriteDf(f, df, app):
    i = 0
    n = len(df)

    while i < n:
        f.write("%d\t%d\t%s\n" % (df.iloc[i,0], df.iloc[i,1], app))
        i=i+1
    
    return i+1

def WriteData(filename, df, deleted_df, retrain=False):
    
    if os.path.exists(filename):
        # Delete Folder code
        os.remove(filename)
    
    f = open(filename,"a")
    
    line1 = WriteDf(f, df, app="1")
    line2 = 0
    
    if(retrain == False):
        line2 = WriteDf(f, deleted_df, app="-1")
    else:
        assert not deleted_df
        
    f.close()

    return line1 + line2

def GenerateMossoInput(df, G, deleted_edges_idx, filename, retrain = False ):

    edgelist = list(G.edges)

    def MakeDfWritten2MossoInput(edgelist, idx_range):

        df = pd.DataFrame()
        df_fromnodeid = []
        df_tonodeid = []

        for idx in idx_range:
            df_fromnodeid.append(edgelist[idx][0])
            df_tonodeid.append(edgelist[idx][1])
        
        df["FromNodeId"] = df_fromnodeid
        df["ToNodeId"] = df_tonodeid

        return df
    
    df = MakeDfWritten2MossoInput(edgelist, range(G.number_of_edges()))
    
    if(retrain == False):
        deleted_df = MakeDfWritten2MossoInput(edgelist, deleted_edges_idx)
    else:
        deleted_df = None

    nline = WriteData(filename, df, deleted_df, retrain = retrain)
    print("having write %d rows to outputfile %s" % (nline, filename))

def RemoveDiEdges(df):
    new_cols = list(df.columns)
    new_cols.reverse()

    removed_df = df[new_cols].copy()
    removed_df.columns = df.columns
    
    return pd.merge(removed_df, df, how = 'inner')

"""
read from .txt format file
"""
def ReadGraphfile2Dataframe(dataset = "gd", subgraph = [1,2,3], filename = "", dataset_folder = "",
                            demo = True, sample = 0.01, random_seed = SEED):
    
    assert subgraph == -1 or len(subgraph) <= dataset_info[dataset]["subgraph_num"]
    assert dataset in dataset_info.keys()
    
    datafolder = os.path.join(dataset_folder, dataset[:2])
    
    df = pd.DataFrame(columns=dataset_dfcolumns)
    
    def MakeSubgraphFilename(number):
        return "subgraph_" + str(number) + ".csv"
    
    def MakeNormalGraphFilename(dataset):
        if(dataset == "mf"):
            return dataset + ".csv"
        # elif(dataset == "mo"):
        #    return dataset + "-TO.txt"
        else:
            return dataset + ".txt"
        
    def AddOffset2Subgraph(df, offset):
        return df.add(offset)
    
    if(subgraph == -1):
        subgraph_range = list(range(1, dataset_info[dataset]["subgraph_num"]+1))
    else:
        subgraph_range = subgraph

    if(dataset_info[dataset]["subgraph_num"] > 1):
        
        offset = 0
        
        for i in subgraph_range:
            subgraph_filename = os.path.join(datafolder, MakeSubgraphFilename(i))
            tmpdf = pd.read_csv(subgraph_filename, header = 0)
            tmpdf.columns = dataset_dfcolumns
            tmpdf = AddOffset2Subgraph(tmpdf, offset)
            offset += len(tmpdf)
            
            df = pd.concat([df, tmpdf], ignore_index = True)

    elif(dataset_info[dataset]["subgraph_num"] == 1):
        
        filename = os.path.join( datafolder, MakeNormalGraphFilename(dataset))
        
        if(dataset == "mf"):
            df = pd.read_csv(filename, header = 0)
        elif("mo" in dataset):
            df = pd.read_csv(filename, delimiter='\t')
        else:
            df = pd.read_csv(filename, skiprows=3, delimiter='\t')
    
        df.columns = dataset_dfcolumns
        
    # generate demo for experiment
    if("mo" not in dataset):
        if(sample < 1 and demo and len(df) > 100000):
            df = df.sample(frac = sample, random_state = SEED)
    
    # remove self loops from df
    self_loop_idx = []
    for i in range(len(df)):
        if(df.iloc[i,0] == df.iloc[i,1]):
            self_loop_idx.append(i)
    print("%d selfloop detected." % len(self_loop_idx))
    df.drop(index = self_loop_idx).reset_index(inplace = True)

    return df

def BuildGraphFromDf(df, dataset, relabel = True):
    
    nrows = len(df)

    print("Dataframe has %d rows." % nrows)

    # initialize graph
    if(dataset_info[dataset]["directed"]):
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # build graph from dataframe
    for i in range(nrows):
        G.add_edge(df.iloc[i,0], df.iloc[i,1])
    
    if("mo" in dataset):
        G = AddVertexinMo(G)
    
    if(relabel):
        mapping = dict(zip(G, range(G.number_of_nodes())))
        G = nx.relabel_nodes(G, mapping)
        
    return G

"""
sample deleted item index
"""
def GenerateDeletedIdx(G, delete_p, delete_vertex):
    deleted_edges_idx = []
    deleted_nodes_idx = []
    
    random.seed(0)

    if(delete_vertex):
        if(delete_p > 0 and delete_p < 1):
            nodes = G.number_of_nodes()
            deleted_nodes_idx = random.sample(range(nodes), int(delete_p*nodes))
    else:
        if(delete_p > 0 and delete_p < 1):
            edges = G.number_of_edges()
            deleted_edges_idx = random.sample(range(edges), int(delete_p*edges))
        
    return deleted_edges_idx, deleted_nodes_idx

def PrintDegreeInfo(G, max_degree = 1):
    
    degree_cnt = collections.defaultdict(int)

    for v in G.nodes:
        if(G.degree[v] <= max_degree):
            degree_cnt[G.degree[v]] += 1

    for d in range(max_degree+1):
        print("%d %d-degree nodes" % (degree_cnt[d], d))

def RemoveItemFromGraphByIdx(G, deleted_vertex, deleted_idx):
    if(deleted_vertex):
        G.remove_nodes_from(deleted_idx)
    else:
        edgelist = list(G.edges)
        for idx in deleted_idx:
            G.remove_edge(edgelist[idx][0], edgelist[idx][1])
    return G

def MakeMossoInputfilename(inputfolder, datasetname, delete_p, retrain):
    filename = '-'.join([datasetname, str(delete_p), "r" if retrain else "d"]) + ".txt"
    return os.path.join(inputfolder, filename)

def CommunityDetection(graph, prefix):
    t1 = time()
    graph_community = nx.community.louvain_communities(graph)
    
    t2 = time()
    print("%s graph community detection timing: %.3f" % (prefix, t2 - t1))
    return graph_community

def AddVertexinMo(G):
    filename = "/data/lizitong/202304-YQQ/dataset-new/mo/mo.txt"
    df = pd.read_csv(filename, header= 0, delimiter='\t')
    #df.columns = ["start","end","timestamp"]
    for i,row in df.iterrows():
        if(int(row['start']) not in G.nodes):
            G.add_node(row['start'])
           
        if(int(row['end']) not in G.nodes):
            G.add_node(row['end'])

    return G

"""
Read .txt format datafile to graph.
gen_delete: whether create a graph that has already removed the target edges/nodes.
    if gen_delete = True, then the returned graph has already removed the target edges/nodes,
    and the deleted_edges would be None.
    otherwise, the returned graph is generated originally from the datafile without
    edges/nodes removal, and deleted_edges contains the edges to be removed.
delete_vertex: whether to delete by vertices or by edges.if delete_vertex = True,
    then delete by vertices, that is, remove all the edges related to a vertices,
    and the returned deleted_nodes is not empty. if delete_vertex = False, then the
    deleted_nodes is returned empty
delete_p: control the edges/nodes deletion proportion.
"""

def ReadDatafile(dataset, subgraph, dataset_folder, filename = "", sample_frac = 1, 
                 gen_delete = False, delete_vertex = False, delete_p = -1, 
                 relabel = True, genMossoInput = False, mossoinputfolder = ""):
    
    df = ReadGraphfile2Dataframe(dataset = dataset, subgraph = subgraph, 
                                 dataset_folder = dataset_folder, 
                                 filename = "", demo = True, sample = sample_frac, 
                                 random_seed = SEED)
    t1 = time()
    G = BuildGraphFromDf(df, dataset = dataset, relabel = relabel)
    # G = AddVertexinMo(G)
    deleted_nodes_idx, deleted_edges_idx = None, None
    t2 = time()
    print("Build graph timing: %.3f, min and max node label:%d and %d" % (t2-t1, min(G.nodes), max(G.nodes)))


    if("mo" not in dataset):
        G.remove_edges_from(nx.selfloop_edges(G))
    
        deleted_edges_idx, deleted_nodes_idx = GenerateDeletedIdx(G, delete_p = delete_p, 
                                                              delete_vertex = delete_vertex)

    if(gen_delete):
        if(delete_vertex):
            deleted_idx = deleted_nodes_idx
        else:
            deleted_idx = deleted_edges_idx
        G = RemoveItemFromGraphByIdx(G, deleted_vertex = delete_vertex, deleted_idx = deleted_idx)

        PrintDegreeInfo(G, max_degree = 1)
        print("min and max node label:%d and %d" % (min(G.nodes), max(G.nodes)))
    else:
        PrintDegreeInfo(G, max_degree = 1)
    
    if(genMossoInput):
        filename = MakeMossoInputfilename(inputfolder = mossoinputfolder, datasetname = dataset, 
                                          delete_p = delete_p, retrain = gen_delete)
        GenerateMossoInput(df, G, deleted_edges_idx, filename, retrain = gen_delete)

    A = nx.adjacency_matrix(G)
    
    return G, A, deleted_nodes_idx, deleted_edges_idx

def GetCosineDistance(a , b):
    dot = np.dot(a, b)
    return dot/(np.linalg.norm(a)*np.linalg.norm(b))

def GetManhattanDistance(a, b):
    
    def Reshape(a):
        
        if(a.shape[0]==1):
            a = a.reshape(a.shape[1])
        return a
    
    a = Reshape(a)
    b = Reshape(b)
    
    return np.sum(np.abs(a - b)) #np.linalg.norm(np.logical_xor(a, b), ord=0)

def GetEdgeCountV0(a_set, b_set, G, p = 0.6):
    cnt = 0
    thres = (len(a_set) * p) *  (len(b_set) * p)
    for u in a_set:
        for v in b_set:
            cnt += G.has_edge(u,v)
    
        if(cnt > thres):
            return 2
    
    if(cnt > thres):
        return 1
    
    return False

def GetEdgeCountV1(a_set, b_set, edgelist, p = 0.36):
    
    cnt = len(list(filter(lambda x: x[0] in a_set and x[1] in b_set, edgelist)))
    
    if(cnt > len(a_set) * len(b_set) * p):
        return True
    
    return False

def GetEdgeCountV2(a_set, b_set, G, p = 0.36):
    # cnt = 0
    
    # t1 = time()
    # src_a_idx = np.where(np.isin(A_nonzero_row, a_set) == 1)
    # t2 = time()
    # dst_a = A_nonzero_col[src_a_idx]
    # t3 = time()
    # cnt = np.sum(np.isin(dst_a, list(b_set)))
    # print("timing t2-t1: %.3f, t3-t2: %.3f" % (t2-t1, t3-t2))
    """
    for u in a_set:
        dst_u_idx = np.where(A_nonzero_row == u)
        dst_u = A_nonzero_col[dst_u_idx]
        cnt += len(b_set & set(dst_u))
    """
    
    a_set_half_right = list(a_set)[:int(len(a_set) * max(p, 1-p))]
    a_set_half_left = list(a_set)[int(len(a_set) * max(p, 1-p)):]

    cnt = 0
    for u in a_set_half_right:
        for v in b_set:
            cnt += G.has_edge(u,v)
    
    if(cnt + len(a_set_half_left) * len(b_set) < len(a_set) * len(b_set) * p):
        return False
    
    for u in a_set_half_left:
        for v in b_set:
            cnt += G.has_edge(u,v)
    
    if(cnt > len(a_set) * len(b_set) * p):
        return True
    
    return False

def ReadSupergraphFromFile(config, graph, retrain = "o"):

    supernode_file, superedge_file = ReadSUGPTOutput(output_folder = config["output_folder"], 
                                                     dataset = config["dataset"], delete_p = config["delete_p"], 
                                                     retrain = retrain)
    print("="*10, "begin to read sugpt-output:", supernode_file, "="*10)
    supergraph = Supergraph(graph, config)
    sn_df = pd.read_csv(supernode_file)
    se_df = pd.read_csv(superedge_file)
    
    # if(sn_df.columns[0] == "Supernode" and sn_df.columns[1] == "Vertex"):
    #     print("Swap our csv output columns.")
    #     sn_df = sn_df[["Vertex","Supernode"]]
    
    for cnt in range(len(sn_df)):
        vertex, sn = sn_df.iloc[cnt, 1], sn_df.iloc[cnt, 0]
        if(sn not in supergraph.supernode.keys()):
            supergraph.supernode[sn] = Supernode(candidate = set(), aggmethod = None, graph = None, se_gen_method = "EDG")
        supergraph.supernode[sn].vertices.add(vertex)

    for cnt in range(len(se_df)):
        svi, svj = se_df.iloc[cnt, 0], se_df.iloc[cnt, 1]
        supergraph.superedge.append([svi, svj])
    return supergraph

def ReadSUGPTOutput(output_folder = "", dataset = "", delete_p = 0.1, retrain = "r"):
    
    if(retrain == "o"):
        supernode_file = output_folder + '-'.join([dataset, "sn", retrain]) + ".csv"
    
        superedge_file = output_folder + '-'.join([dataset, "se", retrain]) + ".csv"
    else:
        supernode_file = output_folder + '-'.join([dataset, str(delete_p), "sn", retrain]) + ".csv"
    
        superedge_file = output_folder + '-'.join([dataset, str(delete_p), "se", retrain]) + ".csv"

    return supernode_file, superedge_file

def SaveSupernode(supergraph, supernode_file, chunk):
    
    def ConcatTmpresult(supernode_df, sv_list, vertices_list):
        assert len(sv_list) == len(vertices_list)
        tmpdf = pd.DataFrame(columns=["Supernode","Vertex"])
        tmpdf["Supernode"] = sv_list
        tmpdf["Vertex"] = vertices_list
        supernode_df = pd.concat([supernode_df, tmpdf], ignore_index=True)
        return supernode_df
    
    supernode_df = pd.DataFrame()
    sv_list = []
    vertices_list = []
        
    cnt = 0
    for sv in supergraph.supernode:
        for vertex in supergraph.supernode[sv].vertices:
            sv_list.append(sv)
            vertices_list.append(vertex)
            cnt += 1
                
            if(cnt > chunk):
                supernode_df = ConcatTmpresult(supernode_df, sv_list, vertices_list)
                
                # clear tmpresult
                cnt = 0
                sv_list.clear()
                vertices_list.clear()
        
    supernode_df = ConcatTmpresult(supernode_df, sv_list, vertices_list)
        
    assert len(supernode_df["Vertex"].unique()) == supergraph.graph.vertices_num
    supernode_df.to_csv(supernode_file, index = False)
    print("save supernode to file: ", supernode_file)
        
def SaveSuperedge(supergraph, superedge_file, chunk):
    
    def ConcatTmpresult(superedge_df, from_list, to_list):
        assert len(from_list) == len(to_list)
        tmpdf = pd.DataFrame(columns=["From","To"])
        tmpdf["From"] = from_list
        tmpdf["To"] = to_list
        superedge_df = pd.concat([superedge_df, tmpdf], ignore_index=True)
        return superedge_df
    
    superedge_df = pd.DataFrame()
    from_list = []
    to_list = []
        
    cnt = 0
    for se in supergraph.superedge:
        from_list.append(se[0])
        to_list.append(se[1])
            
        if(cnt > chunk):
            superedge_df = ConcatTmpresult(superedge_df, from_list, to_list)
            
            # clear tmpresult
            cnt = 0
            from_list.clear()
            to_list.clear()
        
    superedge_df = ConcatTmpresult(superedge_df, from_list, to_list)
        
    superedge_df.to_csv(superedge_file, index = False)
    print("save superedge to file: ", superedge_file)
    
def SaveSupergraph(supergraph, supernode_file, superedge_file, chunk = 10000, retrain = "r", dp = 0.1, output_folder = "./"):
    
    if(supernode_file == ""):
        supernode_file = output_folder + '-'.join([supergraph.dataset, str(dp), "sn", retrain]) + ".csv"

    if(superedge_file == ""):
        superedge_file = output_folder + '-'.join([supergraph.dataset, str(dp), "se", retrain]) + ".csv"

    SaveSupernode(supergraph, supernode_file, chunk)
    SaveSuperedge(supergraph, superedge_file, chunk)

def SaveOriginalSupergraph(supergraph, supernode_file, superedge_file, chunk = 10000, output_folder = "./"):
    
    if(supernode_file == ""):
        supernode_file = output_folder + '-'.join([supergraph.dataset, "sn", "o" ]) + ".csv"

    if(superedge_file == ""):
        superedge_file = output_folder + '-'.join([supergraph.dataset, "se", "o"]) + ".csv"

    SaveSupernode(supergraph, supernode_file, chunk)
    SaveSuperedge(supergraph, superedge_file, chunk)

def ReadSupergraphFromFilev0(supergraph, supernode_file, superedge_file):
    sn_df = pd.read_csv(supernode_file)
    se_df = pd.read_csv(superedge_file)
    
    if(sn_df.columns[0] == "Supernode" and sn_df.columns[1] == "Vertex"):
        print("Swap our csv output columns.")
        sn_df = sn_df[["Vertex","Supernode"]]
    
    for cnt in range(len(sn_df)):
        vertex, sn = sn_df.iloc[cnt, 0], sn_df.iloc[cnt, 1]
        supergraph[sn].vertices.add(vertex)

    return

def ProcessDynamicDatasetv0(filename):

    df = pd.read_csv(filename, header = None, delimiter=' ')
    df.columns = ["start","end", "tmpstamp"]
    
    def str_map(x):
        return ''.join(sorted(x))
    df["dup"] = df["start"].astype('str') + '.' + df["end"].astype('str')
    df["dup"] = df["dup"].map(str_map)
    df = df.drop_duplicates("dup",keep = 'first', ignore_index=True)
    
    # frac = 10
    # part_len = math.floor(len(df)/frac)
    init_frac = 0.5
    part_len = math.floor(len(df) * init_frac)
    N = 10
    window_frac = math.floor(part_len/N)

    nrows = len(df)

    # initialize graph
    G = nx.DiGraph()

    # build graph from dataframe
    for i in range( nrows ):
        G.add_edge(df.iloc[i,0], df.iloc[i,1])   

    # relabel
    mapping = dict(zip(G, range(G.number_of_nodes())))
    df["start"] = df["start"].map(mapping)
    df["end"] = df["end"].map(mapping)
    
    df = df.drop(columns = 'dup', axis = 1)
    df.to_csv("/data/lizitong/202304-YQQ/dataset-new/mo/mo.txt", header=True, sep = "\t", index = False)

    for i in range(N+1):

        if( i == 0 ):
            #T0
            output_file = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + ".txt"
            tmpdf = df.iloc[ : part_len, [0,1]]
            tmpdf.to_csv(output_file, header=True, sep = "\t", index = False)
            print("written %d rows to %s " % (len(tmpdf), output_file))
        else:
            tmpdf_insert = df.iloc[part_len + window_frac*(i-1) : part_len + window_frac*(i), [0,1]]
            output_file_insert = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + "-ins.txt"
            tmpdf_insert.to_csv(output_file_insert, header=True, sep = "\t", index = False)
            
            tmpdf_remove = df.iloc[window_frac*(i-1) : window_frac*(i), [0,1]]
            output_file_remove = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + "-del.txt"
            tmpdf_remove.to_csv(output_file_remove, header=True, sep = "\t", index = False)
            
            tmpdf_truth = df.iloc[window_frac*(i) : part_len + window_frac*(i), [0,1]]
            output_file_truth = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + ".txt"
            tmpdf_truth.to_csv(output_file_truth, header=True, sep = "\t", index = False)
            
            print("written %d, %d, %d rows to %s, %s, %s " % (len(tmpdf_insert), 
                                                      len(tmpdf_remove),
                                                      len(tmpdf_truth),
                                                      output_file_insert,
                                                      output_file_remove,
                                                      output_file_truth))
    return

def ProcessDynamicDataset(filename):

    df = pd.read_csv(filename, header = None, delimiter=' ')
    df.columns = ["start","end", "timestamp"]
    
    def str_map(x):
        return ''.join(sorted(x))
    df["dup"] = df["start"].astype('str') + '.' + df["end"].astype('str')
    df["dup"] = df["dup"].map(str_map)
    df = df.drop_duplicates("dup",keep = 'first', ignore_index=True)
    
    # frac = 10
    # part_len = math.floor(len(df)/frac)
    time_median = int(df["timestamp"].median())
    time_min = int(df["timestamp"].min())
    N = 10
    window_frac = math.floor((time_median-time_min)/N)
    print("time_median: %d, time_min: %d" % (time_median, time_min))
    nrows = len(df)

    # initialize graph
    G = nx.DiGraph()

    # build graph from dataframe
    for i in range( nrows ):
        G.add_edge(df.iloc[i,0], df.iloc[i,1])   

    # relabel
    mapping = dict(zip(G, range(G.number_of_nodes())))
    df["start"] = df["start"].map(mapping)
    df["end"] = df["end"].map(mapping)
    
    df = df.drop(columns = 'dup', axis = 1)
    df.to_csv("/data/lizitong/202304-YQQ/dataset-new/mo/mo.txt", header=True, sep = "\t", index = False)

    for i in range(N+1):

        if( i == 0 ):
            #T0
            output_file = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + ".txt"
            tmpdf = df[df["timestamp"].between(time_min, time_median)].iloc[ :, [0,1]]
            tmpdf.to_csv(output_file, header=True, sep = "\t", index = False)
            print("written %d rows to %s " % (len(tmpdf), output_file))
        else:
            tmpdf_insert = df[df["timestamp"].between(time_median + window_frac*(i-1), time_median + window_frac*(i))].iloc[ :, [0,1]]
            output_file_insert = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + "-ins.txt"
            tmpdf_insert.to_csv(output_file_insert, header=True, sep = "\t", index = False)
            
            tmpdf_remove = df[df["timestamp"].between(time_min + window_frac*(i-1), time_min + window_frac*(i))].iloc[ :, [0,1]]
            # df.iloc[window_frac*(i-1) : window_frac*(i), [0,1]]
            output_file_remove = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + "-del.txt"
            tmpdf_remove.to_csv(output_file_remove, header=True, sep = "\t", index = False)
            
            tmpdf_truth = df[df["timestamp"].between(time_min + window_frac*(i), time_median + window_frac*(i))].iloc[ :, [0,1]]
            # df.iloc[window_frac*(i) : part_len + window_frac*(i), [0,1]]
            output_file_truth = "/data/lizitong/202304-YQQ/dataset-new/mo/mo-T" + str(i) + ".txt"
            tmpdf_truth.to_csv(output_file_truth, header=True, sep = "\t", index = False)
            
            print("written %d, %d, %d rows to %s, %s, %s " % (len(tmpdf_insert), 
                                                      len(tmpdf_remove),
                                                      len(tmpdf_truth),
                                                      output_file_insert,
                                                      output_file_remove,
                                                      output_file_truth))
    return

def ProcessDynamicDatasetMoSSo(filename, output_folder):

    df = pd.read_csv(filename, header = None, delimiter=' ')
    df.columns = ["start","end", "timestamp"]
    
    def str_map(x):
        return ''.join(sorted(x))
    df["dup"] = df["start"].astype('str') + '.' + df["end"].astype('str')
    df["dup"] = df["dup"].map(str_map)
    df = df.drop_duplicates("dup",keep = 'first', ignore_index=True)
    
    df["selfloop"] = df["start"] - df["end"]
    df.drop(df[df['selfloop'] == 0].index, inplace=True)
    
    # frac = 10
    # part_len = math.floor(len(df)/frac)
    time_median = int(df["timestamp"].median())
    time_min = int(df["timestamp"].min())
    N = 10
    window_frac = math.floor((time_median-time_min)/N)
    print("time_median: %d, time_min: %d" % (time_median, time_min))
    nrows = len(df)

    # initialize graph
    G = nx.DiGraph()

    # build graph from dataframe
    for i in range( nrows ):
        G.add_edge(df.iloc[i,0], df.iloc[i,1])

    # relabel
    mapping = dict(zip(G, range(G.number_of_nodes())))
    df["start"] = df["start"].map(mapping)
    df["end"] = df["end"].map(mapping)
    
    df = df.drop(columns = 'dup', axis = 1)
    df = df.drop(columns = 'selfloop', axis = 1)
    # df.to_csv("/data/lizitong/202304-YQQ/dataset-new/mo/mo.txt", header=False, sep = "\t", index = False)

    df_dynamic = df[df["timestamp"].between(time_min, time_median)].iloc[ :, [0,1]]
    df_dynamic[2] = 1
    print("T0 dataset length: %d" % len(df_dynamic))

    for i in range(1,N+1):

        tmpdf_insert = df[df["timestamp"].between(time_median + window_frac*(i-1), time_median + window_frac*(i))].iloc[ :, [0,1]]
        tmpdf_insert[2] = 1

        tmpdf_remove = df[df["timestamp"].between(time_min + window_frac*(i-1), time_min + window_frac*(i))].iloc[ :, [0,1]]
        tmpdf_remove[2] = -1

        df_dynamic = df_dynamic.append(tmpdf_insert)
        df_dynamic = df_dynamic.append(tmpdf_remove)

        print(print("T%d dataset length: %d, insert %d edges, delete %d edges" % (i,len(df_dynamic),
                                                                                  len(tmpdf_insert), len(tmpdf_remove))))
            
        tmpdf_truth = df[df["timestamp"].between(time_min + window_frac*(i), time_median + window_frac*(i))].iloc[ :, [0,1]]
        tmpdf_truth[2] = 1
        output_file_truth = os.path.join( output_folder, "mo-T" + str(i) + ".txt")
        tmpdf_truth.to_csv(output_file_truth, header=False, sep = "\t", index = False)
            
        print("written %d rows to %s " % (len(tmpdf_truth),output_file_truth))
    
    dynamic_file = os.path.join( output_folder, "mo_dynamic.txt")
    df_dynamic.to_csv(dynamic_file, header=False, sep = "\t", index = False)
    print("written %d rows to %s " % (len(df_dynamic), dynamic_file))

    return

def ReadEdges(filename, datafolder):

    filename = os.path.join( datafolder, "mo", filename)

    edges = []
    df = pd.read_csv(filename, delimiter='\t')

    for i,row in df.iterrows():
        edges.append([int(row['start']),int(row['end'])])
        
    return edges

if __name__ == "__main__":
    # ProcessDynamicDataset(filename = "/data/lizitong/202304-YQQ/dataset-new/mo/mo_notrelabeled.txt")
    ProcessDynamicDatasetMoSSo(filename = "/data/lizitong/202304-YQQ/dataset-new/mo/mo_notrelabeled.txt", 
                              output_folder = "/data/lizitong/202304-YQQ/kdd20-mosso/mosso/dataset_mo")