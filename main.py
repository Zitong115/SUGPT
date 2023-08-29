# -*- coding: utf-8 -*-
"""
Author: zitong li
Date: 2023.4.
"""

from argparse import ArgumentParser
import math
import random
from sys import exit
from copy import deepcopy

import arraytree
import evaluation
import graph
import utils
import os
import supergraph_dask as supergraph

parser = ArgumentParser(prog = "Forgettable Graph Summarization")
parser.add_argument("--config", type = str, default = "C:/research/GraphSummarization/mycode-0619/src-0807/test.yaml")
parser.add_argument("--dataset", type = str, default = "")
parser.add_argument("--delete_p", type = float, default = 0)
parser.add_argument("--candidate_earlystop_thres", type = int, default = 0)
parser.add_argument("--evaluate_mosso", type = bool, default = False)
parser.add_argument("--evaluate_type", type = str, default = "")
args = parser.parse_args()

def ExpSingleDp(sg, config, dp, deleted_idx):
    
    # g1 = graph.Graph(config = config, dp = dp, gen_delete = False, eval_mosso = True)

    if(config["vertex_deletion"]):
        sg.VertexDelete(deleted_vertices = deleted_idx)
        sg.ShowInfo()
    else:
        sg.EdgeDelete(deleted_edge_idx = deleted_idx)
        sg.ShowInfo()
    
    g2 = graph.Graph(config = config, dp = dp, gen_delete = True)

    Evaluator = evaluation.Evaluator(supergraph = sg, 
                                     supernode = None, superedge = None, 
                                     evaluated_file = None, 
                                     evaluated_file_type = evaluation.MY_OUTPUT_FILE_TYPE, 
                                     original_graph = g2.G, 
                                     evaluation_type = config["evaluation_type"],
                                     dataset = config["dataset"])
    Evaluator.Evaluate(vertex2supernode = None)

    utils.SaveSupergraph(supergraph=sg, supernode_file = "", 
                         superedge_file = "", dp = dp, retrain = "d", output_folder = config["output_folder"])
    """
    print("="*10, "SUMMARIZE FROM SCRATCH", "="*10)

    t1 = utils.time()
    g2.VerticesAlignmentByCol()
    g2.genGroups()
    
    sg2 = supergraph.Supergraph(graph = g2, config = config, dp = dp)
    sg2.genSupernodeForAll()
    sg2.genSuperedgeForAll()

    t2 = utils.time()
    sg2.ShowInfo()
    
    Evaluator = evaluation.Evaluator(supergraph = sg2, 
                                     supernode = None, superedge = None, 
                                     evaluated_file = None, 
                                     evaluated_file_type = evaluation.MY_OUTPUT_FILE_TYPE, 
                                     original_graph = g2.G, 
                                     evaluation_type = config["evaluation_type"],
                                     dataset = config["dataset"])
    Evaluator.Evaluate(vertex2supernode = None)

    print("Summarize original graph timing: %.3f" % (t2-t1))

    utils.SaveSupergraph(supergraph=sg2, supernode_file = "", 
                         superedge_file = "", dp = dp, retrain = "r", output_folder = config["output_folder"])
    """
if __name__ == "__main__":
    
    config = utils.ReadYmlConfig(args.config)
    
    if(args.dataset):
        config["dataset"] = args.dataset
    if(args.delete_p):
        config["delete_p"] = args.delete_p
    if(args.candidate_earlystop_thres):
        config["candidate_earlystop_thres"] = args.candidate_earlystop_thres
    if(args.evaluate_mosso):
        config["evaluate_mosso"] = args.evaluate_mosso
    if(args.evaluate_type):
        config["evaluation_type"] = [args.evaluate_type]

    # if(config["dataset"] == "mt"):
    #     config["candidate_earlystop_thres"] = 10
    print("Experimenting on %s dataset, delete_p: %.3f" % (config["dataset"],config["delete_p"]))
    print("config:", config)

    """
    for dp in [0.1, 0.2, 0.3, 0.4, 0.5]:
        #g1 = graph.Graph(config = config, dp = dp, gen_delete = False, eval_mosso = True)
        g1 = graph.Graph(config = config, dp = dp, gen_delete = True, eval_mosso = True)
    """

    # To evaluate Mosso output
    if(config["evaluate_mosso"]):
        mosso_outputfile = utils.MakeMossoInputfilename(config["mosso_outputfolder"], config["dataset"], config["delete_p"], retrain = False)
        print("="*10, "begin to evaluate mosso-output:", mosso_outputfile, "="*10)
        g = graph.Graph(config = config, gen_delete = True, eval_mosso = True)
        
        Evaluator = evaluation.Evaluator(supergraph = None, supernode = None, superedge = None, 
                                         evaluated_file = mosso_outputfile,
                                         evaluated_file_type = evaluation.MOSSO_OUTPUT_FILE_TYPE, 
                                         original_graph = g.G, 
                                         dataset = config["dataset"], 
                                         evaluation_type = config["evaluation_type"])
        Evaluator.Evaluate()
        exit(0)
        
    g1 = graph.Graph(config = config, gen_delete = False)
    
    t1 = utils.time()
    if(os.path.exists(config["output_folder"] + '-'.join([config["dataset"], "sn", "o"]) + ".csv") and
        os.path.exists(config["output_folder"] + '-'.join([config["dataset"], "se", "o"]) + ".csv")):
        sg = utils.ReadSupergraphFromFile(config, g1, retrain = "o")
    else:
        g1.VerticesAlignmentByCol()
        g1.genGroups()
        
        sg = supergraph.Supergraph(graph = g1, config = config)
        sg.genSupernodeForAll()

        assert sg.graph.vertices_num == len(sg.vertex2supernode.keys())
        
        sg.genSuperedgeForAll()

    t2 = utils.time()
    
    sg.ShowInfo()

    Evaluator = evaluation.Evaluator(supergraph = sg, 
                                     supernode = None, superedge = None, 
                                     evaluated_file = None, 
                                     evaluated_file_type = evaluation.MY_OUTPUT_FILE_TYPE, 
                                     original_graph = g1.G, 
                                     evaluation_type = config["evaluation_type"],
                                     dataset = config["dataset"])
    
    Evaluator.Evaluate(vertex2supernode = None)
    print("Summarize original graph timing: %.3f, len(sg.vertex2supernode): %d" % (t2-t1, len(sg.vertex2supernode)))
    
    utils.SaveOriginalSupergraph(supergraph=sg, supernode_file = "", 
                          superedge_file = "", output_folder = config["output_folder"])
    
    #sg_original = deepcopy(sg)

    print("="*10, "DECREMENTAL EXPERIMENT", "="*10)
    
    for dp in [0.1, 0.2, 0.3, 0.4, 0.5]:
        
        print("="*10, "DECREMENTAL EXPERIMENT EVALUATION DELETE_P ", dp,  "="*10)
        tmpsg = deepcopy(sg)
        deleted_edges_idx, deleted_nodes_idx = tmpsg.GenEdgeDeleteIdx(dp)
        ExpSingleDp(tmpsg, config, dp, deleted_edges_idx)
        print("="*10, "FINISH DECREMENTAL EXPERIMENT EVALUATION DELETE_P ", dp,  "="*10)
        