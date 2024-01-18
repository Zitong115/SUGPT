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
import supergraph_0103 as supergraph

parser = ArgumentParser(prog = "Forgettable Graph Summarization")
parser.add_argument("--config", type = str, default = "C:/research/GraphSummarization/mycode-0619/src-1116/test.yaml")
parser.add_argument("--dataset", type = str, default = "")
parser.add_argument("--delete_p", type = float, default = 0)
parser.add_argument("--candidate_earlystop_thres", type = int, default = 0)
parser.add_argument("--evaluate_mosso", type = bool, default = False)
parser.add_argument("--evaluate_type", type = str, default = "")
args = parser.parse_args()

def cmpSupernode(sg1, sg2):

    print("="*5, "supernode check", "="*5)

    sg1_vertices = set(sg1.graph.G.nodes())
    sg2_vertices = set(sg2.graph.G.nodes())

    larger_uns_cnt = 0
    larger_res_cnt = 0

    for sv in sg1.supernode.keys():

        cur1 = sg1.supernode[sv]
        sg1_vertices -= cur1.vertices

        for v in cur1.vertices:
            cur2 = sg2.supernode[sg2.vertex2supernode[v]]

            # if(len(cur1.vertices - cur2.vertices) or len(cur2.vertices - cur1.vertices)):
            #     print("differ sv: ", cur1.vertices, list(cur1.ele_cnt.items()), cur1.stop_round, cur2.vertices, list(cur2.ele_cnt.items()), cur2.stop_round)
            
            if(len(cur1.vertices) < len(cur2.vertices)):
                larger_res_cnt += 1
                if(larger_res_cnt < 10):
                    print("differ sv: ", sv, cur1.vertices, list(cur1.ele_cnt.items()), cur1.stop_round, sg2.vertex2supernode[v], cur2.vertices, list(cur2.ele_cnt.items()), cur2.stop_round)
            elif(len(cur1.vertices) > len(cur2.vertices)):
                larger_uns_cnt += 1
                if(larger_uns_cnt < 10):
                    print("differ sv: ", sv, cur1.vertices, list(cur1.ele_cnt.items()), cur1.stop_round, sg2.vertex2supernode[v], cur2.vertices, list(cur2.ele_cnt.items()), cur2.stop_round)

            sg2_vertices -= cur2.vertices
    
    print("larger_uns_cnt:" , larger_uns_cnt, "larger_res_cnt:" , larger_res_cnt)

def cmpGroup(sg1, sg2):

    assert sum(sg1.graph.A - sg2.graph.A).sum() == 0

    print("="*5, "groups check", "="*5)
    flag = False

    for treeno in sg1.graph.trees.keys():
        
        if(treeno not in sg2.graph.trees.keys()):
            flag = True
            break

        groups = sg1.graph.trees[treeno].group.keys()

        for groupno in groups:
            
            if(groupno not in sg2.graph.trees[treeno].group.keys()):
                if(len(sg1.graph.trees[treeno].group[groupno].vertices)):
                    # print("differ in treeno - groupno; group in sg1 is not empty: ", treeno, groupno, sg1.graph.trees[treeno].group[groupno].vertices)
                    flag = True
            else:
                if(len(sg1.graph.trees[treeno].group[groupno].vertices - sg2.graph.trees[treeno].group[groupno].vertices) or 
                   len(sg1.graph.trees[treeno].group[groupno].vertices - sg2.graph.trees[treeno].group[groupno].vertices)):
                    
                    # print("differ group: ", treeno, groupno, sg1.graph.trees[treeno].group[groupno].vertices, sg2.graph.trees[treeno].group[groupno].vertices)
                    flag = True
    if(flag):
        print("group check failed.")
        exit(1)
    else:
        print("group check passed.")

def ExpSingleDp(sg, config, dp, deleted_idx):
    
    # g1 = graph.Graph(config = config, dp = dp, gen_delete = False, eval_mosso = True)

    s0_vertex2supernode = sg.vertex2supernode.copy()
    s0_ordered_tree = sg.ordered_tree

    if(config["vertex_deletion"]):
        sg.VertexDelete(deleted_vertices = deleted_idx)
        sg.ShowInfo()
    else:
        uninfluenced_sv_set = sg.EdgeDelete(deleted_edge_idx = deleted_idx)
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

    # utils.SaveSupergraph(supergraph=sg, supernode_file = "", 
    #                      superedge_file = "", dp = dp, retrain = "d", output_folder = config["output_folder"])
    
    
    print("="*10, "SUMMARIZE FROM SCRATCH", "="*10)

    t1 = utils.time()
    g2.VerticesAlignmentByCol()
    g2.genGroups()
    
    sg2 = supergraph.Supergraph(graph = g2, config = config, dp = dp, ordered_tree = s0_ordered_tree, vertex2supernode = s0_vertex2supernode)
    cmpGroup(sg, sg2)
    
    sg2.genSupernodeForAll()
    sg2.genSuperedgeForAll()
    
    t2 = utils.time()
    sg2.ShowInfo()
    
    cmpSupernode(sg, sg2)
    
    Evaluator = evaluation.Evaluator(supergraph = sg2, 
                                     supernode = None, superedge = None, 
                                     evaluated_file = None, 
                                     evaluated_file_type = evaluation.MY_OUTPUT_FILE_TYPE, 
                                     original_graph = g2.G, 
                                     evaluation_type = config["evaluation_type"],
                                     dataset = config["dataset"])
    Evaluator.Evaluate(vertex2supernode = None)

    print("Summarize original graph timing: %.3f" % (t2-t1))
    
    # utils.SaveSupergraph(supergraph=sg2, supernode_file = "", 
    #                      superedge_file = "", dp = dp, retrain = "r", output_folder = config["output_folder"])
    
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

    print("Experimenting on %s dataset, delete_p: %.3f" % (config["dataset"],config["delete_p"]))
    print("config:", config)

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

    print("="*10, "TEMPORAL EXPERIMENT", "="*10)
    
    for t in [1,2,3,4,5,6,7,8,9]:
        
        print("="*10, "TIMESTAMP ", t, "="*10)

        print("="*10, "IN/DECREMENTAL", "="*10)

        if(t > 0):
            insert_edges = utils.ReadEdges(filename = "mo-T" + str(t) + "-ins.txt", datafolder=config["dataset_folder"])
            removed_edges = utils.ReadEdges(filename = "mo-T" + str(t) + "-del.txt", datafolder=config["dataset_folder"])
        else:
            removed_edges = insert_edges
            insert_edges = utils.ReadEdges(filename = "mo-T" + str(t) + ".txt", datafolder=config["dataset_folder"])
        
        print("insert %d edges, remove %d edges" % (len(insert_edges), len(removed_edges)))

        sg.EdgeAddDel(insert_edges, removed_edges)
        sg.ShowInfo()
        
        g1 = graph.Graph(config = config, gen_delete = False, dataset = "mo-T" + str(t))
        Evaluator = evaluation.Evaluator(supergraph = sg, 
                                     supernode = None, superedge = None, 
                                     evaluated_file = None, 
                                     evaluated_file_type = evaluation.MY_OUTPUT_FILE_TYPE, 
                                     original_graph = g1.G, 
                                     evaluation_type = config["evaluation_type"],
                                     dataset = config["dataset"])
    
        Evaluator.Evaluate(vertex2supernode = None)
        
        print("="*10, "RESUMMARIZATION", "="*10)

        t1 = utils.time()
        
        g1.VerticesAlignmentByCol()
        g1.genGroups()
            
        sg_re = supergraph.Supergraph(graph = g1, config = config)
        sg_re.genSupernodeForAll()

        assert sg_re.graph.vertices_num == len(sg_re.vertex2supernode.keys())
            
        sg_re.genSuperedgeForAll()

        t2 = utils.time()
        
        sg_re.ShowInfo()

        Evaluator = evaluation.Evaluator(supergraph = sg_re, 
                                        supernode = None, superedge = None, 
                                        evaluated_file = None, 
                                        evaluated_file_type = evaluation.MY_OUTPUT_FILE_TYPE, 
                                        original_graph = g1.G, 
                                        evaluation_type = config["evaluation_type"],
                                        dataset = config["dataset"])
        
        Evaluator.Evaluate(vertex2supernode = None)
        print("re-Summarize original graph timing: %.3f" % (t2-t1))

        print("="*10, "TEMPORAL EXPERIMENT DONE", t,  "="*10)
    
    """
    print("="*10, "Resummarization temporal EXPERIMENT", "="*10)

    for t in [1,2,3,4,5,6,7,8,9]:
        g1 = graph.Graph(config = config, gen_delete = False, dataset = "mo-T" + str(t) )
    
        t1 = utils.time()
        
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
        print("re-Summarize original graph timing: %.3f, len(sg.vertex2supernode): %d" % (t2-t1, len(sg.vertex2supernode)))
    
    # utils.SaveOriginalSupergraph(supergraph=sg, supernode_file = "", 
    #                       superedge_file = "", output_folder = config["output_folder"])
    """