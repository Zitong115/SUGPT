This is the code for paper "SUGPT: Efficient Graph Unsummarization for the Right to Be Forgotten".

# Dataset

The datasets are downloaded from the website: https://snap.stanford.edu/data/.

The dataset format is the same as the directly downloaded version. For example, the head of the "email-EuAll" dataset text file is:

> Directed graph (each unordered pair of nodes is saved once): Email-EuAll.txt 
> 
> Email network of a large European Research Institution (directed edge means at least one email was sent between October 2003 and March 2005)
> 
> Nodes: 265214 Edges: 420045
> 
> FromNodeId	ToNodeId
> 
> 0	1
> 
> 0	4
> 
> 0	5
> 
> 0	8


# Usage

To run this code, please install the networkx package first.

Before running the code, you may edit the config file (see example "test.yaml"), customize the dataset path. To reproduce the result of SUGPT, you can set the "evaluate_mosso" and "evaluate_GSS" in the config file to "False", and ignore fields "mosso_outputfolder" and "mossoinputfolder".

The "H","D" and "T" fields have the meaning as illustated in the paper. H is the height of trie. D is the threshold to distinguish whether two vertices are close in a trie. T is the threshold for generating superedges.

You may run the code using the following cmd:

>python main.py --config test.yaml
