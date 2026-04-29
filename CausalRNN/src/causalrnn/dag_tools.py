import numpy as np
import networkx as nx
import itertools, string

def sample_random_dag(nodes, p_edge=0.5):
    '''
    Generate a random DAG over the specified nodes by creating an upper-triangular
    adjacency matrix and then randomly permuting the node labels.
    '''
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Upper-triangular adjacency matrix to ensure acyclicity
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p_edge:
                adj_matrix[i, j] = 1

    # Create graph from the adjacency matrix (nodes 0, 1, ..., n-1)
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Randomly permute the node labels
    permuted_labels = np.random.permutation(nodes)
    mapping = {i: permuted_labels[i] for i in range(num_nodes)}

    # Relabel nodes to the permuted labels
    G = nx.relabel_nodes(G, mapping)

    return G

def sample_multiple_experiences_from_dag(G, n_experiences=100, p_spont=0.2, p_cause=0.8, n_samples=75, intersample_interval=5,
                                         hold_time=0):
    return np.array([
        sample_experience_from_dag(G, p_spont=p_spont, p_cause=p_cause, n_samples=n_samples, 
                                   intersample_interval=intersample_interval, hold_time=hold_time)
        for _ in range(n_experiences)
    ])

def sample_experience_from_dag(G, p_spont=0.2, p_cause=0.8, n_samples=75, intersample_interval=5, hold_time=0):
    seq_len = (n_samples - 1) * intersample_interval + 1
    nodes = list(G.nodes)
    n_nodes = len(nodes)

    full_duration = seq_len + hold_time
    
    data = {}
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        
        if not parents:
            # If a node has no parents, it activates only from the leak probability
            p_on = np.full(n_samples, p_spont)
        else:
            # Node has parents
            parent_activity = np.stack([data[p] for p in parents], axis=1)
            num_active_parents = parent_activity.sum(axis=1)
            prob_failure_from_causes = (1 - p_cause) ** num_active_parents
            prob_node_off = (1 - p_spont) * prob_failure_from_causes
            p_on = 1 - prob_node_off
        data[node] = np.random.binomial(1, p_on)

    # node_activity = np.stack([data[key] for key in data.keys()], axis=1)  # shape: (n_samples, 3)
    node_activity = np.stack([data[n] for n in nodes], axis=1)

    experience = np.zeros((full_duration, n_nodes), dtype=np.float32)
    for i in range(n_samples):
        t = i * intersample_interval
        if t >= seq_len:
            break
        experience[t] = node_activity[i]
    return experience.T

def instantiate_dag(nodes, adj_matrix):
    return nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph, nodelist=nodes)

def is_acyclic(adj_matrix):
    '''
    Check if a directed graph (given by an adjacency matrix) is acyclic
    by attempting a NetworkX topological sort.
    '''
    A = np.asarray(adj_matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('adj_matrix must be a square 2D array')

    G = nx.from_numpy_array(A.astype(bool), create_using=nx.DiGraph)

    try:
        order = list(nx.topological_sort(G))
        return True
    except nx.NetworkXUnfeasible:
        return False

def enumerate_dags(n_nodes):
    '''
    Enumerates all labeled directed acyclic graphs (DAGs) for a given number of nodes.
    '''
    assert n_nodes <= 26, 'Only up to 26 nodes supported with A-Z labels.'
    nodes = list(string.ascii_uppercase[:n_nodes])
    idx = {lab: i for i, lab in enumerate(nodes)}

    # List all possible directed edges, excluding self-loops
    possible_edges = list(itertools.permutations(nodes, 2))
    labeled_dags = []
    
    # Iterate over all possible numbers of edges (from 0 to the max possible)
    for r in range(len(possible_edges) + 1):
        # Find all combinations of edges for the current size r
        for edges in itertools.combinations(possible_edges, r):
            adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
            for u, v in edges:
                adj_matrix[idx[u], idx[v]] = 1
            
            # If the graph is acyclic, add its edge list to our collection
            if is_acyclic(adj_matrix):
                labeled_dags.append(instantiate_dag(nodes, adj_matrix))
                
    return labeled_dags

def empty_dag(nodes):
    '''
    Return an empty DAG (no edges) over the specified node labels.
    '''
    G = nx.DiGraph()
    G.add_nodes_from(list(nodes))
    return G

def complete_dag_from_order(nodes_ordered):
    '''
    Return a complete DAG consistent with a provided topological order.

    If nodes_ordered = [A, B, C], edges are A->B, A->C, B->C.
    '''
    order = list(nodes_ordered)
    n = len(order)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = 1
    return instantiate_dag(order, adj)

def complete_dag_fwd(nodes):
    '''
    Complete DAG consistent with the given node order.
    '''
    return complete_dag_from_order(nodes)

def complete_dag_rev(nodes):
    '''
    Complete DAG consistent with the reverse of the given node order.
    '''
    return complete_dag_from_order(list(reversed(list(nodes))))


def single_edge_dag(nodes, source, target):
    '''
    DAG with exactly one directed edge: source -> target.
    All nodes are included.
    '''
    nodes = list(nodes)
    if source == target:
        raise ValueError('source and target must be different')
    if source not in nodes or target not in nodes:
        raise ValueError('source and target must be in nodes')

    n = len(nodes)
    idx = {lab: i for i, lab in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=int)
    adj[idx[source], idx[target]] = 1
    return instantiate_dag(nodes, adj)

def single_edge_dag_from_query(nodes, query):
    '''
    Convenience wrapper for queries represented as a tuple/list (source, target).
    '''
    if len(query) != 2:
        raise ValueError('query must be (source, target)')
    return single_edge_dag(nodes, query[0], query[1])

def dedupe_dags_by_adjacency(dags, nodes):
    '''
    Deduplicate DAGs by adjacency matrix (with a fixed node ordering).
    '''
    unique = []
    seen = set()
    nodelist = list(nodes)
    for G in dags:
        A = nx.to_numpy_array(G, nodelist=nodelist, dtype=np.uint8)
        key = A.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(G)

    return unique

def seed_dags_true_false_plus_singletons(nodes):
    '''
    Returns the DAG set:
      - empty DAG
      - complete DAG (forward order)
      - complete DAG (reverse order)
      - single-edge DAG for every ordered query (a->b), a != b

    For n=2, we remove duplicates (complete DAGs equal single-edge DAGs).
    '''
    nodes = list(nodes)
    dags = [empty_dag(nodes), complete_dag_fwd(nodes), complete_dag_rev(nodes)]
    queries = [(a, b) for a in nodes for b in nodes if a != b]
    dags.extend([single_edge_dag(nodes, a, b) for (a, b) in queries])
    dags_deduped = dedupe_dags_by_adjacency(dags, nodes)
    return dags_deduped