import networkx as nx
from collections import deque

def node_ordering(graph, method, node_feat_name, last_order=[]):
    """
    Return the ordering of the input graph without modifying it.
    Some graphs may not have contiunous node ids.
        For example, a graph may have node ids:
        {'2', '9', '10'} (unordered set).   节点id不连续
        graph.nodes() could return different ordering
        if g is loaded again from the disk,
        but once it is loaded, graph.nodes() would be consistent.
        The returned ordering in this case could be
        [2, 0, 1],
        denoting '10' --> '2' --> '9'
        if graph.nodes() == ['2', '9', '10'],
        and the returned mapping could be
        {'10': '2', '2': '9', '9': '10'}.
        To sum up, ordering is about integer indexing into
        g.nodes(), while mapping is about relabeling nodes
        regardless of the randomness of g.nodes().
    :param graph:
    :param method: 'bfs', 'degree', None.
    :param node_feat_name:
    :param last_order: List of node hashes in desired ending order or empty list. Ordering
                       ignores any matches in graph.nodes() that match last_order, then puts any
                       nodes that match last_order at the end of the returned ordering. Use to
                       force any nodes to be at the end of the ordering (specifically
                       supersource node).
    :return: ordering is a list of integers that can be used to
             reorder the N by D node input matrix or the N by N adj matrix
             by self.dense_node_inputs[self.order, :] or
             self.adj[np.ix_(self.order, self.order)].
             mapping is a dict mapping original node id --> new node id.  # 原节点id到新节点id的map
    """
    # Holds the nodes to append at the end, so make sure that they're valid nodes.
    sort_ignore_nodes = [node for node in last_order if node in graph.nodes()]   # sort_ignore_nodes =  last_order = []
    # Get the ordered sequence with highest degree first. We presort the nodes so that the input
    # is stable before doing other sorts because
    nodes = _sorted_nodes_based_on_node_deg_and_types(sorted(graph.nodes()), graph, node_feat_name)   # 基于节点的度对节点排序  aids700nef node_feat_name='type'
    if method == 'bfs':
        seq = [node for node in _bfs_seq(graph, nodes[0], node_feat_name, sort_ignore_nodes)]   # 节点 从 nodes[0]开始做广度优先搜索
    elif method == 'degree':
        seq = [node for node in nodes]   # 按度排列节点
    else:
        raise RuntimeError('Unknown ordering method {}'.format(method))
    origin_seq = [node for node in graph.nodes()]   # 原始的节点id
    ordering = []   # 
    for e in seq:   # 按bfs排序 的节点e 在origin_seq中的位置  如果5是bfs的第3个及诶按
        ordering.append(origin_seq.index(e))
    orig_nodes = graph.nodes()  # has randomness :(
    mapping = {orig_nodes[orig]: sorted(orig_nodes)[final]
               for final, orig in enumerate(ordering)}
    return ordering, mapping

def _sorted_nodes_based_on_node_deg_and_types(nodes, graph, node_feat_name):
    # For DiGraphs (supersource or similar), we want focus on out_degree, so a node with no
    # out_degree is at the end.
    if type(graph) == nx.Graph:
        degree_fn = graph.degree
    elif type(graph) == nx.DiGraph:
        degree_fn = graph.out_degree
    else:
        raise RuntimeError('Unidentified input graph type: {}'.format(type(graph)))
    # Sorting is based on 1. highest degree, 2. type (optional) 3. node.
    if node_feat_name:
        types = nx.get_node_attributes(graph, node_feat_name)
        decorated = [(degree_fn()[node], types[node], node)
                     for node in nodes]
        decorated.sort(key=lambda k: (-k[0], k[1], k[2]))
        return [node for deg, type, node in decorated]
    else:
        decorated = [(degree_fn()[node], node) for node in nodes]
        decorated.sort(key=lambda k: (-k[0], k[1]))
        return [node for deg, node in decorated]


def _bfs_seq(graph, start_id, node_feat_name, sort_ignore_nodes=[]):
    # sort_ignore_nodes is a list of nodes that should be ignored during the bfs search. We use
    # this as a special check for nodes to ignore that will be appended at the end of the ordering.
    dictionary = dict(_stable_bfs_successors(graph, start_id, sort_ignore_nodes))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbors = dictionary.get(current)
            if neighbors is not None:
                neighbors = _sorted_nodes_based_on_node_types(neighbors, graph,
                                                              node_feat_name)
                next += neighbors
        output += next
        start = next
    # Put the ignored nodes in order at the end.
    output.extend(sort_ignore_nodes)
    assert len(output) == len(graph.nodes()), 'Mismatch graph nodes and output, something is ' \
                                              'probably wrong with the BFS successors with an ' \
                                              'ignored node.'
    return output


def _stable_bfs_successors(graph, start_id, sort_ignore_nodes=[]):
    # Replacement for nx.bfs_successors(). Provides stable bfs ordering by checking all
    # neighbors first, then adding relevant ones to the queue, but adding them sorted by id.
    queue = deque()
    seen_nodes = set()
    bfs_successors = {}

    # Init with the start node.
    seen_nodes.add(start_id)
    queue.append(start_id)

    while queue:
        current = queue.popleft()
        bfs_successors[current] = []
        neighbors = graph.neighbors(current)
        # Sort the neighbors first, then add them if they haven't been seen yet.
        for neighbor in sorted(neighbors):
            if neighbor in sort_ignore_nodes:
                seen_nodes.add(neighbor)
            if neighbor not in seen_nodes:
                bfs_successors[current].append(neighbor)
                seen_nodes.add(neighbor)
                queue.append(neighbor)
    # Make sure we've seen all the nodes. This is an assert check for disconnected graphs since
    # they are not currently handled.
    assert(len(seen_nodes) == len(graph.nodes()))
    return bfs_successors

def _sorted_nodes_based_on_node_types(nodes, graph, node_feat_name):
    if node_feat_name:
        types = nx.get_node_attributes(graph, node_feat_name)
        decorated = [(types[node], node) for node in nodes]
        decorated.sort()
        return [node for type, node in decorated]
    else:
        return nodes