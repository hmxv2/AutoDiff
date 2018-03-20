from .utils import find_topo_sort

class Executor:
  """Executor computes values for a given subset of nodes in a computation graph.""" 
  def __init__(self, eval_node_list):
    """
    Parameters
    ----------
    eval_node_list: list of nodes whose values need to be computed.
    """
    self.eval_node_list = eval_node_list

  def run(self, feed_dict):
    """Computes values of nodes in eval_node_list given computation graph.
    Parameters
    ----------
    feed_dict: list of variable nodes whose values are supplied by user.

    Returns
    -------
    A list of values for nodes in eval_node_list. 
    """
    node_to_val_map = dict(feed_dict)
    # Traverse graph in topological sort order and compute values for all nodes.
    topo_order = find_topo_sort(self.eval_node_list)
    for node in topo_order:
      if node not in node_to_val_map:
        node_to_val_map[node] = node.op.compute(node, [node_to_val_map[x] for x in node.inputs])

    # Collect node values.
    node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
    return node_val_results
