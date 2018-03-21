from .utils import find_topo_sort, sum_node_list
from .op import oneslike_op

def gradients(output_node, node_list):
  """Take gradient of output node with respect to each node in node_list.

  Parameters
  ----------
  output_node: output node that we are taking derivative of.
  node_list: list of nodes that we are taking derivative wrt.

  Returns
  -------
  A list of gradient values, one for each node in node_list respectively.

  """

  # a map from node to a list of gradient contributions from each output node
  node_to_output_grads_list = {}
  # Special note on initializing gradient of output_node as oneslike_op(output_node):
  # We are really taking a derivative of the scalar reduce_sum(output_node)
  # instead of the vector output_node. But this is the common case for loss function.
  node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
  # a map from node to the gradient of that node
  node_to_output_grad = {}
  # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
  reverse_topo_order = reversed(find_topo_sort([output_node]))
  for node in reverse_topo_order:
    if node in node_to_output_grads_list:
      # numeric value of the gradient of the current node
      node_to_output_grad[node] = sum_node_list(node_to_output_grads_list[node])
      # create the grad node and pass the numeric value to it
      grads = node.op.gradient(node, node_to_output_grad[node])
      for idx, input_node in enumerate(node.inputs):
        if input_node not in node_to_output_grads_list:
          node_to_output_grads_list[input_node] = []
        node_to_output_grads_list[input_node].append(grads[idx])

  # Collect results for gradients requested.
  grad_node_list = [node_to_output_grad[node] for node in node_list]
  return grad_node_list
