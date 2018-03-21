from .node import Node
import numpy as np

class Op(object):
  """Op represents operations performed on nodes."""
  def __call__(self):
    """Create a new node and associate the op object with the node.
    
    Returns
    -------
    The new node object.
    """
    new_node = Node()
    new_node.op = self
    return new_node

  def compute(self, node, input_vals):
    """Given values of input nodes, compute the output value.

    Parameters
    ----------
    node: node that performs the compute.
    input_vals: values of input nodes.

    Returns
    -------
    An output value of the node.
    """
    raise NotImplementedError

  def gradient(self, node, output_grad):
    """Given value of output gradient, compute gradient contributions to each input node.

    Parameters
    ----------
    node: node that performs the gradient.
    output_grad: value of output gradient summed from children nodes' contributions

    Returns
    -------
    A list of gradient contributions to each input node respectively.
    """
    raise NotImplementedError


class AddOp(Op):
  """Op to element-wise add two nodes."""
  def __call__(self, node_A, node_B):
    new_node = Op.__call__(self)
    new_node.inputs = [node_A, node_B]
    new_node.name = '({}+{})'.format(node_A.name, node_B.name)
    return new_node

  def compute(self, node, input_vals):
    """Given values of two input nodes, return result of element-wise addition."""
    assert len(input_vals) == 2
    return input_vals[0] + input_vals[1]

  def gradient(self, node, output_grad):
    """Given gradient of add node, return gradient contributions to each input."""
    return [output_grad, output_grad]
  
  def __str__(self):
    return 'Add_Op'
  
  __repr__ = __str__


class AddByConstOp(Op):
  """Op to element-wise add a node by a constant."""
  def __call__(self, node, const_val):
    new_node = Op.__call__(self)
    new_node.const_attr = const_val
    new_node.inputs = [node]
    new_node.name = '({}+{})'.format(node.name, str(const_val))
    return new_node

  def compute(self, node, input_vals):
    """Given values of input node, return result of element-wise addition."""
    assert len(input_vals) == 1
    return input_vals[0] + node.const_attr

  def gradient(self, node, output_grad):
    """Given gradient of add node, return gradient contribution to input."""
    return [output_grad]
  
  def __str__(self):
    return 'Add_Const_Op'
  
  __repr__ = __str__


class NegOp(Op):
  """Op to element-wise negate a node"""
  def __call__(self, node):
    new_node = Op.__call__(self)
    new_node.inputs = [node]
    new_node.name = '(-{})'.format(node.name)
    return new_node

  def compute(self, node, input_vals):
    """return the result of element-wise negative value"""
    assert len(input_vals) == 1
    return -input_vals[0]

  def gradient(self, node, output_grad):
    """Given gradient of the node, return gradient contribution to input."""
    return [-output_grad]

  def __str__(self):
    return 'Neg_Op'
  
  __repr__ = __str__


class MulOp(Op):
  """Op to element-wise multiply two nodes."""
  def __call__(self, node_A, node_B):
    new_node = Op.__call__(self)
    new_node.inputs = [node_A, node_B]
    new_node.name = '({}*{})'.format(node_A.name, node_B.name)
    return new_node

  def compute(self, node, input_vals):
    """Given values of two input nodes, return result of element-wise multiplication."""
    assert len(input_vals) == 2
    return input_vals[0] * input_vals[1]

  def gradient(self, node, output_grad):
    """Given gradient of multiply node, return gradient contributions to each input."""
    return [output_grad * node.inputs[1], output_grad * node.inputs[0]]

  def __str__(self):
    return 'Mul_Op'
  
  __repr__ = __str__


class MulByConstOp(Op):
  """Op to element-wise multiply a nodes by a constant."""
  def __call__(self, node, const_val):
    new_node = Op.__call__(self)
    new_node.const_attr = const_val
    new_node.inputs = [node]
    new_node.name = '({}*{})'.format(node.name, str(const_val))
    return new_node

  def compute(self, node, input_vals):
    """Given values of input node, return result of element-wise multiplication."""
    assert len(input_vals) == 1
    return input_vals[0] * node.const_attr

  def gradient(self, node, output_grad):
    """Given gradient of multiplication node, return gradient contribution to input."""
    return [output_grad * node.const_attr]

  def __str__(self):  
    return 'Mul_Const_Op'
  
  __repr__ = __str__


class ReciprOp(Op):
  """Op to element-wise recipr a node"""
  def __call__(self, node):
    new_node = Op.__call__(self)
    new_node.inputs = [node]
    new_node.name = '(1/{})'.format(node.name)
    return new_node
  
  def compute(self, node, input_vals):
    """return the result of element-wise recipr value"""
    assert len(input_vals) == 1
    return 1.0 / input_vals[0]

  def gradient(self, node, output_grad):
    """Given gradient of the node, return gradient contribution to input."""
    return [-output_grad / (node.inputs[0] * node.inputs[0])]

  def __str__(self):
    return 'Recipr_Op'
  
  __repr__ = __str__


class MatMulOp(Op):
  """Op to matrix multiply two nodes."""
  def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
    """Create a new node that is the result a matrix multiple of two input nodes.

    Parameters
    ----------
    node_A: lhs of matrix multiply
    node_B: rhs of matrix multiply
    trans_A: whether to transpose node_A
    trans_B: whether to transpose node_B

    Returns
    -------
    Returns a node that is the result a matrix multiple of two input nodes.
    """
    new_node = Op.__call__(self)
    new_node.matmul_attr_trans_A = trans_A
    new_node.matmul_attr_trans_B = trans_B
    new_node.inputs = [node_A, node_B]
    new_node.name = 'MatMul({},{},{},{})'.format(node_A.name, node_B.name, str(trans_A), str(trans_B))
    return new_node

  def compute(self, node, input_vals):
    """Given values of input nodes, return result of matrix multiplication."""
    assert len(input_vals) == 2
    lhs = input_vals[0].T if node.matmul_attr_trans_A else input_vals[0]
    rhs = input_vals[1].T if node.matmul_attr_trans_B else input_vals[1]
    return np.matmul(lhs, rhs)

  def gradient(self, node, output_grad):
    """Given gradient of multiply node, return gradient contributions to each input.
      
    Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
    """
    return [
      self.__call__(output_grad, node.inputs[1], False, True),
      self.__call__(node.inputs[0], output_grad, True, False),
    ]
  
  def __str__(self):
    return 'MatMul_Op'
  
  __repr__ = __str__


class PlaceholderOp(Op):
  """Op to feed value to a nodes."""
  def __call__(self, name=None):
    """Creates a variable node."""
    new_node = Op.__call__(self)
    new_node.name = name
    return new_node

  def compute(self, node, input_vals):
    """No compute function since node value is fed directly in Executor."""
    assert False, "placeholder values provided by feed_dict"

  def gradient(self, node, output_grad):
    """No gradient function since node has no inputs."""
    return None
  
  def __str__(self):
    return 'Placeholder_Op'
  
  __repr__ = __str__


class ZerosLikeOp(Op):
  """Op that represents a constant np.zeros_like."""
  def __call__(self, node_A):
    """Creates a node that represents a np.zeros array of same shape as node_A."""
    new_node = Op.__call__(self)
    new_node.inputs = [node_A]
    new_node.name = 'Zeroslike({})'.format(node_A.name)
    return new_node

  def compute(self, node, input_vals):
    """Returns zeros_like of the same shape as input."""
    assert(isinstance(input_vals[0], np.ndarray))
    return np.zeros(input_vals[0].shape)

  def gradient(self, node, output_grad):
    return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
  """Op that represents a constant np.ones_like."""
  def __call__(self, node_A):
    """Creates a node that represents a np.ones array of same shape as node_A."""
    new_node = Op.__call__(self)
    new_node.inputs = [node_A]
    new_node.name = 'Oneslike({})'.format(node_A.name)
    return new_node

  def compute(self, node, input_vals):
    """Returns ones_like of the same shape as input."""
    if isinstance(input_vals[0], np.ndarray):
      return np.ones(input_vals[0].shape)
    else:
      return np.ones(1)

  def gradient(self, node, output_grad):
    return [zeroslike_op(node.inputs[0])]


class ExpOp(Op):
  """Op that performs exp function"""
  def __call__(self, node):
    """Perform element-wise exp, return a new node"""
    new_node = Op.__call__(self)
    new_node.inputs = [node]
    new_node.name = 'Exp({})'.format(node.name)
    return new_node

  def compute(self, node, input_vals):
    """Return the exp value"""
    assert(len(input_vals) == 1)
    return np.exp(input_vals[0])
  
  def gradient(self, node, output_grad):
    return [output_grad * exp_op(node.inputs[0])]


class LogOp(Op):
  """Op that performs log function(base e)"""
  def __call__(self, node):
    new_node = Op.__call__(self)
    new_node.inputs = [node]
    new_node.name = 'Log({})'.format(node.name)
    return new_node

  def compute(self, node, input_vals):
    assert(len(input_vals) == 1)
    return np.log(input_vals[0])
  
  def gradient(self, node, output_grad):
    return [output_grad / node.inputs[0]]


class ReduceSumOp(Op):
  """Op that performs sum over all elements"""
  def __call__(self, node):
    new_node = Op.__call__(self)
    new_node.inputs = [node]
    new_node.name = 'ReduceSum({})'.format(node.name)
    return new_node
  
  def compute(self, node, input_vals):
    assert(len(input_vals) == 1)
    assert(isinstance(input_vals[0], np.ndarray))
    return np.sum(input_vals[0])
  
  def gradient(self, node, output_grad):
    """
    According to the autograd module in TensorFlow and PyTorch
    y = reduce_sum(x)
    dy / dx = ones_like(x)

    *in Tensorflow
    y = reduce_sum(x) + ones(z)
    dy / dx = ones_like(x) * z.shape[0]

    y = reduce_sum(x) * ones(z)
    dy / dx = ones_like(x) * reduce_sum(z)

    *in PyTorch
    it is not supported(report error when adding a scalar variable
    to a tensor)

    In machine learning, we usually use reduce_sum as the last operation
    therefore, here it only consider the gradient without further calc
    or plus a scalar value, for example, the regularization term
    """
    return [output_grad * oneslike_op(node.inputs[0])]


# Create global singletons of operators.
add_op = AddOp()
neg_op = NegOp()
recipr_op = ReciprOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
exp_op = ExpOp()
log_op = LogOp()
reduce_sum_op = ReduceSumOp()
