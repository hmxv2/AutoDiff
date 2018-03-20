class Node(object):
  """Node in a computation graph."""
  def __init__(self):
    """Constructor, new node is indirectly created by Op object __call__ method.
      
      Instance variables
      ------------------
      self.inputs: the list of input nodes.
      self.op: the associated op object, 
        e.g. add_op object if this node is created by adding two other nodes.
      self.const_attr: the add or multiply constant,
        e.g. self.const_attr=5 if this node is created by x+5.
      self.name: node name for debugging purposes.
    """
    self.inputs = []
    self.op = None
    self.const_attr = None
    self.name = ""

  def __add__(self, other):
    """Adding two nodes return a new node."""
    from .op import add_op, add_byconst_op
    if isinstance(other, Node):
      new_node = add_op(self, other)
    else:
      # Add by a constant stores the constant in the new node's const_attr field.
      # 'other' argument is a constant
      new_node = add_byconst_op(self, other)
    return new_node

  def __mul__(self, other):
    """Multiply two nodes return a new node."""
    from .op import mul_op, mul_byconst_op
    if isinstance(other, Node):
      new_node = mul_op(self, other)
    else:
      # Multiply by a constant stores the constant in the new node's const_attr field.
      # 'other' argument is a constant
      new_node = mul_byconst_op(self, other)
    return new_node


  # Allow left-hand-side add and multiply.
  __radd__ = __add__
  __rmul__ = __mul__

  def __str__(self):
    """Allow print to display node name.""" 
    return self.name

  __repr__ = __str__


def Variable(name):
  """User defined variables in an expression.  
    e.g. x = Variable(name = "x")
  """
  from .op import placeholder_op
  placeholder_node = placeholder_op()
  placeholder_node.name = name
  return placeholder_node