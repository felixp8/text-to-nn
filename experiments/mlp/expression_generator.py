import numpy as np

from numpy import square, abs, sqrt, sin, cos, exp, log
# cube = lambda x: x ** 3

### Constants defining expression possibilities
# Not using exponentiation (**) for now because many argument restrictions

OPERATORS = ['+', '-', '*', '/'] # '%', '**'
MODIFIERS = ['sin', 'cos', 'square', 'abs', 'exp', 'log', 'sqrt'] # 'exp', 'log', 'sqrt', 'cube'

OPERATOR_CONSTRAINTS = {
    '/': {
        'left': None, # anything can be numerator
        'right': [(-np.inf, -1e-2), (1e-2, np.inf)], # denominator != 0
    },
}

MODIFIER_CONSTRAINTS = {
    'exp': [(-np.inf, 4.)], # arbitrary cutoff to prevent overflow / large values
    'log': [(1e-2, np.inf)], # must be > 0
    'sqrt': [(1e-2, np.inf)], # must be >= 0
}

### Actual sampling function

def expr_sampler(num_inputs, depth_decay=0.6, var_decay=0.2, const_prob=0.3, modifier_prob=0.3):
    """Generates an arbitrary string of a expression/function of inputs.
    Essentially builds out a tree, with operators at the node between branches,
    and constants or input variables as leaves. Single-argument modifier functions
    (e.g. sin, cos, log) are inserted also at random
    
    Parameters
    ----------
    num_inputs : int
        number of inputs to the function
    depth_decay : float, default 0.6
        decay rate of probability of branching vs. leaf node
        with increasing depth
    var_decay : float, default 0.2
        decay rate of relative probability of each input variable
        after each occurrence. e.g. with var_decay = 0.2 and 
        num_inputs = 5, if input 'i1' is chosen once, its
        probability decreases from 1 / 5 to 0.2 / 4.2
    const_prob : float, default 0.3
        at a leaf node, the probability that a random constant
        is used instead of an input variable
    modifier_prob : float, default 0.3
        the probability that a given node will be input to
        a modifier function
    
    Returns
    -------
    string
        string with input variables 'i0', 'i1', ... that
        can be evaluated with python `eval()`
    """
    # just something to scale probabilities of each variable
    # so we get more variety
    assert num_inputs > 0
    rel_probs = np.ones(num_inputs)

    # over-coded helper func for enforcing certain argument constraints
    def enforce_constraint(expr, constraint=None):
        if constraint is None:
            return expr
        elif len(constraint) > 1:
            constraint = constraint[np.random.choice(len(constraint))]
        else:
            constraint = constraint[0]
        
        if not any([(f'i{n}' in expr) for n in range(num_inputs)]): # constant argument
            expr_val = eval(expr)
            if expr_val > constraint[0] and expr_val < constraint[1]:
                return expr
            else:
                if (np.inf in constraint):
                    expr = str(round(np.random.exponential() + constraint[0], 4))
                elif (-np.inf in constraint):
                    expr = str(round(constraint[1] - np.random.exponential(), 4))
                else: # TODO: make more reasonable ranges if this is ever necessary
                    expr = str(round(np.random.uniform(constraint[0], constraint[1]), 4))
            return expr

        if (np.inf in constraint) or (-np.inf in constraint): # one-sided
            modifier = np.random.choice(['abs', 'square']) # could also use exp but...
            expr = f'{modifier}({expr})'
            if (-np.inf in constraint):
                bound = round(constraint[1] - np.random.exponential(), 4)
                expr = f'(-{expr} + {bound})'
            else:
                bound = round(constraint[0] + np.random.exponential(), 4)
                expr = f'({expr} + {bound})'
        else:
            modifier = np.random.choice(['sin', 'cos'])
            expr = f'{modifier}({expr})'
            max_range = (constraint[1] - constraint[0])
            samp_range = round(max_range * np.random.uniform(0.2, 1.0), 4)
            center = round(np.random.uniform(
                constraint[0] + samp_range / 2 + 1e-2, 
                constraint[1] - samp_range / 2 - 1e-2), 4)
            expr = f'({samp_range} * {expr} + {center})'
        return expr

    # recursive function to build tree
    def build_expr_tree(probs, depth=0):
        if np.random.rand() < modifier_prob:
            modifier = np.random.choice(MODIFIERS)
        else:
            modifier = None

        if np.random.rand() < (depth_decay ** depth):
            root = np.random.choice(OPERATORS)
        else:
            if np.random.rand() < (1 - const_prob):
                root = np.random.choice(len(probs), p=probs / probs.sum())
                probs[root] *= var_decay
                expr = f'i{root}'
            else:
                root = round(np.random.normal(loc=0., scale=2.), 4) # constant
                expr = str(root)
            if modifier is not None:
                if MODIFIER_CONSTRAINTS.get(modifier, None):
                    expr = enforce_constraint(expr, MODIFIER_CONSTRAINTS.get(modifier))
                expr = f'{modifier}({expr})'
            return expr, probs

        if np.random.rand() < 0.5:
            left, probs = build_expr_tree(probs=probs, depth=depth+1)
            right, probs = build_expr_tree(probs=probs, depth=depth+1)
        else:
            right, probs = build_expr_tree(probs=probs, depth=depth+1)
            left, probs = build_expr_tree(probs=probs, depth=depth+1)

        if OPERATOR_CONSTRAINTS.get(root, None):
            constraint = OPERATOR_CONSTRAINTS.get(root)
            left = enforce_constraint(left, constraint.get('left', None))
            right = enforce_constraint(right, constraint.get('right', None))

        expr = f'({left} {root} {right})'
        if modifier is not None:
            if MODIFIER_CONSTRAINTS.get(modifier, None):
                expr = enforce_constraint(expr, MODIFIER_CONSTRAINTS.get(modifier))
                if not expr.startswith('('):
                    expr = f'({expr})'
            expr = f'{modifier}{expr}'
        return expr, probs

    expr = build_expr_tree(probs=rel_probs)[0]
    return expr

if __name__ == "__main__":
    i0 = np.random.normal(0., 3.)
    i1 = np.random.normal(0., 3.)
    i2 = np.random.normal(0., 3.)
    i3 = np.random.normal(0., 3.)
    i4 = np.random.normal(0., 3.)

    expr = expr_sampler(5, depth_decay=0.5, var_decay=0.05, const_prob=0.3)
    print(expr)
    print(eval(expr))