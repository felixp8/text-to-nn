# TODO: transition everything to sympy!!!!

import math
import numpy as np

from numpy import square, abs, sqrt, sin, cos, exp, log
cube = lambda x: x ** 3

### Constants defining expression possibilities
# Not using exponentiation (**) for now because many argument restrictions

OPERATORS = ['+', '-', '*', '/'] # '%', '**'
MODIFIERS = ['sin', 'cos', 'square', 'exp', 'log', 'sqrt', 'cube'] # 'abs', 

OPERATOR_PROBS = [0.3, 0.3, 0.3, 0.1] # division is sort of a pain in the butt
MODIFIER_PROBS = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

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

CONST_ROUNDING = 0

### Actual sampling function

def expr_sampler(
    num_inputs, 
    depth_decay=0.6, 
    var_decay=0.2, 
    const_prob=0.3, 
    modifier_prob=0.3,
    leaf_prob=0.0,
    operators=OPERATORS,
    modifiers=MODIFIERS,
    operator_probs=OPERATOR_PROBS,
    modifier_probs=MODIFIER_PROBS,
    const_rounding=CONST_ROUNDING,
):
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

    if const_rounding == 0:
        round_const = lambda x: int(round(x))
    else:
        round_const = lambda x: round(x, const_rounding)

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
                    expr = str(round_const(np.random.exponential() + math.ceil(constraint[0])))
                elif (-np.inf in constraint):
                    expr = str(round_const(math.floor(constraint[1]) - np.random.exponential()))
                else: # TODO: make more reasonable ranges if this is ever necessary
                    expr = str(round_const(np.random.uniform(math.ceil(constraint[0]), math.floor(constraint[1]))))
            return expr

        if (np.inf in constraint) or (-np.inf in constraint): # one-sided
            modifier = np.random.choice(['square']) # could also use exp but...
            expr = f'{modifier}({expr})'
            if (-np.inf in constraint):
                if np.random.rand() < const_prob:
                    bound = round_const(math.floor(constraint[1]) - np.random.exponential())
                    expr = f'(-{expr} + {bound})'
                else:
                    expr = f'(-{expr})'
            else:
                if np.random.rand() < const_prob:
                    bound = round_const(math.ceil(constraint[0]) + np.random.exponential())
                    expr = f'({expr} + {bound})'
                else:
                    expr = f'({expr})'
        else:
            modifier = np.random.choice(['sin', 'cos'])
            expr = f'{modifier}({expr})'
            max_range = (math.floor(constraint[1]) - math.ceil(constraint[0]))
            samp_range = round_const(max_range * np.random.uniform(0.2, 1.0))
            center = round_const(np.random.uniform(
                math.ceil(constraint[0] + samp_range / 2 + 1e-2), 
                math.floor(constraint[1] - samp_range / 2 - 1e-2)))
            expr = f'({samp_range} * {expr} + {center})'
        return expr

    # recursive function to build tree
    def build_expr_tree(probs, depth=0):
        if probs.sum() < 1e-6:
            if np.random.rand() < (1 - const_prob):
                return None, probs

        if np.random.rand() < modifier_prob:
            modifier = np.random.choice(modifiers, p=modifier_probs)
        else:
            modifier = None

        if np.random.rand() < ((1 - leaf_prob) * (depth_decay ** depth)):
            root = np.random.choice(operators, p=operator_probs)
        else:
            if np.random.rand() < (1 - const_prob) and probs.sum() > 1e-6:
                root = np.random.choice(len(probs), p=probs / probs.sum())
                probs[root] *= var_decay
                expr = f'i{root}'
            else:
                root = round(np.random.uniform(low=0., high=6.), CONST_ROUNDING) # constant
                root *= np.sign(np.random.normal())
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

        if (left is None) and (right is None):
            return None, probs
        elif (left is None):
            expr = f'{right}'
        elif (right is None):
            expr = f'{left}'
        else:
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

    expr = expr_sampler(
        num_inputs=3, 
        depth_decay=0.3, 
        var_decay=0.0, 
        const_prob=0.0,
        modifier_prob=0.0,
        leaf_prob=0.05,
        operators=["+", "-", "*"],
        operator_probs=[1/3, 1/3, 1/3],
    )
    print(expr)
    print(eval(expr))