import numpy as np
import pandas as pd
import h5py

from numpy import square, abs, sqrt, sin, cos, exp, log

from datamodule import validate_expr


def map_inputs(expr, num_inputs, target_names):
    """rename inputs from i0, etc. to something else, e.g. x, y, z"""
    assert len(target_names) == num_inputs
    for n in range(num_inputs):
        expr = expr.replace(f'i{n}', target_names[n])
    return expr


def build_paren_list(expr):
    """parse out parentheses"""
    if '(' not in expr and ')' not in expr:
        return expr
    paren_list = []
    paren_stack = []
    for i in range(len(expr)):
        char = expr[i]
        if char == '(':
            paren_stack.append(len(paren_list))
            paren_list.append([i])
        elif char == ')':
            paren_idx = paren_stack.pop(-1)
            paren_list[paren_idx].append(i)
    return paren_list


def reformat_square(expr):
    """replace square with ^2 for readability"""
    if 'square' in expr:
        paren_list = build_paren_list(expr)
        start_idx = expr.find('square')
        square_paren = None
        for paren in paren_list:
            if paren[0] == start_idx + 6:
                square_paren = paren
                break
        if square_paren is None:
            raise ValueError("no parens found associated with square")
        end_idx = square_paren[-1]
        pre_str = expr[:start_idx]
        post_str = expr[end_idx+1:]
        arg = expr[square_paren[0]:square_paren[-1]+1]
        expr = pre_str + f'{arg}^2' + reformat_square(post_str)
    return expr


def clean_expr(expr):
    # main things that are done:
    # 1) removing extra parentheses at very beginning and end of expression
    # 2) removing extra double parentheses , e.g. ((...)) -> (...)
    # 3) changing instances of square(...) to (...)^2, as it is presumably more common notation?

    # 1)
    paren_list = build_paren_list(expr)
    if [0, len(expr) - 1] in paren_list:
        expr = expr[1:-1]
    
    # 2)
    paren_list = np.array(build_paren_list(expr))
    extra = np.all((paren_list[:, None, :] - paren_list[None, :, :]) == np.array([1, -1])[None, None, :], axis=-1)
    remove = np.nonzero(extra)[0]
    if len(remove) > 0:
        remove_idx = np.sort(paren_list[remove, :].flatten())[::-1]
        for idx in remove_idx:
            expr = expr[:idx] + expr[idx+1:]
    
    # 3)
    expr = reformat_square(expr)
        
    return expr
    
