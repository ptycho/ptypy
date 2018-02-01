'''
useful utilities from ptypy that should be ported to gpu. These don't ahve external dependencies
'''
import numpy as np

def abs2(input):
    '''
    
    :param input. An array that we want to take the absolute value of and square. Can be inplace. Can be complex or real. 
    :return: The real valued abs**2 array
    '''
    
    return np.multiply(input, input.conj()).real


def sum_to_buffer(in1, outshape, in1_addr, out1_addr):
    '''
    :param in1. An array that we want to take the absolute value of and square. Can be inplace. Can be complex or real.
    :param outshape. An array that we want to take the absolute value of and square. Can be inplace. Can be complex or real.
    :param in1_addr. An array that we want to take the absolute value of and square. Can be inplace. Can be complex or real.
    :param out1_addr. An array that we want to take the absolute value of and square. Can be inplace. Can be complex or real.
    :return: The real valued abs**2 array
    '''
    out1 = np.zeros(outshape, dtype=in1.dtype)
    for i1, o1 in zip(in1_addr, out1_addr):
        out1[o1[0]] += in1[i1[0]]
    return out1
