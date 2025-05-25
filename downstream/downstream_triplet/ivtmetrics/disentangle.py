#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation triplet component filtering .
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
(c) camma, icube, unistra
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np

#%%%%%%%%% COMPONENT FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Disentangle(object):
    """
    Class: filter a triplet prediction into the components (such as instrument i, verb v, target t, instrument-verb iv, instrument-target it, etc)    
    @args
    ----
        url: str. path to the dictionary map file of the dataset decomposition labels            
    @params
    ----------
    bank :   2D array
        holds the dictionary mapping of all components    
    @methods
    ----------
    extract(input, componet): 
        call filter a component labels from the inputs labels     
    """

    def __init__(self, url="maps.txt"):
        self.bank = self.map_file()
#        self.bank = np.genfromtxt(url, dtype=int, comments='#', delimiter=',', skip_header=0)
        
    def decompose(self, inputs, component):
        """ Extract the component labels from the triplets.
            @args:
                inputs: a 1D vector of dimension (n), where n = number of triplet classes;
                        with values int(0 or 1) for target labels and float[0, 1] for predicted labels.
                component: a string for the component to extract; 
                        (e.g.: i for instrument, v for verb, t for target, iv for instrument-verb pair, it for instrument-target pair and vt (unused) for verb-target pair)
            @return:
                output: int or float sparse encoding 1D vector of dimension (n), where n = number of component's classes.
        """
        txt2id = {'ivt':0, 'i':1, 'v':2, 't':3, 'iv':4, 'it':5, 'vt':6} 
        key    = txt2id[component]
        index  = sorted(np.unique(self.bank[:,key]))
        output = []
        for idx in index:
            same_class  = [i for i,x in enumerate(self.bank[:,key]) if x==idx]
            y           = np.max(np.array(inputs[same_class]))
            output.append(y)        
        return output
    
    def extract(self, inputs, component="i"):
        """
        Extract a component label from the triplet label
        @args
        ----
        inputs: 2D array,
            triplet labels, either predicted label or the groundtruth
        component: str,
            the symbol of the component to extract, choose from
            i: instrument
            v: verb
            t: target
            iv: instrument-verb
            it: instrument-target
            vt: verb-target (not useful)
        @return
        ------
        label: 2D array,
            filtered component's labels of the same shape and data type as the inputs
        """      
        if component == "ivt":
            return inputs
        else:
            component = [component]* len(inputs)
            return np.array(list(map(self.decompose, inputs, component)))

    def map_file(self):
        return np.array([ 
                       [0,0,0,0,0,0],
                       [1,0,0,1,0,1],
                       [2,0,0,2,0,2],
                       [3,0,0,3,0,3],
                       [4,0,0,4,0,4],
                       [5,0,0,5,0,5],
                       [6,0,0,6,0,6],
                       [7,0,1,0,1,0],
                       [8,0,1,2,1,2],
                       [9,0,1,3,1,3],
                       [10,0,1,4,1,4],
                       [11,0,2,0,2,0],
                       [12,0,2,1,2,1],
                       [13,0,2,2,2,2],
                       [14,0,2,3,2,3],
                       [15,0,2,4,2,4],
                       [16,0,2,7,2,7],
                       [17,0,3,0,3,0],
                       [18,0,3,2,3,2],
                       [19,0,3,3,3,3],
                       [20,0,3,4,3,4],
                       [21,0,9,9,4,8],
                       [22,1,0,0,5,9],
                       [23,1,0,1,5,10],
                       [24,1,0,2,5,11],
                       [25,1,0,3,5,12],
                       [26,1,0,4,5,13],
                       [27,1,0,6,5,15],
                       [28,1,1,0,6,9],
                       [29,1,1,2,6,11],
                       [30,1,1,3,6,12],
                       [31,1,1,4,6,13],
                       [32,1,3,2,7,11],
                       [33,1,3,3,7,12],
                       [34,1,3,4,7,13],
                       [35,1,4,1,8,10],
                       [36,1,4,2,8,11],
                       [37,1,4,3,8,12],
                       [38,1,4,4,8,13],
                       [39,1,4,5,8,14],
                       [40,1,4,6,8,15],
                       [41,1,4,7,8,16],
                       [42,1,6,0,9,9],
                       [43,1,6,3,9,12],
                       [44,1,6,4,9,13],
                       [45,1,9,9,10,17],
                       [46,2,0,0,11,18],
                       [47,2,0,1,11,19],
                       [48,2,0,2,11,20],
                       [49,2,0,3,11,21],
                       [50,2,0,4,11,22],
                       [51,2,0,6,11,23],
                       [52,2,7,8,12,24],
                       [53,2,9,9,13,25],
                       [54,3,0,0,14,26],
                       [55,3,0,3,14,28],
                       [56,3,0,4,14,29],
                       [57,3,4,0,15,26],
                       [58,3,4,1,15,27],
                       [59,3,4,4,15,29],
                       [60,3,4,5,15,30],
                       [61,3,4,6,15,31],
                       [62,3,4,7,15,32],
                       [63,3,6,0,16,26],
                       [64,3,6,3,16,28],
                       [65,3,6,4,16,29],
                       [66,3,9,9,17,33],
                       [67,4,0,0,18,34],
                       [68,4,0,1,18,35],
                       [69,4,0,2,18,36],
                       [70,4,0,3,18,37],
                       [71,4,0,4,18,38],
                       [72,4,4,1,19,35],
                       [73,4,4,2,19,36],
                       [74,4,4,3,19,37],
                       [75,4,4,4,19,38],
                       [76,4,4,5,19,39],
                       [77,4,4,6,19,40],
                       [78,4,4,7,19,41],
                       [79,4,9,9,20,42],
                       [80,5,8,0,21,43],
                       [81,5,8,2,21,44],
                       [82,5,8,3,21,45],
                       [83,5,8,4,21,46],
                       [84,5,8,6,21,47],
                       [85,5,9,9,22,48],
                       [86,6,5,3,23,49],
                       [87,6,5,4,23,50],
                       [88,6,9,9,24,51]
                       ])