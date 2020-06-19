#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import pylab
import numpy as np
import os

def from_number_sequence_to_character_sequence(number_sequence, separator=';'):
    character_sequence = ''
    number_strings = [str(n) for n in number_sequence]
    return separator.join(number_strings)

def merge_two_overlapping_character_sequences(seq1, seq2, alignment_length=175):
    start = seq1.index(seq2[:alignment_length])
    print 'start', start
    return seq1[:start] + seq2
    
def from_character_sequence_to_number_sequence(character_sequence, separator=';'):
    return map(float, character_sequence.split(';'))
    
def merge_two_overlapping_number_sequences(r1, r2, alignment_length=175):
    print 'r1'
    print len(r1)
    print 'r2'
    print len(r2)
    c1 = from_number_sequence_to_character_sequence(r1)
    c2 = from_number_sequence_to_character_sequence(r2)
    c = merge_two_overlapping_character_sequences(c1, c2, alignment_length)
    
    r = from_character_sequence_to_number_sequence(c)
    return r
    
def test():
    r1 = range(10)
    r2 = range(7, 31)
    
    print merge_two_overlapping_number_sequences(r1, r2, alignment_length=5)
    
def main():
    directory  = '/nfs/ruche/proxima2a-spool/2017_Run4/2017-09-07/com-proxima2a/RAW_DATA/Commissioning/undulator/full_beam'
    filename = 'gap_8.7b_results.pickle'
    
    r = pickle.load(open(os.path.join(directory, filename)))
    
    calibrated_diode = r['calibrated_diode']['observations']
    
    complete_observation =calibrated_diode[0][1]
    #print 'start', complete_observation
    for k, observation in enumerate(calibrated_diode[1:]):
        print k
        complete_observation = merge_two_overlapping_number_sequences(complete_observation, observation[1])
        
    print len(complete_observation)
    pylab.plot(complete_observation, '-')
    pylab.show()
    
if __name__ == '__main__':
    main()