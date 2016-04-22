#!/usr/bin/python
'''testing spot finding routines'''

import multiprocessing
import time
import os

#chunk = 67
n_cpu = 4 
template = '../cbf/aav8-3_3_%05d.cbf'
first = 1
last = 200


def get_single_wedge(start, images_in_wedge):
    return [start + j for j in range(images_in_wedge)]
        
def get_wedges(start, nimages, n_cpu):
    iterations, rest = divmod(nimages, n_cpu)
    wedges = []
    for i in range(iterations):
        wedges.append(get_single_wedge(start+i*n_cpu, n_cpu))
    if rest:
        wedges.append(get_single_wedge(start+iterations*n_cpu, rest))
    return wedges

#def get_wedges(first, last, chunk):
    #wedges = []
    #starts = range(first, last, chunk)
    #ends = range(chunk, last+1, chunk)
    #print starts
    #print ends
    #for (s, e) in zip(starts, ends):
        #wedges.append(range(s, e+1))
    #return wedges

def execute_line(wedge, k ):
    line = 'dials.find_spots %s' % (' '.join([template]*len(wedge)))
    line = line % (tuple(wedge))
    line += ' nproc=1'
    #print line
    #return
    pdir = 'sf_%s' % k
    try:
        os.mkdir(pdir)
    except:
        pass
    os.chdir(pdir)
    os.system(line)
    print line
    os.chdir('../')

wedges = get_wedges(first, last, int((1+last-first)/n_cpu))
print wedges

start = time.time()
jobs = []
for k, wedge in enumerate(wedges): 
    job = multiprocessing.Process(target=execute_line, args=(wedge, k))
    jobs.append(job)
    job.start()
for job in jobs:
    job.join()

end = time.time()

print 'total processing time %6.4f s, which is %6.4f s per image' % (end-start, (end-start)/(last-first))
        
