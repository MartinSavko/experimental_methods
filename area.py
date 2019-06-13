#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import copy

def shift(vertical_shift, horizontal_shift):
    s = np.array([[1., 0.,    vertical_shift], 
                  [0., 1.,  horizontal_shift], 
                  [0., 0.,               1.]])
    return s

def scale(vertical_scale, horizontal_scale):
    s = np.diag([vertical_scale, horizontal_scale, 1.])
    return s

class area:
    def __init__(self, range_y=1, range_x=1, rows=3, columns=1, center_y=0, center_x=0):
        self.range_y = range_y
        self.range_x = range_x
        self.rows = rows
        self.columns = columns
        self.center_x = center_x
        self.center_y = center_y
        
        self.start_y = center_y - range_y/2.
        self.start_x = center_x - range_x/2.
        self.end_y = center_y + range_y/2.
        self.end_x = center_x + range_x/2.
        
        start = np.array([self.start_y, self.start_x])
        end = np.array([self.end_y, self.end_x])
        self.extent = end - start
        self.center = start + self.extent/2.
        self.shape = (self.rows, self.columns)
        
    def get_grid_and_points(self):
        vertical = np.linspace(0, 1, self.rows)
        horizontal = np.linspace(0, 1, self.columns)

        positions = itertools.product(vertical, horizontal)
        points = np.array([np.array(position) for position in positions])
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        
        points = np.dot(shift(-0.5, -0.5), points.T).T
        points = np.dot(scale(*self.extent), points.T).T
        points = np.dot(shift(*self.center), points.T).T

        points[:, 2] += np.arange(points.shape[0])
        
        indexes = map(int, points[:, 2])
        points = dict(zip(indexes, points[:, :2]))
        
        grid = np.reshape(indexes, (self.rows, self.columns))
        
        return grid, points
    
    def get_position_sequence(self, grid):
        return grid.ravel()
    
    def get_jump_sequence(self, grid, against_gravity=False):
        if against_gravity:
            jump_sequence = zip(grid[:, -1], grid[:, 0])
        else:
            jump_sequence = zip(grid[:, 0], grid[:, -1])
        return jump_sequence
        
    def get_horizontal_raster(self, grid, start=1):
        g = copy.deepcopy(grid)
        if np.__version__ >= '1.8.3':
            g[start::2, ::] = g[start::2, ::-1]
        else:
            raster = []
            for k, line in enumerate(g):
                if k % 2 == 1:
                    line = line[::-1]
                raster.append(line)
            g = np.array(raster)
        return g
        
    def get_vertical_raster(self, grid, start=1):
        g = copy.deepcopy(grid)
        if np.__version__ >= '1.8.3':
            g[::, start::2] = g[::-1, start::2]
        else:
            g = self.get_horizontal_raster(g.T).T
        return g
   
    def get_linearized_point_jumps(self, jumps, points):
        return [(points[jump[0]], points[jump[1]]) for jump in jumps]
    
def test():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-y', '--range_y', default=1, type=float, help='Vertical scan range')
    parser.add_option('-x', '--range_x', default=1, type=float, help='Horizontal scan range')
    parser.add_option('-r', '--rows', default=5, type=int, help='Number of rows')
    parser.add_option('-c', '--columns', default=3, type=int, help='Number of columns')
    parser.add_option('-a', '--center_y', default=1, type=float, help='Vertical origin')
    parser.add_option('-b', '--center_x', default=1, type=float, help='Horizontal origin')
    
    options, args = parser.parse_args()
    
    a = area(**vars(options))
    
    grid, points =  a.get_grid_and_points()
    print 'grid'
    print grid
    
    print 'grid.T'
    print grid.T
    
    print 'points'
    print points
    vr = a.get_vertical_raster(grid)
    hr = a.get_horizontal_raster(grid)
    print 'vertical raster'
    print vr
    print 'vertical raster linearized positions'
    print a.get_position_sequence(vr.T)
    print 'vertical raster linearized jumps'
    vertical_jumps = a.get_jump_sequence(vr.T)
    print vertical_jumps
    print a.get_linearized_point_jumps(vertical_jumps, points)
    
    print 'horizontal raster'
    print hr
    print 'horizontal raster linearized positions'
    print a.get_position_sequence(hr)
    print 'horizontal raster linearized jumps'
    horizontal_jumps = a.get_jump_sequence(hr)
    print horizontal_jumps
    print a.get_linearized_point_jumps(horizontal_jumps, points)
    
def main():
    import sys
    
    if len(sys.argv) > 0:
        what = sys.argv[1]
    else:
        print 'Please specify what should I aling'
        
if __name__ == '__main__':
    test()