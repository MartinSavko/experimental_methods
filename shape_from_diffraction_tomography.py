#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import zmq
import time
import pickle
import copy
import numpy as np
import open3d as o3d
import pylab
import scipy.ndimage as ndi
from goniometer import get_points_in_goniometer_frame, get_origin, get_voxel_calibration, get_distance, get_reduced_point
from diffraction_tomography import diffraction_tomography
from skimage.measure import regionprops
from scipy.spatial import distance_matrix
from reconstruct import principal_axes

def get_reconstruction(request, port=8900, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print('Connecting to server ...')
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://localhost:%d' % port)
    socket.send(pickle.dumps(request))
    reconstruction = pickle.loads(socket.recv())
    if verbose:
        print('Received reconstruction in %.4f seconds' % (time.time() - start))
    return reconstruction

def get_calibration(vertical_step_size, horizontal_step_size):
    calibration = np.ones((3,))
    calibration[0] = horizontal_step_size
    calibration[1: ] = vertical_step_size
    return calibration

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('-d', '--directory', default='/nfs/data2/excenter/2023-04-25T12:19:45.158775', type=str, help='directory')
    parser.add_argument('-d', '--directory', default='/nfs/data2/excenter/2023-04-15T17:22:01.257118', type=str, help='directory')
    parser.add_argument('-n', '--name_pattern', default='excenter', type=str, help='name_pattern')
    parser.add_argument('-m', '--min_spots', default=7, type=int, help='min_spots')
    parser.add_argument('-t', '--threshold', default=0.125, type=float, help='threshold')
    parser.add_argument('-D', '--display', action='store_true', help='Display')
    parser.add_argument('-M', '--method', default='xds', type=str, help='Analysis method')
    parser.add_argument('-r', '--ratio', default=5, type=int, help='Horizonta/Vertical step size ratio')
    parser.add_argument('-o', '--horizontal_beam_size', default=0.01, type=float, help='horizontal beam size')
    parser.add_argument('-R', '--detector_row_spacing', default=1, type=int, help='detector vertical pixel size')
    parser.add_argument('-C', '--detector_col_spacing', default=1, type=int, help='detector horizontal pixel size')
    args = parser.parse_args()
    print('args', args)
    
    dt = diffraction_tomography(directory=args.directory,
                                name_pattern=args.name_pattern)
    
    parameters = dt.get_parameters()
    print('parameters')
    for p in ['scan_start_angles', 'ntrigger', 'nimages']:
        print(parameters[p])
    
    detector_rows = parameters['nimages']
    detector_cols = int(args.horizontal_beam_size/parameters['vertical_step_size']) #args.ratio
    
    pcd_filename = os.path.join('%s_%s.pcd' % (dt.get_template(), args.method))
    obj_filename = os.path.join('%s_%s_raddose3d.obj' % (dt.get_template(), args.method))
    img_filename = os.path.join('%s_%s_reconstruction2d.jpg' % (dt.get_template(), args.method))
    txt_filename = os.path.join('%s_%s.txt' % (dt.get_template(), args.method))
    results_filename = os.path.join('%s_%s.results' % (dt.get_template(), args.method))
    
    if args.method == 'xds':
        xds_results = dt.get_xds_results()
        results = xds_results[:].astype('float') # dozor_results[:]
    else:
        dozor_results = dt.get_dozor_results()
        results = dozor_results[:,1].astype('float')
        
    results[results<args.min_spots] = 0.
    results[results<results.mean()] = 0.
    
    reference_position = dt.get_reference_position()
    result_position = dt.get_result_position()
    
    if args.display and False:
        pylab.figure(2, figsize=(16, 9))
        pylab.plot(results)
        pylab.show()
        try:
            pylab.figure(1, figsize=(16, 9))
            pylab.grid(0)
            pylab.imshow(parameters['rgbimage'])
        except:
            pass
    results /= results.max()
    raw_projections = []
    for k in range(parameters['ntrigger']):
        k_start = int(k*parameters['nimages'])
        k_end = int((k+1)*parameters['nimages'])
        line = results[k_start: k_end][::-1]
        line[line<=line.max()*args.threshold] = 0
        projection = np.zeros((detector_rows, detector_cols)) + np.reshape(line, (detector_rows, 1))
        raw_projections.append(projection)
    
    try:
        angles = np.deg2rad(parameters['scan_start_angles'])
    except:
        angles = []
        for k in range(parameters['ntrigger']):
            angles.append(np.deg2rad(k*90))
    
    print('angles', np.rad2deg(angles))

    calibration = get_voxel_calibration(parameters['vertical_step_size'], args.horizontal_beam_size/detector_cols)
    origin = get_origin(parameters, position_key='reference_position')
    
    number_of_projections = len(angles)
    projections = np.zeros((detector_cols, number_of_projections, detector_rows))
    
    for k in range(len(raw_projections)):
        projection = raw_projections[k]
        projection[projection>0] = 1
        projections[:, k, :] = projection.T 
    
    center_of_mass = ndi.center_of_mass(projections)
    print('projections shape, center_of_mass', projections.shape, center_of_mass, projections.max(), projections.mean())
    
    #vertical_correction = 0.
    vertical_correction = result_position['AlignmentZ'] - reference_position['AlignmentZ']
    vertical_correction /= calibration[-1]
    #if not np.isnan(center_of_mass[2]):
        #vertical_correction =  - (center_of_mass[2] - detector_rows/2)
    #else:
        #vertical_correction = 0.
    print('vertical_correction', vertical_correction)
    
    request = {'projections': projections,
               'angles': angles,
               'detector_rows': detector_rows,
               'detector_cols': detector_cols,
               'detector_col_spacing': args.detector_col_spacing,
               'detector_row_spacing': args.detector_row_spacing,
               'vertical_correction': vertical_correction}
    
    reconstruction = get_reconstruction(request, verbose=True)
    reconstruction_thresholded = reconstruction>0.95*reconstruction.max()
    reconstruction_2d = np.mean(reconstruction_thresholded, axis=0) > 0
    analysis_results = {}
    try:
        props = regionprops(reconstruction_2d.astype(np.uint8))[0]
        print('props', props)
        for prop in ['centroid', 'area', 'orientation', 'axis_major_length', 'axis_minor_length']:
            print(prop, props[prop])
            analysis_results[prop] = props[prop]
    except IndexError:
        for prop in ['centroid', 'area', 'orientation', 'axis_major_length', 'axis_minor_length']:
            analysis_results[prop] = 0.
    
    centroid = np.array(analysis_results['centroid'])[::-1]
    orientation = analysis_results['orientation']
    amaxl = analysis_results['axis_major_length']
    aminl = analysis_results['axis_minor_length']
    
    #R = np.array([[np.cos(orientation), np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    #amaxp = np.dot(R, np.array([-amaxl/2, 0])) + np.array(centroid).T
    #aminp = np.dot(R, np.array([0, aminl/2])) + np.array(centroid).T
                          
    amaxp = np.array([-np.sin(orientation) * amaxl, -np.cos(orientation) * amaxl])/2 + centroid
    aminp = np.array([np.cos(orientation) * aminl, -np.sin(orientation) * aminl])/2 + centroid
    
    maxl = np.vstack([centroid, amaxp])
    minl = np.vstack([centroid, aminp])
    
    print('reconstruction', reconstruction.shape, reconstruction.max(), reconstruction.mean())
    objectpoints = np.argwhere(reconstruction_thresholded)
    analysis_results['volume_voxels'] = len(objectpoints)
    voxel_mm3 = np.prod(calibration)
    analysis_results['voxel_mm^3'] = voxel_mm3
    analysis_results['volume_mm^3'] = len(objectpoints) * voxel_mm3
    analysis_results['axis_major_length_mm'] = analysis_results['axis_major_length'] * np.mean(calibration[1:])
    analysis_results['axis_minor_length_mm'] = analysis_results['axis_minor_length'] * np.mean(calibration[1:])
    
    print('#objectpoints', len(objectpoints))
    print('objectpoints.shape', objectpoints.shape)
    #print('objectpoints[:10]', objectpoints[:10])
    #>args.threshold*reconstruction.max())
    
    #center = np.array(reconstruction.shape)/2 #np.array([detector_cols/2, detector_rows/2, detector_rows/2])
    center = np.array([2., centroid[1], centroid[0]])
    analysis_results['calibration'] = calibration
    analysis_results['origin'] = origin
    analysis_results['center'] = center
    analysis_results['reference_position'] = reference_position
    analysis_results['result_position'] = result_position
    
    #center[2] += vertical_correction
    print('center', center)
    print('calibration', calibration)
    print('origin', origin)
    objectpoints_mm = get_points_in_goniometer_frame(objectpoints, calibration, origin[:3], center=center, directions=np.array([-1, 1, 1])) 
    #positive_pixel is negative_movement for CentringX
    #negative_pixel is negative_movement for CentringY
    print('result_position (cx, cy, ay)')
    print(result_position['CentringX'], result_position['CentringY'], result_position['AlignmentY'])
    print('objectpoints_px mean')
    print(np.mean(objectpoints, axis=0))
    print('objectpoints_mm median')
    print(np.median(objectpoints_mm, axis=0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(objectpoints_mm)
    pcd.estimate_normals()
    o3d.io.write_point_cloud(pcd_filename, pcd)
    
    try:
        pcd_rd3_points = objectpoints_mm[:, [0, 2, 1]]
        pcd_rd3_points -= pcd_rd3_points.mean(axis=0) 
        pcd_rd3_points *= 1000
        pcd_rd3 = o3d.geometry.PointCloud()
        pcd_rd3.points = o3d.utility.Vector3dVector(pcd_rd3_points)
        rd3_mesh, rd3_points = pcd_rd3.compute_convex_hull()
        rd3_mesh.compute_vertex_normals()
        rd3_mesh.compute_triangle_normals()
        o3d.io.write_triangle_mesh(obj_filename, rd3_mesh)
        
        hull_mesh, hull_points = pcd.compute_convex_hull()
        hull_mesh.compute_triangle_normals()
        hull_mesh.compute_vertex_normals()
        hull_points_mm = objectpoints_mm[hull_points]
        hull_points_px = objectpoints[hull_points]
    except:
        hull_points_mm = objectpoints_mm
        hull_points_px = objectpoints
        
    dm = distance_matrix(hull_points_mm, hull_points_mm)
    print('max dm', np.max(dm))
    
    extreme_points = np.unravel_index(np.argmax(dm), dm.shape)
    print('argmax dm', extreme_points)
    
    point1_mm = hull_points_mm[extreme_points[0]]
    point2_mm = hull_points_mm[extreme_points[1]]
    point1_px = hull_points_px[extreme_points[0]]
    point2_px = hull_points_px[extreme_points[1]]
   
    print('objectpoints extreme mm ', point1_mm, point2_mm)
    print('objectpoints extreme px ', point1_px, point2_px)
    
    ep1 = copy.copy(result_position)
    ep1['CentringX'], ep1['CentringY'], ep1['AlignmentY'] = point1_mm
        
    ep2 = copy.copy(result_position)
    ep2['CentringX'], ep2['CentringY'], ep2['AlignmentY'] = point2_mm
    analysis_results['extreme_points'] = [ep1, ep2]
   
    pca = principal_axes(reconstruction_2d, verbose=True)
    pca_center = pca[-1]
    pca_s = pca[-2]
    pca_e = pca[1]
    print('pca_s', pca_s)
    print('pca_e, sqrt(pca_e)', np.round(pca_e, 3), np.round(np.sqrt(pca_e), 3))
    
    try:
        print('ratio of eig1/eig2 %.3f, sqrt(eig1/eig2) %.3f' % (pca_e[0]/pca_e[1], np.sqrt(pca_e[0]/pca_e[1])))
        print('ratio of major and minor axes %.3f' % (amaxl/aminl))
        #amaxp = np.array([-np.sin(orientation) * amaxl, -np.cos(orientation) * amaxl])/2 + centroid
        #amaxp_0p95_shift_px = 0.9*np.dot(R , np.array([-amaxl/2, 0]))
        #amaxp_0p95_shift_px = np.dot(pca_s.T, np.array([[amaxl, 0], [0, aminl]]))
    except:
        pass
    #amaxp_0p95_shift_px = 0.9*amaxl*np.array([-np.sin(orientation), -np.cos(orientation)])/2
    #print('amaxp_0p95_shift_px from major', amaxp_0p95_shift_px)
    
    #amaxp_0p95_a_px = centroid + amaxp_0p95_shift_px
    #amaxp_0p95_b_px = centroid - amaxp_0p95_shift_px
    
    #print('centroid from major', centroid)
    #print('amaxp_0p95_a_px from major', amaxp_0p95_a_px)
    #print('amaxp_0p95_b_px from maror', amaxp_0p95_b_px)
    
    
    amaxp_0p95_shift_px = 0.45 * pca_s.T[0,:] * amaxl
    print('amaxp_0p95_shift_px from pca', amaxp_0p95_shift_px)
    
    amaxp_0p95_a_px = pca_center + amaxp_0p95_shift_px # (pca_e[0]/np.sum(pca_e))*100
    amaxp_0p95_b_px = pca_center - amaxp_0p95_shift_px # (pca_e[0]/np.sum(pca_e))*100
    
    print('centroid from pca', pca_center)
    print('amaxp_0p95_a_px from pca', amaxp_0p95_a_px)
    print('amaxp_0p95_b_px from pca', amaxp_0p95_b_px)
    
    #positive_pixel is negative_movement for CentringX
    #negative_pixel is negative_movement for CentringY
    
    amaxp_0p95_shift_mm = amaxp_0p95_shift_px * calibration[1:] - 0.005
    print('amaxp_0p95_shift_mm', amaxp_0p95_shift_mm)
    
    pca_point_a = copy.copy(result_position)
    pca_point_b = copy.copy(result_position)
    
    pca_point_a['CentringX'] += -amaxp_0p95_shift_mm[0]
    pca_point_a['CentringY'] += amaxp_0p95_shift_mm[1]
    
    pca_point_b['CentringX'] -= -amaxp_0p95_shift_mm[0]
    pca_point_b['CentringY'] -= amaxp_0p95_shift_mm[1]
    analysis_results['pca_points'] = [pca_point_a, pca_point_b]
    
    
    print('extreme_points')
    print([get_reduced_point(point, keys=['CentringX', 'CentringY']) for point in analysis_results['extreme_points']])
    print('get_distance(extreme_points)', get_distance(*analysis_results['extreme_points']))
    print('pca points')
    print([get_reduced_point(point, keys=['CentringX', 'CentringY']) for point in analysis_results['pca_points']])
    print('get_distance(pca_points)', get_distance(*analysis_results['pca_points']))
    print('distances extreme vs pca')
    print(get_distance(analysis_results['extreme_points'][0], analysis_results['pca_points'][0]))
    print(get_distance(analysis_results['extreme_points'][1], analysis_results['pca_points'][1]))
    print(get_distance(analysis_results['extreme_points'][1], analysis_results['pca_points'][0]))
    print(get_distance(analysis_results['extreme_points'][0], analysis_results['pca_points'][1]))
    

    #reconstruction_2d_pca = np.dot(reconstruction_2d, pca_s)
    
    f = open(results_filename, 'wb')
    pickle.dump(analysis_results, f)
    f.close()
    
    f = open(txt_filename, 'w')
    for key in analysis_results:
        #if type(analysis_results[key]) is float:
        v = analysis_results[key]
        print('%s, %s, type %s' % (key, v, type(v)))
        try:
            if type(v) is int or key == 'area':
                ark = '%d' % v
            elif abs(v) > 1e-9:
                ark = '%.9f' % v
            else:
                ark = v
        except:
            ark = v
        f.write('%s: %s\n' % (key, ark))
    f.close()
    
    pylab.figure(figsize=(16, 9))
    pylab.imshow(reconstruction_2d, label='reconstruction mean')
    pylab.grid(False)
    pylab.title('vertical_correction plus %s' % args.name_pattern)
    
    ax = pylab.gca()
    c = pylab.Circle(centroid, radius=3, color='red')
    e1 = pylab.Circle(point1_px[::-1], radius=2)
    e2 = pylab.Circle(point2_px[::-1], radius=2)
    aa = pylab.Circle(amaxp_0p95_a_px[::-1], radius=2, color='cyan')
    ab = pylab.Circle(amaxp_0p95_b_px[::-1], radius=2, color='cyan')
    pca_c = pylab.Circle(pca_center[::-1], radius=2, color='green')

    p1 = pca_center + amaxp_0p95_shift_px # pca_s[0,:].T*(pca_e[0]/np.sum(pca_e))*100
    p2 = pca_center + 0.45*pca_s.T[1,:]*aminl # pca_s[1,:].T*(pca_e[1]/np.sum(pca_e))*100
    pylab.plot([pca_center[1], p1[1]], [pca_center[0], p1[0]], label='eig1')
    pylab.plot([pca_center[1], p2[1]], [pca_center[0], p2[0]], label='eig2')

    ax.add_patch(c)
    ax.add_patch(pca_c)
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.add_patch(aa)
    ax.add_patch(ab)
    pylab.plot(maxl[:,0], maxl[:,1], label='axis_major')
    pylab.plot(minl[:,0], minl[:,1], label='axis_minor')
    pylab.legend()
    
    pylab.savefig(img_filename)
    #pylab.figure()
    #pylab.imshow(reconstruction_2d_pca, 'in own coordinates')
    if args.display:
        pylab.show()
        
    if args.display:
        pcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([pcd], window_name='reconstructed crystal volume')
    
if __name__ == '__main__':
    main()
