#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
_total_start = time.time()

import os
import zmq
import h5py
import simplejpeg
import traceback
import pickle
import numpy as np
import open3d as o3d
import random
import pylab
import scipy.ndimage as ndi
from scipy.optimize import minimize
from optical_path_report import circle_model_residual, projection_model_residual, select_better_model, create_mosaic, circle_model, projection_model, circle_projection_model
from goniometer import get_points_in_goniometer_frame, get_voxel_calibration, get_position_from_vector, get_vector_from_position
from camera import camera
cam = camera()
print('all imports done in %.3f seconds' % (time.time() - _total_start))

def get_initial_parameters(aspect, name=None):
    c = np.mean(aspect)
    try:
        r = 0.5 * (max(aspect) - min(aspect))
    except:
        traceback.print_exc()
        print('name', name)
        print(aspect)
        try:
            r = np.std(aspect)/np.sin(np.pi/4)
        except:
            r = 0.
    alpha = np.random.rand()*np.pi
    return c, r, alpha

def get_notion_string(notion):
    if type(notion) is list:
        notion_string = ','.join(notion)
    else:
        notion_string = notion
    return notion_string

def principal_axes(array, verbose=False):
    #https://github.com/pierrepo/principal_axes/blob/master/principal_axes.py
    _start = time.time()
    if array.shape[1] != 3:
        xyz = np.argwhere(array==1)
    else:
        xyz = array[:, :]
    
    coord = np.array(xyz, float)
    center = np.mean(coord, 0)
    coord = coord - center
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    order = np.argsort(e_values)[::-1]
    eigenvalues = np.array(e_values[order])
    eigenvectors = np.array(e_vectors[:, order])
    _end = time.time()
    if verbose:
        print('principal axes')
        print('intertia tensor')
        print(inertia)
        print('eigenvalues')
        print(eigenvalues)
        print('eigenvectors')
        print(eigenvectors)
        print('principal_axes calculated in %.3f seconds' % (_end-_start))
        print()
    return inertia, eigenvalues, eigenvectors, center

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
        print('Received reconstruction in %.3f seconds' % (time.time() - start))
    return reconstruction

def get_predictions(request, port=8901, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print('Connecting to server ...')
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://localhost:%d' % port)
    socket.send(pickle.dumps(request))
    predictions = pickle.loads(socket.recv())
    if verbose:
        print('Received predictions in %.3f seconds' % (time.time() - start))
    return predictions

def get_images(history_master, as_jpegs=True):
    images = [jpeg for jpeg in history_master['history_images'][()]]
    if not as_jpegs:
        images = np.array([simplejpeg.decode_jpeg(jpeg) for jpeg in images])
    return images

def get_angles(history_master):
    return np.deg2rad(history_master['history_state_vectors'][()][:, 0])

def get_zooms(history_master):
    return history_master['history_state_vectors'][()][:, 9]

def get_positions(history_master):
    return history_master['history_state_vectors'][()][:,0:8]

def get_origin(master):
    #motor_names=['Omega', 'Kappa', 'Phi', 'CentringX', 'CentringY', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'ScintillatorVertical', 'Zoom']
    origin = np.median(get_positions(master)[:,[3,4,6,7]], axis=0)
    return origin
    
def get_raw_projections(predictions, notion='foreground', notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'foreground': 5}, threshold=0.5, min_size=32):
    raw_projections = []
    for k in range(len(predictions[notion_indices[notion]])):
        present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, notion_mask = get_notion_prediction(predictions, notion, k=k)
        if present:
            raw_projections.append((present, (r, c, h, w), notion_mask))
    return raw_projections

def get_calibration(master):
    return get_voxel_calibration(*cam.get_calibration(zoomposition=np.median(get_zooms(master))))

def get_origin(master):
    #motor_names=['Omega', 'Kappa', 'Phi', 'CentringX', 'CentringY', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'ScintillatorVertical', 'Zoom']
    origin = np.median(get_positions(master)[:,[3,4,6,7]], axis=0)
    return origin

def get_kappa_phi(master):
    kappa_phi = np.median(get_positions(master)[:,[1, 2]], axis=0)
    return kappa_phi

def fit_aspect(angles, aspect, aspect_name=None, minimize_method='nelder-mead', debug=False):
    initial_parameters = get_initial_parameters(aspect, name=aspect_name)
                                                
    fit_circle = minimize(circle_model_residual, 
                            initial_parameters, 
                            method=minimize_method, 
                            args=(angles, aspect))
    
    fit_projection = minimize(projection_model_residual, 
                                initial_parameters, 
                                method=minimize_method, 
                                args=(angles, aspect))
    
    if debug:
        print('aspect', aspect, 'initial_parameters', initial_parameters, 'optimized_parameters circle, projection', fit_circle.x, fit_projection.x)
    
    fit, k = select_better_model(fit_circle, fit_projection)
    
    result = {'fit_circle': fit_circle, 
              'fit_projection': fit_projection, 
              'fit': fit, 
              'k': k}
    
    if k == 1:
        result['best_model'] = circle_model
    else:
        result['best_model'] = projection_model
        
    return result

def get_extreme_point(projection, orientation='horizontal', extreme_direction=1):
    xyz = np.argwhere(projection != 0)
    
    pa = principal_axes(projection)
    
    S = pa[-2]
    center = pa[-1]
    
    xyz_0 = xyz - center
    
    xyz_S = np.dot(xyz_0, S)
    xyz_S_on_axis = xyz_S[np.isclose(xyz_S[:, 1], 0, atol=1)]
    
    mino = xyz_S[np.argmin(xyz_S[:, 0])]
    mino_on_axis = xyz_S_on_axis[np.argmin(xyz_S_on_axis[:, 0])]
    maxo = xyz_S[np.argmax(xyz_S[:, 0])]
    maxo_on_axis = xyz_S_on_axis[np.argmax(xyz_S_on_axis[:, 0])]
    
    mino_0_s = np.dot(mino, np.linalg.inv(S)) + center
    maxo_0_s = np.dot(maxo, np.linalg.inv(S)) + center

    mino_0_s_on_axis = np.dot(mino_on_axis, np.linalg.inv(S)) + center
    maxo_0_s_on_axis = np.dot(maxo_on_axis, np.linalg.inv(S)) + center
    
    if orientation == 'horizontal':
        if extreme_direction*mino_0_s[1] > extreme_direction*maxo_0_s[1]:
            extreme_point_out = mino_0_s
            extreme_point_out_on_axis = mino_0_s_on_axis
            extreme_point_ini = maxo_0_s
            extreme_point_ini_on_axis = maxo_0_s_on_axis
        else:
            extreme_point_out = maxo_0_s
            extreme_point_out_on_axis = maxo_0_s_on_axis
            extreme_point_ini = mino_0_s
            extreme_point_ini_on_axis = mino_0_s_on_axis
    else:
        if extreme_direction*mino_0_s[0] > extreme_direction*maxo_0_s[0]:
            extreme_point_out = mino_0_s
            extreme_point_out_on_axis = mino_0_s_on_axis
            extreme_point_ini = maxo_0_s
            extreme_point_ini_on_axis = maxo_0_s_on_axis
        else:
            extreme_point_out = maxo_0_s
            extreme_point_out_on_axis = maxo_0_s_on_axis
            extreme_point_ini = mino_0_s
            extreme_point_ini_on_axis = mino_0_s_on_axis
    return extreme_point_out, extreme_point_ini, extreme_point_out_on_axis, extreme_point_ini_on_axis, pa
    
def get_analysis(predictions, notions, angles, minimize_method='nelder-mead', debug=False):

    _start = time.time()
    
    analysis = {}
    analysis['original_image_shape'] = predictions['original_image_shape']
    analysis['angles'] = angles
    detector_rows, detector_cols = predictions['descriptions'][0]['prediction_shape']
    number_of_projections = len(angles)
    analysis['detector_rows'] = detector_rows
    analysis['detector_cols'] = detector_cols
    analysis['number_of_projections'] = number_of_projections
    
    for notion in notions:
        notion_string = get_notion_string(notion)
        analysis[notion_string] = get_notion_analysis(predictions, notion, angles, minimize_method=minimize_method, debug=debug)
        
    return analysis
    
def get_notion_analysis(predictions, notion, angles, minimize_method='nelder-mead', debug=False):
    print('notion', notion)
    notion_string = get_notion_string(notion)
    
    notion_analysis = {}
    descriptions = predictions['descriptions'][notion_string]
    valid_centroids = []
    valid_angles = []
    principal_axes = []
    extreme_points_out = []
    extreme_points_ini = []
    extreme_points_out_on_axis = []
    extreme_points_ini_on_axis = []
    
    for description, angle in zip(descriptions, angles):
        if description['present']:
            valid_angles.append(angle)
            epo = description['epo']
            epi = description['epi']
            epooa = description['epooa']
            epioa = description['epioa']
            pa = description['pa']
            principal_axes.append(pa)
            extreme_points_out.append(epo)
            extreme_points_ini.append(epi)
            extreme_points_out_on_axis.append(epooa)
            extreme_points_ini_on_axis.append(epioa)
            bbox = description['r'], description['c'], description['h'], description['w'], description['area']
            valid_centroids.append(bbox)
        
    notion_analysis['principal_axes'] = principal_axes
    extreme_points_out = np.array(extreme_points_out)
    extreme_points_ini = np.array(extreme_points_ini)
    extreme_points_out_on_axis = np.array(extreme_points_out_on_axis)
    extreme_points_ini_on_axis = np.array(extreme_points_ini_on_axis)            
    centroids = np.array(valid_centroids)
    centroid_verticals = centroids[:, 0]
    centroid_horizontals = centroids[:, 1]
    heights = centroids[:, 2]
    widths = centroids[:, 3]
    areas = centroids[:, 4]
    extreme_verticals = extreme_points_out[:, 0]
    extreme_horizontals = extreme_points_out[:, 1]
    extreme_ini_verticals = extreme_points_ini[:, 0]
    extreme_ini_horizontals = extreme_points_ini[:, 1]
    extreme_out_on_axis_verticals = extreme_points_out_on_axis[:, 0]
    extreme_out_on_axis_horizontals = extreme_points_out_on_axis[:, 1]
    extreme_ini_on_axis_verticals = extreme_points_ini_on_axis[:, 0]
    extreme_ini_on_axis_horizontals = extreme_points_ini_on_axis[:, 1]
    
    fit_start = time.time()
    fits = {}
    
    fits['aspects'] = {'heights': heights, 
                       'widths': widths, 
                       'areas': areas,
                       'centroid_verticals': centroid_verticals, 
                       'centroid_horizontals': centroid_horizontals, 
                       'extreme_verticals': extreme_verticals, 
                       'extreme_horizontals': extreme_horizontals, 
                       'extreme_ini_verticals': extreme_ini_verticals, 
                       'extreme_ini_horizontals': extreme_ini_horizontals, 
                       'extreme_out_on_axis_verticals': extreme_out_on_axis_verticals,
                       'extreme_out_on_axis_horizontals': extreme_out_on_axis_horizontals,
                       'extreme_ini_on_axis_verticals': extreme_ini_on_axis_verticals,
                       'extreme_ini_on_axis_horizontals': extreme_ini_on_axis_horizontals}
    
    notion_analysis['valid_angles'] = valid_angles
    fits['results'] = {}
    
    for aspect in fits['aspects']:
        fits['results'][aspect] = fit_aspect(valid_angles, fits['aspects'][aspect], minimize_method=minimize_method, debug=debug)
        
    fit_end = time.time()
    print('Fit took %.3f seconds' % (fit_end - fit_start))
    
    notion_analysis['fits'] = fits
    
    
    if debug and notion=='loop':
        for k in range(7):
            i = np.random.randint(0, len(angles))
            pylab.imshow(descriptions[i]['notion_mask'])
            pylab.grid(False)
            print('i: %d' % i)
            print('angle %.2f' % angles[i])
            print('extreme_point_out', extreme_points_out[i])
            print('extreme_point_ini', extreme_points_ini[i])
            center = principal_axes[i][-1]
            s = principal_axes[i][-2]
            e = principal_axes[i][1]
            p1 = center + s[0,:]*(e[0]/np.sum(e))*100
            p2 = center + s[1,:]*(e[1]/np.sum(e))*100
            pylab.plot([center[1], p1[1]], [center[0], p1[0]], label=e[0])
            pylab.plot([center[1], p2[1]], [center[0], p2[0]], label=e[1])
            pylab.plot(extreme_points_out[i][1], extreme_points_out[i][0], 'o', color='red', label='extreme_out')
            pylab.plot(extreme_points_ini[i][1], extreme_points_ini[i][0], 'o', color='magenta', label='extreme_ini')
            pylab.plot(extreme_points_out_on_axis[i][1], extreme_points_out_on_axis[i][0], 'o', color='blue', label='extreme_out_on_axis')
            pylab.plot(extreme_points_ini_on_axis[i][1], extreme_points_ini_on_axis[i][0], 'o', color='green', label='extreme_ini_on_axis')
            pylab.legend()
            pylab.show()
            
    if debug:
        test_angles = np.linspace(0, 2*np.pi, 360)
        for k, aspect in enumerate(fits['aspects']):
            angles = fits['angles']
            data = fits['aspect'][aspect]
            model = fits['results'][aspect]['best_model'](test_angles, *fits['results'][aspect]['fit'].x)
            pylab.figure(k+1, figsize=(8, 6))
            pylab.title('%s %s' % (notion, aspect))
            pylab.plot(np.rad2deg(angles), data, 'go', label='%s' % aspect)
            pylab.plot(np.rad2deg(test_angles), model, 'g-', label='%s fit' % aspect)
            pylab.xlabel('Omega [deg]')
        pylab.legend()
        pylab.show()

    return notion_analysis


def get_volume(notion, descriptions, analysis, origin, calibration, original_image_shape=(1200, 1600), detector_col_spacing=1, detector_row_spacing=1):
    
    print('notion', notion)
    notion_string = get_notion_string(notion)
    
    notion_analysis = analysis[notion_string]
    descriptions = descriptions[notion_string]
    angles = analysis['angles']
    detector_rows = analysis['detector_rows']
    detector_cols = analysis['detector_cols']
    number_of_projections = analysis['number_of_projections']
    
    projections = np.zeros((detector_cols, number_of_projections, detector_rows))
    valid_angles = []
    valid_index = 0
    print('%d projections' % len(descriptions))
    for description, angle in zip(descriptions, angles):
        if description['present']:
            projections[:, valid_index, :] = description['notion_mask'].T
            valid_angles.append(angle)
            valid_index += 1
    projections = projections[:, :valid_index, :]
    
    center_of_mass = ndi.center_of_mass(projections)
    print('projections shape, center_of_mass', projections.shape, center_of_mass, projections.max(), projections.mean())
    
    vertical_correction = detector_rows/2 - notion_analysis['fits']['results']['extreme_verticals']['fit'].x[0]
    
    _start = time.time()
    request = { 'projections': projections,
                'angles': valid_angles,
                'detector_rows': detector_rows,
                'detector_cols': detector_cols,
                'detector_col_spacing': detector_col_spacing,
                'detector_row_spacing': detector_row_spacing,
                'vertical_correction': vertical_correction}
    
    reconstruction = get_reconstruction(request, verbose=True)
    _end = time.time()
    print('reconstruction done in %.3f seconds (%.4f from start)' % (_end-_start, _end-_total_start))
    
    _start = time.time()
    print('reconstruction', reconstruction.shape, reconstruction.max(), reconstruction.mean())
    objectpoints = np.argwhere(reconstruction>0.95*reconstruction.max()) 
    print('#objectpoints', len(objectpoints))
    print('objectpoints.shape', objectpoints.shape)
    
    cols_ratio = original_image_shape[1]/detector_cols
    rows_ratio = original_image_shape[0]/detector_rows
    calibration[0] *= cols_ratio
    calibration[1:] *= rows_ratio
    
    reference_position = get_position_from_vector(origin, keys=['CentringX', 'CentringY', 'AlignmentY', 'AlignmentZ'])
    print('reference_position', reference_position)
    #center = np.array([detector_cols/2, detector_rows/2, detector_rows/2])
    center = np.array([reconstruction.shape[0]/2] + list(np.mean(objectpoints, axis=0)[1:])) #np.array(reconstruction.shape)/2
    print('center', center)

    objectpoints_mm = get_points_in_goniometer_frame(objectpoints, calibration, origin[:3], center=center, directions=np.array([-1, 1, 1]))
    
    print('objectpoints_mm median')
    print(np.median(objectpoints_mm, axis=0))
    
    pca3d = principal_axes(objectpoints_mm, verbose=True) # inertia, eigenvalues, eigenvectors, center
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(objectpoints_mm)
    pcd.estimate_normals()
    _end = time.time()
    print('3d coordinates calculated in %.3f seconds (%.4f from start)' % (_end-_start, _end-_total_start))

    return pcd 

def get_sample_description(predictions):
    
    # aoi stands for area of interest
    aoi_keys = ['detected', 'width', 'height', 'position', 'thickness', 'angle_min', 'angle_max']
    extreme_keys = ['position', 'vertical_fit_error', 'horizontal_fit_error']
    crystal = ['detected', 'position', 'width' , 'height']
    
    sample_description = {}
    
    return sample_description
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-H', '--history', default='/nfs/data2/excenter/2023-04-15T17:22:01.257118/403841_Sat_Apr_15_17:19:14_2023_history.h5', type=str, help='history')
    parser.add_argument('-D', '--display', action='store_true', help='Display')
    parser.add_argument('-d', '--debug', action='store_true', help='debug')
    parser.add_argument('-R', '--detector_row_spacing', default=1, type=int, help='detector vertical pixel size')
    parser.add_argument('-C', '--detector_col_spacing', default=1, type=int, help='detector horizontal pixel size')
    parser.add_argument('-j', '--as_jpegs', action='store_true', help='query using jpeg strings instead of raw arrays')
    parser.add_argument('-f', '--from_scratch', action='store_true', help='do not use saved predictions, get the new ones')
    parser.add_argument('-n', '--notions', default=['foreground', 'crystal', 'loop', 'stem', ['crystal', 'loop'], ['crystal', 'loop', 'stem']], help='notions to consider during the analysis')
    parser.add_argument('-m', '--minimize_method', default='nelder-mead', type=str, help='scipy.optimize.minimize algorithm (nelder-mead by default)')
    parser.add_argument('-s', '--save', action='store_true', help='save predicitons and analysis resuls') 
    args = parser.parse_args()
    print('args', args)
    
    master = h5py.File(args.history, 'r')
    angles = get_angles(master)
    calibration = get_calibration(master)
    origin = get_origin(master)
    print('calibration', calibration)
    print('origin', origin)
    
    predictions_filename = args.history.replace('.h5', '.predictions')
    pcd_filename = args.history.replace('.h5', '.pcd')
    pca_filename = args.history.replace('.h5', '.pca')
    analysis_filename = args.history.replace('.h5', '.analysis')
    
    if os.path.isfile(predictions_filename) and not bool(args.from_scratch):
        predictions = pickle.load(open(predictions_filename, 'rb'))
    else:
        _start = time.time()
        images = get_images(master, as_jpegs=bool(args.as_jpegs))
        _end = time.time()
        print('%d images loaded in %.3f seconds (since start %.4f)' % (len(images), _end - _start, _end-_total_start))
        
        request = {'to_predict': images,
                   'description': args.notions}
    
        predictions = get_predictions(request, verbose=True)
        f = open(predictions_filename, 'wb')
        pickle.dump(predictions, f)
        f.close()
    
    analysis = get_analysis(predictions, args.notions, angles, minimize_method=args.minimize_method, debug=args.debug)

    if args.display:
        notions_aspects = [('foreground', ('extreme_verticals', 'extreme_horizontals')), ('loop', ('widths', 'heights')), (['crystal', 'loop_inside', 'loop'], ('widths', 'heights', 'centroid_verticals', 'centroid_horizontals')), ('crystal', ('widths', 'heights', 'centroid_verticals', 'centroid_horizontals'))]
        
        test_angles = np.linspace(0, 2*np.pi, 360)
        
        k = 0
        for notion, aspects in notions_aspects:
            notion_string = get_notion_string(notion)
            notion_analysis = analysis[notion_string]
            valid_angles = notion_analysis['valid_angles']
            fits = notion_analysis['fits']
            for aspect in aspects:
                k += 1
                pylab.figure(k, figsize=(8, 6))
                pylab.title('%s %s' % (notion, aspect))
                data = fits['aspects'][aspect]
                model = fits['results'][aspect]['best_model'](test_angles, *fits['results'][aspect]['fit'].x)
                pylab.plot(np.rad2deg(valid_angles), data, 'go', label='%s' % aspect)
                pylab.plot(np.rad2deg(test_angles), model, 'g-', label='%s fit' % aspect)
                
        pylab.legend()
        pylab.show()
    
    if args.save:
        f = open(analysis_filename, 'wb')
        pickle.dump(analysis, f)
        f.close()
        
        f = open(pca_filename, 'wb')
        pickle.dump(analysis['foreground']['pca'], f)
        f.close()
        
    descriptions = predictions['descriptions']
    volume = get_volume('foreground', descriptions, analysis, origin, calibration, original_image_shape=analysis['original_image_shape'])

    if args.display:
        o3d.visualization.draw_geometries([volume.paint_uniform_color([1., 0.706, 0.])], mesh_show_wireframe=True, window_name='reconstructed sample volume')
        o3d.io.write_point_cloud(pcd_filename, volume)
        
if __name__ == '__main__':
    main()
