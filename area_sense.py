#!/usr/bin/env python

import pickle
import os
import commands
import re
import numpy
import scipy.ndimage
import glob
import scipy.misc
import traceback

def get_nspots_nimage(a):
    results = {}
    for line in a:
        try:
            nimage, nspots, nspots_no_ice, total_intensity = map(int, re.findall('\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\|', line)[0])
            #nspots, nimage = map(int, re.findall('Found (\d*) strong pixels on image (\d*)', line)[0])
            results[nimage] = {}
            results[nimage]['dials_spots'] = nspots_no_ice
            results[nimage]['dials_all_spots'] = nspots
            results[nimage]['dials_total_intensity'] = total_intensity
        except:
            print traceback.print_exc()
    return results

def save_results(results_file, results):
    f = open(results_file, 'w')
    pickle.dump(results, f)
    f.close()
    
def get_parameters(directory, name_pattern):
    return pickle.load(open(os.path.join(directory, '%s_parameters.pickle' % name_pattern)))
    
def get_results(directory, name_pattern, parameters):
    results_file = os.path.join(directory, '%s_%s' % (name_pattern, 'results.pickle'))
    if not os.path.isfile(results_file):
        process_dir = os.path.join(directory, '%s_%s' % ('process', name_pattern) )
        if not os.path.isdir(process_dir):
            os.mkdir(process_dir)
        print 'process_dir', process_dir
        if not os.path.isfile('%s/dials.find_spots.log' % process_dir):
            #if len(glob.glob('%s_*.cbf' % os.path.join(process_dir, name_pattern))) < parameters['number_of_rows'] * parameters['number_of_columns']:
                #convert_line = 'ssh p10 "cd %s; ~/bin/H5ToCBF.py -n 24 -m ../%s_master.h5"' % (process_dir, name_pattern)
                #os.system(convert_line)
            #spot_find_line = 'ssh p10 "source /usr/local/dials-v1-2-4/dials_env.sh; cd %s ; echo $(pwd); dials.find_spots nproc=24 %s_*.cbf"' % (process_dir, name_pattern) 
            spot_find_line = 'ssh process1 "source /usr/local/dials-v1-3-3/dials_env.sh; cd %s ; echo $(pwd); dials.find_spots shoebox=False per_image_statistics=True spotfinder.filter.ice_rings.filter=True nproc=80 ../%s_master.h5"' % (process_dir, name_pattern)
            #spot_find_line = 'ssh p10 "source /usr/local/dials-v1-2-4/dials_env.sh; cd %s ; echo $(pwd); dials.find_spots shoebox=True spotfinder.filter.ice_rings.filter=True nproc=24 ../%s_master.h5"' % (process_dir, name_pattern) 
            print 'pwd', commands.getoutput('pwd')
            print spot_find_line
            os.system(spot_find_line)
        a = commands.getoutput("grep '|' %s/dials.find_spots.log" % process_dir ).split('\n')
        save_results(results_file, get_nspots_nimage(a))
    return pickle.load(open(results_file))
  
def invert(z):
    z_inverted = z[:,::-1]
    z_raster = numpy.zeros(z.shape)
    for k in range(len(z)):
        if k%2 == 1:
            z_raster[k] = z_inverted[k]
        else:
            z_raster[k] = z[k]
    return z_raster
    
def get_z(parameters, results):
    number_of_rows = parameters['number_of_rows']
    number_of_columns = parameters['number_of_columns']
    points = parameters['cell_positions']
    indexes = parameters['indexes']
    
    z = numpy.zeros((number_of_rows, number_of_columns))

    if parameters['scan_axis'] == 'horizontal':
        for r in range(number_of_rows):
            for c in range(number_of_columns):
                try:
                    z[r,c] = results[int(points[r,c,2])]['dials_spots']
                except KeyError:
                    z[r,c] = 0
        z = raster(z)
        z = mirror(z) 
    
    if parameters['scan_axis'] == 'vertical':
        z = numpy.ravel(z)
        for n in range(len(z)):
            try:
                z[n] = results[n+1]['dials_spots']
            except:
                z[n] = 0
        z = numpy.reshape(z, (number_of_columns, number_of_rows))
        z = raster(z)
        z = z.T
        z = mirror(z)
    #print 'z'
    #print z
    return z

def mirror(grid):
    return raster(grid,k=0,l=1)
    
def raster(grid, k=0, l=2):
    gs = grid.shape
    orderedGrid = []
    for i in range(gs[0]):
        line = grid[i, :]
        if (i + 1) % l == k:
            line = line[: : -1]
        orderedGrid.append(line)
    return numpy.array(orderedGrid)
    
def scale_z(z, scale):
    return scipy.ndimage.zoom(z, scale)
    
def rotate_z(z, angle):
    return scipy.ndimage.rotate(z, angle)
    
def generate_full_grid_image(z, center, angle=0, fullshape=(493, 659)):
    empty = numpy.zeros(fullshape)
    gd1, gd2 = z.shape
    cd1, cd2 = center
    start1 = int(cd1-gd1/2.)
    end1 = int(cd1+gd1/2.) 
    start2 = int(cd2-gd2/2.)
    end2 = int(cd2+gd2/2.) 
    s1 = 0
    s2 = 0
    e1 = gd1 + 1
    e2 = gd2 + 1
    if start1 < 0:
        s1 = -start1 + 1 
        start1 = 0
    if end1 > fullshape[0]:
        e1 = e1 - (end1 - fullshape[0]) - 2
        end1 = fullshape[0] + 1
    if start2 < 0:
        s2 = -start2 + 1
        start2 = 0
    if end2 > fullshape[1]:
        e2 = e2 - (end2 - fullshape[1]) - 1 
        end2 = fullshape[1] + 1
    empty[start1: end1, start2: end2] = z[s1: e1, s2: e2]
    full = empty
    return full
    
def xyz(z):
    shape = z.shape
    x, y = numpy.meshgrid(range(shape[0]), range(shape[1]))
    x -= (shape[0] - 1)
    x *= -1
    y = numpy.transpose(y)
    y = numpy.transpose(y)
    print 'shapes', x.shape, y.shape, z.shape
    return x,y,z

def plot_surface(X, Y, Z):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_countour(X, Y, Z):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2)
    
    levels = np.linspace(0, 1000, 100)
    cs = axs[0].contourf(X, Y, Z, levels=levels)
    fig.colorbar(cs, ax=axs[0], format="%.2f")
    
    levels = np.linspace(0, 1000, 10)
    cs = axs[1].contourf(X, Y, Z, levels=levels)
    fig.colorbar(cs, ax=axs[1], format="%.2f")
    plt.show()
    
def plot_surface_wire(X, Y, Z, filename='resultFigure.png', stride=1):
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm
    import matplotlib.pyplot as plt

    fig = plt.figure(filename.replace('.png', ''), figsize=plt.figaspect(0.5))
    # surface
    
    ax = fig.add_subplot(1, 3, 1, projection='3d', title='Grey')
    surf = ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, cmap=cm.Greys, linewidth=0, antialiased=True)
    ax.view_init(elev=8., azim=-49.)
    fig.colorbar(surf, shrink=0.5, aspect=15)
    
    ax = fig.add_subplot(1, 3, 2, projection='3d', title='Bone')
    surf = ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, cmap=cm.bone, linewidth=0, antialiased=True)
    ax.view_init(elev=8., azim=-49.)
    fig.colorbar(surf, shrink=0.5, aspect=15)
    
    # wire
    ax = fig.add_subplot(1, 3, 3, projection='3d', title='Wireframe')
    wire = ax.plot_wireframe(X, Y, Z, rstride=stride, cstride=stride)
    ax.view_init(elev=8., azim=-49.)
    ## mesh
    #ax = fig.add_subplot(1, 4, 3, projection='3d', title='Wireframe')
    #wire = ax.mesh(X, Y, Z, rstride=stride, cstride=stride)

    # save and display
    plt.savefig(filename)
    plt.show()

def plot_wire_frame(X, Y, Z):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

    plt.show()


def main():
    import optparse
    
    parser = optparse.OptionParser()

    parser.add_option('-n', '--name_pattern' , default='grid', type=str, help='Template of files with the scan results, (default: %default)')
    parser.add_option('-d', '--directory', default='/tmp', type=str, help='Directory with the scan results, (default: %default)')
    
    options, args = parser.parse_args()

    print options, args
    
    directory = options.directory
    name_pattern = options.name_pattern
    
    optical_image_name = os.path.join(directory, name_pattern + '_optical_bw.png')
    
    parameters = get_parameters(directory, name_pattern)
    results = get_results(directory, name_pattern, parameters)
    
    vertical_range = parameters['vertical_range']
    horizontal_range = parameters['horizontal_range']
    beam_position_vertical = parameters['beam_position_vertical']
    beam_position_horizontal = parameters['beam_position_horizontal']
    camera_calibration_horizontal = parameters['camera_calibration_horizontal']
    camera_calibration_vertical = parameters['camera_calibration_vertical']
    number_of_rows = parameters['number_of_rows']
    number_of_columns = parameters['number_of_columns']
    
    center = numpy.array((beam_position_vertical, beam_position_horizontal))
    calibration = numpy.array((camera_calibration_vertical, camera_calibration_horizontal))
    shape = numpy.array((number_of_rows, number_of_columns))
    lengths = numpy.array((vertical_range, horizontal_range))
    
    z = get_z(parameters, results)
    
    grid_shape_on_real_image_in_pixels = lengths / calibration

    scale = grid_shape_on_real_image_in_pixels[::-1] / shape[::-1]

    print 'grid_shape_on_real_image_in_pixels %s' % grid_shape_on_real_image_in_pixels
    print 'scale %s ' % scale
    
    z_scaled = scipy.ndimage.zoom(z, scale[::-1])
    scaled_scan_image_name =  os.path.join(directory, name_pattern+'_z_scaled.png')
    scipy.misc.imsave(scaled_scan_image_name, z_scaled)

    z_full = generate_full_grid_image(z_scaled, center)

    o = scipy.misc.imread(optical_image_name, flatten=1)
    scan_image_name =  os.path.join(directory, name_pattern+'_scan.png')
    bw_overlay_image_name =  os.path.join(directory, name_pattern + '_bw_overlay.png')
    color_overlay_image_name = os.path.join(directory, name_pattern + '_overlay.png')
    contour_overlay_image_name = os.path.join(directory, name_pattern + '_countour_overlay.png')
    filter_overlay_image_name = os.path.join(directory, name_pattern + '_filter_overlay.png')

    scipy.misc.imsave(scan_image_name, z_full)
    scipy.misc.imsave(bw_overlay_image_name, z_full + o)
    os.system('composite -dissolve %s %s %s %s' % (55, scan_image_name, optical_image_name, color_overlay_image_name))

    grid_contour = z_full * ( z_full > 0.33 * z.max() ) * ( z_full < 0.66 * z.max() )
    contour_image_name = os.path.join(directory, name_pattern+'_contour.png')
    scipy.misc.imsave(contour_image_name, grid_contour)
    os.system('composite -dissolve %s %s %s %s' % (55, contour_image_name, optical_image_name, contour_overlay_image_name))

    grid_filter = ( z_full > 0.77 * z.max() ) * 255
    filter_image_name = os.path.join(directory, name_pattern+'_filter.png')
    scipy.misc.imsave(filter_image_name, grid_filter)
    os.system('composite -dissolve %s %s %s %s' % (55, filter_image_name, optical_image_name, filter_overlay_image_name))
    
if __name__ == '__main__':
    main()
