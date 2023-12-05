#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pyfive
import os.path
import re

# This script was originally developed by Andreas Förster at DECTRIS based on work by Marcus Mueller.
# Please note that this is not an official DECTRIS product and neither endorsed nor supported by DECTRIS.
# Please report errors and problems to docandreas@gmail.com.

# Original XDS_from_H5 script modified by Martin Savko (synchrotron Soleil) to rely on pyfive to access header
# information instead of albula API.

XDS_header_lines = """!*****************************************************************************
!
! XDS.INP template for ! %(family)s %(detector)s with %(sensor).2f mm thick silicon sensors.
!
!    Characters to the right of an exclamation mark are comments.
!
!    This file was autogenerated by XDS_from_H5.py (Oct 2015).
!    Please check default values before processing.
!
! For questions and comments please contact docandreas@gmail.com.
!
!*****************************************************************************
"""

XDS_detector_lines = """
!====================== DETECTOR PARAMETERS ==================================
 DETECTOR=%(family)s
 MINIMUM_VALID_PIXEL_VALUE=0
 OVERLOAD= %(cutoff)i ! taken from HDF5 header item
! /entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff
 SENSOR_THICKNESS=%(sensor).2f ! [mm]
!SILICON=-1.0
 QX=%(pixsize_x).3f  QY=%(pixsize_y).3f  ! [mm]
"""

XDS_main_lines = """
 TRUSTED_REGION=0.0 1.41 !Relative radii limiting trusted detector region

 DIRECTION_OF_DETECTOR_X-AXIS= 1.0 0.0 0.0
 DIRECTION_OF_DETECTOR_Y-AXIS= 0.0 1.0 0.0 ! 0.0 cos(2theta) sin(2theta)

!====================== JOB CONTROL PARAMETERS ===============================
 JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT
!JOB= DEFPIX INTEGRATE CORRECT
 
! MAXIMUM_NUMBER_OF_JOBS= 6  !Speeds up COLSPOT & INTEGRATE on multicore machine
! MAXIMUM_NUMBER_OF_PROCESSORS= 12 !<32;ignored by single cpu version of xds
!SECONDS=0   !Maximum number of seconds to wait until data image must appear
!TEST=1     !Test flag. 1,2 additional diagnostics and images

!====================== GEOMETRICAL PARAMETERS ===============================
!ORGX and ORGY are often close to the image center, i.e. ORGX=NX/2, ORGY=NY/2
 ORGX= %(orgx).1f  ORGY= %(orgy).1f    !Detector origin (pixels).  ORGX=NX/2; ORGY=NY/2
 DETECTOR_DISTANCE= %(dist).2f   ! [mm]

 ROTATION_AXIS= 1.0 0.0 0.0

! Optimal choice is 0.5*mosaicity (REFLECTING_RANGE_E.S.D.= mosaicity)
 OSCILLATION_RANGE=%(osc_range).5f            ! [deg] (>0)

 X-RAY_WAVELENGTH=%(wavelength).4f           ! [A]
 INCIDENT_BEAM_DIRECTION=0.0 0.0 1.0
 FRACTION_OF_POLARIZATION=0.99 !default=0.5 for unpolarized beam
 POLARIZATION_PLANE_NORMAL= 0.0 1.0 0.0

!======================= CRYSTAL PARAMETERS =================================
 SPACE_GROUP_NUMBER=0   !0 for unknown crystals; cell constants are ignored.
 UNIT_CELL_CONSTANTS= 0 0 0 0 0 0

! You may specify here the x,y,z components for the unit cell vectors if
! known from a previous run using the same crystal in the same orientation
!UNIT_CELL_A-AXIS=
!UNIT_CELL_B-AXIS=
!UNIT_CELL_C-AXIS=

!Optional reindexing transformation to apply on reflection indices
!REIDX=   0  0 -1  0  0 -1  0  0 -1  0  0  0

 FRIEDEL'S_LAW= TRUE ! Default is TRUE.

!REFERENCE_DATA_SET= CK.HKL   ! Name of a reference data set (optional)

!==================== SELECTION OF DATA IMAGES ==============================
!Generic file name and format (optional) of data images
 NAME_TEMPLATE_OF_DATA_FRAMES=%(name_template)s ! HDF5
"""

XDS_tail_lines = """
!==================== DATA COLLECTION STRATEGY (XPLAN) ======================
!                       !!! Warning !!!
! If you processed your data for a crystal with unknown cell constants and
! space group symmetry, XPLAN will report the results for space group P1.

!STARTING_ANGLE=  0.0      STARTING_FRAME=1
!used to define the angular origin about the rotation axis.
!Default:  STARTING_ANGLE=  0 at STARTING_FRAME=first data image

!RESOLUTION_SHELLS=10 6 5 4 3 2 1.5 1.3 1.2

!STARTING_ANGLES_OF_SPINDLE_ROTATION= 0 180 10

!TOTAL_SPINDLE_ROTATION_RANGES=30.0 120 15

!====================== INDEXING PARAMETERS =================================
!Never forget to check this, since the default 0 0 0 is almost always correct!
!INDEX_ORIGIN= 0 0 0          ! used by "IDXREF" to add an index offset

!Additional parameters for fine tuning that rarely need to be changed
!INDEX_ERROR=0.05 INDEX_MAGNITUDE=8 INDEX_QUALITY=0.8
 SEPMIN=4.0       ! default is 6 for other detectors
 CLUSTER_RADIUS=2 ! default is 3 for other detectors
!MAXIMUM_ERROR_OF_SPOT_POSITION=3.0
!MAXIMUM_ERROR_OF_SPINDLE_POSITION=2.0
!MINIMUM_FRACTION_OF_INDEXED_SPOTS=0.5

!============== DECISION CONSTANTS FOR FINDING CRYSTAL SYMMETRY =============
!Decision constants for detection of lattice symmetry (IDXREF, CORRECT)
 MAX_CELL_AXIS_ERROR= 0.03 ! Maximum relative error in cell axes tolerated
 MAX_CELL_ANGLE_ERROR= 2.0 ! Maximum cell angle error tolerated

!Decision constants for detection of space group symmetry (CORRECT).
!Resolution range for accepting reflections for space group determination in
!the CORRECT step. It should cover a sufficient number of strong reflections.
 TEST_RESOLUTION_RANGE= 8.0 4.5
 MIN_RFL_Rmeas= 50  ! Minimum #reflections needed for calculation of Rmeas
 MAX_FAC_Rmeas= 2.0 ! Sets an upper limit for acceptable Rmeas

!================= PARAMETERS CONTROLLING REFINEMENTS =======================
 REFINE(IDXREF)= BEAM AXIS ORIENTATION CELL ! POSITION
 REFINE(INTEGRATE)= POSITION ORIENTATION ! BEAM CELL AXIS
 REFINE(CORRECT)= POSITION BEAM ORIENTATION CELL AXIS

!================== CRITERIA FOR ACCEPTING REFLECTIONS ======================
 VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS= 6000 30000 !Used by DEFPIX
                   !for excluding shaded parts of the detector.

 INCLUDE_RESOLUTION_RANGE= 65.0 %(reso_range).1f !Angstroem; used by DEFPIX,INTEGRATE,CORRECT

!used by CORRECT to exclude ice-reflections
!EXCLUDE_RESOLUTION_RANGE= 3.93 3.87 !ice-ring at 3.897 Angstrom
!EXCLUDE_RESOLUTION_RANGE= 3.70 3.64 !ice-ring at 3.669 Angstrom
!EXCLUDE_RESOLUTION_RANGE= 3.47 3.41 !ice-ring at 3.441 Angstrom
!EXCLUDE_RESOLUTION_RANGE= 2.70 2.64 !ice-ring at 2.671 Angstrom
!EXCLUDE_RESOLUTION_RANGE= 2.28 2.22 !ice-ring at 2.249 Angstrom
!EXCLUDE_RESOLUTION_RANGE= 2.102 2.042 !ice-ring at 2.072 Angstrom - strong
!EXCLUDE_RESOLUTION_RANGE= 1.978 1.918 !ice-ring at 1.948 Angstrom - weak
!EXCLUDE_RESOLUTION_RANGE= 1.948 1.888 !ice-ring at 1.918 Angstrom - strong
!EXCLUDE_RESOLUTION_RANGE= 1.913 1.853 !ice-ring at 1.883 Angstrom - weak
!EXCLUDE_RESOLUTION_RANGE= 1.751 1.691 !ice-ring at 1.721 Angstrom - weak

!MINIMUM_ZETA=0.05 !Defines width of 'blind region' (XPLAN,INTEGRATE,CORRECT)

!WFAC1=1.0  !This controls the number of rejected MISFITS in CORRECT;
        !a larger value leads to fewer rejections.
!REJECT_ALIEN=20.0 ! Automatic rejection of very strong reflections

!============== INTEGRATION AND PEAK PROFILE PARAMETERS =====================
!Specification of the peak profile parameters below overrides the automatic
!determination from the images
!Suggested values are listed near the end of INTEGRATE.LP
!BEAM_DIVERGENCE=   0.80         !arctan(spot diameter/DETECTOR_DISTANCE)
!BEAM_DIVERGENCE_E.S.D.=   0.080 !half-width (Sigma) of BEAM_DIVERGENCE
!REFLECTING_RANGE=  0.780 !for crossing the Ewald sphere on shortest route
!REFLECTING_RANGE_E.S.D.=  0.113 !half-width (mosaicity) of REFLECTING_RANGE

! NUMBER_OF_PROFILE_GRID_POINTS_ALONG_ALPHA/BETA=15!used by: INTEGRATE
! NUMBER_OF_PROFILE_GRID_POINTS_ALONG_GAMMA=15     !used by: INTEGRATE

!DELPHI= 6.0!controls the number of reference profiles and scaling factors
!CUT=2.0    !defines the integration region for profile fitting
!MINPK=75.0 !minimum required percentage of observed reflection intensity

!======= PARAMETERS CONTROLLING CORRECTION FACTORS (used by: CORRECT) =======
!MINIMUM_I/SIGMA=3.0 !minimum intensity/sigma required for scaling reflections
!NBATCH=-1  !controls the number of correction factors along image numbers
!REFLECTIONS/CORRECTION_FACTOR=50   !minimum #reflections/correction needed
!PATCH_SHUTTER_PROBLEM=TRUE         !FALSE is default
!STRICT_ABSORPTION_CORRECTION=TRUE  !FALSE is default
!CORRECTIONS= DECAY MODULATION ABSORPTION

!=========== PARAMETERS DEFINING BACKGROUND AND PEAK PIXELS =================
!STRONG_PIXEL=3.0                              !used by: COLSPOT
!A 'strong' pixel to be included in a spot must exceed the background
!by more than the given multiple of standard deviations.

!MAXIMUM_NUMBER_OF_STRONG_PIXELS=1500000       !used by: COLSPOT

!SPOT_MAXIMUM-CENTROID=3.0                     !used by: COLSPOT

 MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT=3          !used by: COLSPOT
!This allows to suppress spurious isolated pixels from entering the
!spot list generated by "COLSPOT".

!NBX=3  NBY=3  !Define a rectangle of size (2*NBX+1)*(2*NBY+1)
!The variation of counts within the rectangle centered at each image pixel
!is used for distinguishing between background and spot pixels.

!BACKGROUND_PIXEL=6.0                          !used by: COLSPOT,INTEGRATE
!An image pixel does not belong to the background region if the local
!pixel variation exceeds the expected variation by the given number of
!standard deviations.

!SIGNAL_PIXEL=3.0                              !used by: INTEGRATE
!A pixel above the threshold contributes to the spot centroid

!FIXED_SCALE_FACTOR=TRUE  !Default is FALSE; used by : INIT,INTEGRATE
"""

detector_families = {
    'pilatus' : {
        'nmodules' : {
            '12M': (5, 24),
            '6M' : (5, 12),
            '2M' : (3 ,8),
            '1M' : (2, 5),
            '300K-W': (3, 1),
            '300K' : (1 ,3),
            '200K' : (1 ,2),
            '100K' : (1, 1),
            },
        'module' : {
            'size': (487, 195),
            'gap': (7, 17),
            'pixel_size': (0.172e-03, 0.172e-03),
            'nchips': (8, 2),
            },
        'chip': {
            'size': (60, 97),
            'gap': (1, 1),
        },
        'sizes' : {}, # will be populated with correct sizes
        },
    'eiger' : {
        'nmodules' : {
            '1M': (1, 2),
            '4M': (2, 4),
            '9M': (3, 6),
            '16M': (4, 8),
            },
        'module' : {
            'size': (1030, 514),
            'gap': (10, 37),
            'pixel_size': (0.075e-03, 0.075e-03),
            'nchips': (4, 2),
            },
        'chip' : {
            'size' : (256, 256),
            'gap' : (2, 2),
        },
        'sizes' : {}, # will be populated with correct sizes
        },
    }

# All interesting parameters
incident_wavelength = "/entry/instrument/beam/incident_wavelength"
software_version = "/entry/instrument/detector/detectorSpecific/software_version"
beam_center_x = "/entry/instrument/detector/beam_center_x"
beam_center_y = "/entry/instrument/detector/beam_center_y"
x_pixel_size = "/entry/instrument/detector/x_pixel_size"
y_pixel_size = "/entry/instrument/detector/y_pixel_size"
detector_distance = "/entry/instrument/detector/detector_distance"
sensor_thickness = "/entry/instrument/detector/sensor_thickness"
nimages = "/entry/instrument/detector/detectorSpecific/nimages"
description = "/entry/instrument/detector/description"
omega_range_average = "/entry/sample/goniometer/omega_range_average"
omega_increment = "/entry/sample/goniometer/omega_increment"
countrate_correction_count_cutoff = "/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff"
resolution_cutoff = 'max resolution'

# The list below contains the parameters to be extracted from H5
parameters = [
    incident_wavelength,
    software_version,
    beam_center_x,
    beam_center_y,
    x_pixel_size,
    y_pixel_size,
    detector_distance,
    sensor_thickness,
    nimages,
    description,
    omega_range_average,
    countrate_correction_count_cutoff,
    resolution_cutoff
    ]

def create_XDS_INP(parameters, file_name):
    lines = []
    description = parameters["/entry/instrument/detector/description"].split()
    family = description[1].lower()
    sensor = float(parameters["/entry/instrument/detector/sensor_thickness"]) * 1000.0
    FAMILY = family.upper()
    det_name = description[2]
    file_template = re.sub("master\.h5", "??????.h5", file_name)
    lines.append(XDS_header_lines % {
        'family': FAMILY,
        'detector': det_name,
        'sensor': sensor,})
    lines.append(XDS_detector_lines % {
        'family': FAMILY,
        'cutoff': int(float(parameters["/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff"])),
        'sensor': sensor,
        'pixsize_x': float(parameters["/entry/instrument/detector/x_pixel_size"]) * 1000.0,
        'pixsize_y': float(parameters["/entry/instrument/detector/y_pixel_size"]) * 1000.0,})
    lines = lines + get_size_specific_lines(fam=family, det=det_name, n_excluded_edge_pixels=0)
    lines.append(XDS_main_lines % {
        'orgx': float(parameters["/entry/instrument/detector/beam_center_x"]),
        'orgy': float(parameters["/entry/instrument/detector/beam_center_y"]),
        'dist': float(parameters["/entry/instrument/detector/detector_distance"]) * 1000.0,
        'osc_range': float(parameters["/entry/sample/goniometer/omega_range_average"]),
        'wavelength': float(parameters["/entry/instrument/beam/incident_wavelength"]),
        'name_template': file_template,})
    first = 1
    last = int(parameters["/entry/instrument/detector/detectorSpecific/nimages"])
    para_images = int(full_parameters["/entry/instrument/detector/detectorSpecific/nimages"])
    rotation = float(full_parameters["/entry/sample/goniometer/omega_range_average"])
    lines.append("\n DATA_RANGE=%i %i\n" % (first, last))
    if (para_images * rotation <= 30):
        if (last > 100):
            bkg = 100
        else:
            bkg = last
        lines.append("\n")
        lines.append(" BACKGROUND_RANGE=%i %i  ! Numbers of first and last data image for background\n" % (first, bkg))
        lines.append("!Five degrees are sufficient\n")
        lines.append("\n")
        lines.append(" SPOT_RANGE= %i %i       ! Image range for finding spots\n" % (first, last))
        lines.append("!Use all images if this range is not sufficient\n")
    elif (para_images * rotation > 30):
        # split spot finding into three 10 degree segments
        bkg = first + int(5/rotation)
        end1   = first + int(10/rotation)
        start2 = first + int(last/2)
        end2   = first + int(last/2) + int(10/rotation)
        start3 = first + last - int(10/rotation) - 1
        end3   = first + last - 1
        lines.append("\n")
        lines.append(" BACKGROUND_RANGE=%i %i  ! Numbers of first and last data image for background\n" % (first, bkg))
        lines.append("!Five degrees are sufficient\n")
        lines.append("\n")
        lines.append(" SPOT_RANGE= %i %i       ! First image range for finding spots\n" % (first, end1))
        lines.append(" SPOT_RANGE= %i %i       ! Second image range for finding spots\n" % (start2, end2))
        lines.append(" SPOT_RANGE= %i %i       ! Third image range for finding spots\n" % (start3, end3))
        lines.append("!Use all images if three ranges are not sufficient\n")
    lines.append(XDS_tail_lines % {
        'reso_range': float(parameters["max resolution"]),})
    return lines

def get_size_specific_lines(fam, det, n_excluded_edge_pixels=0):
    param_lines = []
    gaps = calculate_gaps(
        detector_families[fam]['sizes'][det],
        detector_families[fam]['module']['size'],
        detector_families[fam]['module']['gap'],
        )
    param_lines.append(' NX= %4d  NY= %4d \n\n' % detector_families[fam]['sizes'][det])
    param_lines.append('!EXCLUSION OF VERTICAL DEAD AREAS OF THE '
        '%s %s DETECTOR \n' % (fam.upper(), det))
    module_edge_comment = ('!EXCLUDING %d ADDITIONAL PIXELS OF THE '
        'MODULE EDGES \n' % n_excluded_edge_pixels)
    if n_excluded_edge_pixels > 0:
        param_lines.append(module_edge_comment)
    # offset is required because XDS.INP pixel values start with 1, not 0
    offset = 1
    for gap in gaps[0]:
        param_lines.append(' UNTRUSTED_RECTANGLE= %4d %4d   %4d %4d \n' % (
            gap[0] - 1 + offset - n_excluded_edge_pixels,
            gap[1] + 1 + offset + n_excluded_edge_pixels,
            0,
            detector_families[fam]['sizes'][det][1] + offset))
    param_lines.append('\n')
    param_lines.append('!EXCLUSION OF HORIZONTAL DEAD AREAS OF THE '
        '%s %s DETECTOR \n' % (fam.upper(), det))
    if n_excluded_edge_pixels > 0:
        param_lines.append(module_edge_comment)
    for gap in gaps[1]:
        param_lines.append(' UNTRUSTED_RECTANGLE= %4d %4d   %4d %4d \n' % (
            0,
            detector_families[fam]['sizes'][det][0] + offset,
            gap[0] - 1 + offset - n_excluded_edge_pixels,
            gap[1] + 1 + offset + n_excluded_edge_pixels))
    return param_lines

def warning():
    return ('\nThis script extracts from a given HDF5 master file all metadata\n'
            'required to write XDS.INP.  The user is prompted for missing metadata.\n'
            '\n'
            'WARNING - This script is a proof-of-principle, pre-alpha.\n'
            'Do not rely on it for anything serious.  Things will go wrong.\n'
            'In particular, this does not work for data collected in ROI mode.\n'
            '\n'
            'Please report shortcomings and errors to andreas.foerster@dectris.com\n')

def help():
    return ('ERROR - You must specify exactly one HDF5 master file:\n'
            '\n'
            'python XDS_from_H5.py <name>_master.h5\n')

permitted_versions = ["1.6.2", "1.6.1", "1.6.0", "1.5.2", "1.5.1", "1.5.0", "1.2.0", "1.2.1", "1.3", "1.3.0", "1.4.0"]
def version_check(version):
    if (str(version) in permitted_versions):
        return 1
    else:
        return 0

zero_values = [0, "0", 0.0, "0.0"]

def isFile(file_input):
    '''This function verifies that the file name entered by the user
    corresponds to a master.h5 file and attaches an extension if necessary.'''
    if os.path.isfile(file_input) and re.search("master\.h5\Z", file_input):
        return file_input
    elif os.path.isfile(file_input + ".h5") and re.search("master\Z", file_input):
        return(file_input + ".h5")
    else:
        return 0

def request_parameter(parameter):
    if (parameter == omega_range_average):
        return raw_input("Please enter the oscillation range in degrees.\n")
    elif (parameter == detector_distance):
        return raw_input("Please enter the detector distance in meters.\n")
    elif (parameter == incident_wavelength):
        return raw_input("Please enter the wavelength in Angstrom.\n")
    elif (parameter == beam_center_x):
        return raw_input("Please enter the x coordinate of the beam center in pixels.\n")
    elif (parameter == beam_center_y):
        return raw_input("Please enter the y coordinate of the beam center in pixels.\n")
    elif (parameter == x_pixel_size):
        return raw_input("Please enter the x coordinate of the pixel size.\n")
    elif (parameter == y_pixel_size):
        return raw_input("Please enter the y coordinate of the pixel size.\n")
    elif (parameter == sensor_thickness):
        return raw_input("Please enter the sensor thickness in meters.\n")
    elif (parameter == nimages):
        return raw_input("Please enter the number of images.\n")
    elif (parameter == description):
        print("Please enter the description of the detector, e.g.")
        return raw_input("Dectris Eiger 4M\n")
    elif (parameter == countrate_correction_count_cutoff):
        return raw_input("Please enter the maximum trusted pixel value.\n")
    elif (parameter == resolution_cutoff):
        #return raw_input("Please enter a resolution limit for processing.\n")
        return 0
    else:
        print("Unknown software version.  Please check.")
        return 0

def calculate_gaps(det_size, mod_size, gap_size):
    """
    Return list of tuples with first and last pixel in each detector gap.

    One list for each detector dimension (x and y).
    Input: total detector size in pixels
        size of a module in pixels
        size of a gap in pixels
    """
    ndims = len(det_size)
    gaps = []
    for dim_index in range(ndims):
        gaps.append([])
        module_start = 0
        while module_start < det_size[dim_index]:
            # First pixel on a module has index 0, Python and C style
            gap_start = module_start + mod_size[dim_index]
            module_start = gap_start + gap_size[dim_index]
            gap_end = module_start - 1
            if module_start < det_size[dim_index]:
                gaps[dim_index].append((gap_start, gap_end))
            else:
                break
    return gaps



# Creates a dictionary of all keys and values in the NeXus tree
def iterate_children(node, nodeDict={}):
    """ iterate over the children of a neXus node """
    if node.type() == dec.DNeXusNode.GROUP:
        for kid in node.children():
            nodeDict = iterate_children(kid, nodeDict)
    else:
        nodeDict[node.path()] = node.value()
    return nodeDict

# Extracts values from HDF5 file according to parameters array
def get_params(hdf5_file):
    extracted = {}
    h5cont = pyfive.File(hdf5_file) #dec.DImageSeries(hdf5_file)
    #neXus_tree = h5cont.neXus()
    #neXus_root = neXus_tree.root()
    #neXus_string_tree = iterate_children(neXus_root)
    if (len(sys.argv) == 2):
        print("Extracting metadata from " + hdf5_file)
        print("Please modify XDS.INP if these numbers are incorrect.\n")
    for i in parameters:
        try:
            extracted[i] = str(h5cont[i].value)
        except:
            extracted[i] = ""
    return extracted

def calculate_size(n_modules, mod_size, gap_size):
    n_gaps = [n - 1 for n in n_modules]
    size = []
    for nmod, ngap, nmodpix, ngappix in zip(n_modules, n_gaps, mod_size, gap_size):
        size.append(nmod * nmodpix + ngap * ngappix)
    return tuple(size)

# populate dicts with sizes of detectors in pixels
for family in detector_families.values():
    for model, n_modules in family['nmodules'].items():
        family['sizes'][model] = calculate_size(n_modules=n_modules,
                mod_size=family['module']['size'],
                gap_size=family['module']['gap'])

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Make sure that XDS.INP does not already exist
        if os.path.isfile ("XDS.INP"):
            print("\nERROR: XDS.INP exists already.  Please rename and rerun script.")
        else:
            # test whether argument 1 is HDF5 file.
            # attach ".h5" if necessary
            clean_file = isFile(sys.argv[1])
            if (clean_file):
                print(warning())
                full_parameters = get_params(clean_file)
                for i, v in full_parameters.iteritems():
                    if (v in zero_values):
                        print(i + " = " + str(v) + "   <== WARNING:  Should this really be 0?")
                        full_parameters[i] = request_parameter(i)
                        print(i + " = " + str(full_parameters[i]))
                    elif (v == "NaN") or (v == ""):
                        print(i + " = " + v + "   <== ERROR:  undefined value.")
                        full_parameters[i] = request_parameter(i)
                        print(i + " = " + str(full_parameters[i]))
                    else:
                        print(i + " = " + str(v))
                para_version = str(full_parameters[software_version])
                if version_check(para_version):
                    param_lines = create_XDS_INP(full_parameters, clean_file)
                    open("XDS.INP", 'w').writelines(param_lines)
                    print("\nFile XDS.INP was created successfully.")
                    if (int(full_parameters["/entry/instrument/detector/detectorSpecific/nimages"]) == 1):
                        print("However, there's not much you can do with one image.\n")
                    else:
                        print("Please verify its contents before processing data.\n")
                else:
                    print("\nThe HDF5 file was created with version %s of the detector firmware" % (para_version))
                    print("This script supports versions 1.5 and up.")
                    print("Please extract metadata with hdfview or h5dump.\n")
            else:
                print(help())
    elif (len(sys.argv) == 3):
        # This assumes the second argument is the rotation range
        # The script will run non-interactively
        # The master.h5 must be specified with its full name
        # An existing XDS.INP will be overwritten
        full_parameters = get_params(sys.argv[1])
        full_parameters["omega_range_average"] = sys.argv[2]
        param_lines = create_XDS_INP(full_parameters, sys.argv[1])
        open("XDS.INP", 'w').writelines(param_lines)
    else:
        print(help())
        exit(-1)

