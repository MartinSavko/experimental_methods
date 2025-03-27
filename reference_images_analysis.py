#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import sys

XDS_PATH = ""
xdsme_home = '/nfs/ruche/share-dev/px1dev/bioxsoft/progs/XDSME/xdsme'

sys.path.insert(0, xdsme_home)
sys.path.insert(0, os.path.join(xdsme_home, 'XDS'))
#sys.path.insert(0, os.path.join(xdsme_home, 'XIO'))
#sys.path.insert(0, os.path.join(xdsme_home, 'XOalign'))
#sys.path.insert(0, os.path.join(xdsme_home, 'XOconv'))
#sys.path.insert(0, os.path.join(xdsme_home, 'XOconv/pycgtypes'))
sys.path.insert(0, os.path.join(xdsme_home, '3rdparty'))
sys.path.insert(0, os.path.join(xdsme_home, 'bin/noarch'))

from xdsme import *

__version__ = "0.0.1.0 based on xdsme 0.6.6.0"
__author__ = "Pierre Legrand (pierre.legrand \at synchrotron-soleil.fr) hereby mutilated by Martin Savko (martin.savko \at synchrotron-soleil.fr)"
__date__ = "2019-10-05"
__copyright__ = "Copyright (c) 2006-2019 Pierre Legrand"
__license__ = "New BSD http://www.opensource.org/licenses/bsd-license.php"

#STRFTIME = '%a %b %d %X %Z %Y'
#STRFTIME2 = '%F %X'
#time_start = time.strftime(STRFTIME)
#DIRNAME_PREFIX = "xdsme_"
#NUM_PROCESSORS = get_number_of_processors()
#SET_NTHREADS = False
## Use a maximum of 16 proc. by job. Change it if you whant another limit.
#WARN_MSG = ""
#VERBOSE = False
#WEAK = False
#ANOMAL = False
#ICE = False
#STRICT_CORR = False
#BEAM_X = 0
#BEAM_Y = 0
#SPG = 0
#STRATEGY = False
#RES_HIGH = 0
#DISTANCE = 0
#OSCILLATION = 0
#ORIENTATION_MATRIX = False
#PROJECT = ""
#WAVELENGTH = 0
#RES_LOW = 50
#FIRST_FRAME = 0
#LAST_FRAME = 0
#REFERENCE = False
#_beam_center_optimize = False
#_beam_center_ranking = "ZSCORE"
#_beam_center_swap = False
#CELL = ""
#XDS_INPUT = ""
#_beam_in_mm = False
#SLOW = False
#FAST = False
#BRUTE = False
#STEP = 1
#OPTIMIZE = 0
#INVERT = False
#XDS_PATH = ""
#RUN_XDSCONV = True
#RUN_AIMLESS = True
#XML_OUTPUT = False
#MINIMAL_COMPL_TO_EXPORT = 0.5

def select_strategy(idxref_results, xds_par, auto=True, LATTICE_GEOMETRIC_FIT_CUTOFF=5):
    "Interactive session to select strategy parameters."
    sel_spgn = SPG #xds_par["SPACE_GROUP_NUMBER"]
    sel_ano =  xds_par["FRIEDEL'S_LAW"]
    valid_inp = False
    bravais_to_spgs = get_BravaisToSpgs()
    # Select LATTICE
    
    while not valid_inp:
        def_sel = 1
        if auto:
            for k, LAT in enumerate(idxref_results["lattices_table"]):
                if LAT.fit <= LATTICE_GEOMETRIC_FIT_CUTOFF:
                    def_sel = k+1
                
        if sel_spgn != 0:
            # choose the lattice solution according to the selected spg.
            i = 0
            for LAT in idxref_results["lattices_table"]:
                if LAT.fit <= LATTICE_GEOMETRIC_FIT_CUTOFF:
                    i += 1
                    if sel_spgn in bravais_to_spgs[LAT.Bravais_type]:
                        def_sel = i
        if not auto:
            selection = raw_input("\n Select a solution number [%d]: " % def_sel)
        else:
            selection = str(def_sel)
        # If the selection is not compatible with the spg, set not valid
        _sel = selection.split()
        selnum = 1
        try:
            if len(_sel) == 1:
                selnum = int(_sel[0])
                valid_inp = True
            elif len(_sel) == 0:
                selnum = def_sel
                valid_inp = True
            else:
                raise(Exception, "Invalid selection input.")
        except Exception as err:
            prnt(err, ERROR)
    sel_lat = idxref_results["lattices_table"][selnum-1]
    if sel_spgn == 0:
        sel_spgn = sel_lat.symmetry_num
    valid_inp = False
    # Select SPACEGROUP
    prnt(" Possible spacegroup for this lattice are:\n")
    for spgsymb in bravais_to_spgs[sel_lat.Bravais_type]:
        prnt("  %15s, number: %3d" % (SPGlib[spgsymb][1], spgsymb))
    while not valid_inp:
        if not auto:
            selection = raw_input("\n Select the spacegroup [%s, %d]: "
                                  % (SPGlib[sel_spgn][1], sel_spgn))
        else:
            selection = '\n'
        _sel = selection.split()
        try:
            if len(_sel) == 1:
                sel_spgn, _spg_info, _spg_str = parse_spacegroup(_sel[0])
                # selSpgS = _spg_info[1]
                valid_inp = True
            elif len(_sel) == 0:
                valid_inp = True
            else:
                raise(Exception, "Invalid selection input.")
            if sel_spgn not in bravais_to_spgs[sel_lat.Bravais_type]:
                valid_inp = False
                msg = "Inconsistant combinaison of Bravais lattice"
                msg += " and spacegroup.\n For this Bravais Lattice"
                msg += " (%s), spacegroup should be one of these:\n\n" % \
                        (sel_lat.Bravais_type)
                for spgsymb in bravais_to_spgs[sel_lat.Bravais_type]:
                    msg += "  %15s, number: %3d\n" % \
                                 (SPGlib[spgsymb][1], spgsymb)
                raise(Exception, msg)
        except Exception as err:
            prnt(err, ERROR)
    valid_inp = False
    # Select ANOMALOUS
    while not valid_inp:
        if sel_ano == "TRUE":
            txt3 = "N/y"
        else:
            txt3 = "Y/n"
        if not auto:
            selection = raw_input(" Anomalous [%s]: " % txt3)
        else:
            selection = 'y'
        try:
            _ans =  selection.strip()
            if _ans == "":
                valid_inp = True
            elif _ans[0] in "Yy":
                xds_par["FRIEDEL'S_LAW"] = "FALSE"
                valid_inp = True
            elif _ans[0] in "Nn":
                xds_par["FRIEDEL'S_LAW"] = "TRUE"
                valid_inp = True
            else:
                raise(Exception, "Invalid answer [Y/N].")
        except Exception as err:
            prnt(err, ERROR)
    prnt("\n Selected  cell paramters:  %s" % sel_lat)
    if sel_spgn > 2:
        sel_lat.idealize()
        prnt(" Idealized cell parameters: %s" % sel_lat.prt())
        xds_par["UNIT_CELL_CONSTANTS"] = sel_lat.prt()
    xds_par["SPACE_GROUP_NUMBER"] = sel_spgn
    return xds_par

class myXDS(XDS):
    
    STRFTIME = '%a %b %d %X %Z %Y'
    STRFTIME2 = '%F %X'
    time_start = time.strftime(STRFTIME)
    DIRNAME_PREFIX = "xdsme_"
    NUM_PROCESSORS = get_number_of_processors()
    SET_NTHREADS = False
    # Use a maximum of 16 proc. by job. Change it if you whant another limit.
    WARN_MSG = ""
    VERBOSE = False
    WEAK = False
    ANOMAL = False
    ICE = False
    STRICT_CORR = False
    BEAM_X = 0
    BEAM_Y = 0
    SPG = 0
    STRATEGY = False
    RES_HIGH = 0
    DISTANCE = 0
    OSCILLATION = 0
    ORIENTATION_MATRIX = False
    PROJECT = ""
    WAVELENGTH = 0
    RES_LOW = 50
    FIRST_FRAME = 0
    LAST_FRAME = 0
    REFERENCE = False
    _beam_center_optimize = False
    _beam_center_ranking = "ZSCORE"
    _beam_center_swap = False
    CELL = ""
    XDS_INPUT = ""
    _beam_in_mm = False
    SLOW = False
    FAST = False
    BRUTE = False
    STEP = 1
    OPTIMIZE = 0
    INVERT = False
    XDS_PATH = ""
    RUN_XDSCONV = True
    RUN_AIMLESS = True
    XML_OUTPUT = False
    MINIMAL_COMPL_TO_EXPORT = 0.5
    
    def __init__(self, obj=None, link_to_images=True, 
                 STRFTIME = '%a %b %d %X %Z %Y',
                 STRFTIME2 = '%F %X',
                 time_start = time.strftime(STRFTIME),
                 DIRNAME_PREFIX = "xdsme_",
                 NUM_PROCESSORS = get_number_of_processors(),
                 SET_NTHREADS = False,
                 # Use a maximum of 16 proc. by job. Change it if you whant another limit.
                 WARN_MSG = "",
                 VERBOSE = False,
                 WEAK = False,
                 ANOMAL = False,
                 ICE = False,
                 STRICT_CORR = False,
                 BEAM_X = 0,
                 BEAM_Y = 0,
                 SPG = 0,
                 STRATEGY = False,
                 RES_HIGH = 0,
                 DISTANCE = 0,
                 OSCILLATION = 0,
                 ORIENTATION_MATRIX = False,
                 PROJECT = "",
                 WAVELENGTH = 0,
                 RES_LOW = 50,
                 FIRST_FRAME = 0,
                 LAST_FRAME = 0,
                 REFERENCE = False,
                 _beam_center_optimize = False,
                 _beam_center_ranking = "ZSCORE",
                 _beam_center_swap = False,
                 CELL = "",
                 XDS_INPUT = "",
                 _beam_in_mm = False,
                 SLOW = False,
                 FAST = False,
                 BRUTE = False,
                 STEP = 1,
                 OPTIMIZE = 0,
                 INVERT = False,
                 XDS_PATH = "",
                 RUN_XDSCONV = True,
                 RUN_AIMLESS = True,
                 XML_OUTPUT = False,
                 MINIMAL_COMPL_TO_EXPORT = 0.5):
        
        self.link_to_images = link_to_images
        
        self.STRFTIME = STRFTIME
        self.STRFTIME2 = STRFTIME2
        self.time_start = time_start
        self.DIRNAME_PREFIX = DIRNAME_PREFIX
        self.NUM_PROCESSORS = NUM_PROCESSORS
        self.SET_NTHREADS = SET_NTHREADS
        # Use a maximum of 16 proc. by job. Change it if you whant another limit.
        self.WARN_MSG = WARN_MSG
        self.VERBOSE = VERBOSE
        self.WEAK = WEAK
        self.ANOMAL = ANOMAL
        self.ICE = ICE
        self.STRICT_CORR = STRICT_CORR
        self.BEAM_X = BEAM_X
        self.BEAM_Y = BEAM_Y
        self.SPG = SPG
        self.STRATEGY = STRATEGY
        self.RES_HIGH = RES_HIGH
        self.DISTANCE = DISTANCE
        self.OSCILLATION = OSCILLATION
        self.ORIENTATION_MATRIX = ORIENTATION_MATRIX
        self.PROJECT = PROJECT
        self.WAVELENGTH = WAVELENGTH
        self.RES_LOW = RES_LOW
        self.FIRST_FRAME = FIRST_FRAME
        self.LAST_FRAME = LAST_FRAME
        self.REFERENCE = REFERENCE
        self._beam_center_optimize = _beam_center_optimize
        self._beam_center_ranking = _beam_center_ranking
        self._beam_center_swap = _beam_center_swap
        self.CELL = CELL
        self.XDS_INPUT = XDS_INPUT
        self._beam_in_mm = _beam_in_mm
        self.SLOW = SLOW
        self.FAST = FAST
        self.BRUTE = BRUTE
        self.STEP = STEP
        self.OPTIMIZE = OPTIMIZE
        self.INVERT = INVERT
        self.XDS_PATH = XDS_PATH
        self.RUN_XDSCONV = RUN_XDSCONV
        self.RUN_AIMLESS = RUN_AIMLESS
        self.XML_OUTPUT = XML_OUTPUT
        self.MINIMAL_COMPL_TO_EXPORT = MINIMAL_COMPL_TO_EXPORT
        
        self.__cancelled = 0
        self.__lastOutp = 0
        self.mode = []
        if XDS_PATH:
            self._execfile = os.path.join(self.XDS_PATH, "xds_par")
        else:
            self._execfile = "xds_par"
        self.running = 0
        self.outp = []
        self.run_dir = "."
        self.status = None
        self.inpParam = XParam()
        self.collect_dir = "./"
        self.link_name_to_image = "img"
        self.running_processes = []
        #
        if sys.version_info.major <= 2:
            if type(obj) == file:
                exec(obj.read(), self.inpParam.__dict__)
                obj.close()
        else:
            if str(type(obj)) == "<class '_io.TextIOWrapper'>":
                exec(obj.read(), self.inpParam.__dict__)
                obj.close()
        if type(obj) == str:
            exec(obj, self.inpParam.__dict__)

    def run_init(self):
        "Runs the 2 first steps: XYCORR and INIT"
        if self.XDS_INPUT:
            self.inpParam.mix(xdsInp2Param(inp_str=self.XDS_INPUT))
        #self.inpParam["TRUSTED_REGION"] = [0, 1.20]
        self.inpParam["JOB"] = "XYCORR", "INIT"
        i1, i2 = self.inpParam["DATA_RANGE"]
        #if "slow" in self.mode:
        # default is min of 3 degrees or 8 images.
        dPhi = self.inpParam["OSCILLATION_RANGE"]
        if BRUTE:
            bkgr =  i1, i1+40
        elif SLOW or WEAK:
            bkgr =  i1, min(i2, min(i1+15, i1+int(7./dPhi)))
        else:
            bkgr =  i1, min(i2, min(i1+7, i1+int(3./dPhi)))
        self.inpParam["BACKGROUND_RANGE"] = bkgr
        self.run(rsave=True)
        res = XDSLogParser("INIT.LP", run_dir=self.run_dir, verbose=1)
        if res.results["mean_background"] < 1.:
            prnt("INIT has found a very LOW mean background.\n" + \
              "  -> Setting FIXED_SCALE_FACTOR for INTEGRATE step.", WARNING)
            self.inpParam["DATA_RANGE_FIXED_SCALE_FACTOR"] = i1, i2, 1.
        return res.results
    
    def run_colspot(self):
        "Runs the COLSPOT step."
        if self.XDS_INPUT:
            self.inpParam.mix(xdsInp2Param(inp_str=self.XDS_INPUT))
        self.inpParam["JOB"] = "COLSPOT",
        if SET_NTHREADS:
            self.inpParam["MAXIMUM_NUM_PROCESSORS"] = 1
            self.inpParam["MAXIMUM_NUMBER_OF_JOBS"] = NUM_PROCESSORS
        _trial = 0

        # DEFAULT=3.2 deg., SLOW=6.4 deg., FAST=1.6 deg.
        dPhi = self.inpParam["OSCILLATION_RANGE"]
        frames_per_colspot_sequence = FRAMES_PER_COLSPOT_SEQUENCE
        if "slow" in self.mode:
            frames_per_colspot_sequence = int(round(6.4/dPhi, 0))
        elif "fast" in self.mode:
            frames_per_colspot_sequence = int(round(1.6/dPhi, 0))
        elif BRUTE:
            frames_per_colspot_sequence = int(round(60./dPhi, 0))
            self.inpParam["VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS"] = \
                 5000, 30000
            self.inpParam["SIGNAL_PIXEL"] = 4.5
        else:
            frames_per_colspot_sequence = int(round(3.2/dPhi, 0))
        if "weak" in self.mode:
            self.inpParam["SIGNAL_PIXEL"] = 4.5
            self.inpParam["MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT"] -= 1
            frames_per_colspot_sequence = int(round(12.8/dPhi, 0))
        # Selecting spot range(s),
        # self.inpParam["SPOT_RANGE"] is set to Collect.imageRanges by the
        # xds export function XIO
        cfo = XIO.Collect("foo_001.bar", rotationAxis=INVERT)
        cfo.imageNumbers = cfo._ranges_to_sequence(self.inpParam["SPOT_RANGE"])
        #
        min_fn, max_fn = self.inpParam["DATA_RANGE"]
        _fpcs = frames_per_colspot_sequence
        _2fpcs = 1 + 2 * frames_per_colspot_sequence

        if (max_fn - min_fn + 1) >= _2fpcs:
            # use two range ex: i-i+2, f-2,f
            # with f at maximum 90 degre distance
            max_frame = min(max_fn, min_fn + int(89./dPhi + _fpcs))
            spot_ranges = ((min_fn, min_fn + _fpcs - 1),
                          (max_frame - _fpcs + 1, max_frame))
        else:
            spot_ranges = (min_fn, min(min_fn + _2fpcs - 1, max_fn)),
        # Restrict to matching collected images...
        #self.inpParam["SPOT_RANGE"] = cfo.lookup_imageRanges(False, \
                                              #mask_range=spot_ranges)
        if BRUTE and len(self.inpParam["SPOT_RANGE"]) == 1:
            self.inpParam["SPOT_RANGE"] = (min_fn, int(89./dPhi + _fpcs)),
        print('self.inpParam["SPOT_RANGE"]', self.inpParam["SPOT_RANGE"])
        self.run(rsave=True)
        _rs = "  Image range(s) for spot collection: "
        for sub_range in self.inpParam["SPOT_RANGE"]:
            _rs += ("  [%d - %d]," % tuple(sub_range))
        prnt(_rs[:-1] + "\n")

        res = XDSLogParser("COLSPOT.LP", run_dir=self.run_dir, verbose=1)
        while res.results["spot_number"] < MIN_SPOT_NUMBER and _trial < 4:
            _trial += 1
            min_pixels = int(self.inpParam["MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT"])
            self.inpParam["MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT"] = max(min_pixels-1, 1)
            self.inpParam["SIGNAL_PIXEL"] -= 1.
            #self.inpParam["SPOT_MAXIMUM_CENTROID"] += 1
            prnt("Insuficiant number of spot (minimum set to %d)." % \
                                                         MIN_SPOT_NUMBER)
            prnt("Recollecting spots. Trial number %d" % _trial)
            self.run(rsave=True)
            res = XDSLogParser("COLSPOT.LP", run_dir=self.run_dir, verbose=1)
        return res.results
    
    def run_idxref(self, beam_center_search=False, ranking_mode="ZSCORE",
                         beam_center_swap=False):
        "Runs the IDXREF step. Can try to search for better beam_center."
        res = None
        test_results = []
        if self.XDS_INPUT:
            self.inpParam.mix(xdsInp2Param(inp_str=self.XDS_INPUT))
        self.inpParam["JOB"] = "IDXREF",
        # this prevent bad spot to be included.
        saved_trusted_region = self.inpParam["TRUSTED_REGION"]
        if saved_trusted_region[1] > 0.98:
            self.inpParam["TRUSTED_REGION"] = [0, 0.98]
        self.run(rsave=True)
        try:
            res = XDSLogParser("IDXREF.LP", run_dir=self.run_dir, verbose=1)
        except XDSExecError as err:
            prnt(" in %s\n" % err[1], ERROR)
            if err[0] == "FATAL" and not (beam_center_swap or 
                                          beam_center_search):
                sys.exit()
        except Exception as err:
            prnt(" %s\n" % err, CRITICAL)

        qx, qy = self.inpParam["QX"], self.inpParam["QY"]
        dist = self.inpParam["DETECTOR_DISTANCE"]
        det_x = vec3(self.inpParam["DIRECTION_OF_DETECTOR_X-AXIS"])
        det_y = vec3(self.inpParam["DIRECTION_OF_DETECTOR_Y-AXIS"])
        det_z = det_x.cross(det_y)
        det_params = dist, det_x, det_y, det_z, qx, qy

        #RD["indexed_percentage"] < 70. or \
        #if beam_center_search or RD["xy_spot_position_ESD"] > 2. or \
        #  RD["z_spot_position_ESD"] > 2*self.inpParam["OSCILLATION_RANGE"]:
        if res:
            test_results.append(res.results)
        if beam_center_swap:
            x, y = self.inpParam["ORGX"], self.inpParam["ORGY"]
            mx, my = self.inpParam["NX"] - x, self.inpParam["NY"] - y
            origins = [[y, x], [mx, my], [my, mx],
                       [ x, my], [y, mx], [mx, y], [my, x]]
            for origin in origins:
                self.inpParam["ORGX"] = origin[0]
                self.inpParam["ORGY"] = origin[1]
                prnt("   Testing beam coordinate: (%.2fmm, %.2fmm) = " % \
                                           (origin[0]*qx, origin[1]*qy) +
                          "  %.1f, %.1f" % (origin[0], origin[1]))
                self.run(rsave=True, verbose=False)
                try:
                    test_results.append(XDSLogParser("IDXREF.LP",
                                         run_dir=self.run_dir,
                                         verbose=0, raiseErrors=True).results)
                except XDSExecError as err:
                    prnt(" in %s" % err, ERROR)
        if beam_center_search:
            RD = res.results
            prnt(" Number of possible beam coordinates: %d" % \
                                 len(RD["index_origin_table"]))
            maxTestOrigin = min(60, len(RD["index_origin_table"]))
            origins = RD["index_origin_table"][:maxTestOrigin]
            for origin in origins:
                # We first need to calculate the beam_origin from the
                # beam_coordinate and beam_vector given in the table
                beam = vec3(origin[7:10])
                beam_origin = get_beam_origin(origin[5:7], beam, det_params)
                self.inpParam["ORGX"] = beam_origin[0]
                self.inpParam["ORGY"] = beam_origin[1]
                self.inpParam["INCIDENT_BEAM_DIRECTION"] = tuple(beam)
                #print("DEBUG:  %7.1f %7.1f  - %7.1f %7.1f" % \)
                #  (coorx, coory, self.inpParam["ORGX"], self.inpParam["ORGY"])
                prnt("   Testing beam coordinate: (%.2fmm, %.2fmm) = " % \
                                           (origin[5]*qx, origin[6]*qy) +
                          "  %.1f, %.1f" % (origin[5], origin[6]))
                self.run(rsave=True, verbose=False)
                try:
                    test_results.append(XDSLogParser("IDXREF.LP",
                                         run_dir=self.run_dir,
                                         verbose=0, raiseErrors=True).results)
                except XDSExecError as err:
                    prnt("\t\tError in", ERROR)
        if beam_center_search or beam_center_swap:
            prnt("\n")
            # Need to lookup in the results for the beam-center giving
            best_index_rank = rank_indexation(test_results, ranking_mode)
            #for o in origins:
            #    print(origins.index(o), o[:-3])
            best_origin = origins[best_index_rank[ranking_mode]-1]
            if VERBOSE:
                prnt(best_index_rank)
                #fmt = "%4i%4i%4i%7.2f%7.2f%8.1f%8.1f%9.5f%9.5f%9.5f"
                prnt("best_index_rank %s" % best_index_rank[ranking_mode])
                #print("best_origin", fmt % tuple(best_origin))
            if beam_center_search:
                best_beam = vec3(best_origin[7:10])
                best_beam_coor = best_origin[5:7]
                best_beam_orig = get_beam_origin(best_beam_coor,
                                             best_beam, det_params)
                self.inpParam["ORGX"], self.inpParam["ORGY"] = best_beam_orig
                self.inpParam["INCIDENT_BEAM_DIRECTION"] = tuple(best_beam)
            else:
                self.inpParam["ORGX"], self.inpParam["ORGY"] = best_origin
            # Running again with updated best parameters
            self.run(rsave=True)
            res = XDSLogParser("IDXREF.LP", run_dir=self.run_dir)
        # Set back the Trusted_region to larger values.
        self.inpParam["TRUSTED_REGION"] = saved_trusted_region
        return res.results
    
    def run_xplan(self, ridx=None):
        if self.XDS_INPUT:
            self.inpParam.mix(xdsInp2Param(inp_str=self.XDS_INPUT))
        "Running the strategy."
        if SET_NTHREADS:
            self.inpParam["MAXIMUM_NUMBER_OF_PROCESSORS"] = NUM_PROCESSORS
            self.inpParam["MAXIMUM_NUMBER_OF_JOBS"] = 1

        select_strategy(ridx, self.inpParam)
        prnt("\n Starting strategy calculation.")
        self.inpParam["JOB"] = "IDXREF",
        self.run(rsave=True)
        res = XDSLogParser("IDXREF.LP", run_dir=self.run_dir, verbose=2)
        # Select just the internal circle of the detector.
        self.inpParam["JOB"] = "DEFPIX", "XPLAN"
        self.run(rsave=True)
        res =  XDSLogParser("XPLAN.LP", run_dir=self.run_dir, verbose=1)
        return res.results
    
    def run_integrate(self, image_ranges):
        "Running INTEGRATE."
        if self.BRUTE:
           self.inpParam["DELPHI"] = 20.
        if self.XDS_INPUT:
            self.inpParam.mix(xdsInp2Param(inp_str=XDS_INPUT))
        if self.SET_NTHREADS:
            self.inpParam["MAXIMUM_NUMBER_OF_PROCESSORS"] = NUM_PROCESSORS
            self.inpParam["MAXIMUM_NUMBER_OF_JOBS"] = 1
        if ("slow" in self.mode) or self.BRUTE:
            self.inpParam["NUMBER_OF_PROFILE_GRID_POINTS_ALONG_ALPHA_BETA"] = 13
            self.inpParam["NUMBER_OF_PROFILE_GRID_POINTS_ALONG_GAMMA"] = 13

        "Runs the 2 first steps: DEFPIX and INTEGRATE"
        self.inpParam["JOB"] = "DEFPIX",
        self.run(rsave=True)
        i1, i2 = self.inpParam["DATA_RANGE"]
        res = myXDSLogParser("DEFPIX.LP", run_dir=self.run_dir, verbose=1)
        if res.results["mean_background_unmasked"] < 1.:
            prnt("   -> Setting FIXED_SCALE_FACTOR for INTEGRATE step.")
            self.inpParam["DATA_RANGE_FIXED_SCALE_FACTOR"] = i1, i2, 1.

        if len(image_ranges) >= 1:
            self.inpParam["JOB"] = "INTEGRATE",
            self.run(rsave=True)
            res = XDSLogParser("INTEGRATE.LP", run_dir=self.run_dir, verbose=1)
            self.check_fileout("INTEGRATE.HKL")
        #else:
        #    #print("\n Error in the INTEGRATE step:")
        #    print("\n Image range:", image_ranges)
        #    print(" Multi-sweep integration not yet implemanted. Sorry.\n")
        #    sys.exit(0)
        return res.results
    
    def run(self, run_dir=None, rsave=None, verbose=True, aasync=False):
        "Control the runing of the xds process and parse the output."
        self.__cancelled = 0
        self.running = 1
        self.step = 0
        self.step_name = ""
        self.outp = []
        self.init_dir = os.getcwd()
        self.aasync = aasync
        if run_dir:
            self.run_dir = run_dir
        if not self.run_dir:
            self.run_dir = "."
        result = 0
        if self.run_dir:
            if not os.path.exists(self.run_dir):
                try:
                    os.mkdir(self.run_dir)
                except OSError as err:
                    prnt(err, ERROR)
                    raise(XIO.XIOError, \
                     ("\nSTOP! Can't create xds working directory: %s\n" % \
                                                              self.run_dir))
            if os.path.isdir(self.run_dir):
                os.chdir(self.run_dir)
                if self.link_to_images:
                    if not os.path.exists(self.link_name_to_image):
                        os.system("ln -sf '%s' %s" % (self.collect_dir, \
                                                    self.link_name_to_image))
                        #os.system("ln -sf .. %s" % (self.link_name_to_image))
                    #else:
                    #    raise XIO.XIOError, \
                    #     "STOP! Can't creat link %s in working directory: %s" \
                    #     % (self.link_name_to_image, self.run_dir)
        opWriteCl("XDS.INP", "%s" % self.inpParam)
        #
        # self.running_processes
        xdsProcess = self._creat_process(self._execfile)
        _init_parse = True
        overloaded_spots = 0
        while self.running:
            self.status = xdsProcess.poll()
            if self.status != self.wait_value:
                self.running = 0
                break
            if self.__cancelled:
                os.kill(xdsProcess.pid, 9)
                break
            if self.wait_value == -1:
                lines = xdsProcess.fromchild.readline()
            else:
                lines = xdsProcess.stdout.readline()
                #lines = xdsProcess.communicate()
            # ilines parsing of stdout
            if self.step_name == "INTEGRATE":
                if _init_parse:
                    prnt("    Processing    Mean #Strong  " +
                         "Estimated   Overloaded\n" +
                         "    Image Range   refl./image   " +
                         "Mosaicity   reflections\n")
                    table_int = []
                    _init_parse = False
                if INTEGRATE_STEP_RE.search(lines):
                    _msg = lines[44:50]+" - "+lines[56:-1]
                    nimages = int(lines[56:-1]) - int(lines[44:50]) + 1
                elif INTEGRATE_STRONG_RE.search(lines):
                    _msg += "%11.0f" % (float(lines.split()[0])/nimages)
                elif INTEGRATE_MOSAICITY_RE.search(lines):
                    _msg += " %11.3f" % float(lines.split()[3])
                    prnt(_msg + " %11d" % overloaded_spots)
                    overloaded_spots = 0
                hit = SCALE_RE.search(lines)
                if hit:
                    table_int = hit.groups()
                    overloaded_spots += int(hit.groups()[3])
            sm = STEPMARK.match(lines)
            if sm:
                self.step += 1
                self.step_name = sm.group(2)
                #if VERBOSE:
                if verbose:
                    prnt("\n --->  Running job: %20s\n" % self.step_name)
            if lines:
                self.outp.append(lines)
        self.step += 1
        self.step_name = "FINISHED"
        if self.__cancelled:
            result = -1
        if rsave:
            saveLastVersion(LP_names)
        #if VERBOSE:
        #    print("End of XDS run")
        os.chdir(self.init_dir)
        return 1
    
    def run_pre_correct(self):
        """Runs a first pass of CORRECT to evaluate high_res and
           point group.
        """
        def _get_cell(_file):
            _txt_file = open(_file,'r').readlines()
            if "XPARM.XDS" in _txt_file[0]:
                return map(float, (_txt_file[3]).split()[1:])
            else:
                return map(float, (_txt_file[7]).split()[1:])

        if self.XDS_INPUT:
            self.inpParam.mix(xdsInp2Param(inp_str=XDS_INPUT))
        # run pointless on INTEGRATE.HKL
        if not is_pointless_installed():
            prnt("Pointless program not installed. XDS will try to guess" +
                 " the Laue group. Have a look at the systematic extinctions",
                 WARNING)
            likely_spg = [["P1", 0],]
            new_cell = False
        else:
            prnt("    Pointless analysis on the INTEGRATE.HKL\n"+"    "+44*"=")
            try:
                likely_spg, new_cell = pointless(dir_name=self.run_dir,
                                                 hklinp="INTEGRATE.HKL")
            except:
                raise
                prnt(" While running Pointless. Skipped", ERROR)
                likely_spg = [["P1", 0],]
                new_cell = False
        self.inpParam["JOB"] = "CORRECT",
        if not self.SPG:
            # run first CORRECT in P1 with the cell used for integration.
            # read the cell parameters from the XPARM.XDS file
            self.inpParam["SPACE_GROUP_NUMBER"] = 1
            try:
                xparm_file = os.path.join(self.run_dir, "XPARM.XDS")
                self.inpParam["UNIT_CELL_CONSTANTS"] = _get_cell(xparm_file)
            except:
                os.chdir("..")
                self.inpParam["UNIT_CELL_CONSTANTS"] = _get_cell(xparm_file)
        # run CORRECT
        self.run(rsave=True)
        res = XDSLogParser("CORRECT.LP", run_dir=self.run_dir, verbose=1)
        prnt("\n  Upper theoritical limit of I/sigma: %8.3f\n" % \
                                             res.results["IoverSigmaAsympt"])

        L, H = self.inpParam["INCLUDE_RESOLUTION_RANGE"]
        newH = res.results["HighResCutoff"]
        if newH > H and not RES_HIGH:
            H = newH
        if self.SPG:
            spg_choosen = self.SPG
        else:
            spg_choosen = likely_spg[0][1]
            # Re-order pointless cell-axes in case of orthorombic SPG.
            spgSplit = likely_spg[0][0].split()
            # if cell is coming from pointless, it need reordering
            # in orthorombic cases
            if new_cell:
                a, b, c, A, B, G = new_cell
                if spg_choosen == 18:
                    if spgSplit[1] == "2":
                        new_cell = [b, c, a, A, B, G]
                    elif spgSplit[2] == "2":
                        new_cell = [a, c, b, A, B, G]
                elif spg_choosen == 17:
                    if spgSplit[1] == "21":
                        new_cell = [b, c, a, A, B, G]
                    elif spgSplit[2] == "21":
                        new_cell = [a, c, b, A, B, G]
            else:
                new_cell = self.inpParam["UNIT_CELL_CONSTANTS"]
            lattice = Lattice(new_cell, symmetry=spg_choosen)
            lattice.idealize()
            self.inpParam["UNIT_CELL_CONSTANTS"] = lattice.cell
            #reidx_mat = likely_spg[0][-1]
            #new_cell = new_reidx_cell(self.inpParam["UNIT_CELL_CONSTANTS"],
        return (L, H), spg_choosen
    
    def run_correct(self, res_cut=(1000, 0), spg_num=0):
        "Runs the last step: CORRECT"
        if res_cut[1]:
            prnt("   ->  New high resolution limit: %.2f Å" % res_cut[1])
            self.inpParam["INCLUDE_RESOLUTION_RANGE"] = res_cut
        if spg_num:
            prnt("   ->  Using spacegroup: %s  #%d" % \
                                           (SPGlib[spg_num][1], spg_num))
        lattice = Lattice(self.inpParam["UNIT_CELL_CONSTANTS"],
                          symmetry=spg_num)
        lattice.idealize()
        self.inpParam["UNIT_CELL_CONSTANTS"] = lattice.cell
        self.inpParam["JOB"] = "CORRECT",
        self.inpParam["SPACE_GROUP_NUMBER"] = spg_num
        self.run(rsave=True)
        res = XDSLogParser("CORRECT.LP", run_dir=self.run_dir, verbose=1)
        prnt("\n  Upper theoritical limit of I/sigma: %8.3f\n" % \
                                             res.results["IoverSigmaAsympt"])
        s = resum_scaling(lpf=os.path.join(self.run_dir,"CORRECT.LP"))
        if not s:
            prnt("while running CORRECT", CRITICAL)
        s["image_start"], s["image_last"] = self.inpParam["DATA_RANGE"]
        s["name"] = os.path.basename(self.inpParam["NAME_TEMPLATE_OF_DATA_FRAMES"])
        s["image_path"] = os.path.realpath("img")
        if self.XML_OUTPUT:
            frame_ID = s["name"].split("?")[0]
            s["hostname"] = os.uname()[1]
            s["osname"] = " ".join(os.uname())
            s["username"] = "autoprocessing" #"%s" % (os.environ["LOGNAME"])
            s["cmd_line"] = " ".join(sys.argv).split(".cbf ")[0]
            s["xdsme_version"] = xdsme_version
            s["xds_version"] = xds_version
            s["exec_time_start"] = time_start
            s["run_dir"] = os.path.realpath(self.run_dir)
            s["run_dir_p"] = os.path.dirname(s["run_dir"])
            aimless_ID = "%saimless" % frame_ID
            s["mtz_out"] = "%s.mtz" % frame_ID[:-1]
            s["aimlessout"] = "%s.log" % aimless_ID
            s["xdsmeout"] = "%s.log" % (self.run_dir)
        prnt(s.last_table)
        prnt(FMT_FINAL_STAT % vars(s))
        if s.absent:
            prnt(FMT_ABSENCES % vars(s))
        if self.inpParam["FRIEDEL'S_LAW"] == "FALSE":
            prnt(FMT_ANOMAL % vars(s))
        if s.compl > self.MINIMAL_COMPL_TO_EXPORT or self.XML_OUTPUT:
            if self.RUN_AIMLESS:
                run_aimless(self.run_dir)
                if XML_OUTPUT:
                    s["exec_time_end"] = time.strftime(STRFTIME)
                    xmlaiml = os.path.join(s["run_dir"], "%s.xml" % aimless_ID)
                    res.results.update(s)
                    xml1 = XML_PROGRAM_PART % res.results
                    xml2 = xml_aimless_to_autoproc(xmlaiml)
                    xml3 = XML_INTEGRATION_PART % res.results
                    xmlo = os.path.join(s["run_dir"], "%sxdsme.xml" % frame_ID)
                    opWriteCl(xmlo, XML_TEMPL_MAIN % (xml1 + xml2 + xml3))
                    simple_xscale(run_dir=self.run_dir)
            if self.RUN_XDSCONV:
                run_xdsconv(self.run_dir)
                
class myXDSLogParser(XDSLogParser):
    
    def __init__(self, filename="", run_dir="",
                 verbose=False, raiseErrors=True, XDS_PATH=""):
        
        XDSLogParser.__init__(self, filename=filename, run_dir=run_dir,
                 verbose=verbose, raiseErrors=raiseErrors)
        
        self.XDS_PATH = XDS_PATH
        
    def get_xds_version(self):
        "Get the version of XDS"
        _execstr = "cd /tmp; %s" % os.path.join(self.XDS_PATH, "xds_par")
        wc_out = self.run_exec_str(_execstr).splitlines()[1:]
        if "license expired" in wc_out[0]:
            prnt(wc_out[0].replace(" licen", " XDS licen"), CRITICAL)
        else:
            return wc_out[0].strip()[24:-12].replace(")","")
        
if __name__ == '__main__':
    
    import getopt

    short_opt =  "123456aAbBc:d:E:f:F:i:IL:O:M:n:p:s:Sr:R:x:y:vw:WSFX"
    long_opt = ["anomal",
                "Anomal",
                "beam-x=",
                "beam-y=",
                "ice",
                "invert",
                "spg=",
                "strategy",
                "high-resolution=",
                "low-resolution=",
                "last-frame",
                "first-frame",
                "cell=",
                "distance",
                "reference=",
                "oscillation",
                "orientation-matrix=",
                "nthreads=",
                "project=",
                "exec=",
                "beam-center-optimize-i",
                "beam-center-optimize-z",
                "beam-center-swap",
                "xds-input=",
                "verbose",
                "optimize",
                "O1","O2","O3","O4","O",
                "wavelength=",
                "slow", "weak", "brute",
                "signal=",
                "xml"]

    if len(sys.argv) == 1:
        print(USAGE)
        sys.exit(2)
    try:
        opts, inputf = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError:
        # print(help information and exit:)
        print(USAGE)
        sys.exit(2)

    STRFTIME = '%a %b %d %X %Z %Y'
    STRFTIME2 = '%F %X'
    time_start = time.strftime(STRFTIME)
    DIRNAME_PREFIX = "xdsme_"
    NUM_PROCESSORS = get_number_of_processors()
    SET_NTHREADS = False
    # Use a maximum of 16 proc. by job. Change it if you whant another limit.
    WARN_MSG = ""
    VERBOSE = False
    WEAK = False
    ANOMAL = False
    ICE = False
    STRICT_CORR = False
    BEAM_X = 0
    BEAM_Y = 0
    SPG = 0
    STRATEGY = False
    RES_HIGH = 0
    DISTANCE = 0
    OSCILLATION = 0
    ORIENTATION_MATRIX = False
    PROJECT = ""
    WAVELENGTH = 0
    RES_LOW = 50
    FIRST_FRAME = 0
    LAST_FRAME = 0
    REFERENCE = False
    _beam_center_optimize = False
    _beam_center_ranking = "ZSCORE"
    _beam_center_swap = False
    CELL = ""
    XDS_INPUT = ""
    _beam_in_mm = False
    SLOW = False
    FAST = False
    BRUTE = False
    SIGNAL_PIXEL = 0
    STEP = 1
    OPTIMIZE = 0
    INVERT = False
    XDS_PATH = ""
    RUN_XDSCONV = True
    RUN_AIMLESS = True
    XML_OUTPUT = False
    MINIMAL_COMPL_TO_EXPORT = 0.5

    for o, a in opts:
        if o == "-v":
            VERBOSE = True
        if o in ("-E", "--exec"):
            XDS_PATH = a
            xdsf = os.path.join(XDS_PATH, "xds_par")
            if not (os.path.isfile(xdsf) and os.access(xdsf, os.X_OK)):
                print("WARNING: no 'xds_par' exec found in path %s." % XDS_PATH, WARNING)
                print("         Using default $PATH location.", WARNING)
                XDS_PATH = ""
            else:
                os.environ['PATH'] = "%s%s%s" % (XDS_PATH, os.pathsep,
                                                           os.environ["PATH"])
        if o in ("-a", "--anomal"):
            ANOMAL = True
            STRICT_CORR = False
        if o in ("-A", "--Anomal"):
            ANOMAL = True
            STRICT_CORR = True
        if o in ("-I", "--ice"):
            ICE = True
        if o[1] in "123456":
            STEP = int(o[1])
        if o in ("-s", "--spg"):
            SPG, _spg_info, _spg_str = parse_spacegroup(a)
        #if o in ("-i", "--xds-input"):
            #XDS_INPUT = a
        if o in ("-i", "--xds-input"):
            XDS_INPUT = a + " ROTATION_AXIS= 0.0 -1.0 0.0"
        else:
            XDS_INPUT = "ROTATION_AXIS= 0.0 -1.0 0.0"

        if o in ("-c", "--cell"):
            CELL = a
        if o in ("-d", "--distance"):
            DISTANCE = float(a)
        if o in ("-f", "--reference"):
            if os.path.isfile(a):
                REFERENCE = str(a)
            else:
                print("Can't open reference file %s." % a, CRITICAL)
        if o in ("-F", "--first-frame"):
            FIRST_FRAME = int(a)
        if o in ("-L", "--last-frame"):
            LAST_FRAME = int(a)
        if o in ("-O", "--oscillation"):
            OSCILLATION = float(a)
        if o in ("-M", "--orientation-matrix"):
            if os.path.isfile(a):
                ORIENTATION_MATRIX = str(a)
            else:
                print("Can't open orientation matrix file %s." % a, CRITICAL)
        if o in ("-n","--nthreads"):
            SET_NTHREADS = True
            NUM_PROCESSORS = int(a)
        if o in ("-p", "--project"):
            PROJECT = str(a).strip("/").replace("/","_")
        if o in ("-S", "--strategy"):
            STRATEGY = True
        if o in ("-w", "--wavelength"):
            WAVELENGTH = float(a)
        if o in ("-r", "--high-resolution"):
            RES_HIGH = float(a)
        if o in ("-R", "--low-resolution"):
            RES_LOW = float(a)
        if o in ("-x", "--beam_x"):
            if "mm" in a:
                _beam_in_mm = True
                a = a.replace("mm","")
            BEAM_X = float(a)
        if o in ("-y", "--beam_y"):
            if "mm" in a:
                _beam_in_mm = True
                a = a.replace("mm","")
            BEAM_Y = float(a)
        if o in ("-b", "--beam-center-optimize-i"):
            _beam_center_optimize = True
            _beam_center_ranking = "ISCORE"
        if o in ("-B", "--beam-center-optimize-z"):
            _beam_center_optimize = True
            _beam_center_ranking = "ZSCORE"
        if o in ("-W", "--beam-center-swap"):
            _beam_center_swap = True
        if o in ("--optimize", "--O"):
            OPTIMIZE = 3
            STEP = 4
        if "--O" in o and len(o) == 4:
            STEP = 4
            try:
                OPTIMIZE = int(o[-1])
            except:
                pass
            if OPTIMIZE > 4:
                OPTIMIZE = 4
        if o in ("-X", "--xml"):
            XML_OUTPUT = True
            RUN_AIMLESS = True
        if o in ("--slow"):
            SLOW = True
        if o in ("--brute"):
            BRUTE = True
        if o in ("--weak"):
            WEAK = True
        if o == "--invert":
            INVERT = True
        if o in ("-h", "--help"):
            print(USAGE)
            sys.exit(2)

    if not inputf:
        prnt("No image file specified.", CRITICAL)
    elif not os.path.isfile(inputf[0]):
        prnt("Image file %s not found.\n" % inputf[0], CRITICAL)
    else:
        # TODO cycle over input_file with try/except to avoid XIOError
        _coll = XIO.Collect(inputf[0])
    if not PROJECT:
        newDir = DIRNAME_PREFIX + _coll.prefix
    else:
        newDir = DIRNAME_PREFIX + PROJECT

    fileHandler = logging.FileHandler("%s.log" % newDir)
    fileHandler.setFormatter(logFormatter)
    prntLog.addHandler(fileHandler)

    _linkimages = False
    if not _coll.isContinuous(inputf):
        #prnt("Discontinous naming scheme, creating links.", WARNING)
        _linkimages = True
        link_dir_name = "img_links"
        inputf = make_xds_image_links(inputf,
                                    os.path.join(newDir,link_dir_name),
                                    "image")
        #collect.setDirectory(link_dir_name)
        #collect.prefix = prefix
    try:
        collect = XIO.Collect(inputf, rotationAxis=INVERT)
        collect.interpretImage()
        collect.image.info()
        collect.lookup_imageRanges(forceCheck=False)

    except XIO.XIOError as _mess:
        prnt(_mess, ERROR)
        prnt("Can't access to file(s) %s.\nStop." % inputf, CRITICAL)

    imgDir = collect.directory
    newPar = collect.export("xds")
    # Update some default values defined by XIO.export_xds:
    # In case no beam origin is defined, take the detector center.
    if newPar["ORGX"] == 0:
        newPar["ORGX"] = newPar["NX"]/2.
    if newPar["ORGY"] == 0:
        newPar["ORGY"] = newPar["NY"]/2.
    # This is to correct the starting angle in case first image is not 1.
    newPar["STARTING_ANGLE"] = newPar["STARTING_ANGLE"] - \
              newPar["OSCILLATION_RANGE"]*(newPar["DATA_RANGE"][0] - 1)
    newPar["SIGNAL_PIXEL"] = 6
    newPar["RESOLUTION_SHELLS"] = 15.0, 7.0, newPar["_HIGH_RESOL_LIMIT"]
    newPar["TEST_RESOLUTION_RANGE"] = 20, newPar["_HIGH_RESOL_LIMIT"]+1.5

    newrun = myXDS()

    if _beam_in_mm:
        BEAM_X = BEAM_X / newPar["QX"]
        BEAM_Y = BEAM_Y / newPar["QY"]
    if ANOMAL:
        newPar["FRIEDEL'S_LAW"] = "FALSE"
    else:
        newPar["FRIEDEL'S_LAW"] = "TRUE"
    if STRICT_CORR:
        newPar["STRICT_ABSORPTION_CORRECTION"] = "TRUE"
    else:
        newPar["STRICT_ABSORPTION_CORRECTION"] = "FALSE" 
    if BEAM_X:
        newPar["ORGX"] = BEAM_X
    if BEAM_Y:
        newPar["ORGY"] = BEAM_Y
    if FIRST_FRAME:
        newPar["DATA_RANGE"][0] = FIRST_FRAME
    if LAST_FRAME:
        newPar["DATA_RANGE"][1] = LAST_FRAME
    if ICE:
        newPar.update(EXCLUDE_ICE_RING)
    if SPG and CELL:
        newPar["SPACE_GROUP_NUMBER"] = SPG
        newPar["UNIT_CELL_CONSTANTS"] = CELL
    elif SPG and not CELL:
        WARN_MSG = "  WARNING: Spacegroup is defined but not cell."
        WARN_MSG += " Waiting for indexation for setting cell."
    elif CELL and not SPG:
        WARN_MSG = "  WARNING: Cell is defined but not spacegroup,"
        WARN_MSG += " setting spacegroup to P1."
        newPar["SPACE_GROUP_NUMBER"] = 1
        newPar["UNIT_CELL_CONSTANTS"] = CELL
    if DISTANCE:
        newPar["DETECTOR_DISTANCE"] = DISTANCE
    if REFERENCE:
        if REFERENCE[0] == "/" or REFERENCE[0] == "~":
            newPar["REFERENCE_DATA_SET"] = REFERENCE
        else:
            newPar["REFERENCE_DATA_SET"] = "../"+REFERENCE
    if OSCILLATION:
        newPar["OSCILLATION_RANGE"] = OSCILLATION
    if ORIENTATION_MATRIX:
        try:
            _spg, cell, omat = _get_omatrix(ORIENTATION_MATRIX)
            SPG, _spg_info, _spg_str = parse_spacegroup(_spg)
            newPar["SPACE_GROUP_NUMBER"] = SPG
            newPar["UNIT_CELL_CONSTANTS"] = cell
            newPar["UNIT_CELL_A_AXIS"] = omat[0]
            newPar["UNIT_CELL_B_AXIS"] = omat[1]
            newPar["UNIT_CELL_C_AXIS"] = omat[2]
        except:
            prnt("Can't import orientation matrix from: %s" % \
                                          ORIENTATION_MATRIX, CRITICAL)
    if WAVELENGTH:
        newPar["X_RAY_WAVELENGTH"] = WAVELENGTH
    if OPTIMIZE in [1, 4]:
        gxparm2xpar(newDir)
    if OPTIMIZE >= 2:
        profiles = getProfilRefPar(os.path.join(newDir, "INTEGRATE.LP"))
        newPar.update(profiles)
        prnt("UPDATED PROFILES: %s" % profiles)
    if OPTIMIZE == 4:
        newPar["REFINE(INTEGRATE)= NONE"]
    if XDS_INPUT:
       newPar.update(xdsInp2Param(inp_str=XDS_INPUT))
    if "_HIGH_RESOL_LIMIT" in newPar:
        newPar["INCLUDE_RESOLUTION_RANGE"] = RES_LOW, \
                                             newPar["_HIGH_RESOL_LIMIT"]
    if RES_HIGH:
        newPar["INCLUDE_RESOLUTION_RANGE"] = RES_LOW, RES_HIGH

    if _linkimages:
        collect.setDirectory(link_dir_name)
    else:
        collect.setDirectory(newrun.link_name_to_image)

    newPar["NAME_TEMPLATE_OF_DATA_FRAMES"] = collect.xdsTemplate

    if "SPECIFIC_KEYWORDS" in newPar.keys():
        specific_keys = newPar["SPECIFIC_KEYWORDS"]
        del newPar["SPECIFIC_KEYWORDS"]
    else:
        specific_keys = ""
    newrun.inpParam.mix(xdsInp2Param(inp_str=xdsinp_base+specific_keys))
    newrun.inpParam.mix(newPar)
    newrun.set_collect_dir(os.path.abspath(imgDir))
    newrun.run_dir = newDir
    # Setting DELPHI as a fct of OSCILLATION_RANGE, MODE and NPROC
    _MIN_DELPHI = 5. # in degree
    #_DELPHI = NUM_PROCESSORS * newrun.inpParam["OSCILLATION_RANGE"]
    _DELPHI = 6 #* newrun.inpParam["OSCILLATION_RANGE"]
    #while _DELPHI < _MIN_DELPHI:
    #    _DELPHI *= 2
    newrun.inpParam["DELPHI"] = _DELPHI
    newrun.inpParam["MAXIMUM_NUMBER_OF_JOBS"] = 1
    if SLOW:
        newrun.inpParam["DELPHI"] *= 2
        newrun.mode.append("slow")
    if WEAK:
        newrun.mode.append("weak")

    xds_version = myXDSLogParser().get_xds_version()
    xdsme_version = __version__
    if XDS_PATH:
        prnt(">> XDS_PATH set to: %s" % XDS_PATH)
    prnt(FMT_VERSION % (xds_version, xdsme_version))
    prnt(FMT_HELLO % vars(newrun.inpParam))
    prnt("  Selected resolution range:       %.2f - %.2f A" % \
                                           newPar["INCLUDE_RESOLUTION_RANGE"])
    prnt("  Number of processors available:    %3d\n" % NUM_PROCESSORS)

    if WARN_MSG:
        prnt(WARN_MSG, WARNING)
    if SPG:
        prnt(_spg_str)

    #newrun.run()
    R1 = R2 = R3 = R4 = R5 = None
    if STEP > 1:
        prnt("\n Starting at step: %d (%s)\n" % (STEP, JOB_STEPS[STEP-1]))
    if STEP <= 1:
        R1 = newrun.run_init()
    if STEP <= 2:
        R2 = newrun.run_colspot()
    if STEP <= 3:
        if RES_HIGH:
            prnt("   Applying a SPOT RESOLUTION CUTOFF: %.2f A" % RES_HIGH)
            # July 2013: spot resolution cutoff is now included in xds
            #newrun.spots_resolution_cutoff(RES_HIGH, verbose=True)
        R3 = newrun.run_idxref(_beam_center_optimize,
                               _beam_center_ranking,
                               _beam_center_swap)
    if R3:
        i = 0
        _selected_cell = []
        prnt("    TABLE OF POSSIBLE LATTICES:\n")
        prnt(" num  Symm  quality  mult     a      b      c   " +
              "alpha   beta  gamma\n "+"-"*67)
        fmt_lat = "%3d) %5s %7.2f  %4d  %s"
        for LAT in R3["lattices_table"]:
            if LAT.fit <= LATTICE_GEOMETRIC_FIT_CUTOFF:
                i += 1
                prnt(fmt_lat % (i, LAT.symmetry_str1,
                            LAT.fit, LAT.multiplicity, LAT))
        # If not multiple possible solutions (like P2, or P1...)try to define
        # unitcell from spacegroup.
        #if _spg and not _cell:
        if (len(collect.imageRanges) > 1) or STRATEGY:
            newrun.run_xplan(ridx=R3)
    if STEP <= 4:
        R4 = newrun.run_integrate(collect.imageRanges)
    if STEP <= 5:
        (h, l), spgn  = newrun.run_pre_correct()
        newrun.run_correct((h, l), spgn)
    
