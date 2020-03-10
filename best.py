#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import re

'''
best --help
BEST:  Commmand Line Interface 
       ========================
SYNOPSIS
best -f {detector} -t {time} [OPTIONS] [FILES DENZO/HKL]
best -f {detector} -t {time} [OPTIONS] -mos [FILES/MOSFLM]
best -f {detector} -t {time} [OPTIONS] -xds [FILES/XDS]

Compulsory arguments
-f {detector}
-t {exposure time}
FILES
DENZO/HKL
    image_file file1.x {file2.x ...}
MOSFLM
    -mos{flm) bestfile.dat bestfile.par bestfile1.hkl {bestfile2.hkl ...}
XDS
    -xds CORRECT.LP BKGPIX.cbf XDS_ASCII_1.HKL {XDS_ASCII_2.HKL ...}
MOSFLM-XDS
   -MXDS bestfile.par BKGINIT.cbf bestfile1.hkl {bestfile2.hkl ...}
OPTIONS
--help gives help message
-i2s <I/SigI>, aimed <I/SigI> at aimed resolution, default 3.0
-i2s max  determine maximal reachable <I/SigI> at aimed resolution
-q minimize total time, default minimize the absorbed dose
-r {aimed resolution in �} , default automatic (by -T), >= ref.frame >= 0.9
-T(otal) {number}, maximum total exposure/measurement time, sec, default unlimited
-DMAX {number}, maximum permited total data collection DoseGy, default unlimited
-a,  Friedel low broken - taking into account anomalous scattering
-asad, strategy for SAD data collection, resolution selected automatically, rot.interval=360 dg.
-Rf {number}, target Rfiedel used for SAD resolution selectiondefault = 0.05
-GpS {GpS} , dose rate, Gray per Second, default 0.0 - radiation damage neglected
-sh(ape) {number}, shape factor, default 1, - increase for large crystal in a small beam
-su(susceptibility) {number}, default 1, - increase for radiation-sensitive crystals
-C(ompliteness) {number}, aimed completeness, default 0.99
-R(redundancy) {number}, aimed redundancy, default automatic
-p(hi) {start range}, user defined rotation range, default auto
-w {number}, minimum rotation width per frame, deg., default 0.05
-pl(an) {file} output plan file
-e {none|min|full}, equivalent complexity level {single line|few lines|complicated}
-g some useful plots generated (plotmtv required), default not
-o {file} plots generated and stored in file
-dna {file} xml formatted data stored in file
-S(peed) {number}, maximum rotation speed, deg/sec, (default fast)
-M(inexposure) {number}, minimum exposure per frame, sec, (default short)
-in {file} calculate statistics for data collection plan read from file
-m {number of detector read outs}, default 1
-d {preset counts} , default time mode
-s {.sca file name}, default - use .x file(s)
-l show configured detector formats
-SAD {no|yes|graph}, strategy for SAD data collection if "yes", "graph" - estimation of resolution for SAD
-low {never|ever|only}, calculate low resolution pass strategy, default "only"
-DamPar calculate plan for rad.damage coefficients determination
-Bonly only B and scale will calculate
-Trans {number}, initial image transmission(%),default 100
-TRmin {number}, minimum transmission (%),default 1
-Npos {number}, number of crystal positions,default 1
-DIS_MAX {number}, max limit crystal-detector distance, default 2000 mm
-DIS_MIN {number}, min limit crystal-detector distance, default 0 mm
'''

class best:
    
    def __init__(self,
                 xds_directory,
                 detector='eiger9m',
                 plan='best.plan',
                 exposure_time=0.025,
                 i2s=1.5,
                 DIS_MAX=1100.,
                 DIS_MIN=100.,
                 GpS=0.,
                 Minexposure=0.0043,
                 Speed=130.,
                 DMAX=20.e6,
                 Trans=100.,
                 g=True):
        
        self.xds_directory = xds_directory
        self.detector = detector
        self.plan = plan
        self.exposure_time = exposure_time
        self.i2s = i2s
        self.DIS_MAX = DIS_MAX
        self.DIS_MIN = DIS_MIN
        self.GpS = GpS
        self.Minexposure = Minexposure
        self.Speed = Speed
        self.DMAX = DMAX
        self.Trans = Trans
        self.g = g
            
    def get_best_line(self):
        
        if self.g == True:
            self.plot = '-g'
        else:
            self.plot = ''
        
        best_line = 'best -f {detector} -t {exposure_time} -M {Minexposure} -i2s {i2s} -S {Speed} -Trans {Trans} -GpS {GpS} -DMAX {DMAX} {plot} -o {plot_file} -dna {dna_file} -xds {xds_directory}/CORRECT.LP {xds_directory}/BKGINIT.cbf {xds_directory}/XDS_ASCII.HKL | tee {plan_file}'.format(detector=self.detector, exposure_time=self.exposure_time, Minexposure=self.Minexposure, Trans=self.Trans, GpS=self.GpS, Speed=self.Speed, DMAX=self.DMAX, plot=self.plot, xds_directory=self.xds_directory, plot_file=os.path.join(self.xds_directory, 'best_plots.mtv'), dna_file=os.path.join(self.xds_directory, 'best_strategy.xml'), plan_file=os.path.join(self.xds_directory, self.plan), i2s=self.i2s)
        
        return best_line
        
    def run(self):
        best_line = self.get_best_line()
        print 'best_line'
        print best_line
        os.system(best_line)
        
    def get_strategy(self):
        l = open('{plan_file}'.format(plan_file=os.path.join(self.xds_directory, self.plan))).read()
                 
        print('BEST strategy')
        print l
            
        '''                         Main Wedge  
                                 ================ 
        Resolution limit is set according to the given max.time               
        Resolution limit =2.48 Angstrom   Transmission =   10.0%  Distance = 275.5mm
        -----------------------------------------------------------------------------------------
                   WEDGE PARAMETERS       ||                 INFORMATION
        ----------------------------------||-----------------------------------------------------
        sub-| Phi  |Rot.  | Exposure| N.of||Over|sWedge|Exposure|Exposure| Dose  | Dose  |Comple-
         We-|start |width | /image  | ima-||-lap| width| /sWedge| total  |/sWedge| total |teness
         dge|degree|degree|     s   |  ges||    |degree|   s    |   s    | MGy   |  MGy  |  %    
        ----------------------------------||-----------------------------------------------------
         1    74.00   0.15     0.015   954|| No  143.10     14.2     14.2   3.540   3.540  100.0
        -----------------------------------------------------------------------------------------
        '''
        subwedge = ' (\d)\s*'
        start = '([\d\.]*)\s*'
        width = '([\d\.]*)\s*'
        exposure = '([\d\.]*)\s*'
        nimages = '([\d]*)\|\|'
        search = subwedge + start + width + exposure + nimages

        wedges = re.findall(search, l)
        '''
        [('1', '74.00', '0.15', '0.063', '767'),
        ('2', '189.05', '0.15', '0.161', '187')]
        '''
        strategy = []
        for wedge in wedges:
            wedge_parameters = {}
            wedge_parameters['order'] = int(wedge[0])
            wedge_parameters['scan_start_angle'] = float(wedge[1])
            wedge_parameters['angle_per_frame'] = float(wedge[2])
            wedge_parameters['exposure_per_frame'] = float(wedge[3])
            wedge_parameters['nimages'] = int(wedge[4])
            wedge_parameters['scan_exposure_time'] = wedge_parameters['nimages'] * wedge_parameters['exposure_per_frame']
            strategy.append(wedge_parameters)
    
        return strategy
        
def main():
    
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option('-d', '--xds_directory', default=None, type=str, help='Directory with XDS processing results')
    parser.add_option('-f', '--detector', default='eiger9m', type=str, help='Detector type')
    parser.add_option('-p', '--plan', default='best.plan', type=str, help='Plan file')
    parser.add_option('-e', '--exposure_time', default=0.025, type=float, help='Exposure time')
    parser.add_option('-I', '--i2s', default=1.5, type=float, help='I over sigma at highest resolution')
    parser.add_option('--DIS_MAX', default=1100., type=float, help='Maximum detector distance')
    parser.add_option('--DIS_MIN', default=100., type=float, help='Minimum detector distance')
    parser.add_option('--GpS', default=0., type=float, help='Dose rate in Grays per second')
    parser.add_option('--Minexposure', default=0.0043, type=float, help='Minimum exposure time')
    parser.add_option('--Speed', default=120., type=float, help='Maximum sample rotation speed in degrees per second')
    parser.add_option('--DMAX', default=20.e6, type=float, help='Maximum dose in Gray')
    parser.add_option('--Trans', default=100., type=float, help='Transmission')
    parser.add_option('-g', action='store_true', help='Generate useful plots')
    
    options, args = parser.parse_args()
    
    b = best(**vars(options))
    
    b.run()
    #strategy = b.get_strategy()
    
    #print 'best strategy for %s' % b.xds_directory
    #print strategy
    
if __name__ == '__main__':
    main()
