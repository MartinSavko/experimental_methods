#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import time
import numpy as np
import shutil
import csv
import pickle

'''
rgl.viewpoint(theta = 45, phi = 30)
rgl.snapshot(filename='/tmp/dose_distribution.png')
#movie3d(spin3d(axis=c(1, 0, 0)), duration=18, dir=getwd())
'''

class raddose:
    template = '''
##############################################################################
#                                 Crystal Block                              #
##############################################################################

Crystal

Type Cuboid             # Cuboid
Dimensions {size_x} {size_y} {size_z}  # Dimensions of the crystal in X,Y,Z in µm.
#Type Polyhedron
#WireframeType OBJ
#ModelFile
#PDB 
PixelsPerMicron 0.1     # The computational resolution
AbsCoefCalc  RD3D       # Tells RADDOSE-3D how to calculate the
                        # Absorption coefficients

# Example case for insulin:
UnitCell  {unit_cell_a} {unit_cell_b} {unit_cell_c}  # unit cell size: a, b, c
                                # alpha, beta and gamma angles default to 90°
NumMonomers  {number_of_monomers}  # number of monomers in unit cell
NumResidues  {number_of_residues}  # number of residues per monomer
#ProteinHeavyAtoms {elements_protein_concentration} 
                                #Zn 0.333 S 6  # heavy atoms added to protein part of the
                                # monomer, i.e. S, coordinated metals,
                                # Se in Se-Met
                                # Note: If a sequence file is used S does not 
                                # need to be added
                                
#SolventHeavyConc {elements_solvent_concentration} #P 425 concentration of elements in the solvent
                                # in mmol/l. Oxygen and lighter elements
                                # should not be specified
SolventFraction {solvent_fraction} # fraction of the unit cell occupied by solvent


##############################################################################
#                                  Beam Block                                #
##############################################################################

Beam

Type Gaussian             # Gaussian profile beam
Flux {flux}               # in photons per second (2e12 = 2 * 10^12)
FWHM {beam_size_x} {beam_size_y} #in µm, X and Y for a Gaussian beam
                          # X=vertical and Y = horizontal for a 
                          # horizontal goniometer
                          # Opposite for a vertical goniometer

Energy {photon_energy}     # in keV

Collimation Rectangular 100 100 # 100 100 # X/Y collimation of the beam in µm
                                # X = vertical and Y = horizontal for a
                                # horizontal goniometer
                                # Opposite for a vertical goniometer



##############################################################################
#                                  Wedge Block                               #
##############################################################################

Wedge {oscillation_start} {oscillation_end}
                          # Start and End rotational angle of the crystal
                          # Start < End
ExposureTime {total_exposure_time} # Total time for entire angular range in seconds

# AngularResolution 2     # Only change from the defaults when using very
                          # small wedges, e.g 5°.
    '''
    #input_filename_template = '{size_x:.1f}_{size_y:.1f}_{size_z:.1f}_{unit_cell_a:.1f}_{unit_cell_b:.1f}_{unit_cell_c:.1f}_{number_of_monomers:d}_{number_of_residues:d}_{elements_protein_concentration:d}_{elements_solvent_concentration:d}_{solvent_fraction:.2f}_{flux:.4f}e12_{beam_size_x:.1f}_{beam_size_y:.1f}_{photon_energy:.2f}_{oscillation_start:.1f}_{oscillation_end:.1f}_{total_exposure_time:.1f}.txt'
    
    def __init__(self,
                 size_x=30.,
                 size_y=30.,
                 size_z=30.,
                 unit_cell_a=78.,
                 unit_cell_b=78.,
                 unit_cell_c=36.,
                 number_of_monomers=1,
                 number_of_residues=200,
                 elements_protein_concentration=0,
                 elements_solvent_concentration=0,
                 solvent_fraction=0.5,
                 flux=1.6e12,
                 beam_size_x=10.,
                 beam_size_y=5.,
                 photon_energy=12.65,
                 oscillation_start=0.,
                 oscillation_end=360.,
                 total_exposure_time=90.,
                 prefix='output-',
                 input_filename_template = '{flux:.4f}e12phs_{photon_energy:.3f}keV_{total_exposure_time:.4f}seconds.txt',
                 output_directory='/nfs/data2/raddose3d'):
        
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.unit_cell_a = unit_cell_a
        self.unit_cell_b = unit_cell_b
        self.unit_cell_c = unit_cell_c
        self.number_of_monomers = number_of_monomers
        self.number_of_residues = number_of_residues
        self.elements_protein_concentration = elements_protein_concentration
        self.elements_solvent_concentration = elements_solvent_concentration
        self.solvent_fraction = solvent_fraction
        self.flux = flux
        self.beam_size_x = beam_size_x
        self.beam_size_y = beam_size_y
        self.photon_energy = photon_energy
        self.oscillation_start = oscillation_start
        self.oscillation_end = oscillation_end
        self.total_exposure_time = total_exposure_time
        self.output_directory = output_directory
        self.prefix = prefix
        self.input_filename_template = input_filename_template
        self.expected_files = ['%sSummary.csv' % self.prefix,
                               '%sSummary.txt' % self.prefix,
                               '%sDoseState.csv' % self.prefix,
                               '%sDoseState.R' % self.prefix,
                               '%sRDE.csv' % self.prefix]
        
        self.model = np.poly1d([-3.37265615e-06,  3.38712165e-04, -1.45199019e-02,  3.46027507e-01,
       -4.99042890e+00,  4.41593748e+01, -2.27685441e+02,  5.55936351e+02])
        self.model_flux = 1.6e12
        self.model_total_exposure_time = 90.
        
    def get_parameters(self):
        parameters = { 
                      "size_x": self.size_x,
                      "size_y": self.size_y,
                      "size_z": self.size_z,
                      "unit_cell_a": self.unit_cell_a,
                      "unit_cell_b": self.unit_cell_b,
                      "unit_cell_c": self.unit_cell_c,
                      "number_of_monomers": self.number_of_monomers,
                      "number_of_residues": self.number_of_residues,
                      "elements_protein_concentration": self.elements_protein_concentration,
                      "elements_solvent_concentration": self.elements_solvent_concentration,
                      "solvent_fraction": self.solvent_fraction,
                      "flux": self.flux,
                      "beam_size_x": self.beam_size_x,
                      "beam_size_y": self.beam_size_y,
                      "photon_energy": self.photon_energy,
                      "oscillation_start": self.oscillation_start,
                      "oscillation_end": self.oscillation_end,
                      "total_exposure_time": self.total_exposure_time,
                      "total_oscillation": self.oscillation_end - self.oscillation_start
                      }
        return parameters
    
    def get_input_filename(self):
        parameters = self.get_parameters()
        parameters['flux'] /= 1e12
        filename = self.input_filename_template.format(**parameters)
        input_filename = os.path.join(self.output_directory, filename)
        return input_filename
    
    def get_output_directory(self):
        return self.output_directory
    
    def save_input_file(self):
        input_filename = self.get_input_filename()
        if os.path.isfile(input_filename):
            return
        if not os.path.isdir(os.path.dirname(input_filename)):
            os.makedirs(os.path.dirname(input_filename))
        text = self.template.format(**self.get_parameters())
        f = open(self.get_input_filename(), 'w')
        f.write(text)
        f.close()
        os.chmod(self.get_input_filename(), 0o777)

    def get_raddose_binary_path(self):
        return '/usr/local/bin/raddose3d.jar'
    
    def get_prefix(self):
        return self.prefix
    
    def run(self):
        self.save_input_file()
        line = 'ssh process1 "java -jar %s -i %s -p %s"' % (self.get_raddose_binary_path(), self.get_input_filename(), os.path.join(self.get_output_directory(), '%s_%s-' % (self.get_prefix(), self.get_template_name())))
        logging.debug('executing: %s' % line)
        os.system(line)
        self.save_summary_pickle()
        
    def get_template_name(self):
        return os.path.basename(self.get_input_filename().replace('.txt', ''))
                                
    def get_summary_pickle_name(self):
        return os.path.join(self.output_directory, '%s.pickle' % self.get_template_name())
    
    def save_summary_pickle(self):
        summary_filename = '%s_%s-Summary.csv' % (self.get_prefix(), self.get_template_name())
        filename = os.path.join(self.output_directory, summary_filename)
        k=0
        while not os.path.isfile(filename) and k<7:
            logging.debug('waiting for summary.csv %s to appear' % filename)
            time.sleep(0.5)
            k+=1
            
        f = open(filename)
        a = csv.reader(f)
        lines = [i for i in a]
        keys = [i.strip() for i in lines[0]]
        values = [float(i.strip()) for i in lines[1]]
        d = dict(list(zip(keys, values)))
        f.close()
        summary_pickle_file = open(self.get_summary_pickle_name(), 'wb')
        pickle.dump(d, summary_pickle_file)
        summary_pickle_file.close()
        os.chmod(self.get_summary_pickle_name(), 0o777)
 
    def get_summary_pickle(self):
        if os.path.isfile(self.get_summary_pickle_name()):
            summary_pickle = pickle.load(open(self.get_summary_pickle_name(), 'rb'), encoding="bytes")
        else:
            self.run()
            summary_pickle = pickle.load(open(self.get_summary_pickle_name(), 'rb'), encoding="bytes")
        return summary_pickle
    
    def get_DWD(self):
        try:
            DWD = self.get_summary_pickle()['DWD']
        except KeyError:
            DWD = -1
        return DWD

    def get_DWD_from_model(self):
        logging.debug('get_DWD_from_model')
        logging.debug('photon energy %.3f ' % self.photon_energy)
        x = self.model(self.photon_energy) 
        logging.debug('photon flux %.3f ' % self.flux)
        x *= (self.flux/self.model_flux) 
        logging.debug('total_exposure_time %.2f' % self.total_exposure_time)
        x *= (self.total_exposure_time/self.model_total_exposure_time)
        return x
        
        
if __name__ == '__main__':
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('-x', '--size_x', default=50., type=float, help='horizontal size of crystal')
    parser.add_option('-y', '--size_y', default=50., type=float, help='vertical size of crystal')
    parser.add_option('-z', '--size_z', default=50., type=float, help='depth of crystal')
    parser.add_option('-a', '--unit_cell_a', default=78., type=float, help='unit cell a parameter')
    parser.add_option('-b', '--unit_cell_b', default=78., type=float, help='unit cell b parameter')
    parser.add_option('-c', '--unit_cell_c', default=36., type=float, help='unit cell c parameter')
    parser.add_option('-m', '--number_of_monomers', default=1, type=int, help='number of monomers')
    parser.add_option('-r', '--number_of_residues', default=200, type=int, help='number of residues')
    parser.add_option('-p', '--elements_protein_concentration', default=0, type=float, help='heavy elements protein concentration')
    parser.add_option('-s', '--elements_solvent_concentration', default=0, type=float, help='heavy elements solvent concentration')
    parser.add_option('-f', '--solvent_fraction', default=0.5, type=float, help='solvent fraction')
    parser.add_option('-F', '--flux', default=1.6e12, type=float, help='flux')
    parser.add_option('-X', '--beam_size_x', default=10., type=float, help='horizontal beam size')
    parser.add_option('-Y', '--beam_size_y', default=5., type=float, help='vertical beam size')
    parser.add_option('-P', '--photon_energy', default=12.65, type=float, help='photon energy in keV')
    parser.add_option('-O', '--oscillation_start', default=0., type=float, help='oscillation start')
    parser.add_option('-E', '--oscillation_end', default=360., type=float, help='oscillation end')
    parser.add_option('-T', '--total_exposure_time', default=90., type=float, help='total exposure time')
    parser.add_option('-D', '--output_directory', default='/nfs/data2/raddose3d', type=str, help='output directory')
    parser.add_option('-n', '--prefix', default='output-', type=str, help='output prefix')
    
    options, args = parser.parse_args()
    print('options, args', options, args)
    rp = raddose(**vars(options))
    rp.run()
    
        
