#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMca import McaAdvancedFit
from PyMca5.PyMca import ConfigDict
#from PyMca import McaAdvancedFit
#from PyMca import ConfigDict

import numpy as np

def main(spectrum='/nfs/data4/2023_Run2/com-proxima2a/2023-03-15/RAW_DATA/Commissioning/pos4_xrf_4_7.dat',
         calib=[0.0, 9.93475667754, -16.1723871876],
         config={'legend': 'pos4_xrf_9', 
                 'file': '/nfs/data2/Martin/Research/mxcube_2021/github/mxcubecore/mxcubecore/configuration/soleil/px2/production/experimental_methods/fit_configuration.cfg', 
                 'min': 0, 
                 'max': 2047, 
                 'htmldir': '/nfs/data4/2023_Run2/com-proxima2a/2023-03-15/ARCHIVE/Commissioning',
                 'sourcename': 'Proxima 2A, Synchrotron SOLEIL',
                 'time': 10,
                 'calibration': [0.0, 9.93475667754*1e-3, -16.1723871876*1e-3]}):
    
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)
    
    data = np.loadtxt(spectrum, skiprows=4)
    
    mca_widget = McaAdvancedFit.McaAdvancedFit()
    mca_widget.dismissButton.hide()
    
    x = data[:, 0]
    y = data[:, 1]
    xmin = config['min']
    xmax = config['max']
    calib = np.ravel(calib).tolist()
    
    
    d = ConfigDict.ConfigDict()
    d.read(config['file'])
    mca_widget.configure(d)
    
    
    #outfile = config['legend']
    #outdir = config['htmldir']
    #sourcename = config['legend']
    #report = McaAdvancedFit.QtMcaAdvancedFitReport.\
             #QtMcaAdvancedFitReport(None, 
                                    #outfile=outfile, 
                                    #outdir=outdir,
                                    #fitresult=None, 
                                    #sourcename=sourcename, 
                                    #plotdict={'logy':False}, 
                                    #table=2)

    #text = report.getText()
    #report.writeReport(text=text)
    mca_widget.setData(x, y, calibration=calib)
    
    #mca_widget._energyAxis = False
    #mca_widget.toggleEnergyAxis()
    
    mca_widget.show()
    app.exec()
    
if __name__ == '__main__':
    main()
    
    
    
