#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object allows to define and carry out a collection of series of wedges of diffraction images of arbitrary slicing parameter and of arbitrary size at arbitrary reference angles.
"""

import os
import time
import pickle
import logging
import traceback
import gevent

import numpy as np
import h5py
import re
import shutil
import subprocess

try:
    import xmlrpclib
except ImportError:
    xmlrpclib = None

from omega_scan import omega_scan


class reference_images(omega_scan):
    actuator_names = ["Omega"]

    specific_parameter_fields = [
        {"name": "scan_start_angles", "type": "", "description": ""},
        {"name": "dose_rate", "type": "", "description": ""},
        {"name": "dose_limit", "type": "", "description": ""},
        {"name": "vertical_scan_length", "type": "", "description": ""},
        {"name": "vertical_step_size", "type": "", "description": ""},
        {"name": "inverse_direction", "type": "", "description": ""},
        {"name": "vertical_motor_speed", "type": "", "description": ""},
        {
            "name": "exposure_time_per_frame",
            "type": "float",
            "description": "frame time in s",
        },
        {
            "name": "keep_originals",
            "type": "bool",
            "description": "keep original master files?",
        },
        {
            "name": "generate_sum",
            "type": "bool",
            "description": "generate summed images ?",
        },
    ]

    def __init__(
        self,
        name_pattern="ref-test_$id",
        directory="/tmp",
        scan_range=1.0,
        scan_exposure_time=0.1,
        scan_start_angles="[0, 90, 180, 270]",
        angle_per_frame=0.1,
        image_nr_start=1,
        vertical_scan_length=0,
        vertical_step_size=0.025,
        inverse_direction=True,
        dose_rate=0.25e6,  # Grays per second
        dose_limit=20e6,  # Grays
        i2s_at_highest_resolution=1.0,
        frames_per_second=None,
        position=None,
        kappa=None,
        phi=None,
        photon_energy=None,
        resolution=None,
        detector_distance=None,
        detector_vertical=None,
        detector_horizontal=None,
        transmission=None,
        flux=None,
        snapshot=None,
        diagnostic=None,
        analysis=None,
        simulation=None,
        parent=None,
        treatment_directory="/dev/shm",
        xmlrpc_server="http://localhost:60006",
        mxcube_parent_id=None,
        mxcube_gparent_id=None,
        keep_originals=False,
        generate_sum=True,
    ):
        logging.debug(
            "reference_images __init__ len(reference_images.specific_parameter_fields) %d"
            % len(reference_images.specific_parameter_fields)
        )

        if hasattr(self, "parameter_fields"):
            logging.debug(
                "reference_images __init__ len(self.parameter_fields) %d"
                % len(self.parameter_fields)
            )
            self.parameter_fields += reference_images.specific_parameter_fields
        else:
            self.parameter_fields = reference_images.specific_parameter_fields[:]

        logging.debug(
            "reference_images __init__ len(self.parameters_fields) %d"
            % len(self.parameter_fields)
        )

        if isinstance(scan_start_angles, str):
            scan_start_angles = eval(scan_start_angles)
        self.scan_start_angles = scan_start_angles
        self.scan_range = float(scan_range)

        self.vertical_scan_length = float(vertical_scan_length)
        self.vertical_step_size = float(vertical_step_size)

        if self.vertical_scan_length != 0 and self.vertical_scan_length != None:
            nimages = int(self.vertical_scan_length / self.vertical_step_size)
            angle_per_frame = self.scan_range / nimages

        ntrigger = len(self.scan_start_angles)
        nimages_per_file = int(self.scan_range / angle_per_frame)

        omega_scan.__init__(
            self,
            name_pattern,
            directory,
            scan_range=scan_range,
            scan_start_angle=self.scan_start_angles[0],
            scan_exposure_time=scan_exposure_time,
            angle_per_frame=angle_per_frame,
            image_nr_start=image_nr_start,
            frames_per_second=frames_per_second,
            position=position,
            kappa=kappa,
            phi=phi,
            photon_energy=photon_energy,
            resolution=resolution,
            detector_distance=detector_distance,
            detector_vertical=detector_vertical,
            detector_horizontal=detector_horizontal,
            transmission=transmission,
            flux=flux,
            snapshot=snapshot,
            ntrigger=ntrigger,
            nimages_per_file=nimages_per_file,
            diagnostic=diagnostic,
            analysis=analysis,
            simulation=simulation,
            parent=parent,
            mxcube_parent_id=mxcube_parent_id,
            mxcube_gparent_id=mxcube_gparent_id,
        )

        self.ntrigger = ntrigger

        self.total_expected_exposure_time = scan_exposure_time * ntrigger
        self.total_expected_wedges = ntrigger

        self.inverse_direction = inverse_direction
        self.dose_rate = dose_rate
        self.dose_limit = dose_limit
        self.i2s_at_highest_resolution = i2s_at_highest_resolution

        self.saved_parameters = self.load_parameters_from_file()

        self.treatment_directory = treatment_directory

        self.format_dictionary = {
            "directory": self.directory,
            "name_pattern": self.name_pattern,
            "treatment_directory": self.treatment_directory,
        }
        self.description = "Reference images, Proxima 2A, SOLEIL, %s" % time.ctime(
            self.timestamp
        )
        if xmlrpclib != None:
            self.server = xmlrpclib.ServerProxy(xmlrpc_server)
        else:
            self.server = False

        self.keep_originals = keep_originals
        self.generate_sum = generate_sum

    def get_nimages_per_file(self):
        if self.saved_parameters is not None:
            return self.saved_parameters["nimages_per_file"]
        return int(round(self.scan_range / self.angle_per_frame))

    def get_exposure_time_per_frame(self):
        if self.saved_parameters is not None:
            if "exposure_time_per_frame" in self.saved_parameters:
                return self.saved_parameters["exposure_time_per_frame"]
        return self.scan_exposure_time / self.get_nimages()

    def get_nimages(self, epsilon=1e-3):
        if self.saved_parameters is not None:
            return self.saved_parameters["nimages"]

        nimages = int(self.scan_range / self.angle_per_frame)
        if abs(nimages * self.angle_per_frame - self.scan_range) > epsilon:
            nimages += 1
        return nimages

    def get_ntrigger(self):
        if self.saved_parameters is not None:
            return self.saved_parameters["ntrigger"]
        return self.ntrigger

    def get_vertical_motor_speed(self):
        if self.saved_parameters is not None:
            return self.saved_parameters["vertical_motor_speed"]
        return self.vertical_scan_length / self.scan_exposure_time

    def get_angle_per_frame(self):
        if self.saved_parameters is not None:
            return self.saved_parameters["angle_per_frame"]
        return self.angle_per_frame

    def save_snapshots(self):
        self.logger.info("save_snapshots")
        snapshots = []
        self.goniometer.insert_backlight()
        for k, scan_start_angle in enumerate(self.scan_start_angles):
            self.goniometer.set_orientation(scan_start_angle)
            imagename, image, image_id = self.camera.save_image(
                "%s_%.2f_rgb.png" % (self.get_template(), scan_start_angle), color=True
            )
            snapshots.append(image)
        self.rgbimage = snapshots
        self.goniometer.extract_backlight()

    def run(self, wait=True):
        self.logger.info("expected files %s" % self.get_expected_files())
        if self.snapshot == True:
            self.save_snapshots()
        task_ids = []
        self.md_task_info = []
        vertical_scan_length = self.get_vertical_scan_length()
        for k, scan_start_angle in enumerate(self.scan_start_angles):
            self.logger.info("scan_start_angle %s" % scan_start_angle)
            if vertical_scan_length == 0:
                task_id = self.goniometer.omega_scan(
                    scan_start_angle,
                    self.scan_range,
                    self.scan_exposure_time,
                    wait=wait,
                )
            else:
                if self.inverse_direction == True:
                    vertical_scan_length = self.get_vertical_scan_length() * pow(-1, k)
                task_id = self.goniometer.vertical_helical_scan(
                    vertical_scan_length,
                    position,
                    scan_start_angle,
                    self.scan_range,
                    self.scan_exposure_time,
                    wait=wait,
                )
            task_ids.append(task_id)
            self.md_task_info.append(self.goniometer.get_task_info(task_id))

    def get_scan_start_angles(self):
        if os.path.isfile(self.get_parameters_filename()):
            return self.get_pickled_file(self.get_parameters_filename())[
                "scan_start_angles"
            ]
        else:
            return self.scan_start_angles

    def analyze(self, block=False):
        self.logger.info(
            "reference_images analysis expected files %s" % self.get_expected_files()
        )
        command = "reference_images.py"

        sense_line = '%s -d %s -n %s -A --scan_start_angles "%s" &' % (
            command,
            self.directory,
            self.name_pattern,
            self.get_scan_start_angles(),
        )
        if block:
            sense_line = sense_line.replace(" &", "")
        if self.keep_originals:
            sense_line = sense_line.replace(
                "-A --scan_start_angles", "-A --keep_originals --scan_start_angles"
            )
        if self.generate_sum:
            sense_line += " --generate_sum"
            
        self.logger.info("analysis line %s" % sense_line)
        os.system(sense_line)
        # subprocess.call(sense_line, shell=True)

    def analyze_online(self, generate_sum=True):
        self.logger.info("analyze")
        try:
            self.rectify_master()
        except:
            self.logger.info(traceback.format_exc())
        self.run_dozor()
        if self.generate_sum or generate_sum:
            try:
                #print('would generate summed images')
                self.generate_summed_h5()
            except:
                self.logger.info("problem in generate_summed_h5()")
                logging.exception(traceback.format_exc())
        strategy = None
        self.run_xds()
        self.run_best()
        strategy = self.parse_best()
        self.logger.info("best_strategy")
        self.logger.info(str(strategy))
        return strategy

    def rectify_master(self, sleeptime=0.5, timeout=3):
        self.logger.info("rectify_master")
        _start = time.time()
        expected_files = self.get_expected_files()
        self.logger.info("expected files:")
        self.logger.info(str(expected_files))
        while not self.expected_files_present() and time.time() - _start < timeout:
            gevent.sleep(sleeptime)
        if not self.expected_files_present():
            logging.debug(
                "expected files not present, exiting rectify_master, please check."
            )
            return -1
        else:
            self.logger.info(
                "rectify_master: all files appeared after %.2f seconds"
                % (time.time() - _start)
            )

        for f in expected_files:
            shutil.copy2("%s/%s" % (self.directory, f), self.treatment_directory)
        if self.keep_originals:
            originals_directory = os.path.join(self.directory, "originals")
            if not os.path.isdir(originals_directory):
                os.makedirs(originals_directory)
            for f in expected_files:
                os.rename(
                    os.path.join(self.directory, f),
                    os.path.join(originals_directory, f),
                )

        m = h5py.File(
            "%s/%s_master.h5" % (self.treatment_directory, self.name_pattern), "r+"
        )

        ntrigger = self.get_ntrigger()
        nimages = self.get_nimages()
        angle_per_frame = self.get_angle_per_frame()

        self.logger.info("ntrigger %s" % str(ntrigger))
        self.logger.info("nimages %s" % str(nimages))
        self.logger.info("angle_per_frame %s" % str(angle_per_frame))

        omega = []
        omega_end = []
        self.scan_start_angles = self.get_scan_start_angles()
        absolute_start = self.scan_start_angles[0]
        print("absolute_start", absolute_start)
        print(
            'range(len(m["/entry/data"].keys()))', range(len(m["/entry/data"].keys()))
        )
        print("self.scan_start_angles", self.scan_start_angles)
        for k in range(len(list(m["/entry/data"].keys()))):
            start = self.scan_start_angles[k]
            end = start + nimages * angle_per_frame
            print("in rectify_master start, end", start, end)
            omega += list(np.arange(start, end, angle_per_frame)[:nimages])
            omega_end += list(
                np.arange(
                    start + angle_per_frame, end + angle_per_frame, angle_per_frame
                )[:nimages]
            )
            image_nr_low = int(1 + (start - absolute_start) / angle_per_frame)
            image_nr_high = image_nr_low + nimages - 1
            print("in rectify_master low, high", image_nr_low, image_nr_high)
            try:
                filename = os.path.basename(
                    m["/entry/data/data_%06d" % (k + 1,)].file.filename
                )
                del m["/entry/data/data_%06d" % (k + 1,)]
                consecutive_wedge_number = int(image_nr_high * angle_per_frame)
                m[
                    "/entry/data/data_%06d" % (consecutive_wedge_number,)
                ] = h5py.ExternalLink(filename, "/entry/data/data")
                m["/entry/data/data_%06d" % (consecutive_wedge_number,)].attrs[
                    "image_nr_low"
                ] = image_nr_low
                m["/entry/data/data_%06d" % (consecutive_wedge_number,)].attrs[
                    "image_nr_high"
                ] = image_nr_high
            except:
                self.logger.info("links seem to be already updated")
        self.logger.info("omega %s" % str(omega))
        m["/entry/sample/goniometer/omega"].write_direct(np.array(omega))
        m["/entry/sample/goniometer/omega_end"].write_direct(np.array(omega_end))

        m.close()

        for f in expected_files:
            shutil.move("%s/%s" % (self.treatment_directory, f), self.directory)
        self.logger.info("rectify_master took %.2f seconds" % (time.time() - _start))

    def generate_summed_h5(self):
        self.logger.info("generate_summed_h5")
        _start = time.time()
        if os.path.isfile(
            "{directory}/{name_pattern}_sum10_master.h5".format(
                **self.format_dictionary
            )
        ):
            self.logger.info("summed images already generated")
            return
        self.format_dictionary["nimages_per_file"] = self.get_nimages_per_file()
        self.format_dictionary["treatment_directory"] = self.treatment_directory
        generate_summed_h5_line = "cd {directory}; /usr/local/experimental_methods/summer_devel.py -n {nimages_per_file} -m {name_pattern}_master.h5 &".format(
            **self.format_dictionary
        )
        if os.uname()[1] != "proxima2a-5":
            generate_summed_h5_line = 'ssh proxima2a-5 "%s" &' % generate_summed_h5_line
        self.logger.info("generate_summed_h5_line %s" % generate_summed_h5_line)
        os.system(generate_summed_h5_line)
        for f in ["data_000001.h5", "master.h5"]:
            a = "%s/%s_%s_%s" % (
                self.treatment_directory,
                self.name_pattern,
                "sum%d" % (self.get_nimages_per_file()),
                f,
            )
            self.logger.info("copying %s to %s " % (a, self.directory))
            shutil.copy2(
                "%s/%s_%s_%s"
                % (
                    self.treatment_directory,
                    self.name_pattern,
                    "sum%d" % (self.get_nimages_per_file()),
                    f,
                ),
                self.directory,
            )
            os.remove(
                "%s/%s_%s_%s"
                % (
                    self.treatment_directory,
                    self.name_pattern,
                    "sum%d" % (self.get_nimages_per_file()),
                    f,
                )
            )
        for f in self.get_expected_files():
            os.remove(os.path.join(self.treatment_directory, f))
        self.logger.info("summed images generation took %.2f" % (time.time() - _start,))

    def generate_cbf(self):
        self.logger.info("generate_cbf")
        _start = time.time()
        generate_cbf_line = "cd {treatment_directory}; /usr/local/bin/H5ToCBF.py -m {directory}/{name_pattern}_master.h5 -d {directory}/process".format(
            **self.format_dictionary
        )
        if os.uname()[1] != "process1":
            generate_cbf_line = 'ssh process1 "%s"' % generate_cbf_line
        self.logger.info("generate_cbf_line %s" % generate_cbf_line)
        os.system(generate_cbf_line)
        os.system("touch {directory}".format(**self.format_dictionary))
        self.create_ordered_cbf_links()
        self.logger.info("generate_cbf took %.2f" % (time.time() - _start,))

    def run_xds(self):
        self.logger.info("run_xds")
        xdsme_directory = "{directory}/process/xdsme_auto_{name_pattern}".format(
                        **self.format_dictionary
                    )
        if os.path.isfile(os.path.join(xdsme_directory, "CORRECT.LP")):
            xds_line = ""
        else:
            try:
                os.makedirs(xdsme_directory)
            except:
                pass
            # xds_line = 'cd {directory}/process; ref_xdsme -p auto_{name_pattern} -i "LIB= /nfs/data/xds-zcbf.so" {directory}/{name_pattern}_cbf/{name_pattern}_??????.cbf.gz'.format(**self.format_dictionary)
            xds_line = 'cd {directory}/process; ref_xdsme -p auto_{name_pattern} -i "LIB= /nfs/data/xds-zcbf.so" {directory}/{name_pattern}_??????.cbf.gz'.format(
                **self.format_dictionary
            )
        # if os.uname()[1] != 'process1':
        # xds_line = "ssh process1 '%s'" % xds_line
        self.logger.info("xds_line %s" % xds_line)
        subprocess.call(xds_line, shell=True)
        
        # best_log_file = '{directory}/{name_pattern}_cbf/process/{name_pattern}_best.log'.format(**self.format_dictionary)
        #best_log_file = "{directory}/process/{name_pattern}_best.log".format(
            #**self.format_dictionary
        #)
        #if os.path.isfile(best_log_file) and os.stat(best_log_file).st_size > 200:
            #return

        #os.environ["besthome"] = "/usr/local/bin"
        #best_line = "echo besthome $besthome; export besthome=/usr/local/bin; /usr/local/bin/best -f eiger9m -t {exposure_time} -e none -i2s 1. -M 0.005 -S 120 -Trans {transmission:.1f} -w 0.001 -GpS {dose_rate} -dna {directory}/process/{name_pattern}_best_strategy.xml -xds {directory}/process/xdsme_auto_{name_pattern}/CORRECT.LP {directory}/xdsme_auto_{name_pattern}/BKGINIT.cbf {directory}/process/xdsme_auto_{name_pattern}/XDS_ASCII.HKL | tee {directory}/process/{name_pattern}_best.log ".format(
            #**{
                #"directory": self.directory,
                #"name_pattern": self.name_pattern,
                #"exposure_time": self.get_exposure_time_per_frame(),
                #"dose_rate": self.get_dose_rate(),
                #"dose_limit": self.get_dose_limit(),
                #"transmission": self.get_transmission(),
            #}
        #)
        #self.logger.info("best_line %s" % best_line)
        #if xds_line != "":
            #total_line = "%s && %s" % (xds_line, best_line)
        #else:
            #total_line = best_line
        #self.logger.info("total_line %s" % total_line)
        #subprocess.call(total_line, shell=True)

    def xds_results_present(self, correct_file, xds_ascii_file, bkginit_file):
        return (os.path.isfile(correct_file) and os.path.isfile(xds_ascii_file) and os.path.isfile(bkginit_file))
    
    def run_best(self, sleeptime=1.0, timeout=3.5):
        self.logger.info("run_best")
        best_log_file = "{directory}/process/{name_pattern}_best.log".format(
            **self.format_dictionary
        )
        if os.path.isfile(best_log_file) and os.stat(best_log_file).st_size > 200:
            return
        best_line = "best -f eiger9m -t {exposure_time} -e none -M 0.005 -S 120 -Trans {transmission:.1f} -w 0.001 -GpS {dose_rate} -DMAX {dose_limit} -dna {directory}/process/{name_pattern}_best_strategy.xml -xds {directory}/process/xdsme_auto_{name_pattern}/CORRECT.LP {directory}/process/xdsme_auto_{name_pattern}/BKGINIT.cbf {directory}/process/xdsme_auto_{name_pattern}/XDS_ASCII.HKL | tee {directory}/process/{name_pattern}_best.log ".format(
            **{
                "directory": self.directory,
                "name_pattern": self.name_pattern,
                "exposure_time": self.get_exposure_time_per_frame(),
                "dose_rate": self.get_dose_rate(),
                "dose_limit": self.get_dose_limit(),
                "transmission": self.get_transmission(),
            }
        )

        correct_file = (
            "{directory}/process/xdsme_auto_{name_pattern}/CORRECT.LP".format(
                **{"directory": self.directory, "name_pattern": self.name_pattern}
            )
        )
        xds_ascii_file = (
            "{directory}/process/xdsme_auto_{name_pattern}/XDS_ASCII.HKL".format(
                **{"directory": self.directory, "name_pattern": self.name_pattern}
            )
        )
        bkginit_file = (
            "{directory}/process/xdsme_auto_{name_pattern}/BKGINIT.cbf".format(
                **{"directory": self.directory, "name_pattern": self.name_pattern}
            )
        )

        start = time.time()
        a = 0
        while not self.xds_results_present(correct_file, xds_ascii_file, bkginit_file) and time.time() - start < timeout:
            a += 1
            os.system(
                "touch {directory}/process/xdsme_auto_{name_pattern}".format(
                    **{"directory": self.directory, "name_pattern": self.name_pattern}
                )
            )
            self.logger.info(f"checking for xds_results to appear ... (attempt {a})")
            
            gevent.sleep(sleeptime)

        if self.xds_results_present(correct_file, xds_ascii_file, bkginit_file):
            os.system(best_line)

    def parse_best(self):
        try:
            l = open(
                "{directory}/process/{name_pattern}_best.log".format(
                    **{"directory": self.directory, "name_pattern": self.name_pattern}
                )
            ).read()
        except:
            l = "BEST strategy is not available"
            print(l)
            strategy = []
            return strategy
        #print("BEST strategy")
        #print(l)

        """                         Main Wedge  
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
        """
        strategy = []
        try:
            resolution = float(re.findall("Resolution limit =([\d\.]*) Angstrom", l)[0])
            transmission = float(re.findall("Transmission[\s=]*([\d\.]*)%", l)[0])
            distance = float(re.findall("Distance[\s=]*([\d\.]*)mm", l)[0])

            subwedge = " (\d)\s*"
            start = "([\d\.]*)\s*"
            width = "([\d\.]*)\s*"
            exposure = "([\d\.]*)\s*"
            nimages = "([\d]*)\|\|"
            search = subwedge + start + width + exposure + nimages

            wedges = re.findall(search, l)
        except IndexError:
            resolution = None
            transmission = None
            distance = None
            wedges = None
            if self.server != False:
                self.server.log_message("BEST analysis did not succeed")
            return strategy
        """
        [('1', '74.00', '0.15', '0.063', '767'),
        ('2', '189.05', '0.15', '0.161', '187')]
        """
        strategy_text = ""
        ls = l.split("\n")
        flag = False
        for line in ls:
            if "Main Wedge" in line:
                flag = True
            if "Phi_start" in line:
                flag = False
                break
            if flag == True:
                strategy_text += "%s\n" % line

        strategy = []
        for wedge in wedges:
            wedge_parameters = {}
            wedge_parameters["resolution"] = float(resolution)
            wedge_parameters["transmission"] = float(transmission)
            wedge_parameters["distance"] = float(distance)
            wedge_parameters["order"] = int(wedge[0])
            wedge_parameters["scan_start_angle"] = float(wedge[1])
            wedge_parameters["angle_per_frame"] = float(wedge[2])
            wedge_parameters["exposure_per_frame"] = float(wedge[3])
            wedge_parameters["nimages"] = int(wedge[4])
            wedge_parameters["scan_exposure_time"] = (
                wedge_parameters["nimages"] * wedge_parameters["exposure_per_frame"]
            )
            strategy.append(wedge_parameters)
        try:
            if self.server != False:
                self.server.log_message("BEST recomends the following parameters:")
                for wedge_parameters in strategy:
                    for key in [
                        "scan_start_angle",
                        "resolution",
                        "transmission",
                        "angle_per_frame",
                        "exposure_per_frame",
                        "nimages",
                    ]:
                        self.server.log_message("%s: %s" % (key, wedge_parameters[key]))
        except:
            self.logger.info("BEST recomends the following parameters:")
            for wedge_parameters in strategy:
                for key in [
                    "scan_start_angle",
                    "resolution",
                    "transmission",
                    "angle_per_frame",
                    "exposure_per_frame",
                    "nimages",
                ]:
                    self.logger.info("%s: %s" % (key, wedge_parameters[key]))

        if resolution > self.get_resolution():
            logging.getLogger("user_level_log").warning(
                "Best results indicate the current sample diffracts beyond currently set resolution, please consider approaching detector or increasing photon energy to measure diffraction to higher resolution."
            )
        return strategy

    def parse_xplan(self):
        pass

    def get_strategy(self):
        strategy = self.parse_best()
        if strategy:
            return strategy


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-n",
        "--name_pattern",
        default="ref-test_$id",
        type=str,
        help="Prefix default=%default",
    )
    parser.add_option(
        "-d",
        "--directory",
        default="/nfs/data/default",
        type=str,
        help="Destination directory default=%default",
    )
    parser.add_option(
        "-r", "--scan_range", default=1.2, type=float, help="Scan range [deg]"
    )
    parser.add_option(
        "-e",
        "--scan_exposure_time",
        default=0.06,
        type=float,
        help="Scan exposure time [s]",
    )
    parser.add_option(
        "-s",
        "--scan_start_angles",
        default="[0, 90, 180, 225, 315]",
        type=str,
        help="Scan start angles [deg]",
    )
    parser.add_option(
        "-a", "--angle_per_frame", default=0.1, type=float, help="Angle per frame [deg]"
    )
    parser.add_option(
        "-f", "--image_nr_start", default=1, type=int, help="Start image number [int]"
    )
    parser.add_option(
        "-v",
        "--vertical_scan_length",
        default=0,
        type=float,
        help="Vertical scan length [mm]",
    )
    parser.add_option(
        "-V",
        "--vertical_step_size",
        default=0.025,
        type=float,
        help="Vertical steps size [mm]",
    )
    # parser.add_option('-I', '--inverse_dierction', action='store_true', help='If set will invese direction of subsequent vertical scans')
    parser.add_option(
        "-R",
        "--dose_rate",
        default=0.25e6,
        type=float,
        help="Dose rate in Grays per second (default=%default)",
    )
    parser.add_option(
        "-L",
        "--dose_limit",
        default=15e6,
        type=float,
        help="Dose limit in Grays (default=%default)",
    )
    parser.add_option(
        "-i",
        "--position",
        default=None,
        type=str,
        help="Gonio alignment position [dict]",
    )
    parser.add_option(
        "-p", "--photon_energy", default=None, type=float, help="Photon energy "
    )
    parser.add_option(
        "-t", "--detector_distance", default=None, type=float, help="Detector distance"
    )
    parser.add_option(
        "-o", "--resolution", default=None, type=float, help="Resolution [Angstroem]"
    )
    parser.add_option("-x", "--flux", default=None, type=float, help="Flux [ph/s]")
    parser.add_option(
        "-m",
        "--transmission",
        default=None,
        type=float,
        help="Transmission. Number in range between 0 and 1.",
    )
    parser.add_option(
        "-T", "--snapshot", action="store_true", help="If set will record snapshots."
    )
    parser.add_option(
        "-A",
        "--analysis",
        action="store_true",
        help="If set will perform automatic analysis.",
    )
    parser.add_option(
        "-D",
        "--diagnostic",
        action="store_true",
        help="If set will record diagnostic information.",
    )
    parser.add_option(
        "-S",
        "--simulation",
        action="store_true",
        help="If set will record diagnostic information.",
    )
    parser.add_option(
        "--keep_originals", action="store_true", help="Keep original hdf5 files?"
    )
    
    parser.add_option(
        "--generate_sum", action="store_true", help="generate_sum?"
    )
    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    ri = reference_images(**vars(options))

    filename = "%s_parameters.pickle" % ri.get_template()

    if not os.path.isfile(filename):
        ri.execute()
    elif options.analysis == True:
        ri.analyze_online()


if __name__ == "__main__":
    main()
