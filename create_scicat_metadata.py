#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author Martin Savko (savko@synchrotron-soleil.fr)
# version 2023-06-22 -- add option to create a smaller master file (excluding pixel_mask, flatfield and diagnostics)


import h5py
import pickle
import os
import time
import shutil
import traceback
import numpy as np
import math
import random
import logging
import sys

log = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter(
    "%(filename)s |%(asctime)s |%(levelname)-7s| %(message)s"
)
stream_handler.setFormatter(stream_formatter)
log.addHandler(stream_handler)
log.setLevel(logging.DEBUG)

groups_classes = [
    ("/entry", "NXentry"),
    ("/entry/instrument", "NXinstrument"),
    ("/entry/instrument/detector", "NXdetector"),
    ("/entry/instrument/detector/detectorSpecific", "NXcollection"),
]

groups_to_copy = [
    "/entry/data",
    "/entry/sample",
    "/entry/instrument/beam",
    "/entry/instrument/detector/geometry/",
    "/entry/instrument/detector/goniometer",
]

required_datasets = [
    "/entry/instrument/detector/beam_center_x",
    "/entry/instrument/detector/beam_center_y",
    "/entry/instrument/detector/bit_depth_image",
    "/entry/instrument/detector/bit_depth_readout",
    "/entry/instrument/detector/count_time",
    "/entry/instrument/detector/countrate_correction_applied",
    "/entry/instrument/detector/description",
    "/entry/instrument/detector/detector_distance",
    "/entry/instrument/detector/detector_number",
    "/entry/instrument/detector/detector_readout_time",
    "/entry/instrument/detector/efficiency_correction_applied",
    "/entry/instrument/detector/flatfield_correction_applied",
    "/entry/instrument/detector/frame_time",
    "/entry/instrument/detector/pixel_mask_applied",
    "/entry/instrument/detector/sensor_material",
    "/entry/instrument/detector/sensor_thickness",
    "/entry/instrument/detector/threshold_energy",
    "/entry/instrument/detector/virtual_pixel_correction_applied",
    "/entry/instrument/detector/x_pixel_size",
    "/entry/instrument/detector/y_pixel_size",
    # u'/entry/instrument/detector/detectorSpecific/pixel_mask',
    "/entry/instrument/detector/detectorSpecific/auto_summation",
    "/entry/instrument/detector/detectorSpecific/calibration_type",
    "/entry/instrument/detector/detectorSpecific/compression",
    "/entry/instrument/detector/detectorSpecific/countrate_correction_bunch_mode",
    "/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff",
    "/entry/instrument/detector/detectorSpecific/data_collection_date",
    "/entry/instrument/detector/detectorSpecific/detector_readout_period",
    "/entry/instrument/detector/detectorSpecific/eiger_fw_version",
    "/entry/instrument/detector/detectorSpecific/element",
    "/entry/instrument/detector/detectorSpecific/frame_count_time",
    "/entry/instrument/detector/detectorSpecific/frame_period",
    "/entry/instrument/detector/detectorSpecific/module_bandwidth",
    "/entry/instrument/detector/detectorSpecific/nframes_sum",
    "/entry/instrument/detector/detectorSpecific/nimages",
    "/entry/instrument/detector/detectorSpecific/nsequences",
    "/entry/instrument/detector/detectorSpecific/ntrigger",
    "/entry/instrument/detector/detectorSpecific/number_of_excluded_pixels",
    "/entry/instrument/detector/detectorSpecific/photon_energy",
    "/entry/instrument/detector/detectorSpecific/roi_mode",
    "/entry/instrument/detector/detectorSpecific/software_version",
    "/entry/instrument/detector/detectorSpecific/summation_nimages",
    "/entry/instrument/detector/detectorSpecific/test_mode",
    "/entry/instrument/detector/detectorSpecific/trigger_mode",
    "/entry/instrument/detector/detectorSpecific/x_pixels_in_detector",
    "/entry/instrument/detector/detectorSpecific/y_pixels_in_detector",
]


def create_new_master(reference_master, new_name, minimal=True):
    _start = time.time()
    if minimal:
        new_m = h5py.File(new_name, "w")
        # for group, NX_class in groups_classes:
        # new_m.create_group(group)
        # new_m[group].attrs.create('NX_class',  NX_class)
        # for group in groups_to_copy:
        # reference_master.copy(group, new_m, name=group, shallow=False, expand_soft=True, expand_refs=True) #expand_external=True)
        # for dataset in required_datasets:
        # if os.path.basename(dataset) in list(reference_master[os.path.dirname(dataset)].keys()) and type(reference_master[dataset]) == h5py.Dataset:
        # new_m.create_dataset(dataset, data=reference_master[dataset][()])
        # else:
        # log.info('attempted creating dataset %s, which is not of the expected type, please check...' % dataset)
        rmf = reference_master.filename
        params = pickle.load(
            open(rmf.replace("_master.h5", "_parameters.pickle"), "rb")
        )
        new_m.create_group("/entry")
        new_m["/entry"].attrs.create("NX_class", "NXentry")
        new_m.create_dataset("/entry/beamline", data="Proxima2A")
        for key in [
            "duration",
            ("description", "title"),
            "start_time",
            "end_time",
            "photon_energy",
            "transmission",
            "flux",
            "detector_distance",
            "kappa",
            "phi",
            "resolution",
            "scan_range",
            "scan_exposure_time",
            "angle_per_frame",
            "frame_time",
            "degrees_per_second",
            "degrees_per_frame",
            "frames_per_second",
            "wavelength",
            "undulator_gap",
            "user_id",
            ("name_pattern", "sample_name"),
        ]:
            try:
                if type(key) is tuple:
                    new_m.create_dataset("/entry/%s" % key[1], data=params[key[0]])
                else:
                    new_m.create_dataset("/entry/%s" % key, data=params[key])
            except:
                print(key, "problem")
        new_m.create_group("/entry/files")
        new_m["/entry/files"].attrs.create("NX_class", "NXdata")
        new_m.create_dataset("/entry/files/master", data=[os.path.basename(rmf)])
        data_filenames = [
            os.path.basename(reference_master["/entry/data/%s" % item].file.filename)
            for item in reference_master["/entry/data"].keys()
        ]
        new_m.create_dataset("/entry/files/data_files", data=data_filenames)
        for fname, data_key in zip(
            data_filenames, reference_master["/entry/data"].keys()
        ):
            new_m["/entry/data/%s" % data_key] = h5py.ExternalLink(
                fname, "/entry/data/data"
            )
        new_m["/entry/data/master"] = h5py.ExternalLink(
            os.path.basename(reference_master.file.filename), "./"
        )
        new_m.close()
    else:
        shutil.copy(reference_master.filename, new_name)
    log.info("create_new_master took %.2f" % (time.time() - _start,))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--master",
        type=str,
        default="/nfs/data4/2023_Run3/com-proxima2a/2023-06-02/RAW_DATA/Nastya/px2-0007/pos14/dosing/point_15_omega_offset_0.015/pass_1_master.h5",
        help="master file",
    )

    args = parser.parse_args()

    master = h5py.File(args.master, "r")
    new_name = args.master.replace("_master.h5", "_scicat_metadata.h5")
    log.info("creating %s" % new_name)
    create_new_master(master, new_name, minimal=True)
