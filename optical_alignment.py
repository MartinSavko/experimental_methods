#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optical alignement procedure

"""
import os
import sys
import time
import zmq
import redis
import logging
import gevent
import traceback
import pickle
import h5py
import pylab
import copy
import pprint

from skimage.measure import marching_cubes
import open3d as o3d
import numpy as np
from scipy.optimize import leastsq, minimize
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
from math import cos, sin, sqrt, radians, atan, asin, acos, pi, degrees

from experiment import experiment
from oav_camera import oav_camera
from speaking_goniometer import speaking_goniometer

from film import film
from cats import cats
from optical_path_report import select_better_model, create_mosaic

from goniometer import goniometer

from useful_routines import (
    get_points_in_goniometer_frame,
    get_voxel_calibration,
    get_position_from_vector,
    get_vector_from_position,
    circle_model, 
    circle_model_residual,
    projection_model,
    projection_model_residual,
    get_vertical_and_horizontal_shift_between_two_positions,
    get_aligned_position_from_reference_position_and_shift,
    get_vertical_and_horizontal_shift_between_two_positions,
)

from shape_from_history import (
    get_reconstruction as _get_reconstruction,
    get_predictions,
    get_notion_string,
    principal_axes,
)

from perfect_realignment import (
    get_critical_points,
    get_vector_from_position,
)

c = cats()
g = goniometer()


def test_puck(puck=9, center=True):
    start = time.time()

    for sample in range(1, 17):
        print(f"sample {sample}")
        _start = time.time()
        c.mount(puck, sample, prepare_centring=center)
        if c.sample_mounted() and center:
            name_pattern = "autocenter_%s_element_%s_%s" % (
                os.getuid(),
                (puck, sample),
                time.asctime().replace(" ", "_").replace(":", ""),
            )
            throw_away_alignment(name_pattern=name_pattern)
        g.set_position(
            {"AlignmentZ": g.md.alignmentzposition + 0.1 * (np.random.random() - 0.5)}
        )
        _end = time.time()
        print(f"sample {sample} took {_end-_start:.4f} seconds")
        print("\n" * 7)
    c.get()
    end = time.time()
    print(f"puck {puck} took {end-start:.4f} seconds")


def throw_away_alignment(
    name_pattern="autocenter_%s_element_%s_%s"
    % (os.getuid(), c.get_mounted_sample_id(), time.asctime().replace(" ", "_")),
    directory=os.path.join(os.environ["HOME"], "manual_optical_alignment"),
    save_history=True,
    scan_range=0,
):
    start = time.time()
    oa = optical_alignment(
        name_pattern=name_pattern,
        directory=directory,
        scan_range=scan_range,
        angles="(0, 90, 180, 270)",
        backlight=True,
        analysis=True,
        conclusion=True,
        move_zoom=False,
        zoom=1,
        save_history=save_history,
    )

    oa.execute()
    end = time.time()
    print(f"sample {name_pattern} aligned in {end-start:.4f} seconds")
    print("\n" * 3)


def get_initial_parameters(aspect, name=None):
    c = np.mean(aspect)
    try:
        r = 0.5 * (max(aspect) - min(aspect))
    except:
        traceback.print_exc()
        print("name", name)
        print(aspect)
        try:
            r = np.std(aspect) / np.sin(np.pi / 4)
        except:
            r = 0.0
    alpha = np.random.rand() * np.pi

    return c, r, alpha


def get_bbox_patch(aoi_bbox, linewidth=2, edgecolor="green", facecolor="none"):
    r, c, h, w, area = aoi_bbox[1:]
    C, R = int(c - w / 2), int(r - h / 2)
    W, H = int(w), int(h)
    aoi_bbox_patch = pylab.Rectangle(
        (C, R), W, H, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor
    )

    return aoi_bbox_patch


def get_click_patch(click, radius=5, color="green"):
    click_patch = pylab.Circle(click[::-1], radius=radius, color=color)

    return click_patch


def annotate_image(image, description, alpha=1):
    pylab.figure(1, figsize=(16, 9))
    pylab.axis("off")
    pylab.grid(False)
    ax = pylab.gca()

    pylab.imshow(image)
    # pylab.imshow(description['hierarchical_mask'], alpha=alpha)
    try:
        aoi_bbox_patch = get_bbox_patch(description["aoi_bbox_px"])
        most_likely_click_patch = get_click_patch(description["most_likely_click_px"])
        extreme_patch = get_click_patch(description["extreme_px"], color="blue")
        end_patch = get_click_patch(description["end_likely_px"], color="red")
        start_patch = get_click_patch(description["start_likely_px"], color="red")
        start_possible_patch = get_click_patch(
            description["start_possible_px"], color="magenta"
        )

        ax.add_patch(aoi_bbox_patch)
        ax.add_patch(most_likely_click_patch)
        ax.add_patch(extreme_patch)
        ax.add_patch(end_patch)
        ax.add_patch(start_patch)
        ax.add_patch(start_possible_patch)
    except:
        traceback.print_exc()

    pylab.show()


def annotate_alignment(results, figsize=(8, 6)):
    fits = results["fits"]
    test_angles = np.linspace(0, 2 * np.pi, 360)
    k = 0
    for aspect in [
        "extreme_shift_mm_verticals",
        "extreme_shift_mm_from_position_verticals",
        "aoi_bbox_mm_heights",
        "extreme_hypotetical_shift_mm_from_position_verticals",
        "crystal_bbox_mm_verticals",
    ]:
        #'extreme_shift_mm_from_position_horizontals', 'aoi_bbox_mm_heights', 'aoi_bbox_mm_widths', 'crystal_bbox_mm_verticals', 'crystal_bbox_mm_areas']:
        k += 1
        if ("verticals" in aspect or "horizontals" in aspect) and "bbox" not in aspect:
            likely_model = circle_model(
                test_angles, *fits["results"][aspect]["fit_circle"].x
            )
        else:
            likely_model = projection_model(
                test_angles, *fits["results"][aspect]["fit_projection"].x
            )
        # best_model = fits['results'][aspect]['best_model'](test_angles, *fits['results'][aspect]['fit'].x)

        experiment_angles = fits["angles"][aspect]
        experiment_data = fits["aspects"][aspect]

        pylab.figure(k, figsize=figsize)
        pylab.title(aspect)
        pylab.plot(
            np.rad2deg(experiment_angles) % 360,
            experiment_data,
            "o",
            color="red",
            label="experiment",
        )
        pylab.plot(
            np.rad2deg(test_angles) % 360, likely_model, color="green", label="model"
        )
        pylab.xlabel("Omega [deg]")
        if aspect == "aoi_bbox_mm_heights":
            c_p, r_p, alpha_p = fits["results"][aspect]["fit"].x
            omega_max = results["omega_max"]
            omega_min = results["omega_min"]
            pylab.vlines(
                [omega_max], c_p - r_p, c_p + r_p, color="magenta", label="omega_max"
            )
            pylab.vlines(
                [omega_min], c_p - r_p, c_p + r_p, color="orange", label="omega_min"
            )
        pylab.legend()
    pylab.show()


def apply_threshold(reconstruction, threshold=0.95):
    volume = reconstruction > threshold * reconstruction.max()
    return volume

class optical_alignment(experiment):
    specific_parameter_fields = [
        {"name": "position", "type": "", "description": ""},
        {"name": "n_angles", "type": "", "description": ""},
        {"name": "angles", "type": "", "description": ""},
        {"name": "zoom", "type": "", "description": ""},
        {"name": "kappa", "type": "", "description": ""},
        {"name": "phi", "type": "", "description": ""},
        {"name": "calibration", "type": "", "description": ""},
        {"name": "beam_position_vertical", "type": "", "description": ""},
        {"name": "beam_position_horizontal", "type": "", "description": ""},
        {"name": "frontlight", "type": "bool", "description": ""},
        {"name": "backlight", "type": "bool", "description": ""},
        {"name": "generate_report", "type": "", "description": ""},
        {"name": "default_background", "type": "", "description": ""},
        {"name": "save_raw_background", "type": "bool", "description": ""},
        {"name": "save_history", "type": "bool", "description": ""},
        {"name": "extreme", "type": "bool", "description": ""},
        {"name": "film_step", "type": "bool", "description": ""},
        {"name": "verbose", "type": "bool", "description": ""},
        {"name": "vertical_clicks", "type": "list", "description": ""},
        {"name": "horizontal_clicks", "type": "list", "description": ""},
        {"name": "omega_clicks", "type": "list", "description": ""},
        {"name": "sample_seen", "type": "bool", "description": ""},
        {"name": "scan_range", "type": "float", "description": "scan_range"},
        {
            "name": "scan_exposure_time",
            "type": "float",
            "description": "scan_exposure_time",
        },
        {"name": "md_task_info", "type": "list", "description": "md_task_info"},
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        angles=[0, 90, 225, 315],
        scan_start_angle=None,
        scan_range=360,  # 360,
        scan_exposure_time=3.6,
        n_angles=25,
        zoom=None,
        kappa=None,
        phi=None,
        position=None,
        frontlight=False,
        backlight=True,
        phiy_direction=+1.0,  # MD2 was -1.0,
        phiz_direction=1.0,
        centringx_direction=+1.0,  # MD2 was -1.0,
        centringy_direction=+1.0,
        analysis=None,
        conclusion=None,
        generate_report=None,
        default_background=False,
        save_raw_background=False,
        save_history=False,
        extreme=False,
        move_zoom=False,
        film_step=-120.0,
        size_of_target=0.050,
        verbose=False,
        parent=None,
        debug=False,
        cats_api=None,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += self.specific_parameter_fields[:]
        else:
            self.parameter_fields = self.specific_parameter_fields[:]

        experiment.__init__(
            self,
            name_pattern,
            directory,
            analysis=analysis,
            conclusion=conclusion,
            cats_api=cats_api,
        )

        self.description = "Optical alignment, Proxima 2A, SOLEIL, %s" % time.ctime(
            self.timestamp
        )
        self.camera = oav_camera()
        self.cats = cats()
        self.goniometer = goniometer()
        self.speaking_goniometer = speaking_goniometer()

        self.n_angles = n_angles
        if type(angles) == str:
            self.angles = eval(angles)
        elif type(angles) in [list, tuple]:
            self.angles = angles
        elif self.n_angles != None:
            self.angles = np.linspace(0, 360, self.n_angles + 1)[:-1]

        if scan_start_angle is None:
            scan_start_angle = self.goniometer.get_omega_position()
        self.scan_start_angle = scan_start_angle
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time

        if zoom is None:
            zoom = self.camera.get_zoom()

        self.zoom = zoom
        self.kappa = kappa
        self.phi = phi

        self.position = self.goniometer.check_position(position)
        self.frontlight = frontlight
        self.backlight = backlight
        self.phiy_direction = phiy_direction
        self.phiz_direction = phiz_direction
        self.centringx_direction = centringx_direction

        if generate_report != True:
            generate_report = False
        self.generate_report = generate_report
        self.default_background = default_background
        self.save_raw_background = save_raw_background
        self.save_history = save_history
        self.extreme = extreme
        self.move_zoom = move_zoom
        self.film_step = film_step
        self.size_of_target = size_of_target
        self.verbose = verbose
        self.parent = parent
        self.debug = debug

        self.foreground_string = get_notion_string("foreground")
        self.crystal_loop_string = get_notion_string(["crystal", "loop"])
        self.possible_string = get_notion_string(["crystal", "loop", "stem"])
        self.last_optical_alignment_results_key = "last_optical_alignment_results"
        self.redis = redis.StrictRedis()
        self.innermost_start_time = None
        self.innermost_end_time = None
        self.vertical_clicks = []
        self.horizontal_clicks = []
        self.omega_clicks = []
        self.sample_seen = None
        self.md_task_info = []
        self.descriptions = None
        self.eagerly = False

    def get_kappa(self):
        if self.kappa is None:
            self.kappa = self.goniometer.get_kappa_position()

        return self.kappa

    def get_phi(self):
        if self.phi is None:
            self.phi = self.goniometer.get_phi_position()

        return self.phi

    def get_position(self):
        if self.position is None:
            if os.path.isfile(self.get_parameters_filename()):
                position = self.get_parameters()["position"]
            else:
                position = self.goniometer.get_aligned_position()
        else:
            position = self.position

        return position

    def get_beam_position_vertical(self):
        if os.path.isfile(self.get_parameters_filename()):
            beam_position_vertical = self.get_parameters()["beam_position_vertical"]
        else:
            beam_position_vertical = self.goniometer.md.beampositionvertical

        return beam_position_vertical

    def get_beam_position_horizontal(self):
        if os.path.isfile(self.get_parameters_filename()):
            beam_position_horizontal = self.get_parameters()["beam_position_horizontal"]
        else:
            beam_position_horizontal = self.goniometer.md.beampositionhorizontal

        return beam_position_horizontal

    def get_beam_position(self):
        v = self.get_beam_position_vertical()
        h = self.get_beam_position_horizontal()
        return np.array([v, h])

    def get_calibration(self):
        if os.path.isfile(self.get_parameters_filename()):
            calibration = self.get_parameters()["calibration"]
        else:
            calibration = np.array(
                [self.goniometer.md.coaxcamscaley, self.goniometer.md.coaxcamscalex]
            )
        return calibration

    def get_zoom(self):
        if os.path.isfile(self.get_parameters_filename()):
            zoom = self.get_parameters()["zoom"]
        else:
            zoom = self.camera.get_zoom()

        return zoom

    def get_reference_position(self):
        if os.path.isfile(self.get_parameters_filename()):
            reference_position = self.get_parameters()["position"]
        else:
            reference_position = self.goniometer.get_aligned_position()
        return reference_position

    def get_result_position(self):
        result_position = None
        if os.path.isfile(self.get_results_filename()):
            try:
                result_position = self.load_results()["result_position"]
            except:
                pass
        if result_position is None:
            result_position = self.goniometer.get_aligned_position()
        return result_position

    def check_previous_results(self):
        try:
            last_results = pickle.loads(
                self.redis.get(self.last_optical_alignment_results_key)
            )
            print("last_results present")
            print(last_results)
            if str(last_results["mounted_sample_id"]) == str(
                self.cats.get_mounted_sample_id()
            ):
                print(
                    "mounted_sample_id is the same as the previous one, will try to make use of it"
                )
                self.goniometer.set_position(last_results["result_position"])
        except:
            traceback.print_exc()
            print("last_results not available")

    def prepare(self):
        _start = time.time()
        self.check_directory(self.directory)
        # self.check_previous_results()
        self.sample_seen = False
        position = self.get_position()
        if self.kappa != None:
            if abs(self.goniometer.get_kappa_position() - self.kappa) > 0.05:
                self.goniometer.set_kappa_position(self.kappa, simple=False)
            position["Kappa"] = self.kappa
        if self.phi != None:
            if abs(self.goniometer.get_phi_position() - self.phi) > 0.05:
                self.goniometer.set_phi_position(self.phi)
            position["Phi"] = self.phi
        if self.get_zoom() != self.zoom:
            self.goniometer.set_zoom(self.zoom, wait=True)

        self.logger.info("about to set position %s" % str(position))
        self.goniometer.set_position(position)
        self.beam_position_vertical = self.get_beam_position_vertical()
        self.beam_position_horizontal = self.get_beam_position_horizontal()
        self.calibration = self.get_calibration()

        if len(self.angles) > 10 or self.scan_range is not None and self.scan_range > 0:
            self.eagerly = False
            if self.goniometer.get_current_phase() != "DataCollection":
                self.goniometer.set_data_collection_phase()
            self.goniometer.disable_fast_shutter()
        else:
            self.eagerly = True

        if not self.frontlight and not self.backlight:
            self.backlight = True
            
        if self.backlight:
            self.goniometer.insert_backlight()
        else:
            self.goniometer.extract_backlight()

        if self.frontlight:
            self.goniometer.insert_frontlight()
        else:
            self.goniometer.extract_frontlight()

        self.logger.info("prepare took %.3f seconds" % (time.time() - _start))

    def is_passive_advisable(self, description, threshold=0.25):
        return description["extreme_distance"] < threshold

    def analyse_single_view(
        self,
        image=None,
        reference_position=None,
        debug=False,
        image_name=None,
        save=True,
    ):
        _start = time.time()

        if image is None:
            image = self.camera.get_image()
        if save:
            try:
                if image_name is None:
                    image_name = "%s_%s.jpg" % (
                        self.get_template().replace(":", ""),
                        f"omega_{self.goniometer.get_omega_position():.2f}",
                    )
                self.camera.save_image(image_name, image=image, color=True)
            except:
                traceback.print_exc()

        analysis = self._get_predictions(image)
        description = analysis["descriptions"][0]

        self.logger.info("analysis took %.2f seconds" % (time.time() - _start))

        return description

    def describe_single_image(
        self,
        image=None,
        center=None,
        calibration=None,
        reference_position=None,
        angle=None,
        debug=None,
    ):
        if center is None:
            center = self.get_beam_position()
        if calibration is None:
            calibration = self.get_calibration()
        if reference_position is None:
            reference_position = self.goniometer.get_aligned_position()
        description = self.analyse_single_view(image=image, debug=debug)
        description = self.add_calibrated_data(
            description, center, calibration, reference_position, angle=angle
        )
        return description

    def _align_eagerly(self, step=0.25, debug=None, save=True):
        if not hasattr(self, "start_run_time"):
            self.start_run_time = time.time()
        self.eagerly = True
        if debug is None:
            debug = self.debug

        zoom = self.get_zoom()
        calibration = self.get_calibration()
        center = self.get_beam_position()

        descriptions = []
        omega_start = self.goniometer.get_omega_position()
        for omega in self.get_angles():
            angle = omega_start + omega
            d, r = divmod(angle, 360)
            self.goniometer.set_position({"Omega": r}, wait=True)
            self.goniometer.wait()
            reference_position = self.goniometer.get_aligned_position()
            real_position = self.goniometer.get_omega_position()
            # print(f"real omega position {real_position} (delta {r-real_position})")
            image = self.camera.get_image()

            description = self.describe_single_image(
                image=image,
                center=center,
                calibration=calibration,
                reference_position=reference_position,
                angle=r,
                debug=debug,
            )

            descriptions.append(description)

            most_likely_click = description["most_likely_click"]
            if most_likely_click[0] == -1:
                reference_position["AlignmentY"] += self.phiy_direction * step
                self.goniometer.set_position(reference_position)
                continue
            else:
                self.sample_seen = True
            aligned_position = description["most_likely_click_aligned_position"]
            self.goniometer.set_position(aligned_position)
            # input("main continue?")
            if debug:
                self.logger.info(
                    f"aligned_position {aligned_position}"
                )
                self.logger.info(
                    "most_likely_click %s (fractional %s) "
                    % (
                        description["most_likely_click_px"],
                        description["most_likely_click"],
                    )
                )
                self.logger.info(
                    "most_likely_click_shift_mm_from_position %s"
                    % description["most_likely_click_shift_mm_from_position"]
                )
                self.logger.info("aoi_bbox %s " % str(description["aoi_bbox"]))
                self.logger.info("aoi_bbox_px %s " % str(description["aoi_bbox_px"]))
                self.logger.info("aoi_bbox_mm %s " % str(description["aoi_bbox_mm"]))
                self.logger.info("aoi_bbox_area %s " % description["aoi_bbox_mm"][5])
                self.logger.info("extreme %s" % description["extreme"])
                self.logger.info("extreme_px %s" % description["extreme_px"])
                self.logger.info(
                    "start point %s (possible %s)"
                    % (description["start_likely_px"], description["start_possible_px"])
                )
                self.logger.info("end point %s" % description["end_likely_px"])
                self.logger.info(
                    "click shift %s mm" % description["most_likely_click_shift_mm"]
                )
                self.logger.info(
                    "extreme_shift %s mm" % description["extreme_shift_mm"]
                )
                self.logger.info(
                    "extreme distance from beam %.3f mm"
                    % description["extreme_distance_mm"]
                )
                annotate_image(image, description)

        if save:
            self.save_descriptions(descriptions)

        return descriptions

    def _get_predictions(self, images):
        request_arguments = {}
        request_arguments["to_predict"] = images
        request_arguments["raw_predictions"] = False
        request_arguments["description"] = [
            "foreground",
            "crystal",
            "loop_inside",
            "loop",
            ["crystal", "loop"],
            ["crystal", "loop", "stem"],
        ]
        return get_predictions(request_arguments)

    def _get_omegas_images(self):
        if not os.path.isfile(self.get_parameters_filename()):
            chistory = self.camera.get_history(
                self.innermost_start_time, self.innermost_end_time
            )
            ghistory = self.speaking_goniometer.get_history(
                self.innermost_start_time, self.innermost_end_time
            )

            ctimestamps = np.array(chistory[0])
            images = chistory[1]

            gtimestamps = np.array(ghistory[0])[:, 0]
            omegas = self.speaking_goniometer.get_values_as_array(ghistory[1])[:, -1]

        else:
            chistory = h5py.File(f"{self.get_template()}_sample_view.h5", "r")
            ghistory = h5py.File(f"{self.get_template()}_goniometer.h5", "r")

            ctimestamps = chistory["history_timestamps"][()]
            images = chistory["history_images"][()]

            gtimestamps = ghistory["timestamps"][()][:, 0]
            omegas = ghistory["values"][()][:, -1]

        t_min, t_max = gtimestamps.min(), gtimestamps.max()
        ip = interp1d(gtimestamps, omegas)

        indices = np.argwhere(
            np.logical_and(t_min <= ctimestamps, ctimestamps <= t_max)
        ).flatten()

        print(f"accepted images  {len(indices)} (of {len(images)})")
        omegas = ip(ctimestamps[indices])
        images = [images[k] for k in indices]

        return omegas, images

    def get_descriptions_filename(self):
        return "%s_descriptions.pickle" % self.get_template()

    def get_descriptions(self, save=True):
        if os.path.isfile(self.get_descriptions_filename()):
            self.descriptions = self.get_pickled_file(self.get_descriptions_filename())
        if self.descriptions is not None:
            return self.descriptions

        _start = time.time()
        omegas, images = self._get_omegas_images()
        predictions = self._get_predictions(images)

        zoom = self.get_zoom()
        calibration = self.get_calibration()
        center = self.get_beam_position()
        reference_position = self.get_reference_position()

        descriptions = []
        for description, angle in zip(predictions["descriptions"], omegas):
            description = self.add_calibrated_data(
                description, center, calibration, reference_position, angle=angle
            )
            descriptions.append(description)

        self.logger.info(
            f"parsing of {len(images)} images took %.2f seconds"
            % (time.time() - _start)
        )

        if save:
            self.save_descriptions(descriptions)
        return descriptions

    def save_descriptions(self, descriptions, mode="wb"):
        f = open(self.get_descriptions_filename(), mode)
        pickle.dump(descriptions, f)
        f.close()

    def get_fits(
        self,
        descriptions,
        minimize_method="nelder-mead",
        keys=[
            "most_likely_click",
            "extreme",
            "start_likely",
            "end_likely",
            "start_possible",
            "aoi_bbox_mm",
            "crystal_bbox_mm",
        ],
        subkeys=["", "px", "shift_mm", "shift_mm_from_position"],
        point_subkeys=["verticals", "horizontals"],
        bbox_subkeys=["verticals", "horizontals", "heights", "widths", "areas"],
        fits=None,
    ):
        description_keys = {}
        extended_keys = []

        for key in keys:
            if "bbox" not in key:
                for subkey in subkeys:
                    description_key = "%s_%s" % (key, subkey) if subkey != "" else key
                    for k, subsubkey in enumerate(point_subkeys):
                        extended_key = "%s_%s" % (description_key, subsubkey)
                        extended_keys.append(extended_key)
                        description_keys[extended_key] = {
                            "description_key": description_key,
                            "index": k,
                        }
            else:
                description_key = key
                for k, subkey in enumerate(bbox_subkeys):
                    extended_key = "%s_%s" % (description_key, subkey)
                    extended_keys.append(extended_key)
                    description_keys[extended_key] = {
                        "description_key": description_key,
                        "index": k + 1,
                    }

        fit_start = time.time()
        if fits is None:
            fits = {"angles": {}, "aspects": {}, "results": {}}

        for key in extended_keys:
            fits["angles"][key] = []
            fits["aspects"][key] = []

        for description in descriptions:
            angle = description["angle"]
            for key in extended_keys:
                description_key = description_keys[key]["description_key"]
                i = description_keys[key]["index"]
                fits["angles"][key].append(np.deg2rad(angle))
                fits["aspects"][key].append(description[description_key][i])

        for aspect in fits["aspects"]:
            try:
                fits["results"][aspect] = self.fit_aspect(
                    fits["angles"][aspect],
                    fits["aspects"][aspect],
                    aspect_name=aspect,
                    minimize_method=minimize_method,
                )
            except:
                print("could not make sense of aspect %s" % aspect)
                logging.info("could not make sense of aspect %s" % aspect)

                print(traceback.print_exc())
                logging.info(traceback.format_exc())
                fits["results"][aspect] = None

        fit_end = time.time()
        self.logger.info("Fits took %.3f seconds" % (fit_end - fit_start))

        return fits

    def get_omega_max_and_min(self, fits, to_look_at="widths"):
        try:
            c_p, r_p, alpha_p = fits["results"]["aoi_bbox_mm_%s" % to_look_at][
                "fit_projection"
            ].x
            self.logger.info("c_p, r_p, alpha_p %s" % str((c_p, r_p, alpha_p)))
            omega_max = alpha_p / 2
            omega_min = alpha_p / 2 + np.pi / 2
            if r_p < 0:
                omega_max, omega_min = omega_min, omega_max
        except:
            omega_max = fits["angles"]["aoi_bbox_mm_%s" % to_look_at][
                np.argmax(fits["aspects"]["aoi_bbox_mm_%s" % to_look_at])
            ]
            omega_min = omega_max + np.pi / 2
            self.logger.info(traceback.format_exc())

        omega_max = np.rad2deg(divmod(omega_max, np.pi)[1])
        omega_min = np.rad2deg(divmod(omega_min, np.pi)[1])

        return omega_max, omega_min

    def get_omega_max_omega_min_height_max_height_min_and_width(self, fits):
        # try:
        # to_look_at = "widths"
        # to_squint_at = "heights"
        # c_p_heights, r_p_heights, alpha_p_heights = fits["results"]["aoi_bbox_mm_heights"][
        # "fit_projection"
        # ].x
        # self.logger.info("c_p_heights, r_p_heights, alpha_p_heights %s" % str((c_p_heights, r_p_heights, alpha_p_heights)))
        # height_max = c_p_heights + np.abs(r_p_heights)
        # height_min = c_p_heights - np.abs(r_p_heights)

        # c_p_widths, r_p_widths, alpha_p_widths = fits["results"]["aoi_bbox_mm_%s" % to_look_at]["fit_projection"].x
        # self.logger.info("c_p_heights, r_p_heights, alpha_p_heights %s" % str((c_p_heights, r_p_heights, alpha_p_heights)))
        # width_max = c_p_widths + np.abs(r_p_widths)
        # width_min = c_p_widths - np.abs(r_p_widths)

        # if r_p_heights > r_p_widths:
        # c_p, r_p, alpha_p = c_p_heights, r_p_heights, alpha_p_heights
        # width = c_p_widths
        # else:
        # c_p, r_p, alpha_p = c_p_widths, r_p_widths, alpha_p_widths
        # width = c_p_heights

        # omega_max = alpha_p / 2.
        # omega_min = alpha_p / 2. + np.pi / 2.
        # if r_p < 0:
        # omega_max, omega_min = omega_min, omega_max
        # except:
        to_look_at = "widths"
        to_squint_at = "heights"
        widths_std = np.std(fits["aspects"]["aoi_bbox_mm_widths"])
        heights_std = np.std(fits["aspects"]["aoi_bbox_mm_heights"])
        if heights_std > widths_std:
            to_look_at, to_squint_at = to_squint_at, to_look_at

        omega_max = fits["angles"]["aoi_bbox_mm_%s" % to_look_at][
            np.argmax(fits["aspects"]["aoi_bbox_mm_%s" % to_look_at])
        ]
        omega_min = omega_max + np.pi / 2

        height_max = max(fits["aspects"]["aoi_bbox_mm_%s" % to_look_at])
        height_min = min(fits["aspects"]["aoi_bbox_mm_%s" % to_look_at])

        width = np.median(fits["aspects"]["aoi_bbox_mm_%s" % to_squint_at])
        self.logger.info(traceback.format_exc())

        omega_max = np.rad2deg(divmod(omega_max, np.pi)[1])
        omega_min = np.rad2deg(divmod(omega_min, np.pi)[1])

        return omega_max, omega_min, height_max, height_min, width

    def get_optimum_zoom(self, height=None, width=None, margin_factor=1.5):
        if height is None:
            height = self.results["height_max_mm"]
        if width is None:
            width = self.results["width_mm"]

        raster = np.array([height, width]) * margin_factor
        self.logger.info("raster %s" % str(raster))

        view_shape = self.get_calibration() * np.array(self.camera.get_shape())
        current_zoom = self.get_zoom()
        magnifications = self.camera.magnifications
        print(f"view_shape {view_shape}")
        print(f"current_zoom {current_zoom}")
        print(f"magnifications {magnifications}")

        possible_increase = (
            np.min(view_shape / raster)
            * magnifications[current_zoom - 1]
            / magnifications
            - 1
        )

        try:
            optimum_zoom = np.argmin(possible_increase[possible_increase >= 0]) + 1
        except:
            optimum_zoom = current_zoom

        if self.verbose:
            self.logger.info("possible increase %s" % str(possible_increase))
            self.logger.info("optimumx zoom: -z %d" % (optimum_zoom))

        return optimum_zoom

    def make_sense_of_descriptions(
        self,
        descriptions=None,
        reference_position=None,
        eagerly=None,
        debug=None,
        orientation="vertical",
    ):
        _start = time.time()
        if descriptions is None:
            descriptions = self.get_descriptions()

        if reference_position is None:
            reference_position = self.get_reference_position()

        if eagerly is None:
            eagerly = self.eagerly

        if debug is None:
            debug = self.debug

        self.logger.info("reference_position %s" % reference_position)

        results = {
            "reference_position": reference_position,
            "mounted_sample_id": self.cats.get_mounted_sample_id(),
        }

        fits = self.get_fits(descriptions)

        # omega_max, omega_min = self.get_omega_max_and_min(fits)
        (
            omega_max,
            omega_min,
            height_max,
            height_min,
            width,
        ) = self.get_omega_max_omega_min_height_max_height_min_and_width(fits)

        reference_position["Omega"] = omega_max
        results["omega_max"] = omega_max
        results["omega_min"] = omega_min
        results["height_max_mm"] = height_max
        results["height_min_mm"] = height_min
        results["width_mm"] = width
        results["optimum_zoom"] = self.get_optimum_zoom(height_max, width)
        results["calibration"] = self.get_calibration()
        results["original_image_shape"] = self.camera.get_shape()
        try:
            results["prediction_shape"] = descriptions[0]["prediction_shape"]
        except:
            traceback.print_exc()

        if eagerly:
            print("in eagerly")
            if not self.sample_seen:
                self.logger.info("sample not seen, returning")
                return -1
            result_position = copy.copy(self.get_result_position())
            if orientation == "vertical":
                omega_axis = "AlignmentZ"  # MD3
            else:
                omega_axis = "AlignmentY"  # MD2

            result_position[omega_axis] = np.median(
                [
                    d["most_likely_click_aligned_position"][omega_axis]
                    for d in descriptions[1:]
                ]
            )
        else:
            print("in carefully")
            fit_vertical = fits["results"]["most_likely_click_shift_mm_verticals"][
                "fit_circle"
            ]
            fit_horizontal = fits["results"]["most_likely_click_shift_mm_horizontals"][
                "fit_circle"
            ]
            result_position = (
                self.goniometer.get_aligned_position_from_fit_and_reference(
                    fit_vertical,
                    fit_horizontal,
                    reference_position,
                    orientation=orientation,
                )
            )

            if orientation == "vertical":
                result_position["AlignmentZ"] = fits["results"][
                    "extreme_shift_mm_horizontals"
                ]["fit_circle"].x[0]
            else:
                result_position["AlignmentZ"] = fits["results"][
                    "extreme_shift_mm_verticals"
                ]["fit_circle"].x[0]

        for description in descriptions:
            description = self.add_hypotetical_data(description, result_position)

        # results['fits'] = fits
        hypotetical_fits = self.get_fits(
            descriptions,
            keys=[
                "most_likely_click",
                "extreme",
                "start_likely",
                "end_likely",
                "start_possible",
            ],
            subkeys=["hypotetical_shift_mm_from_position"],
            point_subkeys=["verticals", "horizontals"],
            fits=fits,
        )

        if eagerly:
            print("I am here and am eager")
            if self.sample_seen:
                aligned_positions = self.get_aligned_positions(
                    hypotetical_fits,
                    reference_position,
                    result_position=result_position,
                    eagerly=eagerly,
                )
            else:
                return -1
        else:
            print("I am there and am not eager")
            aligned_positions = self.get_aligned_positions(
                hypotetical_fits, reference_position
            )
            result_position = aligned_positions["most_likely_click"]
            volume = self.get_volume(descriptions, results, fits)

        results["result_position"] = result_position
        results["aligned_positions"] = aligned_positions

        self.logger.info("resulting AlignmentZ %.3f" % result_position["AlignmentZ"])
        self.logger.info("resulting Omega max %.3f" % result_position["Omega"])
        self.logger.info("aoi max height %.3f" % results["height_max_mm"])
        self.logger.info("aoi width %.3f" % results["width_mm"])

        _end = time.time()
        duration = _end - _start
        self.logger.info(
            f"making sense of {len(descriptions)} views took {duration:.2f} seconds"
        )

        if debug:
            self.report_on_fits(fits)
            annotate_alignment(results)

        self.results = results

        return self.results

    def report_on_fits(self, fits):
        try:
            results = fits["results"]
            key = "fit_circle"
            for notion in [
                "extreme_shift_mm_verticals",
                "extreme_shift_mm_from_position_verticals",
                "extreme_hypotetical_shift_mm_from_position_verticals",
                "extreme_shift_mm_horizontals",
                "extreme_shift_mm_from_position_horizontals",
                "extreme_hypotetical_shift_mm_from_position_horizontals",
            ]:
                self.logger.info(
                    f'{notion.replace("shift_mm", "").replace("_", " ")} {results[notion][key].x}'
                )
        except:
            self.logger.info(traceback.format_exc())

    def get_aligned_positions(
        self,
        fits,
        reference_position,
        points=[
            "extreme",
            "start_likely",
            "end_likely",
            "start_possible",
            "most_likely_click",
        ],
        result_position=None,
        eagerly=False,
    ):
        aligned_positions = {}
        for point in points:
            vertical_key = "%s_shift_mm_verticals" % point
            horizontal_key = "%s_shift_mm_horizontals" % point
            fit_vertical = fits["results"][vertical_key]["fit_circle"]
            fit_horizontal = fits["results"][horizontal_key]["fit_circle"]
            aligned_position = (
                self.goniometer.get_aligned_position_from_fit_and_reference(
                    fit_vertical, fit_horizontal, reference_position
                )
            )
            aligned_positions[point] = aligned_position

        for point in points:
            vertical_key = "%s_hypotetical_shift_mm_from_position_verticals" % point
            horizontal_key = "%s_hypotetical_shift_mm_from_position_horizontals" % point
            fit_vertical = fits["results"][vertical_key]["fit_circle"]
            fit_horizontal = fits["results"][horizontal_key]["fit_circle"]
            aligned_position = (
                self.goniometer.get_aligned_position_from_fit_and_reference(
                    fit_vertical, fit_horizontal, reference_position
                )
            )
            aligned_positions["%s_hypotetical" % point] = aligned_position

        if eagerly:
            print("result_position")
            pprint.pprint(result_position)
            print("most_likely_click")
            pprint.pprint(aligned_positions["most_likely_click"])

            shift = get_vector_from_position(
                result_position
            ) - get_vector_from_position(aligned_positions["most_likely_click"])
            print("shift", shift)
            for point in points:
                shifted_point = get_position_from_vector(
                    get_vector_from_position(aligned_positions[point]) + shift
                )
                aligned_positions[point] = shifted_point

        return aligned_positions

    def get_projections(self, descriptions, notion="foreground"):
        notion_string = get_notion_string(notion)
        detector_rows, detector_cols = descriptions[0][notion_string][
            "notion_mask"
        ].shape

        number_of_projections = len(descriptions)

        projections = np.zeros((detector_rows, number_of_projections, detector_cols))
        valid_angles = []
        valid_index = 0
        print("%d projections" % len(descriptions))
        for description in descriptions:
            if description["present"]:
                projections[:, valid_index, :] = description[notion_string][
                    "notion_mask"
                ]  # .T
                valid_angles.append(np.deg2rad(description["angle"]))
                valid_index += 1
        projections = projections[:, :valid_index, :]

        center_of_mass = ndi.center_of_mass(projections)
        print(
            "projections shape, center_of_mass",
            projections.shape,
            center_of_mass,
            projections.max(),
            projections.mean(),
        )

        return projections, valid_angles

    def get_reconstruction(
        self,
        projections,
        angles,
        axis_correction=0.0,
        detector_col_spacing=1,
        detector_row_spacing=1,
    ):
        detector_rows, detector_cols = projections.shape[0], projections.shape[-1]

        request = {
            "projections": projections,
            "angles": angles,
            "detector_rows": detector_cols,
            "detector_cols": detector_rows,
            "detector_col_spacing": detector_col_spacing,
            "detector_row_spacing": detector_row_spacing,
            "vertical_correction": axis_correction,
        }

        reconstruction = _get_reconstruction(request, verbose=True)
        print(
            "reconstruction (shape, max, mean)",
            reconstruction.shape,
            reconstruction.max(),
            reconstruction.mean(),
        )
        return reconstruction

    def get_volume(
        self,
        descriptions=None,
        results=None,
        fits=None,
        do_axis_correction=True,
        notion="foreground",
        threshold=0.75,
        save=True,
    ):
        if descriptions is None:
            descriptions = self.get_descriptions()
        if results is None:
            results = self.make_sense_of_descriptions(descriptions)
        if fits is None:
            fits = self.get_fits(descriptions)

        projections, angles = self.get_projections(descriptions, notion=notion)
        detector_rows, detector_cols = projections.shape[0], projections.shape[-1]

        axis_correction = 0  # detector_cols / 2
        default_axis_position = detector_cols / 2
        if do_axis_correction:
            axis_position = self.get_axis_position(
                fits,
                projections,
                default=default_axis_position,
            )
            print(f"axis_position {axis_position:.2f} vs {default_axis_position:.2f}")
            axis_correction = detector_cols / 2 - axis_position
            print("axis_correction", axis_correction)

        reconstruction = self.get_reconstruction(projections, angles, axis_correction)

        volume = apply_threshold(reconstruction, threshold=threshold)

        if save:
            mesh_mm = self.get_mesh_mm(volume, results, save=save)
            pcd_mm = self.get_pcd_mm(volume, results, save=save)

        return volume

    def get_axis_position(self, fits=None, projections=None, default=None):
        # axis_position = (
        # fits["results"]["extreme_px_verticals"]["fit"].x[0]
        # * detector_rows
        # / original_image_shape[0]
        # )

        axis_position = ndi.center_of_mass(projections)[-1]

        # if fits is not None:
        # axis_position = fits["results"]["extreme_px_horizontals"]["fit"].x[0]
        # elif projections is not None:
        # axis_position = ndi.center_of_mass(projections)[-1]
        # else:
        # axis_position = default

        print("estimated rotation axis position %.3f" % axis_position)

        return axis_position

    def get_points_from_volume(self, volume=None):
        if volume is None:
            volume = self.get_volume()
        objectpoints = np.argwhere(volume)
        print("objectpoints.shape", objectpoints.shape)
        return objectpoints

    def get_surface_mesh_from_volume(self, volume=None):
        if volume is None:
            volume = self.get_volume()
        helper_volume = np.zeros(tuple(np.array(volume.shape) + 2))

        helper_volume[1:-1, 1:-1, 1:-1] = volume
        v, f, n, c = marching_cubes(helper_volume, gradient_direction="descent")
        v -= 1

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v),
            o3d.utility.Vector3iVector(f),
        )
        mesh.vertex_normals = o3d.utility.Vector3dVector(n)
        # mesh.compute_triangle_normals()

        return mesh

    def get_mesh_px(self, volume=None, save=True):
        name = self.get_mesh_px_name()
        if os.path.isfile(name):
            mesh_px = o3d.io.read_triangle_mesh(name)
        else:
            mesh_px = self.get_surface_mesh_from_volume(volume)
            if save:
                self.save_mesh(mesh_px, name)
        return mesh_px

    def get_points_px(self, volume=None, save=False):
        points_px = self.get_points_from_volume(volume)
        if save:
            pcd_px = self.get_pcd(points_px)
            self.save_pcd(pcd_px, self.get_pcd_px_name())
        return points_px

    def get_pcd_px(self, volume=None, save=True):
        name = self.get_pcd_px_name()
        if volume is None and os.path.isfile(name):
            pcd_px = o3d.io.read_point_cloud(name)
        else:
            points_px = self.get_points_px(volume)
            pcd_px = self.get_pcd(points_px)
            if save:
                self.save_pcd(pcd_px, name)
        return pcd_px

    def get_critical_points(
        self,
        results=None,
        points=["extreme", "most_likely_click", "start_likely", "start_possible"],
        keys=["CentringX", "CentringY", "AlignmentY"],
    ):
        name = self.get_results_filename()
        if os.path.isfile(name) and results is None:
            results = self.load_results()
        critical_points = {}
        for point in points:
            critical_points[point] = get_vector_from_position(
                results["aligned_positions"][point], keys=keys
            )
        return critical_points

    def get_gonio_points(
        self,
        points,
        results,
        directions=np.array([1, 1, 1]),
        order=[1, 2, 0],
    ):
        origin = self.get_origin(results)
        center = self.get_center(results)
        voxel_calibration = self.get_voxel_calibration(results)

        print(f"origin {origin}")
        print(f"center {center}")
        print(f"voxel_calibration {voxel_calibration}")

        gonio_points = get_points_in_goniometer_frame(
            points,
            voxel_calibration,
            origin[:3],
            center=center,
            directions=directions,
            order=order,
        )
        return gonio_points

    def get_mesh_mm(self, volume=None, results=None, save=False):
        name = self.get_mesh_mm_name()
        if os.path.isfile(name):
            mesh_mm = o3d.io.read_triangle_mesh(name)
        else:
            mesh_px = self.get_mesh_px(volume, save=save)
            mesh_mm = copy.copy(mesh_px)
            vertices_px = np.asarray(mesh_px.vertices)
            vertices_mm = self.get_gonio_points(vertices_px, results)
            mesh_mm.vertices = o3d.utility.Vector3dVector(vertices_mm)
            if save:
                self.save_mesh(mesh_mm, name)
        return mesh_mm

    def get_pcd_mm(self, volume=None, results=None, save=False):
        name = self.get_pcd_mm_name()
        if os.path.isfile(name):
            pcd_mm = o3d.io.read_point_cloud(name)
        else:
            points_px = self.get_points_px(volume, save=save)
            points_mm = self.get_gonio_points(points_px, results)
            pcd_mm = self.get_pcd(points_mm)
            if save:
                self.save_pcd(pcd_mm, name)
        return pcd_mm
        # print("objectpoints_mm median")
        # print(np.median(objectpoints_mm, axis=0))

        # pca3d = principal_axes(
        # objectpoints_mm, verbose=True
        # )  # inertia, eigenvalues, eigenvectors, center

    def get_origin(self, results):
        reference_position = results["reference_position"]
        # reference_position = results["position"]
        origin = get_vector_from_position(
            reference_position,
            keys=["CentringX", "CentringY", "AlignmentY"]
            # reference_position, keys=["AlignmentY", "CentringX", "CentringY"],
        )
        return origin

    def get_voxel_calibration(self, results):
        voxel_calibration = get_voxel_calibration(*results["calibration"])
        original_image_shape = results["original_image_shape"]
        detector_rows, detector_cols = results["prediction_shape"]

        cols_ratio = original_image_shape[1] / detector_cols
        rows_ratio = original_image_shape[0] / detector_rows

        voxel_calibration[0] *= rows_ratio  # cols_ratio
        voxel_calibration[1:] *= cols_ratio  # rows_ratio
        return voxel_calibration

    def get_center(self, results):
        detector_rows, detector_cols = results["prediction_shape"]
        # center = np.array([detector_cols / 2, detector_rows, detector_rows])
        center = np.array([detector_rows / 2, detector_cols, detector_cols])
        return center

    def get_mesh_mm_name(self):
        mesh_mm_name = "%s_mm.obj" % self.get_template()
        return mesh_mm_name

    def get_mesh_px_name(self):
        mesh_px_name = "%s_px.obj" % self.get_template()
        return mesh_px_name

    def get_pcd_px_name(self):
        pcd_px_name = "%s_px.pcd" % self.get_template()
        return pcd_px_name

    def get_pcd_mm_name(self):
        pcd_mm_name = "%s_mm.pcd" % self.get_template()
        return pcd_mm_name

    def save_mesh(self, mesh, filename):
        o3d.io.write_triangle_mesh(filename, mesh)

    def get_pcd(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        return pcd

    def save_pcd(self, pcd, filename):
        _start = time.time()
        o3d.io.write_point_cloud(filename, pcd)
        print(
            "point cloud %s save took %.4f seconds" % (filename, time.time() - _start)
        )

    def add_calibrated_data(
        self, description, center, calibration, reference_position, angle=None
    ):
        original_shape = description["original_shape"]
        prediction_shape = description["prediction_shape"]
        scale = original_shape / prediction_shape
        description["scale"] = scale

        if angle is not None:
            reference_position["Omega"] = angle

        omega = reference_position["Omega"]

        description["angle"] = omega
        description["reference_position"] = reference_position

        for bbox_key in ["aoi_bbox", "crystal_bbox"]:
            bbox_px = list(description["aoi_bbox"][:])
            bbox_px[1] *= original_shape[0]
            bbox_px[2] *= original_shape[1]
            bbox_px[3] *= original_shape[0]
            bbox_px[4] *= original_shape[1]
            bbox_px.append(bbox_px[3] * bbox_px[4])
            description["%s_px" % bbox_key] = bbox_px

            bbox_mm = list(bbox_px[:])
            bbox_mm[1] = (bbox_px[1] - center[0]) * calibration[0]
            bbox_mm[2] = (bbox_px[2] - center[1]) * calibration[1]
            bbox_mm[3] *= calibration[0]
            bbox_mm[4] *= calibration[1]
            bbox_mm[5] = bbox_mm[3] * bbox_mm[4]
            description["%s_mm" % bbox_key] = bbox_mm

        def add_point_calibrated_data(description, key, reference_position):
            px = description[key] * original_shape
            shift_mm = [np.nan, np.nan]
            distance = np.nan
            aligned_position = None
            shift_mm_from_position = [np.nan, np.nan]
            if description[key][0] >= 0:
                shift_mm = (px - center) * calibration
                distance = np.linalg.norm(shift_mm)
                aligned_position = get_aligned_position_from_reference_position_and_shift(
                    reference_position,
                    shift_mm[1],
                    shift_mm[0],
                    omega=omega,
                )
                shift_mm_from_position = get_vertical_and_horizontal_shift_between_two_positions(
                    aligned_position, reference_position
                )

            description["%s_px" % key] = px
            description["%s_shift_mm" % key] = shift_mm
            description["%s_distance_mm" % key] = distance
            description["%s_aligned_position" % key] = aligned_position
            description["%s_shift_mm_from_position" % key] = shift_mm_from_position

            return description

        for key in [
            "most_likely_click",
            "extreme",
            "end_likely",
            "start_likely",
            "start_possible",
        ]:
            description = add_point_calibrated_data(
                description, key, reference_position
            )

        return description

    def add_hypotetical_data(self, description, hypotetical_reference_position):
        for key in [
            "most_likely_click",
            "extreme",
            "end_likely",
            "start_likely",
            "start_possible",
        ]:
            aligned_position = description["%s_aligned_position" % key]
            try:
                shift_mm_from_position = get_vertical_and_horizontal_shift_between_two_positions(
                    aligned_position, hypotetical_reference_position
                )
            except:
                #print("problem in add_hypotetical_data")
                #print("aligned_position", aligned_position)
                #print("reference_position", hypotetical_reference_position)
                shift_mm_from_position = np.array([np.nan, np.nan])
            description[
                "%s_hypotetical_shift_mm_from_position" % key
            ] = shift_mm_from_position

        return description

    def fit_aspect(
        self,
        angles,
        aspect,
        aspect_name=None,
        minimize_method="nelder-mead",
        debug=False,
    ):
        initial_parameters = get_initial_parameters(aspect, name=aspect_name)

        fit_circle = minimize(
            circle_model_residual,
            initial_parameters,
            method=minimize_method,
            args=(angles, aspect),
        )

        fit_projection = minimize(
            projection_model_residual,
            initial_parameters,
            method=minimize_method,
            args=(angles, aspect),
        )

        if debug:
            print(
                "aspect",
                aspect,
                "initial_parameters",
                initial_parameters,
                "optimized_parameters circle, projection",
                fit_circle.x,
                fit_projection.x,
            )

        fit, k = select_better_model(fit_circle, fit_projection)

        result = {
            "fit_circle": fit_circle,
            "fit_projection": fit_projection,
            "fit": fit,
            "k": k,
        }

        if k == 1:
            result["best_model"] = circle_model
        else:
            result["best_model"] = projection_model

        return result

    def get_start_run_time(self):
        return self.innermost_start_time

    def get_end_run_time(self):
        return self.innermost_end_time

    def run(self):
        _start = time.time()
        if not hasattr(self, "start_run_time"):
            self.start_run_time = _start

        self.innermost_start_time = time.time()
        if self.eagerly:
            descriptions = self._align_eagerly()
            self.innermost_end_time = time.time()
            self.scan_exposure_time = 0.
        else:
            self.sample_seen = True
            task_id = self.goniometer.omega_scan(
                scan_start_angle=self.scan_start_angle,
                scan_range=self.scan_range,
                scan_exposure_time=self.scan_exposure_time,
            )
            self.goniometer.wait_for_task_to_finish(task_id)
            self.innermost_end_time = time.time()
            self.md_task_info.append(self.goniometer.get_task_info(task_id))
            descriptions = self.get_descriptions()

        print(f"innermost took {self.innermost_end_time - self.innermost_start_time:.2f} seconds (exposure time {self.scan_exposure_time:.2f} seconds)")
        
        self.descriptions = descriptions
        self.end_run_time = time.time()
        self.logger.info(
            "run took %.3f seconds"
            % (self.innermost_end_time - self.innermost_start_time)
        )

    def clean(self):
        self.goniometer.enable_fast_shutter()
        super().clean()

    def analyze(self):
        self.make_sense_of_descriptions()

    def conclude(self):
        print("In conclude")
        if not self.sample_seen:
            return -1
        if self.extreme:
            result_position = self.results["aligned_positions"]["extreme"]
        else:
            result_position = self.results["result_position"]
        print("setting result position")
        pprint.pprint(result_position)
        
        self.goniometer.set_position(result_position)

        if self.move_zoom == True:
            self.goniometer.set_zoom(self.results["optimum_zoom"])
        self.goniometer.save_position()
        self.redis.set(
            self.last_optical_alignment_results_key, pickle.dumps(self.results)
        )
        self.goniometer.insert_frontlight()
        self.end_conclusion_time = time.time()


def kappa_phi_realign(
    name_pattern="kappa_phi_%s_%s" % (os.getuid(), time.asctime().replace(" ", "_")),
    directory="%s/manual_optical_alignment" % os.getenv("HOME"),
    positions={},
    source=None,
):
    g = goniometer()
    kappa = g.get_kappa_position()
    phi = g.get_phi_position()

    oa = optical_alignment(
        name_pattern=f"{name_pattern}_k_{kappa:.2f}_p_{phi:.2f}_eager_zoom_1",
        directory=directory,
        scan_range=0,
        backlight=True,
        analysis=True,
        conclusion=True,
        move_zoom=False,
        save_history=True,
        zoom=1,
    )
    oa.execute()

    oa = optical_alignment(
        name_pattern=f"{name_pattern}_k_{kappa:.2f}_p_{phi:.2f}_careful_zoom_1",
        directory=directory,
        scan_range=360,
        backlight=True,
        analysis=True,
        conclusion=True,
        move_zoom=True,
        save_history=True,
        zoom=1,
    )
    oa.execute()

    oa = optical_alignment(
        name_pattern=f"{name_pattern}_k_{kappa:.2f}_p_{phi:.2f}_careful_zoom_X",
        directory=directory,
        scan_range=360,
        backlight=True,
        analysis=True,
        conclusion=True,
        move_zoom=False,
        save_history=True,
        zoom=None,
    )
    oa.execute()

    if source:
        transformed_positions = get_transformed_positions(
            positions, source, oa.get_pcd_mm_name()
        )
        g.set_position(transformed_positions[0])


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-n",
        "--name_pattern",
        default="autocenter_%s_%s"
        % (os.getuid(), time.asctime().replace(" ", "_").replace(":", "")),
        type=str,
        help="Prefix default=%default",
    )
    parser.add_option(
        "-d",
        "--directory",
        default="%s/manual_optical_alignment" % os.getenv("HOME"),
        type=str,
        help="Destination directory default=%default",
    )
    parser.add_option(
        "-g",
        "--n_angles",
        default=24,
        type=int,
        help="Number of equidistant angles to collect at. Takes precedence over angles parameter if specified",
    )
    parser.add_option(
        "-a",
        "--angles",
        default="(0, 90, 180, 225, 315)",
        type=str,
        help="Specific angles to collect at",
    )
    parser.add_option("-f", "--frontlight", action="store_true", help="Use frontlight")
    parser.add_option("-b", "--backlight", action="store_true", help="Use backlight")
    parser.add_option(
        "-r", "--scan_range", default=360, type=float, help="Range of angles"
    )
    parser.add_option("-z", "--zoom", default=None, type=int, help="Zoom")
    parser.add_option("-p", "--position", default=None, type=str, help="Position")
    parser.add_option(
        "-K", "--kappa", default=None, type=float, help="Kappa orientation"
    )
    parser.add_option("-P", "--phi", default=None, type=float, help="Phi orientation")
    parser.add_option(
        "-A",
        "--analysis",
        action="store_true",
        help="If set will perform automatic analysis.",
    )
    parser.add_option(
        "-C",
        "--conclusion",
        action="store_true",
        help="If set will move the motors upon analysis.",
    )
    parser.add_option(
        "-R",
        "--generate_report",
        action="store_true",
        help="If set will generate report.",
    )

    parser.add_option("--extreme", action="store_true", help="Go for extreme point.")
    parser.add_option(
        "--move_zoom",
        action="store_true",
        help="If set will change zoom to the one corresponding to the biggest still containing the whole loop.",
    )
    parser.add_option(
        "--save_history", action="store_true", help="If set will save raw images."
    )
    parser.add_option("-F", "--film_step", default=-120.0, type=float, help="Film step")
    parser.add_option(
        "-S",
        "--size_of_target",
        default=0.05,
        type=float,
        help="Size of target at the end of the sample (e.g. loop)",
    )

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    if options.scan_range <= 0.0:
        options.scan_range = None
        eagerly = True
    else:
        eagerly = False
    oa = optical_alignment(**vars(options))

    filename = "%s_parameters.pickle" % oa.get_template()

    print("filename %s" % filename)

    if not os.path.isfile(filename):
        print("filename %s not found executing" % filename)
        oa.execute()
        # oa.analyse_single_view(debug=True)
        # desc_filename = '/tmp/descriptions.pickle'

        # descriptions = oa.run()
        # descriptions = oa._align_eagerly(debug=True)
        # descriptions = oa._align_carefully()

        # if os.path.isfile(desc_filename):
        # descriptions = pickle.load(open(desc_filename, 'rb'))
        # else:
        # descriptions = oa._align_eagerly(debug=False)
        # f = open(desc_filename, 'wb')
        # pickle.dump(descriptions, f)
        # f.close()
        # oa.make_sense_of_descriptions(descriptions, eagerly=eagerly, debug=True)
    elif options.analysis == True:
        oa.analyze()
        if options.conclusion == True:
            oa.conclude()


if __name__ == "__main__":
    main()
