#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import pickle
import os
import subprocess
import re
import numpy as np
import scipy.ndimage as nd
import glob
import scipy.misc
import traceback
import gevent

from skimage import img_as_float, img_as_int
from skimage.filters import (
    threshold_otsu,
    threshold_triangle,
    threshold_local,
    sobel,
    rank,
    gaussian,
)
from skimage.measure import regionprops, label
from skimage.morphology import (
    closing,
    square,
    rectangle,
    opening,
    disk,
    remove_small_objects,
    dilation,
)
from skimage.feature import match_template

# try:
from imageio import imsave

##from scipy.misc import imsave
# from skimage.io import imsave

from matplotlib.patches import Rectangle, Circle
import pylab

from area import area
from goniometer import goniometer
from diffraction_experiment import diffraction_experiment
from diffraction_experiment_analysis import diffraction_experiment_analysis

from useful_routines import get_focus_and_orthogonal_from_position


class raster_scan_analysis:
    def __init__(
        self,
        name_pattern,
        directory,
        threshold=0.5,
        min_spots=10,
        horizontal_direction=-1,
        vertical_direction=1,
    ):
        self.name_pattern = name_pattern
        self.directory = directory
        self.threshold = threshold
        self.min_spots = min_spots
        self.horizontal_direction = horizontal_direction
        self.vertical_direction = vertical_direction
        self.parameters = None
        self.results = None
        self.z = None
        self.goniometer = goniometer()
        self.diffraction_experiment = diffraction_experiment(
            self.name_pattern, self.directory
        )
        self.dea = diffraction_experiment_analysis(self.name_pattern, self.directory)

    def get_directions(self):
        return np.array([self.vertical_direction, self.horizontal_direction])

    def get_template(self):
        return os.path.join(self.directory, self.name_pattern)

    def get_parameters_filename(self):
        return os.path.join("%s_parameters.pickle" % self.get_template())

    def get_results_filename(self):
        return os.path.join("%s_results_analysis.pickle" % self.get_template())

    def get_dials_raw_results_filename(self):
        return os.path.join("%s_raw_results.pickle" % self.get_template())

    def get_parameters(self):
        if self.parameters == None:
            self.parameters = pickle.load(open(self.get_parameters_filename(), "rb"))
        return self.parameters

    def get_z(self, method="tioga"):
        if self.z is not None:
            return self.z

        parameters = self.get_parameters()

        number_of_rows = parameters["number_of_rows"]
        number_of_columns = parameters["number_of_columns"]
        inverse_direction = parameters["inverse_direction"]

        if "against_gravity" in parameters:
            against_gravity = parameters["against_gravity"]
        else:
            against_gravity = False

        try:
            points = parameters["cell_positions"]
        except KeyError:
            points = parameters["points"]

        try:
            indexes = parameters["indexes"]
        except KeyError:
            indexes = parameters["grid"]

        z = np.zeros((number_of_rows, number_of_columns))

        if method == "dozor":
            results = self.diffraction_experiment.get_dozor_results(blocking=True)
        elif method == "xds":
            print("method is xds!")
            results = self.diffraction_experiment.get_xds_results()  # blocking=True)
            print(results)
        elif method == "dials":
            results = self.diffraction_experiment.get_dials_raw_results()
        elif method == "tioga":
            results = self.dea.get_tioga_results()

        print
        if parameters["scan_axis"] in ["horizontal", b"horizontal"]:
            if method == "dials":
                z = np.ravel(z)
                for n in range(len(z)):
                    try:
                        z[n + 1] = results[n + 1]["dials_spots"]
                    except:
                        pass
            elif method == "dozor":
                print("checkpoint 1")
                z = results[:, 2]
            # elif method == 'xds':
            # z = results[:]
            else:
                z = results[:]

            z = np.reshape(z, (number_of_columns, number_of_rows))
            if inverse_direction == True:
                z = self.raster(z, k=0)
            z = self.mirror(z)

        else:  # parameters['scan_axis'] in ['vertical', b'vertical']:
            z = np.ravel(z)
            if method == "dials":
                for n in range(len(z)):
                    try:
                        z[n + 1] = results[n + 1]["dials_spots"]
                    except:
                        pass
            elif method == "dozor":
                print("checkpoint 2")
                z = results[:, 2]
            else:
                z = results[:]

            z = np.reshape(z, (number_of_columns, number_of_rows))
            if inverse_direction == True:
                z = self.raster(z, k=0)
            if against_gravity == True:
                z = self.mirror(z)
                print("checkpoint 2.5 max(z)", z.max())
            z = z.T
            z = self.mirror(z)
            print("checkpoint 3 max(z)", z.max())
        self.z = z
        print("z", z)
        print("checkpoint 4 max(z)", z.max())
        return self.z

    def mirror(self, grid):
        return self.raster(grid, k=0, l=1)

    def raster(self, grid, k=0, l=2):
        gs = grid.shape
        orderedGrid = []
        for i in range(gs[0]):
            line = grid[i, :]
            if (i + 1) % l == k:
                line = line[::-1]
            orderedGrid.append(line)
        return np.array(orderedGrid)

    def invert(self, z):
        z_inverted = z[:, ::-1]
        z_raster = np.zeros(z.shape)
        for k in range(len(z)):
            if k % 2 == 1:
                z_raster[k] = z_inverted[k]
            else:
                z_raster[k] = z[k]
        return z_raster

    def get_reference_position(self):
        return self.get_parameters()["reference_position"]

    def get_extent(self):
        vertical_range = self.get_parameters()["vertical_range"]
        horizontal_range = self.get_parameters()["horizontal_range"]
        extent = np.array([vertical_range, horizontal_range])
        return extent

    def get_min_max(self):
        extent = self.get_extent()
        origin = self.get_origin()
        horizontal_min = origin[1]
        vertical_max = origin[0]
        horizontal_max = horizontal_min - extent[1]
        vertical_min = vertical_max + extent[0]
        return [horizontal_min, horizontal_max, vertical_min, vertical_max]

    def get_optical_image_min_max(self):
        center = self.get_center()
        image = self.get_image()
        calibration = self.get_calibration()
        extent = np.array(image.shape[:2]) * calibration
        horizontal_min = center[1] - extent[1] / 2.0
        horizontal_max = center[1] + extent[1] / 2.0
        vertical_min = center[0] + extent[0] / 2.0
        vertical_max = center[0] - extent[0] / 2.0
        return [horizontal_max, horizontal_min, vertical_min, vertical_max]

    def get_calibration(self):
        camera_calibration_vertical = self.get_parameters()[
            "camera_calibration_vertical"
        ]
        camera_calibration_horizontal = self.get_parameters()[
            "camera_calibration_horizontal"
        ]
        calibration = np.array(
            (camera_calibration_vertical, camera_calibration_horizontal)
        )
        return calibration

    def get_pixel_beam_center(self):
        beam_position_vertical = self.get_parameters()["beam_position_vertical"]
        beam_position_horizontal = self.get_parameters()["beam_position_horizontal"]
        pixel_beam_center = np.array((beam_position_vertical, beam_position_horizontal))
        return pixel_beam_center

    def get_cell_size(self):
        return self.get_extent() / self.get_shape()

    def get_grid_shape_in_pixels(self):
        return self.get_extent() / self.get_calibration()

    def get_shape(self):
        return np.array(self.get_parameters()["grid"].shape)

    def get_optical_scale(self):
        optical_scale = self.get_grid_shape_in_pixels() / self.get_shape()
        return optical_scale

    def get_image_shape(self):
        return self.get_parameters()["image"].shape[:2]

    def get_scaled_z_overlay(self, scaled_z):
        fullshape = self.get_image_shape()
        center = self.get_coordinates_of_beam_on_optical_image()
        empty = np.zeros(fullshape)
        gd1, gd2 = scaled_z.shape
        cd1, cd2 = center
        start1 = int(cd1 - gd1 / 2.0)
        end1 = int(cd1 + gd1 / 2.0)
        start2 = int(cd2 - gd2 / 2.0)
        end2 = int(cd2 + gd2 / 2.0)
        s1 = 0
        s2 = 0
        e1 = gd1 + 1
        e2 = gd2 + 1
        if start1 < 0:
            s1 = -start1 + 1
            start1 = 0
        if end1 > fullshape[0]:
            e1 = e1 - (end1 - fullshape[0]) - 2
            end1 = start1 + (e1 - s1)  # fullshape[0] + 1
        if start2 < 0:
            s2 = -start2 + 1
            start2 = 0
        if end2 > fullshape[1]:
            e2 = e2 - (end2 - fullshape[1]) - 1
            end2 = start2 + (e2 - s2)  # fullshape[1] + 1

        empty[start1:end1, start2:end2] = scaled_z[s1:e1, s2:e2]
        z_overlay = empty
        return z_overlay

    def get_z_scaled(self, z=None):
        if z is None:
            z = self.get_z()
        return nd.zoom(z, self.get_optical_scale())

    def get_center(self):
        along_center = self.get_reference_position()["AlignmentY"]
        if self.get_parameters()["use_centring_table"]:
            (
                focus_center,
                orthogonal_center,
            ) = self.get_focus_and_orthogonal(self.get_reference_position())
        else:
            orthogonal_center = self.get_reference_position()["AlignmentZ"]

        return np.array([orthogonal_center, along_center])

    def get_origin(self):
        origin = (
            self.get_center() + self.get_motor_directions() * self.get_extent() / 2.0
        )
        return origin

    def get_center_of_mass_z(self, threshold=0.8):
        Zx = self.get_best_of_z(threshold=threshold)
        return nd.center_of_mass(Zx)

    def get_grid_directions(self):
        return np.array([+1, -1])

    def get_motor_directions(self):
        return np.array([-1, +1])

    def get_center_of_mass_z_mm_from_origin(self):
        com = self.get_center_of_mass_z()
        cell_size = self.get_cell_size()
        return com * cell_size

    def get_center_of_mass_absolute_position(self):
        return (
            self.get_origin()
            + self.get_grid_directions() * self.get_center_of_mass_z_mm_from_origin()
        )

    def get_optimum(self):
        com = self.get_center_of_mass_z()
        optimum = (
            self.get_origin()
            - self.get_motor_directions() * self.get_center_of_mass_z_mm_from_origin()
        )
        return optimum

    def get_optimum_position(self):
        shift = self.get_shift()
        shift[1] *= -1
        reference_position = self.get_reference_position()
        optimum_position = (
            self.goniometer.get_aligned_position_from_reference_position_and_shift(
                reference_position,
                shift[1],
                shift[0],
                AlignmentZ_reference=reference_position["AlignmentZ"],
            )
        )
        return optimum_position

    def get_shift(self):
        return self.get_optimum() - self.get_center()

    def get_pixel_position_in_mm(self, column, row):
        shape = self.get_shape()
        horizontal_min, horizontal_max, vertical_min, vertical_max = self.get_min_max()
        horizontal_position = (
            horizontal_min
            - float(column)
            / self.get_shape()[1]
            * np.abs(horizontal_max - horizontal_min)
            - self.get_cell_size()[1] * 0.5
        )
        if row == 0.0:
            vertical_position = np.nan
        else:
            vertical_position = vertical_max + float(row) / self.get_shape()[
                0
            ] * np.abs(vertical_max - vertical_min)
        return horizontal_position, vertical_position

    def get_image(self, color=False):
        if color:
            image = self.get_parameters()["image"]
        else:
            image = self.get_parameters()["image"].mean(axis=2)
        return img_as_float(image)

    def get_image_pixel_calibration(self):
        vertical = self.get_parameters()["camera_calibration_vertical"]
        horizontal = self.get_parameters()["camera_calibration_horizontal"]
        return np.array([vertical, horizontal])

    def get_coordinates_of_beam_on_optical_image(self):
        vertical = self.get_parameters()["beam_position_vertical"]
        horizontal = self.get_parameters()["beam_position_horizontal"]
        return np.array([vertical, horizontal])

    def get_raster_scaled_to_optical_image(self, raster):
        cell_size = self.get_cell_size()
        image_pixel_calibration = self.get_image_pixel_calibration()
        return nd.zoom(raster, self.get_optical_scale())

    def get_denoised_z(self, min_spots=None):
        if min_spots is None:
            min_spots = self.min_spots
        z = self.get_z()
        denoised_z = copy.deepcopy(z)
        denoised_z[denoised_z <= min_spots] = 0.0
        return denoised_z

    def get_filtered_z(self, min_spots=None):
        if min_spots is None:
            min_spots = self.min_spots
        denoised_z = self.get_denoised_z(min_spots=min_spots)
        filtered_z = copy.deepcopy(denoised_z)
        filtered_z[filtered_z >= min_spots] = 1
        filtered_z = remove_small_objects(filtered_z == 1, min_size=50)
        # filtered_z = closing(filtered_z, selem=rectangle(5,1))
        filtered_z = closing(filtered_z)
        return filtered_z

    def get_best_of_z(self, min_spots=None, threshold=None, label=False):
        if min_spots is None:
            min_spots = self.min_spots
        if threshold is None:
            threshold = self.threshold
        denoised_z = self.get_denoised_z(min_spots=min_spots)
        result = np.zeros(denoised_z.shape)
        if denoised_z.max() > min_spots:
            result = copy.deepcopy(denoised_z)
            result[denoised_z < threshold * denoised_z.max()] = 0
            if label:
                result[denoised_z >= threshold * denoised_z.max()] = 1
        return result

    def get_denoised_z_scaled_to_optical_image(self):
        denoised_z = self.get_denoised_z()
        return self.get_raster_scaled_to_optical_image(denoised_z)

    def get_filtered_z_scaled_to_optical_image(self):
        filtered_z = self.get_filtered_z()
        return self.get_raster_scaled_to_optical_image(filtered_z)

    def get_best_of_z_scaled_to_optical_image(self):
        best_of_z = self.get_best_of_z()
        return self.get_raster_scaled_to_optical_image(best_of_z)

    def get_regionprops(self, raster):
        raster_props = regionprops(remove_small_objects(img_as_int(raster), min_size=2))
        return raster_props

    def get_orientation(self):
        return self.get_parameters()["scan_start_angle"]

    def get_shape_characteristics(
        self,
        label_image=None,
        min_spots=None,
        threshold=None,
        match_template_threshold=0.1,
    ):
        if label_image == None:
            # label_image = self.get_filtered_z(min_spots=min_spots)
            label_image = self.get_best_of_z(
                min_spots=min_spots, threshold=threshold, label=True
            )
        top_pattern = np.array([[0], [1]])
        mt = match_template(label_image, top_pattern, pad_input=True)

        tops = mt > match_template_threshold
        bottoms = mt < -match_template_threshold

        top_indices = np.nan_to_num(np.apply_along_axis(nd.center_of_mass, 0, tops)[0])
        bottom_indices = np.nan_to_num(
            np.apply_along_axis(nd.center_of_mass, 0, bottoms)[0]
        )
        center_indices = np.nan_to_num(
            np.apply_along_axis(nd.center_of_mass, 0, label_image)[0]
        )

        top_coordinates = np.array(
            [
                self.get_pixel_position_in_mm(col, row)
                for col, row in enumerate(top_indices)
            ]
        )
        bottom_coordinates = np.array(
            [
                self.get_pixel_position_in_mm(col, row)
                for col, row in enumerate(bottom_indices)
            ]
        )
        center_coordinates = np.array(
            [
                self.get_pixel_position_in_mm(col, row)
                for col, row in enumerate(center_indices)
            ]
        )
        heights = np.array(label_image.sum(axis=0)) * self.get_cell_size()[0]

        return top_coordinates, bottom_coordinates, center_coordinates, heights

    def get_focus_and_orthogonal(self, position):
        focus, orthogonal = get_focus_and_orthogonal_from_position(position)
        return focus, orthogonal

    def save_overlay_image(self, imagename=None):
        if imagename is None:
            imagename = "%s_z.png" % self.get_template()
        imsave(imagename, self.get_denoised_z())

    def save_report(self):
        z = self.get_z()

        print("ranges", self.get_extent())
        print("shape", self.get_shape())
        print("center", self.get_center())
        print("origin", self.get_origin())
        print("cell_size", self.get_cell_size())
        print("center of diffraction in cell sizes", self.get_center_of_mass_z())
        print(
            "center of diffraction in mm from origin",
            self.get_center_of_mass_z_mm_from_origin(),
        )
        print(
            "center of diffraction in absolute motor positions",
            self.get_center_of_mass_absolute_position(),
        )
        optimum = self.get_optimum()
        print("optimum", optimum)
        print("shift", self.get_shift())

        reference_position = self.get_reference_position()
        print("reference_position", reference_position)

        optimum_position = self.get_optimum_position()
        # self.goniometer.get_aligned_position_from_reference_position_and_x_and_y(reference_position, optimum[1], optimum[0])

        print("optimum_position", optimum_position)

        optimum_mark = Circle((optimum[-1], optimum[0]), color="red")

        denoised_z = self.get_denoised_z(min_spots=self.min_spots)
        filtered_z = self.get_filtered_z(min_spots=self.min_spots)
        best_of_z = self.get_best_of_z(min_spots=self.min_spots)
        best_of_z_label = self.get_best_of_z(label=True)

        raster_props = self.get_regionprops(filtered_z)
        for labeled_object in raster_props:
            print("Label %d" % labeled_object.label)
            locs_px = np.array(labeled_object.convex_image.shape)
            locs_mm = locs_px * self.get_cell_size()
            print(
                "The heigth and width of diffracting volume as estimated from conves_image is %d %d (px), %6.4f %6.4f (mm)"
                % tuple(list(locs_px) + list(locs_mm))
            )
            loci_px = np.array(labeled_object.image.shape)
            loci_mm = locs_px * self.get_cell_size()
            print(
                "The heigth and width of diffracting volume as estimated from image is %d %d (px), %6.4f %6.4f (mm)"
                % tuple(list(loci_px) + list(loci_mm))
            )
            locfi_px = np.array(labeled_object.filled_image.shape)
            locfi_mm = locs_px * self.get_cell_size()
            print(
                "The heigth and width of diffracting volume as estimated from filled_image is %d %d (px), %6.4f %6.4f (mm)"
                % tuple(list(loci_px) + list(loci_mm))
            )
            pixel_area = labeled_object.filled_area
            area_mm = pixel_area * np.prod(self.get_cell_size())
            print("The filled area is %d (px), %6.4f (mm^2)" % (pixel_area, area_mm))
            print(
                "The ellipse fit (px): major_axis %6.4f, minor_axis %6.4f, eccentricity %6.4f, orientation %6.4f %6.4f (rad, degrees)"
                % (
                    labeled_object.major_axis_length,
                    labeled_object.minor_axis_length,
                    labeled_object.eccentricity,
                    labeled_object.orientation,
                    np.degrees(labeled_object.orientation),
                )
            )

        (
            top_coordinates,
            bottom_coordinates,
            center_coordinates,
            heights,
        ) = self.get_shape_characteristics()

        results = {}
        results["optimum_position"] = optimum_position
        results["top_coordinates"] = top_coordinates
        results["center_coordinates"] = center_coordinates
        results["bottom_coordinates"] = bottom_coordinates
        results["heights"] = heights

        top_coordinates_aligned_positions = []
        center_coordinates_aligned_positions = []
        bottom_coordinates_aligned_positions = []

        top_coordinates_aligned_positions = [
            self.goniometer.get_aligned_position_from_reference_position_and_x_and_y(
                reference_position, c[0], c[1]
            )
            for c in center_coordinates
        ]
        center_coordinates_aligned_positions = [
            self.goniometer.get_aligned_position_from_reference_position_and_x_and_y(
                reference_position, c[0], c[1]
            )
            for c in center_coordinates
        ]
        bottom_coordinates_aligned_positions = [
            self.goniometer.get_aligned_position_from_reference_position_and_x_and_y(
                reference_position, c[0], c[1]
            )
            for c in center_coordinates
        ]

        results["top_coordinates_aligned_positions"] = top_coordinates_aligned_positions
        results[
            "center_coordinates_aligned_positions"
        ] = center_coordinates_aligned_positions
        results[
            "bottom_coordinates_aligned_positions"
        ] = bottom_coordinates_aligned_positions

        f = open(self.get_results_filename(), "wb")
        pickle.dump(results, f)
        f.close()

        top_color = "magenta"
        bottom_color = "cyan"
        center_color = "blue"

        fig, axes = pylab.subplots(2, 4, figsize=(20, 12))
        ax = axes.flatten()

        ax[0].imshow(
            self.get_image(color=True), extent=self.get_optical_image_min_max()
        )
        optimum_mark = Circle(
            (optimum[-1], optimum[0]),
            color="red",
            transform=ax[0].transData,
            radius=0.002,
        )
        ax[0].add_patch(optimum_mark)

        for col, row in bottom_coordinates:
            c = Circle(
                (col, row), color=bottom_color, transform=ax[0].transData, radius=0.002
            )
            ax[0].add_patch(c)

        for col, row in top_coordinates:
            c = Circle(
                (col, row), color=top_color, transform=ax[0].transData, radius=0.002
            )
            ax[0].add_patch(c)

        for col, row in center_coordinates:
            c = Circle(
                (col, row), color=center_color, transform=ax[0].transData, radius=0.002
            )
            ax[0].add_patch(c)

        ax[0].set_title("optical image, zoom %d" % self.get_parameters()["camera_zoom"])

        ax[1].imshow(z, extent=self.get_min_max())
        optimum_mark = Circle(
            (optimum[-1], optimum[0]),
            color="red",
            transform=ax[1].transData,
            radius=0.002,
        )
        ax[1].add_patch(optimum_mark)
        ax[1].set_title("z")

        ax[2].imshow(denoised_z, extent=self.get_min_max())
        optimum_mark = Circle(
            (optimum[-1], optimum[0]),
            color="red",
            transform=ax[2].transData,
            radius=0.002,
        )
        ax[2].add_patch(optimum_mark)
        ax[2].set_title("denoised_z")

        ax[3].imshow(
            remove_small_objects(img_as_int(filtered_z), min_size=4),
            extent=self.get_min_max(),
        )
        optimum_mark = Circle(
            (optimum[-1], optimum[0]),
            color="red",
            transform=ax[3].transData,
            radius=0.002,
        )
        ax[3].add_patch(optimum_mark)
        ax[3].set_title("filtered_z")

        ax[4].imshow(best_of_z, extent=self.get_min_max())
        optimum_mark = Circle(
            (optimum[-1], optimum[0]),
            color="red",
            transform=ax[4].transData,
            radius=0.002,
        )
        ax[4].add_patch(optimum_mark)
        ax[4].set_title(
            "best_of_z, min_spots=%d, threshold=%.1f" % (self.min_spots, self.threshold)
        )

        ax[5].imshow(
            self.get_filtered_z(min_spots=self.min_spots), extent=self.get_min_max()
        )

        for col, row in bottom_coordinates:
            c = Circle(
                (col, row), color=bottom_color, transform=ax[5].transData, radius=0.002
            )
            ax[5].add_patch(c)

        for col, row in top_coordinates:
            c = Circle(
                (col, row), color=top_color, transform=ax[5].transData, radius=0.002
            )
            ax[5].add_patch(c)

        for col, row in center_coordinates:
            c = Circle(
                (col, row), color=center_color, transform=ax[5].transData, radius=0.002
            )
            ax[5].add_patch(c)

        ax[5].set_title("filtered_z")

        ax[6].imshow(best_of_z_label, extent=self.get_min_max())
        ax[6].set_title("best_of_z_label")
        optimum_mark = Circle(
            (optimum[-1], optimum[0]),
            color="red",
            transform=ax[6].transData,
            radius=0.002,
        )
        ax[6].add_patch(optimum_mark)

        ax[7].imshow(self.get_z_scaled(best_of_z), extent=self.get_min_max())
        ax[7].set_title("best of z scaled to optical image")
        optimum_mark = Circle(
            (optimum[-1], optimum[0]),
            color="red",
            transform=ax[7].transData,
            radius=0.002,
        )
        ax[7].add_patch(optimum_mark)
        fig.suptitle(self.name_pattern)
        for axi in ax:
            axi.grid(False)

        try:
            pylab.savefig(
                "%s_raster_report.png" % os.path.join(self.directory, self.name_pattern)
            )
        except:
            print("could not save raster_report.png")

        optical_image_name = os.path.join(
            self.directory, self.name_pattern + "_optical_bw.png"
        )
        imsave(optical_image_name, self.get_image(color=False))

        scan_image_name = os.path.join(self.directory, self.name_pattern + "_scan.png")
        bw_overlay_image_name = os.path.join(
            self.directory, self.name_pattern + "_bw_overlay.png"
        )
        color_overlay_image_name = os.path.join(
            self.directory, self.name_pattern + "_overlay.png"
        )
        contour_overlay_image_name = os.path.join(
            self.directory, self.name_pattern + "_countour_overlay.png"
        )
        filter_overlay_image_name = os.path.join(
            self.directory, self.name_pattern + "_filter_overlay.png"
        )

        z_full = self.get_scaled_z_overlay(self.get_z_scaled(z))
        o = self.get_image(color=False)

        imsave(scan_image_name, z_full)
        imsave(bw_overlay_image_name, z_full + o)
        os.system(
            "composite -dissolve %s %s %s %s"
            % (55, scan_image_name, optical_image_name, color_overlay_image_name)
        )

        grid_contour = z_full * (z_full > 0.33 * z.max()) * (z_full < 0.66 * z.max())
        contour_image_name = os.path.join(
            self.directory, self.name_pattern + "_contour.png"
        )
        imsave(contour_image_name, grid_contour)
        os.system(
            "composite -dissolve %s %s %s %s"
            % (55, contour_image_name, optical_image_name, contour_overlay_image_name)
        )

        grid_filter = (z_full > 0.77 * z.max()) * 255
        filter_image_name = os.path.join(
            self.directory, self.name_pattern + "_filter.png"
        )
        imsave(filter_image_name, grid_filter)

        os.system(
            "composite -dissolve %s %s %s %s"
            % (55, filter_image_name, optical_image_name, filter_overlay_image_name)
        )

        os.system(
            "eog %s &" % os.path.join(self.directory, self.name_pattern + "*.png")
        )

        pylab.show()


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option(
        "-n",
        "--name_pattern",
        default="7c_side_on_2nd_mount",
        type=str,
        help="Template of files with the scan results, (default: %default)",
    )
    parser.add_option(
        "-d",
        "--directory",
        default="/nfs/ruche/proxima2a-spool/Martin/Research/radiation_damage/puck23/1/raster_scan",
        type=str,
        help="Directory with the scan results, (default: %default)",
    )
    parser.add_option(
        "-t",
        "--threshold",
        default=0.5,
        type=float,
        help="Threshold value in fraction of maximum (default=%default)",
    )
    parser.add_option(
        "-m",
        "--min_spots",
        default=7,
        type=int,
        help="Minimum acceptable number of diffraction spots (default=%default)",
    )

    options, args = parser.parse_args()

    print(options, args)

    rsa = raster_scan_analysis(
        options.name_pattern,
        options.directory,
        min_spots=options.min_spots,
        threshold=options.threshold,
    )

    # z = rsa.get_z()
    optimum_position = rsa.get_optimum_position()
    # z_full = rsa.get_scaled_z_overlay(rsa.get_z_scaled(z))

    rsa.save_report()
    # import pylab
    # pylab.figure()
    # pylab.imshow(z)
    # pylab.figure()
    # pylab.imshow(z_full)
    # pylab.show()


if __name__ == "__main__":
    main()
