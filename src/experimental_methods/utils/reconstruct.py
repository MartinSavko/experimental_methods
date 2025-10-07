#!/usr/bin/env python

import os
import sys

sys.path.insert(0, "/usr/local/bin")
import time
import numpy as np
import pickle
import scipy.ndimage as ndi

# from thucyd.eigen import orient_eigenvectors
import astra
import open3d as o3d
from scipy.optimize import minimize
from predict import get_predictions, get_notion_prediction
from optical_path_report import (
    circle_model_residual,
    projection_model_residual,
    select_better_model,
    create_mosaic,
    circle_model,
    projection_model,
    circle_projection_model,
)
from experimental_methods.utils.useful_routines import principal_axes

from segmentation import get_hierarchical_mask_from_predictions
import pylab


def get_images_omegas_calibrations(
    directory,
    extension,
    model_img_size=(256, 320),
    omega_search_string="color_zoom_[\d]*_([\d\.]*).jpg|.*omega_([\d\.]*)_zoom.*.jpg|color_([\d\.]*).jpg|color_kappa_.*_omega_([\d\.]*).jpg",
    zoom_search_string=".*_zoom_([\d]*)_.*jpg",
):
    _start = time.time()
    import re
    import glob
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from camera import calibrations as pixel_calibrations

    omega_match = re.compile(omega_search_string)
    zoom_match = re.compile(zoom_search_string)
    imagenames = glob.glob(os.path.join(directory, extension))

    to_predict = []
    angles = []
    calibrations = []
    to_predict_imagenames = []

    for imagename in imagenames:
        if "background" in imagename:
            continue
        angle = None
        omega = omega_match.findall(imagename)[0]

        if omega[0] != "":
            angle = omega[0]
        elif omega[1] != "":
            angle = omega[1]
        elif omega[2] != "":
            angle = omega[2]
        elif omega[3] != "":
            angle = omega[3]

        if angle is not None:
            angles.append(np.deg2rad(float(angle)))
            to_predict.append(
                img_to_array(
                    load_img(imagename, target_size=model_img_size), dtype="float32"
                )
            )
            to_predict_imagenames.append(imagename)
            zoom = int(zoom_match.findall(imagename)[0])
            calibrations.append(pixel_calibrations[zoom])
        else:
            continue

    _end = time.time()
    print("loaded %d images in %.4f seconds" % (len(to_predict), _end - _start))
    return to_predict, angles, calibrations, to_predict_imagenames


def get_reconstruction(
    projections,
    angles,
    detector_rows,
    detector_cols,
    detector_row_spacing=1.0,
    detector_col_spacing=1.0,
    vertical_correction=0.0,
):
    print("vertical_correction", vertical_correction)
    proj_geom = astra.create_proj_geom(
        "parallel3d",
        detector_col_spacing,
        detector_row_spacing,
        detector_cols,
        detector_rows,
        angles,
    )
    proj_geom_vec = astra.geom_2vec(proj_geom)
    proj_geom_cor = astra.geom_postalignment(proj_geom, (0, vertical_correction))
    projections_id = astra.data3d.create("-sino", proj_geom_cor, projections)
    vol_geom = astra.create_vol_geom(
        2 * detector_rows, 2 * detector_rows, detector_cols
    )
    reconstruction_id = astra.data3d.create("-vol", vol_geom, 0)
    alg_cfg = astra.astra_dict("BP3D_CUDA")
    alg_cfg["ProjectionDataId"] = projections_id
    alg_cfg["ReconstructionDataId"] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)
    _start = time.time()
    astra.algorithm.run(algorithm_id, 150)
    _end = time.time()
    print("volume reconstructed in %.4f seconds" % (_end - _start))
    reconstruction = astra.data3d.get(reconstruction_id)

    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projections_id)

    return reconstruction


def reconstruct(
    directory="",
    extension="*.jpg",
    search_string="color_zoom_[\d]*_([\d\.]*).jpg|.*omega_([\d\.]*)_zoom.*.jpg|color_([\d\.]*).jpg",
    model_img_size=(256, 320),
    threshold=0.5,
    display=False,
    verbose=False,
    reconstruction_basename="reconstruction.npy",
    pcd_basename="reconstruction.pcd",
    png_basename="reconstruction.png",
    fits_basename="fits.pickle",
    predictions_basename="predictions.pickle",
    projections_basename="projections.npy",
    notions=[
        "foreground",
        "crystal",
        "loop",
        "stem",
        "loop_inside",
        "pin",
        ["crystal", "loop_inside", "loop", "stem"],
    ],
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    minimize_method="nelder-mead",
    initial_parameters=[80.0, 25.0, np.pi / 8],
    generate_report=False,
    save_raw_reconstructions=False,
):
    _start = time.time()

    fits_filename = os.path.join(directory, fits_basename)
    predictions_filename = os.path.join(directory, predictions_basename)

    (
        to_predict,
        angles,
        calibrations,
        to_predict_imagenames,
    ) = get_images_omegas_calibrations(
        directory, extension, model_img_size=model_img_size
    )

    request_arguments = {}
    request_arguments["to_predict"] = to_predict
    request_arguments["model_img_size"] = model_img_size
    request_arguments["save"] = False
    request_arguments["prefix"] = ""

    _start = time.time()
    predictions = get_predictions(request_arguments)
    _end = time.time()

    f = open(predictions_filename, "wb")
    pickle.dump(predictions, f)
    f.close()

    duration = _end - _start
    print(
        "got %d predictions in %.4f seconds (%.4f seconds per image)"
        % (len(to_predict), duration, duration / len(to_predict))
    )

    shape = predictions[0].shape
    print("predictions.shape", shape)
    detector_rows = shape[1]
    detector_cols = shape[2]
    number_of_projections = shape[0]
    all_fits = {}
    all_fits["all_angles"] = angles
    for notion in notions:
        notion_string = ",".join(notion) if type(notion) is list else notion
        # if type(notion) is list:
        # notion_string = ','.join(notion)
        # else:
        # notion_string = notion

        print("notion", notion_string)
        notion_projections_filename = os.path.join(
            directory, "%s_%s" % (notion_string, projections_basename)
        )
        notion_pcd_filename = os.path.join(
            directory, "%s_%s" % (notion_string, pcd_basename)
        )
        notion_reconstruction_filename = os.path.join(
            directory, "%s_%s" % (notion_string, reconstruction_basename)
        )
        notion_png_filename = os.path.join(
            directory, "%s_%s" % (notion_string, png_basename)
        )

        projections = np.zeros((detector_cols, number_of_projections, detector_rows))
        heights = []
        widths = []
        centroid_verticals = []
        centroid_horizontals = []
        rightmost_point_verticals = []
        rightmost_point_horizontals = []
        areas = []
        fit_angles = []
        valid_projection_index = 0
        for k, angle in enumerate(angles):  # range(number_of_projections):
            (
                present,
                r,
                c,
                h,
                w,
                r_max,
                c_max,
                area,
                notion_prediction,
            ) = get_notion_prediction(predictions, notion, k=k, threshold=threshold)
            if not np.isnan(present):
                heights.append(h)
                widths.append(w)
                centroid_verticals.append(r)
                centroid_horizontals.append(w)
                rightmost_point_verticals.append(r_max)
                rightmost_point_horizontals.append(c_max)
                areas.append(area)
                fit_angles.append(angle)
                projections[:, valid_projection_index, :] = (
                    (notion_prediction[:, :] > threshold).astype("float32").T
                )
                valid_projection_index += 1

        if valid_projection_index == 0:
            continue

        projections = projections[:, :valid_projection_index, :]
        if save_raw_reconstructions:
            np.save(notion_projections_filename, projections)

        projection_sum = np.sum(np.sum(projections, axis=2) > 0, axis=1)

        median = np.median(projection_sum[np.argwhere(projection_sum > 0)])

        too_few_to_count = np.argwhere(projection_sum <= max(1, int(0.25 * median)))
        projections[too_few_to_count, :, :] = 0

        # fits
        fit_start = time.time()
        fits = {}
        fits["aspects"] = {
            "heights": heights,
            "widths": widths,
            "centroid_verticals": centroid_verticals,
            "centroid_horizontals": centroid_horizontals,
            "rightmost_point_verticals": rightmost_point_verticals,
            "rightmost_point_horizontals": rightmost_point_horizontals,
            "areas": areas,
        }
        fits["angles"] = fit_angles
        fits["results"] = {}
        for aspect in fits["aspects"]:
            initial_parameters = [
                np.mean(fits["aspects"][aspect]),
                np.std(fits["aspects"][aspect]) / np.sin(np.pi / 4.0),
                np.random.rand() * np.pi,
            ]
            fit_circle = minimize(
                circle_model_residual,
                initial_parameters,
                method=minimize_method,
                args=(fits["angles"], fits["aspects"][aspect]),
            )
            fit_projection = minimize(
                projection_model_residual,
                initial_parameters,
                method=minimize_method,
                args=(fits["angles"], fits["aspects"][aspect]),
            )
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
            fits["results"][aspect] = {
                "fit_circle": fit_circle,
                "fit_projection": fit_projection,
                "fit": fit,
                "k": k,
            }
        fit_end = time.time()
        print("Fit took %.4f seconds" % (fit_end - fit_start))

        all_fits[notion_string] = fits

        if notion == "foreground":
            center_of_mass = ndi.center_of_mass(projections)
            print(
                "projections shape, center_of_mass", projections.shape, center_of_mass
            )
            if not np.isnan(center_of_mass[2]):
                vertical_correction = detector_rows // 2 - center_of_mass[2]
            else:
                vertical_correction = 0

        reconstruction = get_reconstruction(
            projections,
            fit_angles,
            detector_rows,
            detector_cols,
            vertical_correction=vertical_correction,
        )
        if save_raw_reconstructions:
            np.save(notion_reconstruction_filename, reconstruction)
        reconstruction[reconstruction < 0] = 0
        reconstruction /= np.max(reconstruction)
        reconstruction = np.round(reconstruction * 255).astype(np.uint8)

        # segment
        for k in range(0, reconstruction.shape[0]):
            reconstruction[k, :, :] = (
                reconstruction[k, :, :] > 0.95 * reconstruction[k, :, :].max()
            )

        inertia, eigenvalues, eigenvectors, center, Vor, Eor = principal_axes(
            reconstruction, verbose=verbose
        )

        objectpoints = np.argwhere(reconstruction == 1)
        if verbose:
            print("objectpoints.shape", objectpoints.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(objectpoints)
        o3d.io.write_point_cloud(notion_pcd_filename, pcd)

        """https://stackoverflow.com/questions/62122925/pointcloud-to-image-in-open3d"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            visible=True
        )  # works for me with False, on some systems needs to be true
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(notion_png_filename, do_render=True)
        time.sleep(1)
        vis.destroy_window()
        if display:
            o3d.visualization.draw_geometries([pcd])

    f = open(fits_filename, "wb")
    pickle.dump(all_fits, f)
    f.close()

    if generate_report:
        from tensorflow.keras.preprocessing.image import save_img

        report_generation_start = time.time()
        images_mosaic = create_mosaic(to_predict)
        labeled_images = [
            get_hierarchical_mask_from_predictions(predictions, k=k)
            for k in range(shape[0])
        ]
        for li, oi in zip(labeled_images, to_predict_imagenames):
            prediction_imagename = oi.replace(".jpg", "_prediction.png")
            save_img(prediction_imagename, np.expand_dims(li, axis=2), scale=False)
        edges_mosaic = create_mosaic(np.array(labeled_images))
        fig, axes = pylab.subplots(1, 2, figsize=(16, 9))
        ax = axes.flatten()
        fig.suptitle(os.path.basename(directory))
        ax[0].imshow(images_mosaic)
        ax[0].set_title("input images")
        ax[0].grid(False)
        ax[1].imshow(edges_mosaic)
        ax[1].set_title("labeled images")
        ax[1].grid(False)
        pylab.savefig(os.path.join(directory, "comparison_mosaic.png"))

        model_angles = np.linspace(0, 2 * np.pi, 1000)
        model_degrees = np.degrees(model_angles)

        for k, notion in enumerate(notions):
            if type(notion) is list:
                notion_string = ",".join(notion)
            else:
                notion_string = notion
            print("notion", notion_string)
            if not notion_string in all_fits:
                continue

            fig, axes = pylab.subplots(1, 2, figsize=(16, 9))
            fig.suptitle("%s: %s" % (os.path.basename(directory), notion_string))
            ax = axes.flatten()

            ax[0].set_xlabel("degrees")
            ax[0].set_ylabel("pixels")
            ax[1].set_xlabel("degrees")
            ax[1].set_ylabel("pixels")
            lines = []
            labels = []
            a = ax[0]
            a.set_title("width & height")
            for aspect, color in zip(["heights", "widths"], ["red", "green"]):
                angles_degrees = np.degrees(all_fits[notion_string]["angles"])
                l = a.plot(
                    angles_degrees,
                    all_fits[notion_string]["aspects"][aspect],
                    "o",
                    color=color,
                    label=aspect,
                )
                lines.append(l)
                labels.append(aspect)
                l = a.plot(
                    model_degrees,
                    circle_projection_model(
                        model_angles,
                        *all_fits[notion_string]["results"][aspect]["fit"].x,
                        k=all_fits[notion_string]["results"][aspect]["k"]
                    ),
                    "--",
                    lw=2,
                    color=color,
                    label="%s_model" % aspect,
                )
                lines.append(l)
                labels.append("%s_model" % aspect)
            # pylab.legend()
            a.legend()
            a = ax[1]
            a.set_title("position")
            for aspect, color in zip(
                [
                    "centroid_verticals",
                    "centroid_horizontals",
                    "rightmost_point_verticals",
                    "rightmost_point_horizontals",
                ],
                ["green", "blue", "orange", "cyan"],
            ):
                angles_degrees = np.degrees(all_fits[notion_string]["angles"])
                l = a.plot(
                    angles_degrees,
                    all_fits[notion_string]["aspects"][aspect],
                    "o",
                    color=color,
                    label=aspect,
                )
                lines.append(l)
                labels.append(aspect)
                l = a.plot(
                    model_degrees,
                    circle_projection_model(
                        model_angles,
                        *all_fits[notion_string]["results"][aspect]["fit"].x,
                        k=all_fits[notion_string]["results"][aspect]["k"]
                    ),
                    "--",
                    lw=2,
                    color=color,
                    label="%s_model" % aspect,
                )
                lines.append(l)
                labels.append("%s_model" % aspect)
            a.legend()
            # a.legend(lines, labels)
            pylab.legend()  # lines, labels)
            pylab.savefig(os.path.join(directory, "%s_report.png" % notion_string))
        if display:
            pylab.show()
    _end = time.time()

    print("reconstruct took %.4f seconds" % (_end - _start))
    return 0


if __name__ == "__main__":
    _start = time.time()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        default="learning_dataset/100161_Sun_Apr_25_19:15:59_2021",
        type=str,
        help="directory",
    )
    parser.add_argument("-D", "--display", action="store_true", help="display")
    parser.add_argument("-f", "--force", action="store_true", help="force")
    parser.add_argument(
        "-g", "--generate_report", action="store_true", help="Generate report"
    )
    parser.add_argument(
        "-R",
        "--save_raw_reconstructions",
        action="store_true",
        help="Save raw reconstructions",
    )
    args = parser.parse_args()
    print(args)
    report_filename = os.path.join(args.directory, "comparison_mosaic.png")
    if not bool(args.force) and os.path.isfile(report_filename):
        _end = time.time()
        sys.exit(
            "report for %s already present. check took %.4f seconds"
            % (args.directory, _end - _start)
        )

    reconstruct(
        directory=os.path.realpath(args.directory),
        display=bool(args.display),
        generate_report=bool(args.generate_report),
        save_raw_reconstructions=bool(args.save_raw_reconstructions),
    )


"""
Notes
    test
    import random
    import pylab
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    #xx, yy, zz = objectpoints
    #mlab.points3d(xx, yy, zz,
                  #mode="cube",
                  #color=(0, 1, 0),
                  #scale_factor=1)
    #mlab.show()
    
    # https://stackoverflow.com/questions/62948421/how-to-create-point-cloud-file-ply-from-vertices-stored-as-numpy-array
    #xyz = np.vstack([xx, yy, zz]).T
    #print('xyz.shape', xyz.shape)
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize

    #o3d.visualization.draw_geometries([pcd])
    
    ## Load saved point cloud and visualize it
    #pcd_load = o3d.io.read_point_cloud(pcd_filename)
    
    #index = 10 # random.randint(0, predictions[0].shape[0]-1)
    #pylab.imshow(to_predict[index]/255.)
    #pylab.axis('Off')
    #pylab.show()
    #print('showing image %d' % index)
    #ipredictions = np.array([np.expand_dims(item[index], 0) for item in predictions], dtype='float32')
    #for p in ipredictions:
        #pylab.imshow(p[0,:,:,0]>threshold)
        #pylab.show()
    #for notion in ['loop_inside', 'loop', 'stem', 'crystal', 'foreground']:
        #print(notion)
        #example = get_notion_prediction(ipredictions, notion)
        #pylab.title(notion)
        #pylab.axis('Off')
        #pylab.imshow(example[-1])
        #pylab.show()
        
    #projections = foregrounds.reshape((shape[1], shape[0], shape[2]))
    #projections = foregrounds.reshape((detector_rows, number_of_projections, detector_cols))
    #projections = predictions[-1].reshape(shape[:3])
"""
