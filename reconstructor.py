#!/usr/bin/env python
# -*- coding: utf-8 -*-

import zmq
import time
import pickle
import traceback
import astra


def get_reconstruction(
    projections,
    angles,
    detector_rows,
    detector_cols,
    detector_row_spacing=1.0,
    detector_col_spacing=1.0,
    vertical_correction=0.0,
    horizontal_correction=0.0,
    volume_rows_factor=2,
    volume_cols_factor=1,
):
    print("projections.shape", projections.shape)
    print("len(angles)", len(angles))
    print("rows, cols:", detector_rows, detector_cols)
    print("detector spacing (R, C)", detector_row_spacing, detector_col_spacing)
    print("correction (V, H)", vertical_correction, horizontal_correction)
    print("volume_factor (R, C)", volume_rows_factor, volume_cols_factor)
    
    proj_geom = astra.create_proj_geom(
        "parallel3d",
        detector_col_spacing,
        detector_row_spacing,
        detector_cols,
        detector_rows,
        angles,
    )
    proj_geom_vec = astra.geom_2vec(proj_geom)
    proj_geom_cor = astra.geom_postalignment(
        proj_geom, (vertical_correction, horizontal_correction)
    )
    projections_id = astra.data3d.create("-sino", proj_geom_cor, projections)
    vol_geom = astra.create_vol_geom(
        volume_rows_factor*detector_rows, volume_rows_factor*detector_rows, volume_cols_factor*detector_cols
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


def serve(port=8900):
    print(f"Reconstrucor will listen on port {port:d}")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    while True:
        request_string = socket.recv()
        request = pickle.loads(request_string)
        print("%s received request" % (time.asctime(),))
        try:
            projections = request["projections"]
            angles = request["angles"]
            if "detector_rows" in request:
                detector_rows = request["detector_rows"]
            else:
                detector_rows = projections.shape[0]
            if "detector_cols" in request:
                detector_cols = request["detector_cols"]
            else:
                detector_cols = projections.shape[-1]
            
            kwargs = {}
            for key in ["detector_row_spacing", "detector_col_spacing", "vertical_correction", "horizontal_correction", "volume_rows_factor", "volume_cols_factor"]:
                if key in request:
                    kwargs[key] = request[key]
                    
            reconstruction = get_reconstruction(
                projections,
                angles,
                detector_rows,
                detector_cols,
                **kwargs,
            )
        except:
            print(traceback.print_exc())
            reconstruction = -1
        reconstruction_string = pickle.dumps(reconstruction)
        socket.send(reconstruction_string)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--port", type=int, default=8900, help="port")

    args = parser.parse_args()

    serve(port=args.port)

