#!/usr/bin/env python
# coding: utf-8

import os
from useful_routines import (
    get_pickled_file,
    images2movie,
    get_lut,
    filter_descriptions,
)
from explore_descriptions import omalovanka

from optical_alignment import optical_alignment

def run_murko(directory, name_pattern, binning=2, blocking=True, force=False, codec="vp80", suffix="webm", volume=True, threshold=None, port=89013):
    oa = optical_alignment(
        directory=directory,
        name_pattern=name_pattern
    )
    
    descriptions = oa.get_descriptions(port=port)
    
    hierarchical_masks = [item["hierarchical_mask"] for item in descriptions]
    
    if threshold:
        descriptions = filter_descriptions(descriptions, threshold=threshold)
                                           
    if volume:
        _ = oa.get_volume(descriptions=descriptions)
        
    
    movie = f"{oa.get_template()}_murko_movie.{suffix}"
    print(f"run_murko {movie}")
    if os.path.isfile(movie) and not force:
        return
    
    if hierarchical_masks[0].shape[-1] > 3:
        lut = get_lut()
        hierarchical_masks = [omalovanka(hm, lut=lut) for hm in hierarchical_masks]
        
    print(f"saving {len(hierarchical_masks)} images to movie")
    images2movie(hierarchical_masks, movie=movie, codec=codec)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="/nfs/data4/2026_Run3/20260017/2026-06-11/ARCHIVE/opti/manu_7_12_20260611_130650_zoom_1_kappa_30.00_phi_0.00_parameters.pickle",
        help="sufficient experiment description",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force execution",
    )
    
    parser.add_argument(
        "-u",
        "--unblock",
        action="store_true",
        help="do not block execution",
    )
    
    parser.add_argument(
        "-c",
        "--codec",
        default="vp80",
        type=str,
        help="codec",
    )
    
    parser.add_argument(
        "-s",
        "--suffix",
        default="webm",
        type=str,
        help="suffix",
    )
    
    parser.add_argument(
        "-p",
        "--port",
        default=89013,
        type=int,
        help="port",
    )
        
    args = parser.parse_args()
    print(f"args {args}")
    
    directory = None
    name_pattern = None
    if args.experiment.endswith("_sample_view_movie.mp4") and os.path.isfile(args.experiment):
        directory = os.path.dirname(args.experiment)
        name_pattern = os.path.basename(args.experiment).replace("_sample_view_movie.mp4", "")
    elif args.experiment.endswith(".pickle"):
        pars = get_pickled_file(args.experiment)
        directory = pars["directory"]
        name_pattern = pars["name_pattern"]
    
    if directory is not None and name_pattern is not None:
        run_murko(directory, name_pattern, blocking=not args.unblock, force=args.force, codec=args.codec, suffix=args.suffix, port=args.port)

if __name__ == "__main__":
    main()
