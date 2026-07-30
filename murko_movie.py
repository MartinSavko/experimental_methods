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

def run_murko(directory, name_pattern, binning=2, blocking=True, force=False, codec="vp80", suffix="webm", volume=True, threshold=None):
    oa = optical_alignment(
        directory=directory,
        name_pattern=name_pattern
    )
    
    descriptions = oa.get_descriptions()
    
    hierarchical_masks = [item["hierarchical_mask"] for item in descriptions]
    
    if threshold:
        descriptions = filter_descriptions(descriptions, threshold=threshold)
                                           
    if volume:
        _ = oa.get_volume(descriptions=descriptions)
        
    lut = get_lut()
    movie = f"{oa.get_template()}_murko_movie.{suffix}"
    print(f"run_murko {movie}")
    if os.path.isfile(movie) and not force:
        return
    rgb = [omalovanka(hm, lut=lut) for hm in hierarchical_masks]
    print(f"saving {len(rgb)} images to movie")
    images2movie(rgb, movie=movie, codec=codec)

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
    
        
    args = parser.parse_args()
    print(f"args {args}")
    
    if args.experiment.endswith("_sample_view_movie.mp4") and os.path.isfile(args.experiment):
        directory = os.path.dirname(args.experiment)
        name_pattern = os.path.basename(args.experiment).replace("_sample_view_movie.mp4", "")
    elif args.experiment.endswith(".pickle"):
        pars = get_pickled_file(args.experiment)
        directory = pars["directory"]
        name_pattern = pars["name_pattern"]
    
    run_murko(directory, name_pattern, blocking=not args.unblock, force=args.force, codec=args.codec, suffix=args.suffix)

if __name__ == "__main__":
    main()
