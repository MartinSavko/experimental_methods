a = """(name_pattern=None,
        directory=None,
        puck=None,
        sample=None,

        photon_energy=13000.,
        transmission=15.,
        resolution=1.5,

        scan_range=400.,
        frame_exposure_time=0.005,

        characterization_scan_range=1.2,
        characterization_scan_start_angles="[0, 45, 90, 135, 180]",
        characterization_frame_exposure_time=0.1,
        characterization_angle_per_frame=0.1,
        
        characterization_transmission=15.,
        
        
        wash=False,
        beam_align=False,
        skip_tomography=False,
        norient=1,
        defrost=0,
        prealign=False,
        enforce_scan_range=True,
        force_transfer=False,
        force_centring=False,
        ignore_top_up=False,
        
        default_directory="/nfs/data4/mechanized_sample_evaluation")"""

b = a.split(",\n")
c = [
    item.replace(" ", "").replace("\n", "").replace("(", "").replace(")", "")
    for item in b
]

spf = []
for item in c:
    name, value = item.split("=")
    value = eval(value)
    description = name.replace("_", " ")
    print(name, value, type(value), description)
    if value is not None:
        typ = type(value)
    else:
        typ = "str"
    f = {"name": name, "type": str(typ)[8:-2], "description": description}
    spf.append(f)

spf

[
    {"name": "name_pattern", "type": "str", "description": "name pattern"},
    {"name": "directory", "type": "str", "description": "directory"},
    {"name": "puck", "type": "str", "description": "puck"},
    {"name": "sample", "type": "str", "description": "sample"},
    {"name": "photon_energy", "type": float, "description": "photon energy"},
    {"name": "transmission", "type": float, "description": "transmission"},
    {"name": "resolution", "type": float, "description": "resolution"},
    {"name": "scan_range", "type": float, "description": "scan range"},
    {
        "name": "frame_exposure_time",
        "type": float,
        "description": "frame exposure time",
    },
    {
        "name": "characterization_scan_range",
        "type": float,
        "description": "characterization scan range",
    },
    {
        "name": "characterization_scan_start_angles",
        "type": str,
        "description": "characterization scan start angles",
    },
    {
        "name": "characterization_frame_exposure_time",
        "type": float,
        "description": "characterization frame exposure time",
    },
    {
        "name": "characterization_angle_per_frame",
        "type": float,
        "description": "characterization angle per frame",
    },
    {
        "name": "characterization_transmission",
        "type": float,
        "description": "characterization transmission",
    },
    {"name": "wash", "type": bool, "description": "wash"},
    {"name": "beam_align", "type": bool, "description": "beam align"},
    {"name": "skip_tomography", "type": bool, "description": "skip tomography"},
    {"name": "norient", "type": int, "description": "norient"},
    {"name": "defrost", "type": int, "description": "defrost"},
    {"name": "prealign", "type": bool, "description": "prealign"},
    {"name": "enforce_scan_range", "type": bool, "description": "enforce scan range"},
    {"name": "force_transfer", "type": bool, "description": "force transfer"},
    {"name": "force_centring", "type": bool, "description": "force centring"},
    {"name": "ignore_top_up", "type": bool, "description": "ignore top up"},
    {"name": "default_directory", "type": str, "description": "default directory"},
]
