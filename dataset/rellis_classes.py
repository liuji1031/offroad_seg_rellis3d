# RELLIS-3D semantic label classes (20 classes including void)
RELLIS_CLASSES = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}

RELLIS_CLASSES_NAME_INV_MAP = {v: k for k, v in RELLIS_CLASSES.items()}

RELLIS_CLASSES_NAMES_SORTED = [
    "void",
    "grass",
    "bush",
    "tree",
    "mud",
    "concrete",
    "puddle",
    "person",
    "rubble",
    "barrier",
    "fence",
    "log",
    "water",
    "pole",
    "vehicle",
    "asphalt",
    "dirt",
    "sky",
    "object",
    "building",
]

# Color map for visualization (RGB format)
RELLIS_COLOR_MAP = {
    0: [0, 0, 0],  # void - black
    1: [108, 64, 20],  # dirt - brown
    3: [0, 102, 0],  # grass - dark green
    4: [0, 255, 0],  # tree - green
    5: [0, 153, 153],  # pole - cyan
    6: [0, 128, 255],  # water - light blue
    7: [0, 0, 255],  # sky - blue
    8: [255, 255, 0],  # vehicle - yellow
    9: [255, 0, 127],  # object - magenta
    10: [64, 64, 64],  # asphalt - dark gray
    12: [255, 0, 0],  # building - red
    15: [102, 0, 0],  # log - dark red
    17: [204, 153, 255],  # person - light purple
    18: [102, 0, 204],  # fence - purple
    19: [255, 153, 204],  # bush - pink
    23: [170, 170, 170],  # concrete - gray
    27: [41, 121, 255],  # barrier - blue
    31: [134, 255, 239],  # puddle - cyan
    33: [99, 66, 34],  # mud - brown
    34: [110, 22, 138],  # rubble - purple
}

RELLIS_COLOR_LIST_SORTED = [
    (RELLIS_CLASSES_NAME_INV_MAP[n], RELLIS_COLOR_MAP[
        RELLIS_CLASSES_NAME_INV_MAP[n]
    ])
    for n in RELLIS_CLASSES_NAMES_SORTED
]

sorted_keys = sorted(RELLIS_CLASSES.keys())
RELLIS_CLASS_INDEX_MAP = {k: i for i, k in enumerate(sorted_keys)}
RELLIS_CLASS_INDEX_MAP_INV = {v: k for k, v in RELLIS_CLASS_INDEX_MAP.items()}
