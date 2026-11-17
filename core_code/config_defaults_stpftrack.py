"""STPFTrack-specific configuration defaults.

The final experiment overrides these defaults in
experiments/ostrack/stpftrack_final.yaml.
"""

STPFTRACK_MODEL_DEFAULTS = {
    "ENABLE": False,
    "FREEZE_BACKBONE": True,
    "TRAIN_HEAD": True,
    "FEAT_SIZE": 24,
    "FIELD_INIT": "center_gaussian",
    "FIELD_SIGMA": 0.25,
    "MODULATION_LAYERS": [2, 5, 8, 11],
    "UPDATE_LAYERS": [2, 5, 8, 11],
    "BOTTLENECK": 64,
    "MOD_SCALE_INIT": 0.0,
    "UPDATE_SCALE_INIT": 0.0,
    "USE_FIELD_LOSS": True,
    "FIELD_LOSS_TYPE": "mse",
    "FIELD_LOSS_WEIGHT": 0.1,
    "USE_FIELD_DEEP_SUPERVISION": False,
    "FIELD_DEEP_SUPERVISION_WEIGHT": 1.0,
    "NUM_SEARCH_FRAMES": 2,
    "PROPAGATE_FIELD": True,
    "FIELD_PROPAGATION_MIX": False,
    "FIELD_PROPAGATION_LAMBDA": 0.5,
    "DETACH_PROPAGATION": True,
    "USE_GT_FIELD_FOR_TRAIN_INIT": True,
    "GT_FIELD_INIT_PROB_START": 1.0,
    "GT_FIELD_INIT_PROB_END": 0.0,
    "GT_FIELD_INIT_DECAY_EPOCHS": 20,
    "USE_PRED_FIELD_FOR_NEXT": True,
    "USE_BOX_PRIOR_IN_UPDATE": False,
    "USE_FIELD_LOGITS": False,
    "USE_CENTERED_FIELD_MODULATION": False,
    "VISUALIZE_FIELD": False,
    "VIS_INTERVAL": 500,
    "VIS_MAX_SAMPLES": 4,
    "VIS_DIR": "field_vis",
}

STPFTRACK_DATA_DEFAULTS = {
    "SEARCH_NUMBER": 2,
}
