# STPFTrack

STPFTrack adapts OSTrack for thermal infrared single-object tracking. It represents
target information as a dense field aligned with the search-token grid. The field
modulates intermediate search features, is updated using template–search
similarity, and carries target information across frames.

[Network overview](assets/04net_overview.pdf) · [Experiment configuration](experiments/ostrack/stpftrack_final.yaml) · [Training log](training_log/stpftrack_final.log)

## Core implementation

| File | Contents |
| --- | --- |
| [stpf.py](core_code/stpf.py) | Target field construction, feature modulation, and field updates |
| [base_backbone_stpftrack.py](core_code/base_backbone_stpftrack.py) | Integration with the ViT backbone |
| [ostrack_forward.py](core_code/ostrack_forward.py) | Target field inputs and outputs in the model forward pass |
| [train_actor_stpftrack.py](core_code/train_actor_stpftrack.py) | Field propagation and supervision during training |
| [test_tracker_stpftrack.py](core_code/test_tracker_stpftrack.py) | Online field updates during tracking |
| [config_defaults_stpftrack.py](core_code/config_defaults_stpftrack.py) | Default settings for STPFTrack |

These files are excerpts from the OSTrack implementation. Running them requires
the full project, model weights, and datasets. The [environment record](environment.yml)
documents the experiment dependencies.

## Tracking results

Precomputed results are available for four datasets.

| Dataset | Results |
| --- | --- |
| LasHeR (thermal infrared) | [lasher_tir](final_results/lasher_tir/) |
| LSOTB-TIR100 | [lsotb100_tir](final_results/lsotb100_tir/) |
| LSOTB-TIR120 | [lsotb_tir](final_results/lsotb_tir/) |
| PTB-TIR | [ptb_tir](final_results/ptb_tir/) |

Each `<sequence>.txt` file stores predicted bounding boxes in `x,y,w,h` format.
Where available, `<sequence>_time.txt` stores per-frame runtimes.
