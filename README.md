# Beyond Perspective: Neural 360-Degree Video Compression

## Acknowledgement

This is the model implementation of [DCVC-HEM-360](.) accepted for ICCV 2025.
The implementation is based on [DCVC-HEM](https://github.com/microsoft/DCVC) ([paper](https://arxiv.org/abs/2207.05894)).

Please find further information on citing this work and the baseline work at the bottom of this page.

## Timeline

- [x] Publish UGC360 dataset (July 2025)
- [x] Publish UGC360 example use (July 2025)
- [ ] Publish UGC360 download helper script (August 2025)
- [x] Publish model weights (July 2025)
- [x] Publish model weights download helper script (July 2025)
- [x] Publish model source code (July 2025)
- [x] Publish pre-trained model loading example use (July 2025) 
- [ ] Publish video encoding/decoding example script (August 2025)

## UGC360 dataset

The dataset is available for download via Huggingface: [UGC360 dataset](https://huggingface.co/datasets/FAU-LMS/UGC360).
Please download and unpack the dataset according to the instructions presented there.

The `UGC360` pytorch dataset subclass in this repository provides a ready-to-use dataset implementation to use the UGC360 dataset for training.
The provided dataset implementation incorporates the proposed **Flow-Guided Reprojection** with configurable parameters.
Example usage:
```python
from datasets import UGC360
import matplotlib.pyplot as plt

dataset = UGC360(
    ["/path/to/ugc360-s.csv",
     "/path/to/ugc360-m.csv"],
    sequence_length=7,     # The output number of frames
    filter_license=None,   # Optionally include specific CC licenses only
    patch_size=(256, 256), # The output patch size (height, width)
    resize_range=512,      # The virtual size of the reprojected frames
    flow_threshold=0.5,    # The flow threshold for the flow guide
    reproject=True,        # Whether to activate patch reprojection
    mipmap_levels=8        # Number of mipmap levels used during reprojection 
)

sample, pos = dataset[39]  # Sample the dataset (usually wrapped by a Dataloader)

plt.imshow(sample[0].permute(1, 2, 0).cpu().detach().numpy())
plt.show()
```

## Model weights

Pre-trained model weights are available for the DCVC-HEM and the extended DCVC-HEM-360 models.
You can download them via the provided [download script](model_weights/download.py).

| Filename                                                                                                       | FGR | vimeo90k | UGC360 | Model        |
|----------------------------------------------------------------------------------------------------------------|-----|---------|--------|--------------|
| [checkpoint_dcvchem_vimeo90k.pth](https://drive.google.com/file/d/1fYXXJf9QsXg-teu44pkyB1BPJByZKz8k)           |     | X       |        | DCVC-HEM     |
| [checkpoint_dcvchem_ugc360.pth](https://drive.google.com/file/d/1eDEdguj7UU_e4VZ0UyHXEGMf_diETJxv)             | X   |         | X      | DCVC-HEM     |
| [checkpoint_dcvchem_ugc360+vimeo90k.pth](https://drive.google.com/file/d/1lsHVPawj5BRFgVhSDnNjMLWvcPyJ1gP0)    | X   | X       | X      | DCVC-HEM     |
| [checkpoint_dcvchem360_ugc360+vimeo90k.pth](https://drive.google.com/file/d/1-CrfeZ9ovkFh4OyojpFKmKWc1dwCSBcr) | X   | X       | X      | DCVC-HEM-360 |

- FGR: Whether Flow-Guided Reprojection was used for training
- vimeo90k: Whether the vimeo90k dataset was used for training
- UGC360: Whether the UGC360 dataset was used for training
- Model: The model these weights refer to

**Important**:

Model weights for DCVC-HEM must be loaded to the DCVC-HEM model from [https://github.com/microsoft/DCVC](https://github.com/microsoft/DCVC).
Model weights for the DCVC-HEM-360 model must be loaded to the DCVC-HEM-360 model available in this repository.

## Model setup

Please make sure to download the pre-trained model weights before proceeding with this step.

```python
from DCVCHEM360.src.models.video_model_posinput import PosInputDMC, PosInputPositions
from DCVCHEM360.src.utils.stream_helper import get_state_dict
from pathlib import Path

# Entropy model only
posinput_positions = (
    PosInputPositions.HYPERPRIOR_ENCODER,
    PosInputPositions.HYPERPRIOR_DECODER,
    PosInputPositions.ENTROPY_MODEL
)

# Checkpoint for DCVC-HEM-360
checkpoint_filepath = Path(__file__).parent / 'model_weights' / 'checkpoint_dcvchem360_ugc360+vimeo90k.pth'

model = PosInputDMC(posinput_positions=posinput_positions)
model_state_dict = get_state_dict(checkpoint_filepath)
model.load_state_dict(model_state_dict)
```

## Prerequisites

* Follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally/) to install Pytorch.
* Make sure to install Pytorch with CUDA support if you want to use the GPU
* Further requirements
    ```
    # Enter your python environment, then execute:
    pip install -r requirements.txt
    ```

## Build the entropy coder

The entropy coder needs to be built to support compressed bitstream writing/parsing.

### On Windows
```bash
cd src
mkdir build
cd build
conda activate $YOUR_PY38_ENV_NAME
cmake ../cpp -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### On Linux
```bash
sudo apt-get install cmake g++
cd src
mkdir build
cd build
conda activate $YOUR_PY38_ENV_NAME
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j
```

## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{regensky2025nvc360,
  title     = {Beyond Perspective: Neural 360-Degree Video Compression},
  author    = {Andy Regensky and Marc Windsheimer and Fabian Brand and André Kaup},
  booktitle = {accepted for the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

The original DCVC-HEM this work is based on is proposed in:
```
@inproceedings{li2022hybrid,
  title={Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```
