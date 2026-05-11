# YuShi: Physically-informed volumetric structure modeling for single-radar nowcasting
**This repository contains the official implementation of Yushi framework.**

---------------------------------------------------------------

## 🚀 Getting Started

### Construct Environment 

```bash
git clone https://github.com/OpenEarthLab/Yushi
cd Yushi
conda create -n yushi python=3.10
conda activate yushi 
```

### Build Module
```
cd Pytorch-Correlation-extension
pip install -e .
cd Oriented1D/models/dwoconv1d
pip install -e .
cd -
cd Oriented1D/models/dwoconv1d_specialized
pip install -e .
```

### Download Checkpoint and Examples
- Download the [checkpoint](https://drive.google.com/file/d/1pbiMmdxPWUfuwTClZSr8nTWiaAVx2k3f/view?usp=drive_link) and [examples](https://drive.google.com/drive/folders/1cy2Si3MUma5he9F36n1f8O6IX7jsZwS5?usp=drive_link) from Google Drive.

### Inference
```
CUDA_VISIBLE_DEVICES=0 python inference.py
```

---------------------------------------------------------------
# Acknowledgements 
We would like to express our gratitude to the following open-source projects, whose high-quality code and research served as a foundation for this work: [Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension), [Oriented1D](https://github.com/princeton-vl/Oriented1D), [NowcastNet](https://codeocean.com/explore/b36ff208-5bac-4473-a72e-9a685e1e76ab) and [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

# Contact Information
Please feel free to contact zpq0316@163.com if you have questions.

