# Segmenting human artifacts for image inpainting.

## 1. Specification of dependencies

This code requires Python 3.8.12 and CUDA 11.2. Create and activate the following conda envrionment.

```
conda update conda
conda env create -f environment.yml
conda activate laion_seg
```

## 2a. Training code

### Dataset details
We expect the dataset to have the following structure:
```
datasets/
|-- human_artifacts/
|---- train/
|------ humans/
|------ masks/
|-- test/
|---- humans/
|---- masks/
```

To train model, run:

```python
python train.py

```

## 3. Pre-trained models

TBA.

| Dataset | Model  |  Comments | Download   |
| ---------- | ---------- | ---------- | ---------- |
| human artifacts |U-Net | U-Net model trained on 10 images | [TBA](TBA) |

## 4. Demo
TBA.

## License 
MIT