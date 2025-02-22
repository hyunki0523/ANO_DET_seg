# MI2RL_ANO_DET_seg

Welcome to the official repository 


## Used Models 
- [nnU-Net V2](https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file) ✓
- [Swin-UNETR](https://arxiv.org/abs/2201.01266) ✓
- [MedNeXt](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_39) ✓
- [STUnet](https://arxiv.org) ✓
- [MedFormer](https://arxiv.org) ✓



## Benchmark Available 
- Liver Tumor Segmentation 5-Fold (**100 epoch**) [(LiTS)](https://www.sciencedirect.com/science/article/pii/S1361841522003085) ✓ 

The Dice coefficient for lesions is calculated as an average across all lesions. For the evaluation code and a detailed list of metrics, please refer to the [evaluation folder](evaluation). 

| Model    | Dice coefficient (Dice)   | accuracy (Acc.)  | specificity (Spe.)  |  sensitivity (Sen.)    | 
|----------|---------------|----------------|--------------|---------------|
| nnUNet3D | 76.29 (±2.98) | 63.22 (±3.68)  | 1.60 (±0.24) | 91.84 (±0.86) | 
| SwinUNETR | 74.21 (±1.92) | 60.74 (±2.22)  | 1.86 (±0.39) | 86.74 (±1.36) | 
| STUnet | 77.31 (±2.34) | 64.41 (±3.04)  | 1.74 (±0.42) | 92.30 (±0.98) | 
| MedNext | 77.44 (±2.03) | 64.57 (±2.67)  | 1.63 (±0.52)  | 93.60 (±1.25) | 
| MedFormer | 76.99 (±2.12) | 63.84 (±2.62)  | 1.81 (±0.38)  | 94.24 (±0.49) | 


- [TriALS](https://www.synapse.org/#!Synapse:syn53285416/wiki/) (in-progress)

## Getting Started

To get started, follow these steps:

1. Clone the Repository
   ```bash
   git clone https://github.com/xmed-lab/TriALS
   ```
2. Create and Activate a Virtual Environment
    ```bash
    conda create -n TriALS python=3.10
    conda activate TriALS
   ```
3. Install Pytorch: Follow the instructions [here](https://pytorch.org/get-started/locally/):
   ```bash 
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
5. Install the Repository 
   ```bash 
    cd TriALS
    pip install -e .
   ```

## Data Preparation
We follow the [nnU-Net V2](https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file) guideline for data preparation, detailed below and accessible [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).


1. Download and Prepare the [MSD](http://medicaldecathlon.com/) Liver Dataset, and extract it into the `data/nnUNet_raw_data_base` directory.
   ```bash 
   gdown https://drive.google.com/uc?id=1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu
   # or use wget
   wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar
   ```
    ```bash 
    tar -xvf Task03_Liver.tar -C data/nnUNet_raw_data_base
   ```

2. Export nnUNet-v2 evironment variables:
   ```bash
   export nnUNet_raw=<path-to>/data/nnUNet_raw_data_base/
   export nnUNet_preprocessed=<path-to>/data/nnUNet_preprocessed/
   export nnUNet_results=<path-to>/data/nnUNet_results/
   ```

3. Convert the [MSD](http://medicaldecathlon.com/) Liver dataset to nnU-Net format:
   ```bash 
    python nnunetv2/dataset_conversion/convert_MSD_dataset.py -i data/nnUNet_raw_data_base/Task03_Liver/
   ```
Sanity Check: Please verify that the dataset is organized in the following structure:
```
data/
├── nnUNet_raw_data_base/
│   ├── Dataset003_Liver/
│   │   ├── imagesTr
│   │   │   ├── liver_1_0000.nii.gz
│   │   │   ├── liver_2_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── liver_1.nii.gz
│   │   │   ├── liver_2.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
```
3. Preprocess the LiTS Dataset: Replace `<DATASET_ID>` in the command below with `3`:
   ```bash
   nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity
   ```
## Model Training
To train the models, follow these instructions:
- General Training Command on GPU 0. **For the preliminary benchmark, all models are trained for 100 epochs**. Please note that all model variants, with the exception of nnUNet, are trained without deep supervision.   
   ```bash
   CUDA_VISIBLE_DEVICES=0 nnUNetv2_train <DATASET_ID> <CONFIGURATION> <FOLD_NUM> -tr <TRAINER>
   ```
    | Model                                                                         | Configuration | Trainer                           |
    |-------------------------------------------------------------------------------|---------------|-----------------------------------|
    | [nnU-Net 3D](https://github.com/MIC-DKFZ/nnUNet)                              | 3d_fullres    | nnUNetTrainer_100epochs                     |
    | [Swin-UNETR](https://arxiv.org/abs/2201.01266)                                | 3d_fullres    | nnUNetTrainerSwinUNETR_100epochs            |
    | [MedNext-L-5](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_39) | 3d_fullres    | nnUNetTrainerV2_MedNeXt_L_kernel5_100epochs |


### Example Training Commands

- **MedNeXT** ⚡
  
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainerV2_MedNeXt_B_kernel5
   ```

- **SAMed** ✂️

Download the checkpoint of original SAM into `checkpoint`
```bash
# sam-b checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth
# sam-h checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
 ```

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 3 2d_p256 0 -tr nnUNetTrainerV2_SAMed_b_r_4
   ```

## Inference 
- Validation Inference

To generate model validation outputs, execute the command below. To acquire probabilities for the optimal configuration, append the --npz flag, noting this requires significant disk space.

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID CONFIGURATION FOLD_NUM -tr TRAINER --val --npz
```


- Inference on unseen
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -tr TRAINER_NAME
```
 
- Best Configuration and Ensembles (in progress)



## Citation
If you utilize the baselines in this repository for your research, please consider citing the relevant papers for [Swin-UNETR](https://arxiv.org/abs/2201.01266), [SegResNet](https://arxiv.org/pdf/1810.11654.pdf), [LightM-UNet](https://arxiv.org/abs/2403.05246v1), [U-Mamba](https://arxiv.org/abs/2401.04722), [MedNext](https://github.com/MIC-DKFZ/MedNeXt), [SAMed](https://arxiv.org/abs/2304.13785), and [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z).



## Acknowledgements

We would like to acknowledge the contributions of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and the authors of the baseline models: [LightM-UNet](https://github.com/mrblankness/lightm-unet), [MedNeXT](https://github.com/MIC-DKFZ/MedNeXt), and [SAMed](https://github.com/hitachinsk/SAMed). This repository builds upon their foundational code and work.
