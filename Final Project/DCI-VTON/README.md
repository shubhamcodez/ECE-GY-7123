# DCI-VTON-Virtual-Try-On
Running all the scripts in my WSL2 Ubuntu machine, which has all the dependencies including CUDA set up. Therefore I did all my testing and calculations in
python files.

Here are the FID and IS score:
![FID Score](./fid.png "FID Score")
![IS Score](./is-score.png "IS Score")


## Getting Started
### Installation
#### Diffusion Model
```
1. Install Python dependencies
```shell
conda env create -f environment.yaml
conda activate dci-vton
```
2. Download the pretrained [vgg](https://drive.google.com/file/d/1rvow8jStPt8t2prDcSRlnf8yzXhrYeGo/view?usp=sharing) checkpoint and put it in `models/vgg/`
#### Warping Module
3. Clone the PF-AFN repository
```shell
git clone https://github.com/geyuying/PF-AFN.git
```
4. Move the code to the corresponding directory
```shell
cp -r DCI-VTON-Virtual-Try-On/warp/train/* PF-AFN/PF-AFN_train/
cp -r DCI-VTON-Virtual-Try-On/warp/test/* PF-AFN/PF-AFN_test/
```
### Data Preparation
#### VITON-HD
1. Download [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset
2. Download pre-warped cloth image/mask from [Google Drive](https://drive.google.com/drive/folders/15cBiA0AoSCLSkg3ueNFWSw4IU3TdfXbO?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1ss8e_Fp3ZHd6Cn2JjIy-YQ?pwd=x2k9) and put it under your VITON-HD dataset

After these, the folder structure should look like this (the unpaired-cloth* only included in test directory):
```
├── VITON-HD
|   ├── test_pairs.txt
|   ├── train_pairs.txt
│   ├── [train | test]
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-warp
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-warp-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── unpaired-cloth-warp
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── unpaired-cloth-warp-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
```
### Inference
#### VITON-HD
Please download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/11BJo59iXVu2_NknKMbN0jKtFV06HTn5K?usp=sharing)
###### Warping Module
To test the warping module, first move the `warp_viton.pth` to `checkpoints` directory:
```shell
mv warp_viton.pth PF-AFN/PF-AFN_test/checkpoints
```
Then run the following command:
```shell
cd PF-AFN/PF-AFN_test
sh test_VITON.sh
```
After inference, you can put the results in the VITON-HD for inference and training of the diffusion model. 
###### Diffusion Model
To quickly test our diffusion model, run the following command:
```shell
python test.py --plms --gpu_id 0 \
--ddim_steps 100 \
--outdir results/viton \
--config configs/viton512.yaml \
--ckpt /CHECKPOINT_PATH/viton512.ckpt \
--dataroot /DATASET_PATH/ \
--n_samples 8 \
--seed 23 \
--scale 1 \
--H 512 \
--W 512 \
--unpaired
```
or just simply run:
```shell
sh inference.sh
```
### Training
#### Warping Module
To train the warping module, just run following commands:
```shell
cd PF-AFN/PF-AFN_train/
sh train_VITON.sh
```
#### Diffusion Model
We utilize the pretrained Paint-by-Example as initialization, please download the pretrained models from [Google Drive](https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view) and save the model to directory `checkpoints`. 

To train a new model on VITON-HD, you should first modify the dataroot of VITON-HD dataset in `configs/viton512.yaml` and then use `main.py` for training. For example,
```shell
python -u main.py \
--logdir models/dci-vton \
--pretrained_model checkpoints/model.ckpt \
--base configs/viton512.yaml \
--scale_lr False
```
or simply run:
```shell
sh train.sh
```