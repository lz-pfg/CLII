# CLII
CLII: Visual-Text Inpainting via Cross-Modal Predictive Interaction

![qua1](./figures/qua1.pdf)
![qua2](./figures/qua2.pdf)

# Get Started

# Create Virtual Environment with Conda
conda create --name ppt python=3.9
conda activate cl

# Install Dependencies
pip install -r requirements.txt


## Datasets

- Training datasets

    Please refer to .
        

- Evaluation datasets, LMDB datasets (developed version) can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1dTI0ipu14Q1uuK4s4z32DqbqF3dJPdkk/view?usp=sharing).



## Pretrained Models

Get the pretrained models from [BaiduNetdisk(passwd:kwck)](https://pan.baidu.com/s/1b3vyvPwvh_75FkPlp87czQ), [GoogleDrive](https://drive.google.com/file/d/1mYM_26qHUom_5NU7iutHneB_KHlLjL5y/view?usp=sharing). 


## Training

```
python main.py --config=configs/train.yaml
```


## Evaluation

```
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/train.yaml --phase test --image_only
```
Additional flags:
- `--checkpoint /path/to/checkpoint` set the path of evaluation model 
- `--test_root /path/to/dataset` set the path of evaluation dataset
- `--model_eval [alignment|vision]` which sub-model to evaluate
- `--image_only` disable dumping visualization of attention masks




## Citation
If you find our method useful for your reserach, please cite
```bash 
@inproceedings{
}
 ```


## Acknowledgements

The code is based on the https://arxiv.org/pdf/2103.06495.pdf, we sincerely thank   for the awesome repo.
