# Towards Better De-raining Generalization via Rainy Characteristics Memorization and Replay

We have released the code for our paper "Towards Better De-raining Generalization via Rainy Characteristics Memorization and Replay", which is submitted to TNNLS. Our code uses MPRNet as the exemplified de-raining network for illustrating our method.

<br>

<div align=center>
<img width="100%" src="imgs/method.png"/>
</div>

<br>

<div align=center>
<img width="100%" src="imgs/result.png"/>
</div>

### Datasets

[[Download link] https://pan.baidu.com/s/1RoBWfAAfR9HIOuIvmnOWAw (pwd: la47) ]

## Install

Please refer to the requirements.txt file in the directory, where we have listed all the dependencies required for setting up the environment.

## Model Weight

[[Download link for trained weight] https://pan.baidu.com/s/1YcOoZ-EkeCTKXYvXEOcBmA (pwd: m4fh) ]

[[Download link for weight of VRGNet] https://pan.baidu.com/s/1dg04evriT8-ourKciAr4Sw (pwd: 6zkm) ]

## Structure

The folder structure should be organized as follows:

```
├── pbs
├── pytorch-gradual-warmup-lr
├── utils
├── Derain
│   ├── syn
│   │   ├── rain100H
│   │   ├── rain100L
│   │   ├── rain1400
│   │   ├── rain1200_light
│   │   ├── rain1200_medium
│   │   ├── rain1200_heavy
│   │   │   ├── train
│   │   │   │   ├── rain
│   │   │   │   ├── norain
│   │   │   ├── test
│   │   │   │   ├── rain
│   │   │   │   ├── norain
│   ├── real
│   │   ├── SPA
│   │   │   ├── rain
│   │   │   ├── norain
├── VRGNet
│   ├── rain100H
│   ├── rain100L
│   ├── rain1400
│   ├── rain1200_light
│   ├── rain1200_medium
│   ├── rain1200_heavy
├── output
...
```

## Preparation

```bash
cd CLGID
pip install natsort argparse 
cd pytorch-gradual-warmup-lr 
python setup.py install 
cd .. 
```

## Training

```bash
python train.py --yaml ./pbs/100H-100L-1400-1200m.yml
```

## Testing

```bash
python test_image.py --checkpoint your_model_pth_path --data_path ./Derain/real/SPA
```
