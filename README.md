# Towards Better De-raining Generalization via Rainy Characteristics Memorization and Replay

We have released the code for our paper "Towards Better De-raining Generalization via Rainy Characteristics Memorization and Replay". Our code uses MPRNet as the exemplified de-raining network for illustrating our method.

<div align=center>
<img width="100%" src="imgs/method.png"/>
</div>

<div align=center>
<img width="100%" src="imgs/result.png"/>
</div>

### Abstract

Current image de-raining methods mainly focus on learning from a fixed set of de-raining data. However, when facing complex and diverse real-world rainy scenarios, they often yield sub-optimal results as the learned specific de-raining mapping rule can only partially cover real-world rain distribution. To address this issue, we propose a novel generalized image de-raining framework that empowers networks to accumulate de-raining knowledge from increasingly abundant de-raining datasets, instead of solely relying on a static dataset, thereby constantly enhancing their generalization ability. Our inspiration originates from the human brain's complementary learning system, which enables humans to constantly memorize a stream of perceived events and acquire generalization across memorized event. Building upon this concept, we attempt to borrow the mechanism of the complementary learning system into our framework. Specifically, we first utilize GANs to learn and store the rainy characteristics of newly arriving data, imitating the learning and memorizing function of hippocampus. Then, we train the de-raining network using both the current data and GANs-generated data to imitates the hippocampus-to-neocortex replay and the interleaved learning. In addition, we adopt the knowledge distillation with replayed data to imitate the consistency between neocortical activity patterns activated by hippocampal replayed events and existing neocortical knowledge. By equipping our framework, the de-raining network is able to accumulate de-raining knowledge from a stream of datasets, thus constantly improving the generalization on unseen real-world rainy images.
Experiments on three representative de-raining networks demonstrate that our framework enables the networks to effectively accumulate knowledge on six datasets and achieve superior generalization results on unseen real-world datasets compared to the SOTA methods. 

### Datasets

[[Download link] https://pan.baidu.com/s/1oqfCJr3T_l3wFM9wpmHW7g (pwd: shdl) ]

## Install

Please refer to the requirements.txt file in the directory, where we have listed all the dependencies required for setting up the environment.

## Model Weight

[[Download link] https://pan.baidu.com/s/171LJz9gHNhrAZCzd3ZoDvg (pwd: 5jno) ]

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