# A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis
### [Project Page](https://xingangpan.github.io/projects/ShadeGAN.html) | [Paper](https://arxiv.org/pdf/2110.15678.pdf)
> **A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis** <br>
> [Xingang Pan](https://xingangpan.github.io/), [Xudong Xu](https://sheldontsui.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/), [Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/), [Bo Dai](http://daibo.info/)<br>
> *NeurIPS2021*

<p align="center">
    <img src="ShadeGAN_demo.gif", width="900">
</p>

In this repository, we present **ShadeGAN**, a generative model for shape-accurate 3D-aware image synthesis.  
Our method adopts a multi-lighting constraint that resolves the shape-color ambiguity and leads to more accurate 3D shapes.

## Requirements

* python>=3.7
* [pytorch](https://pytorch.org/)>=1.8.1
* other dependencies
    ```sh
    pip install -r requirements.txt
    ```

## Lighting priors and pretrained weights

To download lighting priors and pretrained weights, simply run:
```sh
sh scripts/download.sh
```

## Testing

#### Rendering Images

```sh
sh scripts/render.sh
```
This would generate images of multiple viewpoints and lightings by default.

#### Extracting 3D Shapes

```sh
python extract_shapes.py weights/pretrain/celeba_noview/generator.pth --curriculum CelebA_ShadeGAN_noview --seed 0 5 8 43 --ema
```

#### Evaluation Metrics

To evaluate metrics, you need to download dataset first as mentioned in Training below.

To generate real images for evaluation run  
```sh
python fid_evaluation.py --dataset CelebA --dataset_path path/to/dataset/\*.jpg
```

To calculate fid/kid/inception scores run  
```sh
python eval_metrics.py weights/pretrain/celeba_view/generator.pth --real_image_dir EvalImages/CelebA_real_images_128 --curriculum CelebA_ShadeGAN_view --num_steps 6 --delta 0.06423 --ema
```
where `delta` denotes the integration range along the ray for volume rendering. We record the `delta` for different pretrained models at `weights/pretrain/delta.txt`.

## Training

#### Download datasets

CelebA: Download at [CelebA website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
Cats: Please follow the instruction at [GRAF](https://github.com/autonomousvision/graf)  
BFM: Please follow the instruction at [Unsup3d](https://github.com/elliottwu/unsup3d)

#### Start Training

Before training, please update the `dataset_path` field in the curriculum to point to your images.

We provide our training scripts under `scripts` folder. For example, to train ShadeGAN on the CelebA dataset, simply run:
```sh
sh scripts/run_celeba.sh
```
This would run on 4 GPUs by default. You may change the number of GPUs by revising `CUDA_VISIBLE_DEVICES` in the scripts.

## Tips

* If the number of GPUs for training is changed, you may need to adjust the `batch_size` in the curriculum to keep the total batchsize the same.  
* In case of 'out of memory', you could increase `batch_split` in the curriculum to reduce memory consumption.  
* For CelebA and BFM, both models depedent and independent of viewing direction are provided. The former has better FID while the latter has slightly better shapes.
* For BFM dataset, training could sometimes fall to the hollow-face solution where the face is concave. To prevent this, you could initialize with our pretrained models such as `weights/pretrain/bfm_noview/pretrain5k-*`.

## Acknowledgement

This code is developed based on the [official pi-GAN implementation](https://github.com/marcoamonteiro/pi-GAN).

## Citation

If you find our work useful in your research, please cite:
```
@inproceedings{pan2021shadegan,
    title   = {A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis},
    author  = {Pan, Xingang and Xu, Xudong and Loy, Chen Change and Theobalt, Christian and Dai, Bo},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2021}
}
```
