# **🌸 Flower Image Generation with GANs**

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow" height="24">
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras" height="24">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" height="24">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" height="24">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" height="24">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" height="24">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white" alt="Matplotlib" height="24">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter" height="24">
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Google Colab" height="24">
</div>

Generating and colorizing flower images with two generative adversarial network architectures — DCGAN and Pix2Pix — built from scratch in TensorFlow/Keras.

## **Overview**

This project explores generative modeling on images of flowers using the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) dataset and a [flower colorization](https://www.kaggle.com/datasets/vaibhavrmankar/colour-the-flower-gan-data) dataset. Two GAN variants were implemented and trained:

* **DCGAN** — generates novel synthetic flower images from random noise
* **Pix2Pix (U-Net + PatchGAN)** — performs conditional image-to-image translation, colorizing grayscale flower sketches into realistic color images

Beyond model training, the project includes **a full data pipeline**: exploratory data analysis, duplicate image detection and removal, dimension/distribution analysis, and augmentation (random cropping, jittering, flipping).

## **Architectures**
### **DCGAN**

| Component | Details |
| :--- | :--- |
| Generator | 	Dense → reshape → 4 × `Conv2DTranspose` upsampling blocks *(BatchNorm + ReLU)*, `tanh` output |
| Discriminator | 5 × `Conv2D` downsampling blocks *(LeakyReLU + Dropout)* → global average pooling → sigmoid |
| Loss | 	Binary cross-entropy (standard adversarial loss) |

### **Pix2Pix**

| Component | Details |
| :--- | :--- |
| Generator | *U-Net* with 8 downsampling / 7 upsampling blocks and skip connections |
| Discriminator | *PatchGAN* — classifies overlapping image patches as real/fake |
| Loss | 	Adversarial loss + L1 loss (λ = 100) for pixel-level fidelity |
| Reference | [U-Net paper](https://arxiv.org/pdf/1505.04597) |

## **Data Pipeline**
1. **EDA** — image count, size, and channel-depth statistics via basic-image-eda
1. **Duplicate detection** — MD5 hashing to identify and remove identical images
1. **Distribution analysis** — visualized height/width/depth distributions across the dataset
1. **Preprocessing** — resizing to 256×256, normalization to [-1, 1]
1. **Augmentation** — random jitter (resize + crop) and random horizontal flips during training

## **Project Structure**
```
├── Lab3_Rudas.ipynb        # Main notebook: EDA, DCGAN, and Pix2Pix implementation
├── plots/                  # Generated sample images and loss curves per checkpoint
```

## **Training**
Both models were trained with the Adam optimizer (`lr=2e-4`, `β1=0.5`) using custom tf.function-decorated training steps, with periodic checkpointing and visual sampling every 1,000 steps to monitor generator/discriminator loss convergence.
```python
# DCGAN
errors = fit_dcgan(train_targets, steps=10000)

# Pix2Pix
fit(train_dataset, test_dataset, steps=40000)
```

## **Results**

### **DCGAN**
From this:
![dcgan](/plots/generated_image_1000.png)
To this:
![dcgan](/plots/generated_image_19000.png)

### **Pix2Pix**
![pix2pix_1](/plots/pix2pix_result_1.png)

From this:
![pix2pix_2](/plots/pix2pix_result_2_from.png)
To this:
![pix2pix_2](/plots/pix2pix_result_2_to.png)


Sample outputs and generator/discriminator loss curves are saved to `./plots/` during training.
![dcgan_losses](/plots/losses_19000.png)

## **Setup**
```bash
pip install git+https://github.com/Soongja/basic-image-eda
```

The notebook expects the dataset to be mounted from Google Drive `(/content/drive/MyDrive/data/flowers_coloring.zip)` when run in Google Colab.

## **Author**
Vladyslava Rudas