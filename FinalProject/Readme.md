# Generative Models for Virtual Try-On Applications

This repository contains the code and models for evaluating advanced generative models for virtual try-on applications. 
Virtual try-on technology allows users to visualize how clothing items would look on their bodies without physically trying them on, 
enhancing the online shopping experience and addressing challenges related to fit and style selection.
We focus on three main models: DCI-VTON, Deep Fashion, and HR-VITON.

## Models

### 1. DCI-VTON

- **Description:** DCI-VTON combines warping and refinement modules powered by diffusion models to create high-fidelity virtual try-on images.
- **Performance:**
  - Fréchet Inception Distance (FID): 0.1473
  - Inception Score (IS): 2.948
- **Code:** [DCI-VTON Repository](link-to-dci-vton-repo)

### 2. Deep Fashion

- **Description:** Deep Fashion employs the Adaptive Content Generating and Preserving Network (ACGPN) to generate photorealistic try-on images with rich details.
- **Performance:**
  - Fréchet Inception Distance (FID): 0.1618
  - Inception Score (IS): 1.016
- **Code:** [Deep Fashion Repository](link-to-deep-fashion-repo)

### 3. HR-VITON

- **Description:** HR-VITON achieves high-resolution virtual try-on with misalignment and occlusion-handled conditions.
- **Performance:**
  - Fréchet Inception Distance (FID): 0.1783
  - Inception Score (IS): 1.432
- **Code:** [HR-VITON Repository](link-to-hr-viton-repo)

## How to Use

1. Clone the respective repositories for each model.
2. Follow the instructions in the README files of each repository to set up the environment and run the code.
3. Use the provided pretrained models to perform virtual try-on tasks or train your models on custom datasets.
