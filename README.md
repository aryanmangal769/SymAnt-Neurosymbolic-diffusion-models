# SymAnt: Neuro-Symbolic Diffusion Models for Action anticipation

Effective action anticipation is critical for intelligent agents in applications like human-machine collaboration and autonomous systems. This work introduces a novel approach utilizing symbolic knowledge in the form of knowledge and scene graphs as context for action anticipation in videos. We propose a joint graph-search approach that reasons over the spatial relationships between scene objects and amends it with relevant information from the knowledge graph, including attributes and affordances. Finally, we introduce a novel approach to discrete diffusion using symbolic knowledge as initialization for the diffusion process, iteratively updating and refining the predicted sequence of future actions. We demonstrate the effectiveness of our method on a set of common yet diverse datasets, including Breakfast, 50 Salads, EPIC Kitchens, and EGTEA Gaze+, across both short and long-horizon prediction tasks. Through our experiments, we demonstrate the effectiveness of our neuro-symbolic approach, outperforming current state-of-the-art methods, and analyze its effectiveness in extensive ablation studies.

<img width="439" alt="image" src="https://github.com/user-attachments/assets/f1b70fe8-5e37-478b-a050-3ce688bc7a67">


## Index

1. [Environment Setup](#setup)
2. [Dataset](#dataset)
4. [Testing](#testing)
3. [Training](#training)

## Setup

In order to build a ```conda``` environment for running our model, run the following command:
```
conda env create -f environment.yml
```

Activate environment using:
```
conda activate symant
```

## Dataset

We train our action anticipation pipeline on two publicly available datasets namely, <i>50Salads</i> and <i>Breakfast</i> dataset. You can download the features from [this link](https://mega.nz/file/O6wXlSTS#wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8). 
<br>

## Testing

You can find the pre-trained weights for our model trained on the 50Salads and breakfast dataset [here](https://drive.google.com/drive/folders/1kG0rV2P-bgI6kNHHK-cxhNliGDt4htHY?usp=sharing). <br>
To test your trained model place it in a directory called ```ckpt/``` inside the home directory and run the following command. 

```
./script_mamba_diff/50s_predict.sh
```

## Training 

To train our model on the 50Salads dataset, run the following command:
```
./script_mamba_diff/50s_train.sh
```
