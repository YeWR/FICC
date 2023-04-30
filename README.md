# FICC

Open-source codebase for **forward-inverse cycle consistency** (FICC),
from "Become a Proficient Player with Limited Data through Watching Pure Videos" at ICLR 2023.

# Pretrain Data

This response use [Atari replay dataset](https://research.google/resources/datasets/dqn-replay/) as pretrain data.

# Usage

Pretrain: `bash pretrain.sh breakout`

| Arguments             | Description                                        |
|-----------------------|----------------------------------------------------|
| `--env`               | Name of the pretrain environment                   |
| `--dataset_path`      | Path of the dataset.                               |
| `--device`            | Select working GPU.                                |
| `--batch_size`        | Mini-batch size.                                   |
| `--lr`                | Initiate learning rate for pretrain.               |
| `--latent_action_dim` | Dimension for latent action                        |
| `--num_embeddings`    | Num of embeddings of latent action generator (LAG) |
| `--l1_penalty`        | L1 penalty for output of dynamics function         |
| `--weight_decay`      | L2 penalty for model parameters                    |


# Bibtex

```
@inproceedings{ye2023become,
title={Become a Proficient Player with Limited Data through Watching Pure Videos},
author={Weirui Ye, and Yunsheng Zhang, and Pieter Abbeel, and Yang Gao},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023}
}
```