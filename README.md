# Personal Narratives Empower Politically Disinclined Individuals to Engage in Political Discussions

[![Conference](https://img.shields.io/badge/WebSci'25-Best_Paper_Honorable_Mention-blue.svg)](https://doi.org/10.1145/3717867.3717899)
[![arXiv](https://img.shields.io/badge/arXiv-2502.20463-b31b1b.svg)](https://arxiv.org/abs/2502.20463)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/tejasvichebrolu/personal-narrative-classifier) 
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

This repository contains the official code, data, and models for the paper **"Personal Narratives Empower Politically Disinclined Individuals to Engage in Political Discussions"**, which received a Best Paper Honorable Mention at the 17th ACM Web Science Conference (WebSci'25).

## Abstract

> Engaging in political discussions is crucial in democratic societies, yet many individuals remain politically disinclined due to various factors such as perceived knowledge gaps, conflict avoidance, or a sense of disconnection from the political system. In this paper, we explore the potential of personal narratives—short, first-person accounts emphasizing personal experiences—as a means to empower these individuals to participate in online political discussions. Using a text classifier that identifies personal narratives, we conducted a large-scale computational analysis to evaluate the relationship between the use of personal narratives and participation in political discussions on Reddit. We find that politically disinclined individuals (PDIs) are more likely to use personal narratives than more politically active users. Personal narratives are more likely to attract and retain politically disinclined individuals in political discussions than other comments. Importantly, personal narratives posted by politically disinclined individuals are received more positively than their other comments in political communities. These results emphasize the value of personal narratives in promoting inclusive political discourse.

## Citation
If you use the code, data, or models from this repository in your research, please cite our work:
```bibtex
@inproceedings{chebrolu2025narratives,
  title={{Personal Narratives Empower Politically Disinclined Individuals to Engage in Political Discussions}},
  author={{Chebrolu, Tejasvi and Kumaraguru, Ponnurangam and Rajadesingan, Ashwin}},
  booktitle={{Proceedings of the 17th ACM Web Science Conference 2025 (Websci '25)}},
  year={{2025}},
  organization={{ACM}},
  doi={10.1145/3717867.3717899}
}
```

## Repository Structure

This repository is organized to facilitate both the use of our pre-trained model and the full replication of our study's findings.

```
.
├── data/                  # Datasets used for training and analysis
│   ├── complete_training_data.csv    # Full labeled dataset for training the classifier
│   ├── hypothesis-datasets/          # Data subsets for replicating statistical analyses
│   │   ├── final_h1.csv              # Data for Hypothesis 1, 4, 5, & RQ1
│   │   ├── h2.csv                    # Data for Hypothesis 2
│   │   └── h3.csv                    # Data for Hypothesis 3 
│   └── validation_100_samples.csv    # Manually verified samples for external validation
├── src/                     # All source code
│   ├── analysis/            # R scripts for statistical analysis
│   │   ├── h1.R
│   │   ├── h2.R
│   │   ├── h3.R
│   │   └── RQ.R
│   ├── plots/               # Scripts to generate plots from the paper
│   │   ├── h1.ipynb
│   │   ├── h2_plot.R
│   │   ├── h3_plot.R
│   │   └── RQ_plot.R
│   ├── train_narrative_classifier.py  # Script to train the classifier from scratch
│   ├── predict_narrative.py         # Script to use the trained model for inference
│   ├── convert_pth_to_hf.py         # Utility to convert .pth to Hugging Face format
│   └── upload_to_hf.py            # Script to upload the model to the Hub
└
```

## Model on the Hugging Face Hub

The final, ready-to-use model is hosted on the Hugging Face Hub for easy access.

*   **Model Page:** [**tejasvi/personal-narrative-classifier**](https://huggingface.co/tejasvichebrolu/personal-narrative-classifier)
