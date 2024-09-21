# PocketLG: Protein Binding Pocket Identification Using Protein Large Language Model and Graph Transformer
This repository contains source code, training models, and data preprocessing code for PocketLG
## Introduction
A protein pocket is a special region on the surface of a protein that is commonly used to interact with other molecules (especially small molecule compounds) for various biological functions. Accurately identifying and understanding the structure of protein pockets can accelerate the development of new drugs and improve existing drugs to reveal disease mechanisms. In this study, we developed a protein data preprocessing tool for sampling the binding domains on the surface of proteins and augmented the sample data to output the data needed for a protein pocket model. And we also propose a deep learning model, PocketLG, to discover which of these samples are binding pockets. Experimental results show that our deep learning model can identify binding pockets on proteins with 91% success rate. This work provides a new technical route for protein binding pocket prediction research, which can greatly contribute to the development of the pharmaceutical industry.

<img src=".\figs\fig1.png" width="100%"/>

## Dataset

**Train and Test data:** Train and Test datasets can be downloaded according to the following links `PDBbind` (http://www.pdbbind.org.cn/download.php).

## Data preprocessing

We designed an end-to-end data preprocessor which implements the two main functions of data sampling and data enhancement, and we reduced the time complexity and space complexity of the protein sampling process. The feature samples required for model learning can be obtained directly and efficiently by simply inputting the protein PDB file and the corresponding pocket PDB file.
<img src=".\figs\fig2.png" width="100%"/>

For COACH420, HOLO4K and SC6K, the preprocessing procedure is the same as in [DeepPocket](https://github.com/devalab/DeepPocket). For PDBbind, the refined set of version 2020 is used in our experiments, in which proteins with more than 50% sequence identity to those in ScPDB are removed to avoid data leakage.
