# scLinguist: a Foundation Model for Cross-Modality Translation in Single-Cell Omics
## Overview
In this work, we introduce scLinguist, a novel cross-modal foundation model based on an encoder–decoder architecture, designed to predict protein abundance from single-cell transcriptomic profiles. Drawing inspiration from multilingual translation models, scLinguist adopts a two-stage learning paradigm: it first performs modality-specific pretraining on large-scale unpaired omics data (e.g., RNA or protein) to capture intra-modality expression patterns, and then conducts post-pretraining on paired RNA–protein data to learn cross-modality mappings. This strategy enables the model to integrate knowledge from both data-rich and data-scarce scenarios, enhancing its generalizability and robustness across diverse biological contexts.


## Installation
We tested our code on a server running Ubuntu 18.04.5 LTS, equipped with NVIDIA H100 GPUs.

git clone https://github.com/xxxxx.git
cd scLinguist
conda create -n scLinguist python=3.8.8
conda activate scLinguist
pip install -r requirements.txt

# install torch
pip install torch==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

python setup.py install


## Tutorial
<!-- * [`finetune`](./integration_examples/horizontal) 
* [`zeroshot`](./integration_examples/vertical) 
* [`fewshot`](./integration_examples/mosaic) 
* [`imputation `](./imputation_examples/)  -->

We provided detailed tutorials on applying scLinguist to various tasks. Please refer to [https://scLinguist.readthedocs.io/en/latest/](https://scLinguist.readthedocs.io/en/latest/).

## Pre-trained model 
The pre-trained models can be downloaded from these links.
| Model name                | Description                                             | Download                                                                                     |
| :------------------------ | :------------------------------------------------------ | :------------------------------------------------------------------------------------------- |
| scLinguist pretrained RNA | Pretrained on over 15 million human cells. | [link](https://drive.google.com/file/d/) |
| scLinguist pretrained Protein | Pretrained on over 11 million human cells. | [link](https://drive.google.com/file/d/) |
| scLinguist post-pretrained RNA-Protein | Post-Pretrained on 3 million paired cells. | [link](https://drive.google.com/file/d/) |



## Data
Source of public datasets:
1. BM dataset: [`CITE-seq`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639) 
2. BMMC dataset: {[`CITE-seq`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=194122)}
3. CBMC dataset: {[`CITE-seq`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10086)}
4. PBMC dataset: {[`REAP-seq`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100501)}
5. Perturb dataset: [`ECCITE-seq`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15306)
6. Heart dataset: {[`CITE-seq`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218392)}
7. Spatial dataset: [`10X Visium`](https://www.10xgenomics.com/datasets/gene-protein-expression-library-of-human-tonsil-cytassist-ffpe-2-standard)

We have compiled the public datasets into h5ad files. Please refer to [google driver](https://). 

## Reproduce results presented in manuscript
To reproduce scLinguist's results, please visit [`reproduce`](./reproduce/) folder.

To reproduce compared methods' results, including [`scTranslator`](https://github.com/TencentAILabHealthcare/scTranslator), [`scButterfly`](https://github.com/BioX-NKU/scButterfly), [`TotalVI`](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/totalVI.html), [`scArches`](https://docs.scarches.org/en/latest/), please visit [`https://github.com/xxxx/scLinguist-notebooks`](https://github.com/xxxx/scLinguist-notebooks).