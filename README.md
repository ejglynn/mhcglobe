<div align="left">

# Mitigating Inequity in MHC Binding Predictions for T Cell Epitope Discovery

</div>


<div align="center">
    
[![DOI:00.0000/2022.01.20.000000](http://img.shields.io/badge/DOI-00.0000/0000.00.00.000000-B31B1B.svg)](https://mhcglobe)

</div>

**Motivation:** Computational tools that predict peptide binding by major histocompatibility complex (MHC) proteins play an essential role in current approaches to harness adaptive immunity to fight viral pathogens and cancers. However, there are >22,000 known class-I MHC allelic variants, and it is unknown how well binding preferences are predicted for most alleles. We introduce a machine learning framework that enables state-of-the-art MHC binding prediction along with per-allele estimates of predictive performance. 

**Results:** We demonstrate stark disparities in how much binding data are associated with HLA alleles of individuals across racial and ethnic groups. Pan-MHC modeling mitigates some of these disparities when predicting MHC-peptide binding, and we devise a strategy to begin to address remaining inequities by leveraging our per-allele predictions of performance. The approaches introduced here further the development of equitable MHC binding models, which are necessary to understand adaptive immune response and to design effective personalized immunotherapies in genetically diverse individuals.


# MHCGlobe & MHCPerf Installation

MHCGlobe and MHCPerf are both easily accessible for model inference and re-training.

1) Download the mhcglobe git repository containing the code.

    `$ git clone https://github.com/ejglynn/mhcglobe.git`

2) Update the `mhcglobe_dir` variable in `src/paths.py` with the full path to your `mhcglobe` folder
    
3) Create and activate a Python3 virtual environment with the following commands:

`python3 -m pip install --user --upgrade pip`

`python3 -m pip install --user virtualenv`

`python3 -m venv env`

4) Install prerequisites in the virtual environment:

`pip3 install jupyter pandas scipy sklearn tensorflow tqdm`

5) From the `mhcglobe` folder, start jupyter:

`jupyter notebook`

On your browser, go to 
