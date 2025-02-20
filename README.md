<div align="left">

# MHCGlobe and MHCPerf

</div>


<div align="center">
    
[![DOI:10.1101/2024.01.30.578103](http://img.shields.io/badge/DOI-10.1073/2024.01.30.578103-B31B1B.svg)](https://doi.org/10.1073/pnas.2405106122)

</div>


**Motivation:** Computational tools that predict peptide binding by major histocompatibility complex (MHC) proteins play an essential role in current approaches to harness adaptive immunity to fight viral pathogens and cancers. However, there are >22,000 known class-I MHC allelic variants, and it is unknown how well binding preferences are predicted for most alleles. We introduce a machine learning framework that enables state-of-the-art MHC binding prediction along with per-allele estimates of predictive performance. 

If you utilize MHCGlobe or MHCPerf in your research please cite:

> E. Glynn,D. Ghersi,& M. Singh,  Toward equitable major histocompatibility complex binding predictions, Proc. Natl. Acad. Sci. U.S.A. 122 (8) e2405106122, https://doi.org/10.1073/pnas.2405106122 (2025).

# MHCGlobe & MHCPerf Installation

MHCGlobe and MHCPerf are both easily accessible for model inference and re-training.

1) Download the mhcglobe git repository containing the code:

    `git clone https://github.com/ejglynn/mhcglobe.git`

2) Update the `mhcglobe_dir` variable in `src/paths.py` with the full path to your `mhcglobe` folder.
3) Download the two pickle files available on [Zenodo](https://zenodo.org/records/14902982?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE0YmI3NDE0LThiNTgtNDRhOS04YWUxLThlN2E2ZWI3ZjdmYyIsImRhdGEiOnt9LCJyYW5kb20iOiI2MzgyZTFhMjMyNjE0YzAyMTA1OGIyNzFhNGE5MzA0OCJ9.1VTHMcnqipmYLAKVvO16GDZRVUnoPtzoQFi4DvG6fqianmCi7Q55wwpqMFWADHEm8Jx1T5d3Xkwaq2B2ZmToog) and place them in the data folder.
    
4) From the `mhcglobe` folder create and activate a Python3 virtual environment with the following commands:

    `python3 -m pip install --user --upgrade pip`

    `python3 -m pip install --user virtualenv`

    `python3 -m venv env`
    
    `source env/bin/activate`

5) Install prerequisites in the virtual environment:

    `pip3 install jupyter pandas scipy sklearn tensorflow tqdm`

6) From the `mhcglobe` folder, start jupyter:

    `jupyter notebook`

On your browser, click on the `MHCGlobe_User_Notebook.ipynb` to open and interact with the notebook.

**To speed things up, output files have already been provided in the `output` folder. If you want to recompute these files, simply delete or rename the `output` folder.**

