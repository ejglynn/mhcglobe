---

<div align="left">

# Mitigating Inequity in MHC Binding Predictions for T Cell Epitope Discovery

</div>

---

<div align="center">
    
[![DOI:00.0000/2022.01.20.000000](http://img.shields.io/badge/DOI-00.0000/0000.00.00.000000-B31B1B.svg)](https://mhcglobe)

</div>

---------

**Motivation:** Computational tools that predict peptide binding by major histocompatibility complex (MHC) proteins play an essential role in current approaches to harness adaptive immunity to fight viral pathogens and cancers. However, there are >22,000 known class-I MHC allelic variants, and it is unknown how well binding preferences are predicted for most alleles. We introduce a machine learning framework that enables state-of-the-art MHC binding prediction along with per-allele estimates of predictive performance. 

**Results:** We demonstrate stark disparities in how much binding data are associated with HLA alleles of individuals across racial and ethnic groups. Pan-MHC modeling mitigates some of these disparities when predicting MHC-peptide binding, and we devise a strategy to begin to address remaining inequities by leveraging our per-allele predictions of performance. The approaches introduced here further the development of equitable MHC binding models, which are necessary to understand adaptive immune response and to design effective personalized immunotherapies in genetically diverse individuals.

# MHCGlobe & MHCPerf Installation
-----------

MHCGlobe and MHCPerf are both easily accessible for model inference and re-training through the Docker image, `ejglynn:mhcglobe:latest`.

1) Install [Docker](https://docs.docker.com/get-docker/)

2) Download the MHCGlobe Docker Image

    `$ sudo docker pull ejglynn:mhcglobe:latest`

3) Create a mhcglobe directory in the users local filesystem, which will be mounted to the mhcglobe docker container so new data and saved models can be added and retrieved from mhcglobe docker container.

    `$ mkdir {local_user_path}/mhcglobe`
    
4) Start the mhcglobe docker instance, with port forwarding to access the jupyter notebooks and models within the docker container.

    `$ sudo docker run -it --rm -v {local_user_path}/mhcglobe:/tf/local/ -p 8888:8888 ejglynn/mhcglobe:latest`
    
5) Following this command, a password will be automatically generated and displayed in the terminal to access the mhcglobe juypter environment in the next step.

6) In the web browser natigate to [http://localhost:8888](http://localhost:8888) to access the MHCGlobe jupyter notebook environment. In the browser, the jupyter notebook will prompt the user for the password mentioned above.

