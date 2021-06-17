# Multi-Omics Predictive Modelling of Cervicovaginal Microenvironment

This repository accompanies the publication: "Integration of multi-omics data improves prediction of cervicovaginal microenvironment in cervical cancer". It includes all published analyses, namely the data analysis and modelling in notebook `a-modelling-HPV.ipynb` and the analysis of the interaction of microbiome and metabolome in the notebook `b-mmvec-HPV.ipynb`.

To reproduce the published results please follow the below setup instructions (they are unique for each of the two notebooks).


## Setup for `a-modelling-HPV.ipynb`

* To run this notebook you should setup a conda environment
with the qiime2-2021.4 distribution installed within:

```shell
wget https://data.qiime2.org/distro/core/qiime2-2021.4-py38-osx-conda.yml
conda env create -n hpv_modelling --file qiime2-2021.4-py38-osx-conda.yml
rm qiime2-2021.4-py38-osx-conda.yml
conda activate hpv_modelling
```

* Within the conda environment install all required packages with (usage of `python -m pip` to ensure that the package is pip installed in the conda environment and not elsewhere):
```shell
python -m pip install git+https://github.com/bokulich-lab/RESCRIPt.git
conda install -c conda-forge -c r --file requirements-modelling.txt
```

* Have fun recreating our published results :).

## Setup for `b-mmvec-HPV.ipynb`
* To run this notebook you should setup a conda environment
with an older version of the QIIME2 distribution installed within, namely 2020.6. 
This is required as the used plugin `mmvec` is currently only supported until this version.
```shell
wget https://data.qiime2.org/distro/core/qiime2-2020.6-py36-osx-conda.yml
conda env create -n hpv_mmvec --file qiime2-2020.6-py36-osx-conda.yml
rm qiime2-2020.6-py36-osx-conda.yml
conda activate hpv_mmvec
```

* Within the activated conda environment install the required dependency mmvec as:
```shell
python -m pip install install git+https://github.com/biocore/mmvec.git
qiime dev refresh-cache

```

* (Optional:) If you want to have a nice jupyter notebook experience feel free to also install the following dependencies:
```shell
conda install -c conda-forge --file requirements-mmvec.txt
```

* Happy reproduction of the results! :) 

## Contact

In case of questions or comments feel free to raise an issue in this repository. 
