# RNA-GPS
Interpretable model for predicting high-resolution RNA sub cellular localization to the following localizations:

* ER Membrane
* Nuclear lamina
* Mito matrix
* Cytosol
* Nucleolus
* Nucleus
* Nuclear pore
* Outer mito membrane

This model is trained on APEX-seq data, which measures RNA localization human HEK293T cells.

## SARS-CoV-2 Analysis

Since viruses reproduce by hijacking human cellular machinery, we can also use this model to generate hypotheses surrounding localization of SARS-CoV-2 RNA transcripts. See analyses in the `covid19` directory for additional information, as well as relevent works below.

## Environment setup

After creating the `rnagps` environment using

```
conda env create -f environment.yml
```

Install `xgboost` with
```
conda install -c conda-forge xgboost=0.82
```

## Relevant works

* Wu, K.E., Parker, K.R., Fazal, F.M., Chang, H., and Zou, J. (2020). RNA-GPS predicts high-resolution RNA subcellular localization and highlights the role of splicing. RNA.
* Wu, K.E., Fazal, F.M., Parker, K.R., Zou, J., and Chang, H.Y. (2020). RNA-GPS Predicts SARS-CoV-2 RNA Residency to Host Mitochondria and Nucleolus. Cell Systems.
