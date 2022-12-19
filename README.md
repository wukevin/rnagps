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

## Installation

Download the codebase via `git clone` and use the following command to create the `rnagps` conda environment.

```
conda env create -f environment.yml
```

Activate the environment with `conda activate rnagps`.

## Usage

After installing the conda environment, you can make predictions by using the script under `bin/predict_localization.py` as follows:

```bash
python bin/predict_localization.py <5'UTR sequence> <CDS equence> <3'UTR sequence>
```

## Relevant works

* Wu, K.E., Parker, K.R., Fazal, F.M., Chang, H., and Zou, J. (2020). RNA-GPS predicts high-resolution RNA subcellular localization and highlights the role of splicing. RNA. [link](https://rnajournal.cshlp.org/content/early/2020/03/27/rna.074161.119.abstract)
* Wu, K.E., Fazal, F.M., Parker, K.R., Zou, J., and Chang, H.Y. (2020). RNA-GPS Predicts SARS-CoV-2 RNA Residency to Host Mitochondria and Nucleolus. Cell Systems. [link](https://www.cell.com/cell-systems/pdf/S2405-4712(20)30237-4.pdf)
