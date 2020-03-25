#!/usr/bin/Rscript 

########################################################
# KP 18_1029
# An R script to compare rsem-quantified abundances using DESeq2
# goal is to import files from kallisto quantification, process with DESeq2
########################################################

########################
# user variables
########################

file_path = '/Users/kevin/Documents/Stanford/zou/chang_rna_localization/rnafinder/data/encode/HeLa' # the directory where the files are found
# anno_file = '/home/krparker/user_data/stemrem201b/final/ref/human/Homo_sapiens.GRCh38.merge.gene_anno.txt.gz' # the name of the file containing the annotations
anno_file = '/Users/kevin/Documents/Stanford/zou/chang_rna_localization/rnafinder/data/Homo_sapiens.GRCh38.90.merge_transcriptome.fa.processed.txt'
# samples = c('RNA_WT_CM_rep1','RNA_WT_CM_rep2','RNA_WT_CM_rep3', 'RNA_DKO_CM_rep1','RNA_DKO_CM_rep2','RNA_DKO_CM_rep3') # the names of the samples (should be the names of the folders)
# sample_types = c('WT','WT','WT','DKO','DKO','DKO') # the sample types 
files = Sys.glob(paste(file_path, "*.tsv.gz", sep="/"))
sample_types = c("cyto", "cyto", "nuc", "nuc")
output_file = 'deseq_hela.txt' # the name of the output file

########################
# process files
########################

# import libraries
library('DESeq2')
library('tximport')
library('tximportData')
library('readr')

# import the annotation file
setwd(file_path)
gene_anno = read.table(anno_file, header=TRUE, colClasses="character")
tx2g = gene_anno[c('transcript', 'gene')]

# import the kallisto files and aggregate to gene level
# tx <- tximport(files, type='kallisto', txOut=TRUE, tx2gene=tx2g)
# $Abundance = TPM
# $counts = expected count
tx <- tximport(files, type='rsem', txOut=TRUE, tx2gene=tx2g)
gene <- summarizeToGene(tx, tx2g)

# prepare sample table for DESeq2
sample_table <- data.frame(condition=factor(sample_types))
rownames(sample_table) <- colnames(gene$counts)

# run DESeq2
dds <- DESeqDataSetFromTximport(gene, colData=sample_table, design = ~ condition)
dds <- DESeq(dds)
res <- results(dds)

# save the output
counts_adj <- counts(dds, normalized=TRUE)
colnames(counts_adj) = paste(colnames(counts_adj), "counts_adj", sep="_")
final <- cbind(as.data.frame(res), counts_adj)
write.table(final, output_file, sep="\t")
