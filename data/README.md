# Code and data for "Learning a deep language model for microbiomes: the power of large scale unlabeled microbiome data"

## Data:

- vocab_embeddings.npy
  - Fixed vocabulary embeddings produced from prior work: [Decoding the language of microbiomes using word-embedding techniques, and applications in inflammatory bowel disease](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007859). Adapted from [here](http://files.cqls.oregonstate.edu/David_Lab/microbiome_embeddings/data/embed/).
- microbiomedata.zip
  - Contains the labels and data for the three datasets used in this study. Specifically, it includes:
    - IBD\_(test|train)_(512|otu).npy and IBD_(test|train)\_labels.npy
    - halfvarson\_(512_otu|otu).npy and halfvarson_IBD_labels.npy
    - schirmer*IBD*(512_otu|otu).npy and schirmer_IBD_labels.npy
    - (test|train)encodings\_(512|1897).npy
  - The data are stored as n_samples x max_sample_size x 2 numpy arrays, containing both the vocab IDs of the taxa in the samples, as well as the abundance values for each taxa. data[:,:,0] will give the vocab IDs, and data[:,:,1] will give the abundances.
  - Files which mention '512' are truncated to only have up to 512 taxa in them (max_sample_size = 512).
  - Note that we refer to the schirmer dataset as HMP2 in the paper.
  - (test|train)encodings\_(512|1897).npy represents the full collection of [American Gut Project](https://doi.org/10.1128%2FmSystems.00031-18) data, regardless of whether that data has IBD labels or not, split into train / test splits.
  - Also contains the folders fruitdata and vegdata containing fruit and vegetable data respectively, and the file README, which documents the contents of the first two folders.
  - American Gut Project, Halfvarson, and Schirmer raw data are available from the NCBI database (accession numbers PRJEB11419, PRJEB18471, and PRJNA398089, respectively). We used the curated data produced by [Tataru and David, 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007859).
- pretrainedmodels.zip
  - Contains a sequence of pretrained discriminator models across different epochs, allowing users to compute embeddings without having to pretrain models themselves. Each model is stored as a pair of a pytorch_model.bin file containing weights and a config.json file containing model config parameters. Each pair is located in its own folder whose name corresponds to epoch. E.g., "5head5layer_epoch60_disc" stores the discriminator model that were trained for 60 epochs. Model checkpoints can be loaded by providing a path to the pytorch_model.bin file in the --load_disc argument of begin.py in microbiome_transformers-master/finetune_discriminator.
- ensemble.zip
  - Contains the result of an ensemble finetuning run, allowing users to perform interpretability / attribution experiments without having to train models themselves. Each model is similarly stored as a pytorch_model.bin file and config.json file in its own folder. E.g., the run3_epoch0_disc folder stores the model from the third finetuning run (with epoch0 reflecting that the finetuning only takes one epoch).
- seqs\_.07_embed.fasta
  - Contains the 16S sequences associated with each taxon vocabulary element of our study, originally produced by prior work: [Decoding the language of microbiomes using word-embedding techniques, and applications in inflammatory bowel disease](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007859). Also available [here](http://files.cqls.oregonstate.edu/David_Lab/microbiome_embeddings/data/embed/seqs_.07_embed.fasta).

## Code/Software:

Note that the Dryad repository stores the code and software discussed here is available at [this](https://doi.org/10.5281/zenodo.13858903) site, which is linked under the "Software" tab on the current page.\
The following software include hardcoded absolute paths to various files of interest (described above). These paths have been changed to be of the form "/path/to/file_of_interest", where the "path/to" portion must be changed to reflect the actual paths on whichever system you run these on.

- Attribution_calculations.ipynb
  - Used to calculate per-sample model prediction scores, per-taxa attribution values (used for interpretability), as well as per-taxa averaged embeddings (used for plotting the taxa). Note the current file is set to compute attributions only for IBD, but can easily be changed for Schirmer/HMP2 and Halfvarson.
- Process_Attributions_No_GPU.ipynb
  - Takes the per-sample prediction scores and the per-taxa attribution values (both from Attribution_calculations.ipynb) and identifies the taxa most and least associated with IBD.
- assign_16S_to_phyla.R
  - An R script that makes phylogenetic assignments to the 16S sequences from seqs\_.07_embed.fasta. Invoke with 'Rscript assign_16S_to_phyla.R' and no arguments.
- run_blast_with_downloads.sh
  - Compares the overlap in ASVs between Halfvarson and AGP versus between HMP2 and AGP. Must have BLAST installed. BLAST parameters are set in file, via the results filtering lines ("awk '$5 < 1e-20 && $8 >= 99' | \\"), that set the e-value to 20^-20 and the percent similarity to 99%, with one line for each of the two pairwise comparisons. Simply run via "bash run_blast_with_downloads.sh".
- Plot_microbiome_transformers_results.ipynb
  - Loads the averaged taxa embeddings (from Attribution_calculations.ipynb) and the vocabulary embeddings (from [Decoding the language of microbiomes using word-embedding techniques, and applications in inflammatory bowel disease](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007859) / vocab_embeddings.npy), as well as the taxonomic assignments (from assign_16S_to_phyla.R), and generates the various TSNE-based plots of the embedding space geometry. It also generates plots to compare the clustering quality of the averaged embeddings and the vocabulary embeddings.
- DeepMicro.zip
  - A modified version of [DeepMicro](https://github.com/minoh0201/DeepMicro), adapted to more easily run the DeepMicro-based baselines included in our paper. Most additional functionality is described in the 'help' strings of the additional arguments and the docstrings of the functions. In particular, since our data include unlabeled samples witch nonetheless contribute to learning an embedding space, we needed to add a "--pretraining_data" argument to allow such data to be included in the self-supervised learning portion of the baselines.
  - "convert_data.py" under the "data" folder serves as a utility to help convert from the coordinate-list format of this study to the one-hot abundance table format expected by DeepMicro.
  - "get_unlabeled_pretraining_data.py" under the "data" folder processes labeled microbiome datasets (fruit, vegetable, and IBD) and extends them with unlabeled data from the American Gut Project (AGP).
  - host*to_ids.py under the data/host_to_indices folder will combine metadata from err-to-qid.txt and AG_mapping.txt (both available at *[https://files.cqls.oregonstate.edu/David_Lab/microbiome_embeddings/data/AG_new](https://files.cqls.oregonstate.edu/David_Lab/microbiome_embeddings/data/AG_new)*) with the sequences in seqs*.07_embed.fasta and the numpy data files to create dictionaries that map from host ids to indices in the numpy files, then store those as pickle files. This allow for future training runs from the transformer or the baselines to block their train / validation / test splits by host id.
  - exps_ae.sh, exps_cae.sh, and exps_baselines.sh are shell scripts with the python commands that run the various DeepMicro-based baselines.
  - "display_results.py" is a helper for accumulating experimental results and displaying them in a table.
- property_pathway_correlations.zip
  - A folder containing the required code and files to run the property and pathway correlation experiments.
  - property_pathway_correlations contains three subfolders:
    - figures: stores output figures such as the heatmap of property - pathway correlation strengths.
      - csvs: contains gen_hists.py, which takes the outputs of significant correlation counts / strength from metabolic_pathway_correlations.R and plots a histogram to compare the property correlations of the initial vocabulary embeddings with those of the learned embeddings. Also contains significant_correlations_tests.py, which applies non-parametric and permutation tests to statistically determine whether the learned embeddings tend to have stronger property correlations. Also reports the effect size via Cliff's Delta and Cohen's d statistics.
      - new_hists: will store the histogram generated from gen_hists.py
    - pathways: stores text and csv outputs, such as the correlation strengths between each property and pathway pair (property*pathway_dict_allsig.txt), the top 20 pathways associated with each property (top20Paths_per_property*(ids|names)\_v2.csv), and list of which pathway is most correlated with each property (property_pathway_dict.txt).
    - metabolic_pathways: contains the code and data required to actually run the correlation tests. The code appears in metabolic_pathway_correlations.R, and simply runs with the command Rscript and no arguments. The data appears in the data subfolder, which itself contains three subfolders:
      - embed: contains embeddings to be loaded by metabolic*pathway_correlations.R, e.g., merged_sequences_embeddings.txt or glove_emb_AG_newfilter.07_100.txt. Also contains a script assemble_new_embs.py, which lets new embeddings txt files be formatted from a pytorch embeddings tensor, such as the one stored in epoch_120_IBD_avg_vocab_embeddings.pth, as well as seqs*.07_embed.txt.
      - AG_new/pathways: contains a bunch of files like "corr_matches_i_i+9.RDS", which store intermediate results of the permutation tests, so they don't all have to be calculated at once. Should be recomputed with each run.
      - pathways: mostly stores various other input and output RDS files:
        - corr_matches.rds : stores intermediate results of statistical significance testing with model embeddings. Recomputed each time.
        - corr_matches_pca.rds : stores prior result of statistical significance testing with PCA embeddings. Loaded from storage by default.
        - filtered_otu_pathway_table.RDS / txt : stores associations of each taxa vocab entry with metabolic pathways, filtered to exclude pathways that are no longer present in KEGG.
        - pathway_table.RDS : updated pathway table saved by metabolic_pathway_correlations.R each run.
        - pca_embedded_taxa.rds : stores PCA embeddings of all the vocab taxa entries.
- microbiome_transformers.zip
  - A backup of our [GitHub repository](https://github.com/QuintinPope/microbiome_transformers) for the model architecture (both generator and discriminator), the pretraining processes for both, as well as the model finetuning scripts. Contains its own READMEs.
  - Has the code for pretraining generator models. See pretrain_generator/train_command.sh and pretrain_generator/README.MD
  - Has the code for using those models to pretrain discriminator models. See pretrain_discriminator/train_command.sh and pretrain_discriminator/README.MD
  - Has the code for finetuning those pretrained discriminator models on the classification data in our study (both within-distribution experiments and out of distribution experiments).
    - See finetune_discriminator/README.MD for general info on finetuning.
    - See finetune_discriminator/run_agp_agp_exps.sh for the commands to run the in-distribution experiments.
    - See finetune_discriminator/run_agp_HF_SH_cross_gen_ensemble_tests.sh to run the out of distribution experiments using an ensemble of models.
    - See finetune_discriminator/run_agp_HF_SH_cross_gen_val_set_tests.sh to run the out of distribution experiments without an ensemble and using a val set for stopping condition.

## File Structures:

**microbiomedata.zip**

```
|____total_IBD_otu.npy
|____IBD_train_512.npy
|____halfvarson_IBD_labels.npy
|____IBD_train_otu.npy
|____test_encodings_512.npy
|____total_IBD_512.npy
|____train_encodings_512.npy
|____schirmer_IBD_labels.npy
|____schirmer_IBD_512_otu.npy
|____fruitdata
| |____FRUIT_FREQUENCY_all_label.npy
| |____FRUIT_FREQUENCY_otu_512.npy
| |____FRUIT_FREQUENCY_binary24_labels.npy
| |____FRUIT_FREQUENCY_all_otu.npy
| |____FRUIT_FREQUENCY_binary34_labels.npy
|____vegdata
| |____VEGETABLE_FREQUENCY_all_label.npy
| |____VEGETABLE_FREQUENCY_binary24_labels.npy
| |____VEGETABLE_FREQUENCY_otu_512.npy
| |____VEGETABLE_FREQUENCY_all_otu.npy
| |____VEGETABLE_FREQUENCY_binary34_labels.npy
|____README
|____schirmer_IBD_otu.npy
|____IBD_test_label.npy
|____IBD_test_512.npy
|____IBD_train_label.npy
|____IBD_test_otu.npy
|____test_encodings_1897.npy
|____halfvarson_otu.npy
|____halfvarson_512_otu.npy
|____total_IBD_label.npy
|____train_encodings_1897.npy
```

**pretrainedmodels.zip**

```
____5head5layer_epoch60_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch30_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch105_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch45_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch90_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch120_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch15_disc
| |____config.json
| |____pytorch_model.bin
|____5head5layer_epoch75_disc
| |____config.json
| |____pytorch_model.bin
```

**ensemble.zip**

```
|____run4_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run8_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run1_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run2_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run10_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run7_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run9_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run5_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run6_epoch0_disc
| |____config.json
| |____pytorch_model.bin
|____run3_epoch0_disc
| |____config.json
| |____pytorch_model.bin
```

**DeepMicro.zip**

```
|____LICENSE
|____deep_env_config.yml
|____DM.py
|____exception_handle.py
|____README.md
|____exps_cae.sh
|____exps_ae.sh
|____exps_baselines.sh
|____results
| |____display_results.py
| |____plots
|____data
| |____host_to_indices
| | |____host_to_ids.py
| |____marker.zip
| |____UserLabelExample.csv
| |____convert_data.py
| |____get_unlabeled_pretraining_data.py
| |____UserDataExample.csv
| |____abundance.zip
|____DNN_models.py
```

**property_pathway_correlations.zip**

```
|____metabolic_pathways
| |____metabolic_pathway_correlations.R
| |____data
| | |____AG_new
| | | |____pathways
| | | | |____corr_matches_141_150.RDS
| | | | |____corr_matches_81_90.RDS
| | | | |____corr_matches_21_30.RDS
| | | | |____corr_matches_51_60.RDS
| | | | |____corr_matches_121_130.RDS
| | | | |____corr_matches_101_110.RDS
| | | | |____corr_matches_61_70.RDS
| | | | |____corr_matches_31_40.RDS
| | | | |____corr_matches_131_140.RDS
| | | | |____corr_matches_181_190.RDS
| | | | |____corr_matches_161_170.RDS
| | | | |____corr_matches_11_20.RDS
| | | | |____corr_matches_1_10.RDS
| | | | |____corr_matches_191_200.RDS
| | | | |____corr_matches_171_180.RDS
| | | | |____corr_matches_71_80.RDS
| | | | |____corr_matches_91_100.RDS
| | | | |____corr_matches_111_120.RDS
| | | | |____corr_matches_41_50.RDS
| | | | |____corr_matches_151_160.RDS
| | |____embed
| | | |____seqs_.07_embed.txt
| | | |____merged_sequences_embeddings.txt
| | | |____assemble_new_embs.py
| | | |____epoch_120_IBD_avg_vocab_embeddings.pth
| | | |____glove_emb_AG_newfilter.07_100.txt
| | |____pathways
| | | |____filtered_otu_pathway_table.RDS
| | | |____pca_embedded_taxa.rds
| | | |____pathway_table.RDS
| | | |____corr_matches.rds
| | | |____filtered_otu_pathway_table.txt
| | | |____corr_matches_pca.rds
|____figures
| |____csvs
| | |____significant_correlations_tests.py
| | |____gen_hists.py
| |____new_hists
|____pathways
| |____top20Paths_per_property_ids_v2.csv
| |____top20Paths_per_property_names_v2.csv
| |____property_pathway_dict_allsig.txt
| |____property_pathway_dict.txt
```

**microbiome_transformers.zip**

```
|____electra_trace.py
|____multitaskfinetune
| |____begin.py
| |____pretrain_hf.py
| |____electra_discriminator.py
| |____dataset.py
| |____startup
|____finetune_discriminator
| |____begin.py
| |____pretrain_hf.py
| |____electra_pretrain_model.py
| |____electra_discriminator.py
| |____run_agp_agp_exps.sh
| |____run_agp_HF_SH_cross_gen_val_set_tests.sh
| |____run_agp_HF_SH_cross_gen_ensemble_tests.sh
| |____hf_startup_3
| |____hf_startup_4
| |____README.MD
| |____dataset.py
| |____torch_rbf.py
|____combine_sets.py
|____pretrain_discriminator
| |____begin.py
| |____pretrain_hf.py
| |____electra_pretrain_model.py
| |____hf_startup
| |____README.MD
| |____train_command.sh
| |____dataset.py
|____benchmark_startup
|____pretrain_generator
| |____begin.py
| |____pretrain_hf.py
| |____electra_pretrain_model.py
| |____hf_startup
| |____README.MD
| |____train_command.sh
| |____dataset.py
|____README.md
|____compress_data.py
|____generate_commands.py
|____attention_benchmark
| |____begin.py
| |____pretrain_hf.py
| |____electra_discriminator.py
| |____hf_startup
| |____dataset.py
|____data_analyze.py
|____benchmarks.py
```

# Usage Instructions

Intended to cover both repeating the experiments we performed in our paper, or extending our methods to new datasets:

- Prepare input data and initial embeddings
  - Vocabulary: Set the initial vocabulary size to accommodate all the unique OTUs/ASVs found in the data, plus special tokens such as mask, padding, and cls tokens.
  - Initial embeddings: Each vocabulary element (including special tokens) is assigned a unique embedding vector.
  - Input data format: Given the highly sparse nature of most microbiome samples relative to vocabulary size, we store each sample’s abundance information in coordinate-list format. I.e., a data file is a numpy array of size (n_samples, max_sample_size, 2), and each sample is stored as a (max_sample_size, 2) array.
- Pretrain a language model on those embeddings
  - ELECTRA generators: Pretrain a sequence of generator models on unsupervised microbiome data. See pretrain_generator/train_command.sh and pretrain_generator/README.MD in microbiome_transformers.zip
  - ELECTRA discriminators: Pretrain a sequence of discriminator models on unsupervised microbiome data using outputs from the previously trained generators to generate substitutions for the original sequences. See pretrain_discriminator/train_command.sh and pretrain_discriminator/README.MD in microbiome_transformers.zip
- Characterize the language model with the following interpretability steps:
  - Perform taxonomic assignments: Use assign_16S_to_phyla.R (or similar R code) to map your sequences to the phylogenetic hierarchy.
  - Attribution calculations: Use Attribution_calculations.ipynb to calculate per-sample model prediction scores, per-taxa attribution values (used for interpretability), as well as per-taxa averaged embeddings (used for plotting the taxa).
  - Embeddings visualizations and embedding space clustering:
    - Provide Plot_microbiome_transformers_results.ipynb with the paths to your per-taxa averaged embeddings calculated above, initial vocabulary embeddings (equivalent of vocab_embeddings.npy), and taxonomic assignments.
    - It will help generate TSNE visualizations of the two embedding spaces, as well as cross-comparisons of where taxa in one embedding space appear in the other embedding space.
    - The notebook contains preset regions for which parts of the two embedding spaces to compare (via bounding boxes with the select_by_rectangles function). These regions will likely not work for a new dataset, so you'll have to change them.
    - Finally, the notebook will also plot graphs comparing the clusterability of the data in the original two embedding spaces (non TSNE), so as to not be fooled by the dimension reduction technique.
  - Identify high-attribution taxa:
    - Process_Attributions_No_GPU.ipynb takes the per-sample prediction scores and the per-taxa attribution values (both from Attribution_calculations.ipynb) and identifies the taxa most and least associated with IBD.
    - It also includes filtration steps for the attribution calculations (e.g., only analyze taxa that appear >= 5 times, only use attribution scores that are confident and correct, etc), reflecting those we used in the paper.
    - The notebook will identify the taxa IDs of the top and bottom attributed taxa, then it will use seqs\_.07_embed.fasta (or similar taxa-ID mapping) to print the 16S sequences associated with those taxa.
  - Pathway correlations:
    - Use assemble_new_embs.py to format pytorch vocab embedding files into the expected format for metabolic_pathway_correlations.R
    - Use metabolic_pathway_correlations.R (in the metabolic_pathways folder of property_pathway_correlations.zip) to produce heatmaps of embedding dim / metabolic pathway correlation strengths, and to save a file with the statistically significant correlation data.
    - Use gen_hists.py (in the figures/csvs folder of property_pathway_correlations.zip) to generate histograms comparing embedding dim / pathway correlation strengths of the initial fixed embeddings with those of the learned contextual embeddings.
    - Use significant_correlations_tests.py (also in the figures/csvs folder of property_pathway_correlations.zip) to apply non-parametric statistical tests to determine whether the distribution of embedding dim / pathway correlation strengths from the learned contextual embeddings is shifted right compared to those from the fixed embeddings.
- Evaluate the language model for downstream task
  - First, account for any patients who have multiple samples in the dataset by blocking out any train / validation / test splits you perform by patient ID. Future steps will assume you have dictionaries (stored as pickle files) that map from some patient ID strings (which just need to be unique per patient) to indices of the data files (i.e., you need one mapping dict per training data file). In general, the way to do this will depend on how your patient metadata is structured. You can look to host_to_ids.py (in DeepMicro.zip) to see how we combined metadata from multiple files and compared that with the different training data numpy files to produce this mapping.
  - To run experiments using our paper's transformer methods:
    - "Within distribution" evaluations: Relevant commands are in finetune_discriminator/run_agp_agp_exps.sh in microbiome_transformers.zip
    - "Out of distribution" evaluations: Relevant commands are in finetune_discriminator/run_agp_HF_SH_cross_gen_ensemble_tests.sh (when using an ensemble of models) and finetune_discriminator/run_agp_HF_SH_cross_gen_val_set_tests.sh (without using an ensemble and when using a val set for stopping condition). Both are in microbiome_transformers.zip
    - See also finetune_discriminator/README.MD in microbiome_transformers.zip for more general information about the finetuning functionality
  - To run experiments using the DeepMicro-derived baseline methods:
    - See exps_ae.sh, exps_cae.sh, and exps_baselines.sh in DeepMicro.zip for the experiment commands (for both in-distribution and out of distribution experiments)
    - Also see README.md in DeepMicro.zip for more general information on using DeepMicro and our modifications to it.

## Changelog:

**01/29/2025**

Updated significant_correlations_tests.py to apply permutation testing and report Cohan's d and Cliff's Delta.

Added run_blast_with_downloads.sh, which reports how many taxa in Halfvarson match to any taxa in AGP and how many taxa in Schirmer match any taxa in AGP. It's a way of comparing which of Schirmer or Halfvarson is more similar to AGP in terms of taxa that are present.

We also slightly clarified the README's language to make it clearer where the software can be found.
