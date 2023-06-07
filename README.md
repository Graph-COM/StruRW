# StruRW
This repository is the implementation for the paper [Structural Re-weighting Improves Graph Domain Adaptation](https://arxiv.org/abs/2306.03221) by Shikun Liu, Tianchun Li, Yongbin Feng, Nhan Tran, Han Zhao, Qiu Qiang, and Pan Li. Current Github repo is under construction.

## Overview ##


## Datasets ##
- Fast simulation datasets are the dataset from Pileup mitigation at the Large Hadron Collider
Our datasets from HEP application are put under the `pileup` folder inside `dataset` folder. We did not include the benchmark datasets here. The DBLP and ACM datasets can be downloaded from the UDAGCN paper github. The Cora and Arxiv data can be found in GOOD survey. 

For the training of each StruRW-based model, go to the corresponding folder
'StruRW_ADV' stands for adversarial training based model, 'StruRW_ERM' stands for the ERM based model and 'StruRW_Mix' stands for the mixup-based model
```
python run_nni.py -d [dataset] -m [method] -b [backbone] --dir_name [dir_name]
```

`dataset` can be choosen from `SBM`, `Pileup`, `dblp_acm`, `cora`, `arxiv`\
`method` can be choosen from `ERM`, `DANN` and `DANN_rw`\
`backbone` can be `GCN` or `GraphSage`\
`dir_name` is the name you want to name your directory, which will saves the log file of the experiment

Other arguments can be passed with specific to models, check the argument list for description

Specific arguments for running different datasets:
Pileup: `num_events`, `balanced`, `train_sig`, `train_PU`, `test_sig`, `test_PU`\
CSBM: `num_nodes`, `sigma`, `ps`, `qs`, `pt`, `qt`\
dblp_acm: `src_name`, `tgt_name`, specify the name from 'dblp' and 'acm'\
cora: `domain_split` can be chosen from 'word' or 'degree'\
arxiv: `domain_split` can be specified as 'degree' if want to run the shift with node degree\
otherwise, use `start_year` and `end_year` to specify the time period for training.

Specific arguments for model:
DANN and StruRW-ADV: `alphatimes`, `alphamin`\
for any with StruRW model: specify `rw_lmda`, `start_epoch`, `rw_freq`

The choice of hyperparameter and their search space has been specified in the appendix of our paper


