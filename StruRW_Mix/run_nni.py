import argparse
import numpy as np
from nni.experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description='Tune SAT')
    parser.add_argument('--name', type=str, required=True, help='name of the experiment')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', required=True)
    parser.add_argument('-m', '--method', type=str, help='method used', required=True)
    parser.add_argument('--dir_name', type=str, required=True)
    parser.add_argument("--src_name", type=str, help='specify for the source dataset name', default = 'dblp')
    parser.add_argument("--tgt_name", type=str, help='specify for the target dataset name', default = 'acm')
    parser.add_argument("--domain_split", type=str, help='specify for the cora split', default = 'word')
    parser.add_argument("--ps", type=float, help="the source intraclass connection probability for CSBM dataset", default=0.2)
    parser.add_argument("--qs", type=float, help="the source interclass connection probability for CSBM dataset", default=0.02)
    parser.add_argument("--pt", type=float, help="the target intraclass connection probability for CSBM dataset", default=0.2)
    parser.add_argument("--qt", type=float, help="the target interclass connection probability for CSBM dataset", default=0.16)
    parser.add_argument('--start_year', type=int, help='training year start for arxiv', default=2005)
    parser.add_argument('--end_year', type=int, help='training year end for arxiv', default=2007)

    parser.add_argument("--gpu_loc", type=str, help='later indicates the later three, ow the former', default='later')
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    if args.method == 'Mixup':
        search_space = {
            'epochs': {'_type': 'choice', '_value': [300]},
            'lr': {'_type': 'choice', '_value': [0.001]},
            'opt_decay_step': {'_type': 'choice', '_value': [50]},
            'opt_decay_rate': {'_type': 'choice', '_value': [0.8]},
            'rw_lmda' : {'_type': 'choice', '_value': [1]},
            'dropout': {"_type": "choice", "_value": [0.5]},
            #'weight_decay': {"_type": "uniform", "_value": [0, 1e-3]},
        }
    elif args.method == 'Mixup_rw':
        search_space = {
            'epochs': {'_type': 'choice', '_value': [300]},
            'lr': {'_type': 'choice', '_value': [0.001]},
            'start_epoch': {'_type': 'choice', '_value': [200]},
            'opt_decay_step': {'_type': 'choice', '_value': [50]},
            'opt_decay_rate': {'_type': 'choice', '_value': [0.8]},
            'rw_freq': {'_type': 'choice', '_value': [10]},
            'rw_lmda': {'_type': 'choice', '_value': [0]},
            'dropout': {"_type": "choice", "_value": [0.5]},
        }
    else:
        search_space = {}
    if args.dataset == "cora" or args.dataset == "arxiv":
        search_space['hidden_dim'] = {'_type': 'choice', '_value': [300]}
    elif args.dataset == "dblp_acm":
        search_space['hidden_dim'] = {'_type': 'choice', '_value': [128]}
    elif args.dataset == "Pileup":
        search_space['hidden_dim'] = {'_type': 'choice', '_value': [50]}
    else:
        search_space['hidden_dim'] = {'_type': 'choice', '_value': [20]}
   
    experiment = Experiment('local')
    experiment.config.experiment_name = args.name
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.trial_gpu_number = 1
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.training_service.use_active_gpu = True
    experiment.config.experiment_working_directory = '../nni-experiments'

    experiment.config.tuner.name = 'GridSearch'
    # experiment.config.tuner.name = 'SMAC'
    if experiment.config.tuner.name == 'GridSearch':
        experiment.config.max_trial_number = int(np.prod([len(v['_value']) for k, v in search_space.items()])) * 2
    else:
        experiment.config.max_trial_number = 10000
    experiment.config.trial_concurrency = 60
    if args.gpu_loc == 'later':
        experiment.config.training_service.gpu_indices = [0, 1, 2, 3, 6, 7]
    else:
        experiment.config.training_service.gpu_indices = [3, 7]
    
    if args.dataset == 'dblp_acm':
        experiment.config.training_service.max_trial_number_per_gpu = 7
    elif args.dataset == 'arxiv':
       if args.domain_split == 'time1':
            experiment.config.training_service.max_trial_number_per_gpu = 2
       else:
            experiment.config.training_service.max_trial_number_per_gpu = 1
    elif args.dataset == 'cora':
        experiment.config.training_service.max_trial_number_per_gpu = 4
    else:
        experiment.config.training_service.max_trial_number_per_gpu = 10

    gpu_ratio = 1 / experiment.config.training_service.max_trial_number_per_gpu

    experiment.config.trial_command = f'python mixup_nni.py -d {args.dataset} -m {args.method}\
        --domain_split {args.domain_split} --src_name {args.src_name} --tgt_name {args.tgt_name} \
        --start_year {args.start_year} --end_year {args.end_year}\
        --ps {args.ps} --qs {args.qs} --pt {args.pt} --qt {args.qt}\
        --gpu_ratio {gpu_ratio} --dir_name {args.dir_name}'

    experiment.run(args.port)
    input('Press enter to quit')
    experiment.stop()


if __name__ == '__main__':
    main()
