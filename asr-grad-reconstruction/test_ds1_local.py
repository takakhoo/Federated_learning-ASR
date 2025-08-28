#!/usr/bin/env python3
"""
Local test script for DS1 experiment - modified to use local paths
"""

import torch
import torch.optim as optim
import argparse
from torch import nn
import sys, os
import time
import numpy as np
import logging
from typing import List, Tuple

# Add the 'modules/deepspeech/src/' directory to the system path
sys.path.insert(0, os.path.abspath('modules/deepspeech/src'))

# Add the src directory to the path
sys.path.insert(0, os.path.abspath('src/'))

from models.ds1 import DeepSpeech1WithContextFrames
from ctc.ctc_loss_imp import *
from data.librisubset import *
from utils.plot import *
from utils.util import *
from loss.loss import *

# Import the reconstruct_dataset function from the main script
from reconstruct_ds1_run_many_sample import reconstruct_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_device_net(FLAGS, use_relu):
    net = DeepSpeech1WithContextFrames(FLAGS.n_context, FLAGS.drop_prob, use_relu=use_relu).to(device)
    return device, net

def main(FLAGS):
    """Main function for running DS1 experiment locally"""
    
    # Create local logging directory
    local_logging_dir = os.path.join(os.getcwd(), 'local_logging')
    os.makedirs(local_logging_dir, exist_ok=True)
    
    # Create experiment-specific directory
    exp_name = f"LOCAL_TEST_DS1_batchstart_{FLAGS.batch_start}_batch_end_{FLAGS.batch_end}_init_{FLAGS.init_method}_opt_{FLAGS.optimizer_name}_lr_{FLAGS.lr}_distfunc_{FLAGS.distance_function}_distfuncweight_{FLAGS.distance_function_weight}_reg_{FLAGS.reg}_regw_{FLAGS.reg_weight}_top-grad-perc_{FLAGS.top_grad_percentage}"
    FLAGS.exp_path = os.path.join(local_logging_dir, exp_name)
    os.makedirs(FLAGS.exp_path, exist_ok=True)
    
    logging.info(f'Logging experiment to {FLAGS.exp_path}')
    
    # ---------------------------------------------------------------------------- #
    #                      Loading network and devices.......                      #
    # ---------------------------------------------------------------------------- #
    device, net = get_device_net(FLAGS, use_relu=False)
    logging.info(f'Device: {device}')
    logging.info(f'Network: {net.__class__.__name__}')
    
    if FLAGS.cpt_resume is not None:
        state_dict = torch.load(FLAGS.cpt_resume)['network']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = 'network.' + k  # Add 'network.' prefix
            new_state_dict[new_key] = v
        net.load_state_dict(new_state_dict)
        logging.info(f'Checkpoint loaded from {FLAGS.cpt_resume}')
    else:
        logging.info('Random init weight')
    
    # ---------------------------------------------------------------------------- #
    #                     Loading dataset ....................                     #
    # ---------------------------------------------------------------------------- #
    try:
        dataset, loader = get_dataset_libri_sampled_folder_subset(net, FLAGS)
        logging.info(f'Dataset loaded successfully. Dataset size: {len(dataset)}')
        logging.info(f'Loader created successfully')
        
        # Test reconstruction on a small batch
        logging.info('Testing reconstruction on first batch...')
        reconstruct_dataset(net, loader, FLAGS)
        
    except Exception as e:
        logging.error(f'Error loading dataset: {e}')
        logging.info('This might be due to missing dataset path or permissions')
        logging.info('The experiment structure is working, but dataset access needs to be configured')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local test of DS1 experiment")
    
    # Add arguments with default values
    parser.add_argument("--batch-start", type=int, default=0, help="index of the start")
    parser.add_argument("--batch-end", type=int, default=1, help="index of the end")
    parser.add_argument("--optimizer_name", type=str, default='Adam', help="Optimizer to use for optimization")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate for optimization")
    parser.add_argument("--zero_lr", type=float, default=100, help="Learning rate for zero order optimization")
    parser.add_argument("--distance_function", type=str, default='cosine', choices=["L1", "L2","cosine", "cosine+l2"], help="Distance function for gradient matching")
    parser.add_argument("--distance_function_weight", type=float, default=1.0, help="weight for the main component of the distance function 0-1")
    parser.add_argument("--reg", type=str, default='None', choices=["L1", "L2", "TV", "None"], help="Type of regularization")
    parser.add_argument("--reg_weight", type=float, default=0.0, help="Weight of the regularization term")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of seeds to try")
    parser.add_argument("--max_iter", type=int, default=2000, help="Number of iterations for the optimization")
    parser.add_argument("--zero_max_iter", type=int, default=200, help="Number of iterations for zero order optimization")
    parser.add_argument("--n_context", type=int, default=6, help="Number of context frames")
    parser.add_argument("--drop_prob", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--init_method", type=str, default='uniform', help="Initialization method")
    parser.add_argument("--cpt_resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--top_grad_percentage", type=float, default=1.0, help="Top gradient percentage")
    parser.add_argument("--batch_min_dur", type=int, default=0, help="Minimum duration of batch")
    parser.add_argument("--batch_max_dur", type=int, default=1000, help="Maximum duration of batch")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--dataset_path", type=str, default='/scratch/f006pq6/datasets/librispeech_sampled_600_file_0s_4s', help="Dataset path")
    
    FLAGS = parser.parse_args()
    
    assert FLAGS.batch_size == 1, "Batch size must be 1"
    
    logging.info("=== DS1 Local Test Experiment ===")
    logging.info(f"Batch start: {FLAGS.batch_start}")
    logging.info(f"Batch end: {FLAGS.batch_end}")
    logging.info(f"Learning rate: {FLAGS.lr}")
    logging.info(f"Optimizer: {FLAGS.optimizer_name}")
    logging.info(f"Distance function: {FLAGS.distance_function}")
    logging.info(f"Dataset path: {FLAGS.dataset_path}")
    
    main(FLAGS)
