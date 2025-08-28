#!/usr/bin/env python3
"""
Debug script to isolate DS2 infinite loss issue
"""

import torch
import sys
import os
import logging

# Add the 'modules/deepspeech/src/' directory to the system path
sys.path.insert(0, os.path.abspath('modules/deepspeech/src'))

# Add the src directory to the path
sys.path.insert(0, os.path.abspath('src/'))

from models.ds2 import DeepSpeech2
from ctc.ctc_loss_imp import batched_ctc_v2
from data.librisubset import get_dataset_libri_sampled_folder_subset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def debug_ds2_step_by_step():
    """Debug DS2 step by step to find where infinite loss occurs"""
    
    logging.info("=== DS2 Debug Session ===")
    logging.info(f"Device: {device}")
    
    # 1. Create model
    net = DeepSpeech2(winlen=0.032, winstep=0.02).to(device)
    logging.info(f"Model created: {net.__class__.__name__}")
    
    # 2. Create dummy data with PROPER length for DS2
    # DS2 needs longer sequences due to convolution layers
    batch_size = 1
    seq_len = 100  # Much longer to account for conv downsampling
    n_features = 257
    
    # Create dummy input
    dummy_input = torch.randn(seq_len, batch_size, n_features).to(device)
    logging.info(f"Dummy input shape: {dummy_input.shape}")
    logging.info(f"Input stats - mean: {dummy_input.mean():.6f}, std: {dummy_input.std():.6f}")
    
    # Create dummy targets
    dummy_targets = torch.randint(1, 29, (batch_size, 6)).to(device)  # 6 tokens
    logging.info(f"Dummy targets: {dummy_targets}")
    
    # 3. Test forward pass
    net.eval()
    with torch.no_grad():
        try:
            output = net(dummy_input)
            logging.info(f"Forward pass successful")
            logging.info(f"Output shape: {output.shape}")
            logging.info(f"Output stats - mean: {output.mean():.6f}, std: {output.std():.6f}")
            logging.info(f"Output min: {output.min():.6f}, max: {output.max():.6f}")
            
            # Check for NaN or inf
            if torch.isnan(output).any():
                logging.error("❌ Output contains NaN values!")
            if torch.isinf(output).any():
                logging.error("❌ Output contains infinite values!")
                
        except Exception as e:
            logging.error(f"❌ Forward pass failed: {e}")
            return
    
    # 4. Test log_softmax
    try:
        log_probs = output.log_softmax(-1)
        logging.info(f"Log softmax successful")
        logging.info(f"Log probs stats - mean: {log_probs.mean():.6f}, std: {log_probs.std():.6f}")
        logging.info(f"Log probs min: {log_probs.min():.6f}, max: {log_probs.max():.6f}")
        
        # Check for NaN or inf
        if torch.isnan(log_probs).any():
            logging.error("❌ Log probs contain NaN values!")
        if torch.isinf(log_probs).any():
            logging.error("❌ Log probs contain infinite values!")
            
    except Exception as e:
        logging.error(f"❌ Log softmax failed: {e}")
        return
    
    # 5. Test CTC loss with PROPER lengths
    try:
        # Use the ACTUAL output length from the model
        actual_seq_len = output.shape[0]
        logging.info(f"Model output length: {actual_seq_len}")
        logging.info(f"Target length: {dummy_targets.shape[1]}")
        
        # Create proper lengths
        input_lengths = torch.tensor([actual_seq_len]).to(device)
        target_lengths = torch.tensor([dummy_targets.shape[1]]).to(device)
        
        logging.info(f"Input lengths: {input_lengths}")
        logging.info(f"Target lengths: {target_lengths}")
        
        ctc_loss = batched_ctc_v2(log_probs, dummy_targets, input_lengths, target_lengths)
        logging.info(f"CTC loss successful: {ctc_loss.item():.6f}")
        
        # Check for NaN or inf
        if torch.isnan(ctc_loss):
            logging.error("❌ CTC loss is NaN!")
        if torch.isinf(ctc_loss):
            logging.error("❌ CTC loss is infinite!")
            
    except Exception as e:
        logging.error(f"❌ CTC loss failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Test gradient computation
    try:
        net.train()
        ctc_loss = batched_ctc_v2(log_probs, dummy_targets, input_lengths, target_lengths)
        
        # Compute gradients
        gradients = torch.autograd.grad(ctc_loss, net.parameters(), create_graph=True)
        logging.info(f"Gradient computation successful")
        
        # Check gradient norms
        total_norm = 0
        for i, grad in enumerate(gradients):
            if grad is not None:
                param_norm = grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                logging.info(f"Param {i} gradient norm: {param_norm:.6f}")
                
                if torch.isnan(grad).any():
                    logging.error(f"❌ Param {i} gradients contain NaN!")
                if torch.isinf(grad).any():
                    logging.error(f"❌ Param {i} gradients contain infinite values!")
        
        total_norm = total_norm ** 0.5
        logging.info(f"Total gradient norm: {total_norm:.6f}")
        
    except Exception as e:
        logging.error(f"❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logging.info("✅ All tests passed! DS2 seems to be working correctly with dummy data.")

def debug_with_real_data():
    """Debug with real dataset to see where the issue occurs"""
    
    logging.info("\n=== Testing with Real Dataset ===")
    
    # Create model
    net = DeepSpeech2(winlen=0.032, winstep=0.02).to(device)
    
    # Create dummy FLAGS for dataset loading
    class DummyFlags:
        def __init__(self):
            self.batch_size = 1
            self.batch_min_dur = 0
            self.batch_max_dur = 1000
            self.batch_start = 0
            self.batch_end = 1
            self.dataset_path = '/scratch/f006pq6/datasets/librispeech_sampled_600_file_0s_4s'
    
    FLAGS = DummyFlags()
    
    try:
        # Load dataset
        dataset, loader = get_dataset_libri_sampled_folder_subset(net, FLAGS)
        logging.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Get first batch
        for batch_idx, (inputs, targets, input_sizes, target_sizes) in enumerate(loader):
            if batch_idx >= 1:  # Only test first batch
                break
                
            logging.info(f"Batch {batch_idx}:")
            logging.info(f"  Inputs shape: {inputs.shape}")
            logging.info(f"  Targets: {targets}")
            logging.info(f"  Input sizes: {input_sizes}")
            logging.info(f"  Target sizes: {target_sizes}")
            
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_sizes = input_sizes.to(device)
            target_sizes = target_sizes.to(device)
            
            # Test forward pass
            net.eval()
            with torch.no_grad():
                output = net(inputs)
                logging.info(f"  Output shape: {output.shape}")
                logging.info(f"  Output stats - mean: {output.mean():.6f}, std: {output.std():.6f}")
                
                if torch.isnan(output).any():
                    logging.error("  ❌ Output contains NaN!")
                if torch.isinf(output).any():
                    logging.error("  ❌ Output contains infinite values!")
            
            # Test log softmax
            log_probs = output.log_softmax(-1)
            logging.info(f"  Log probs stats - mean: {log_probs.mean():.6f}, std: {log_probs.std():.6f}")
            
            if torch.isnan(log_probs).any():
                logging.error("  ❌ Log probs contain NaN!")
            if torch.isinf(log_probs).any():
                logging.error("  ❌ Log probs contain infinite values!")
            
            # Test CTC loss
            ctc_loss = batched_ctc_v2(log_probs, targets, input_sizes, target_sizes)
            logging.info(f"  CTC loss: {ctc_loss.item():.6f}")
            
            if torch.isnan(ctc_loss):
                logging.error("  ❌ CTC loss is NaN!")
            if torch.isinf(ctc_loss):
                logging.error("  ❌ CTC loss is infinite!")
            
            break
            
    except Exception as e:
        logging.error(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with dummy data first
    debug_ds2_step_by_step()
    
    # Test with real data
    debug_with_real_data()
