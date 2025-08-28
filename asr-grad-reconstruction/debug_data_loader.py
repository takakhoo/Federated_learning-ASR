#!/usr/bin/env python3
"""
Debug script to understand the actual data loader format
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
from data.librisubset import get_dataset_libri_sampled_folder_subset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def debug_data_loader():
    """Debug the actual data loader format"""
    logger.info("🔍 DEBUGGING DATA LOADER FORMAT")
    
    # Create model
    net = DeepSpeech2(winlen=0.032, winstep=0.02).to(device)
    
    # Create flags
    class DebugFlags:
        def __init__(self):
            self.batch_size = 1
            self.batch_min_dur = 2000
            self.batch_max_dur = 4000
            self.batch_start = 0
            self.batch_end = 3
            self.dataset_path = '/scratch/f006pq6/datasets/librispeech_sampled_600_file_0s_4s'
    
    FLAGS = DebugFlags()
    
    try:
        # Load dataset
        dataset, loader = get_dataset_libri_sampled_folder_subset(net, FLAGS)
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Examine first few batches
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= 3:
                break
                
            logger.info(f"\n📊 BATCH {batch_idx}:")
            logger.info(f"  Type: {type(batch_data)}")
            logger.info(f"  Length: {len(batch_data)}")
            
            # Examine each element
            for i, element in enumerate(batch_data):
                logger.info(f"  Element {i}:")
                logger.info(f"    Type: {type(element)}")
                logger.info(f"    Shape: {getattr(element, 'shape', 'No shape')}")
                logger.info(f"    Content preview: {str(element)[:100]}...")
                
                if isinstance(element, list):
                    logger.info(f"    List length: {len(element)}")
                    if len(element) > 0:
                        logger.info(f"    First item type: {type(element[0])}")
                        logger.info(f"    First item shape: {getattr(element[0], 'shape', 'No shape')}")
                
                elif isinstance(element, torch.Tensor):
                    logger.info(f"    Tensor dtype: {element.dtype}")
                    logger.info(f"    Tensor device: {element.device}")
                    
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_loader()
