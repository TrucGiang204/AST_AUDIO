#!/usr/bin/env python3
"""
Complete Training Pipeline with AudioLDM Synthetic Data Augmentation

This script demonstrates the full integration of:
1. Data imbalance analysis
2. AudioLDM-based synthetic data generation
3. Enhanced dataset with synthetic data
4. ADD-RSC model training with balanced data

Usage:
    python train_with_audioldm.py --data_folder /path/to/icbhi --generate_synthetic --train_model
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

# Add current directory to path
sys.path.append('.')

# Import project modules
from util.icbhi_synthetic_dataset import ICBHISyntheticDataset, create_enhanced_dataset
from util.icbhi_dataset import ICBHIDataset
from util.augmentation import SpecAugment
from models.ast import ASTModel
from models.resnet import ResNet50
from audioldm_generator import RespiratoryAudioGenerator
from analyze_data_distribution import analyze_distribution

import warnings
warnings.filterwarnings("ignore")


class AudioLDMTrainingPipeline:
    """Complete training pipeline with AudioLDM synthetic data augmentation."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set class information
        if args.class_split == 'lungsound' and args.n_cls == 4:
            args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
        else:
            raise NotImplementedError("Only 4-class lungsound classification is supported")
        
        print(f"Training pipeline initialized on device: {self.device}")
        print(f"Target classes for augmentation: {args.cls_list[1:]}")  # Skip normal class
    
    def analyze_data_imbalance(self):
        """Analyze data distribution and determine augmentation needs."""
        print("\n" + "="*60)
        print("STEP 1: DATA IMBALANCE ANALYSIS")
        print("="*60)
        
        results = analyze_distribution(self.args.data_folder)
        
        # Determine augmentation plan for minority classes
        augmentation_plan = {}
        train_counts = results['train_counts']
        
        # Find majority class count (usually normal)
        max_count = max(train_counts.values())
        
        # Calculate augmentation for target classes (1, 2, 3)
        for class_idx in [1, 2, 3]:  # crackle, wheeze, both
            current_count = train_counts.get(class_idx, 0)
            if current_count > 0:
                # Calculate samples needed to balance with majority class
                samples_needed = int((max_count - current_count) * self.args.augmentation_ratio)
                if samples_needed > 0:
                    augmentation_plan[class_idx] = min(samples_needed, self.args.max_synthetic_per_class)
        
        print(f"\nAugmentation plan:")
        for class_idx, num_samples in augmentation_plan.items():
            class_name = self.args.cls_list[class_idx]
            print(f"  Class {class_idx} ({class_name}): {num_samples} synthetic samples")
        
        return augmentation_plan, results
    
    def generate_synthetic_data(self, augmentation_plan):
        """Generate synthetic data using AudioLDM."""
        print("\n" + "="*60)
        print("STEP 2: AUDIOLDM SYNTHETIC DATA GENERATION")
        print("="*60)
        
        if not augmentation_plan:
            print("No augmentation needed based on current data distribution.")
            return None
        
        # Initialize AudioLDM generator
        generator = RespiratoryAudioGenerator(
            model_id=self.args.audioldm_model,
            device=self.device
        )
        
        # Generate synthetic data
        generation_results = generator.generate_augmentation_dataset(
            augmentation_plan=augmentation_plan,
            output_dir=self.args.synthetic_output_dir
        )
        
        return generation_results
    
    def create_datasets(self):
        """Create training and validation datasets with synthetic data."""
        print("\n" + "="*60)
        print("STEP 3: DATASET CREATION WITH SYNTHETIC DATA")
        print("="*60)
        
        # Define transforms
        if self.args.model == 'ast':
            # AST-specific transforms
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                SpecAugment(args=self.args)
            ])
        else:
            # ResNet transforms
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                SpecAugment(args=self.args)
            ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)) if self.args.model != 'ast' else transforms.ToTensor()
        ])
        
        # Create enhanced training dataset with synthetic data
        synthetic_data_dir = self.args.synthetic_output_dir if self.args.use_synthetic else None
        
        train_dataset = create_enhanced_dataset(
            train_flag=True,
            transform=train_transform,
            args=self.args,
            synthetic_data_dir=synthetic_data_dir,
            augmentation_ratio=self.args.synthetic_ratio,
            print_flag=True
        )
        
        # Create validation dataset (no synthetic data)
        val_dataset = ICBHIDataset(
            train_flag=False,
            transform=val_transform,
            args=self.args,
            print_flag=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, train_dataset, val_dataset
    
    def create_model(self):
        """Create the ADD-RSC model."""
        print("\n" + "="*60)
        print("STEP 4: MODEL INITIALIZATION")
        print("="*60)
        
        if self.args.model == 'ast':
            model = ASTModel(
                label_dim=self.args.n_cls,
                fstride=10, tstride=10,
                input_fdim=128, input_tdim=1024,
                imagenet_pretrain=True,
                audioset_pretrain=False,
                model_size='base384',
                adaptive_freq_filter=True,
                adaptive_diff_denoise=True,
                denoise_d_model=self.args.denoise_d_model,
                denoise_num_heads=self.args.denoise_num_heads
            )
        else:
            model = ResNet50(
                num_classes=self.args.n_cls,
                adaptive_freq_filter=True,
                adaptive_diff_denoise=True,
                denoise_d_model=self.args.denoise_d_model,
                denoise_num_heads=self.args.denoise_num_heads
            )
        
        model = model.to(self.device)
        
        print(f"Model: {self.args.model}")
        print(f"Number of classes: {self.args.n_cls}")
        print(f"Adaptive Differential Denoising enabled")
        
        return model
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        print("🫁 AudioLDM-Enhanced ADD-RSC Training Pipeline")
        print("="*60)
        
        # Step 1: Analyze data imbalance
        if self.args.analyze_data or self.args.generate_synthetic:
            augmentation_plan, analysis_results = self.analyze_data_imbalance()
        else:
            augmentation_plan = {}
        
        # Step 2: Generate synthetic data
        if self.args.generate_synthetic and augmentation_plan:
            generation_results = self.generate_synthetic_data(augmentation_plan)
            
            # Save pipeline results
            pipeline_results = {
                'augmentation_plan': augmentation_plan,
                'analysis_results': analysis_results,
                'generation_results': generation_results,
                'args': vars(self.args)
            }
            
            results_file = os.path.join(self.args.save_dir, 'audioldm_pipeline_results.json')
            os.makedirs(self.args.save_dir, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            print(f"Pipeline results saved to: {results_file}")
        
        # Step 3: Create datasets
        if self.args.train_model:
            train_loader, val_loader, train_dataset, val_dataset = self.create_datasets()
            
            # Step 4: Create model
            model = self.create_model()
            
            # Print final dataset statistics
            if hasattr(train_dataset, 'get_synthetic_info'):
                synthetic_info = train_dataset.get_synthetic_info()
                print(f"\nFinal Training Dataset Statistics:")
                print(f"  Total samples: {len(train_dataset)}")
                print(f"  Synthetic samples: {synthetic_info['num_synthetic_samples']}")
                print(f"  Augmentation ratio: {synthetic_info['augmentation_ratio']:.2f}")
            
            print(f"\n🎯 Ready for training with enhanced balanced dataset!")
            print(f"   Training samples: {len(train_dataset)}")
            print(f"   Validation samples: {len(val_dataset)}")
            
            return {
                'model': model,
                'train_loader': train_loader,
                'val_loader': val_loader,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset
            }
        
        print(f"\n✅ AudioLDM pipeline completed successfully!")
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AudioLDM-Enhanced ADD-RSC Training Pipeline')
    
    # Data arguments
    parser.add_argument('--data_folder', type=str, default='./ICBHI/ICBHI_final_database',
                       help='Path to ICBHI dataset folder')
    parser.add_argument('--synthetic_output_dir', type=str, default='./synthetic_respiratory_sounds',
                       help='Output directory for synthetic audio files')
    parser.add_argument('--save_dir', type=str, default='./save',
                       help='Directory to save models and results')
    
    # Pipeline control
    parser.add_argument('--analyze_data', action='store_true',
                       help='Analyze data distribution')
    parser.add_argument('--generate_synthetic', action='store_true',
                       help='Generate synthetic data using AudioLDM')
    parser.add_argument('--train_model', action='store_true',
                       help='Train the ADD-RSC model with enhanced dataset')
    parser.add_argument('--use_synthetic', action='store_true', default=True,
                       help='Use synthetic data during training')
    
    # Synthetic data generation
    parser.add_argument('--audioldm_model', type=str, default='cvssp/audioldm-s-full-v2',
                       help='AudioLDM model ID from HuggingFace')
    parser.add_argument('--augmentation_ratio', type=float, default=0.5,
                       help='Target augmentation ratio for imbalanced classes')
    parser.add_argument('--max_synthetic_per_class', type=int, default=200,
                       help='Maximum synthetic samples per class')
    parser.add_argument('--synthetic_ratio', type=float, default=0.3,
                       help='Ratio of synthetic to original samples in training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='ast', choices=['ast', 'resnet50'],
                       help='Model architecture to use')
    parser.add_argument('--n_cls', type=int, default=4,
                       help='Number of classes')
    parser.add_argument('--class_split', type=str, default='lungsound',
                       help='Class split type')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    
    # Audio processing
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Audio sample rate')
    parser.add_argument('--desired_length', type=int, default=8,
                       help='Target audio length in seconds')
    parser.add_argument('--n_mels', type=int, default=128,
                       help='Number of mel filter banks')
    parser.add_argument('--nfft', type=int, default=1024,
                       help='FFT size')
    parser.add_argument('--pad_types', type=str, default='repeat',
                       help='Padding type for audio')
    
    # Augmentation
    parser.add_argument('--raw_augment', type=int, default=0,
                       help='Number of raw audio augmentations')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup',
                       help='SpecAugment policy')
    parser.add_argument('--specaug_mask', type=str, default='mean',
                       help='SpecAugment mask type')
    
    # ADD denoising parameters
    parser.add_argument('--denoise_d_model', type=int, default=256,
                       help='Hidden size of denoising transformer')
    parser.add_argument('--denoise_num_heads', type=int, default=8,
                       help='Number of attention heads in denoising transformer')
    
    # Default values for compatibility
    parser.add_argument('--test_fold', type=str, default='official')
    parser.add_argument('--stetho_id', type=int, default=-1)
    
    return parser.parse_args()


def main():
    """Main function to run the complete pipeline."""
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_folder) and (args.analyze_data or args.generate_synthetic or args.train_model):
        print(f"Error: Data folder '{args.data_folder}' does not exist.")
        print("Please download the ICBHI dataset or specify the correct path.")
        return
    
    if not (args.analyze_data or args.generate_synthetic or args.train_model):
        print("Please specify at least one action: --analyze_data, --generate_synthetic, or --train_model")
        return
    
    # Initialize and run pipeline
    pipeline = AudioLDMTrainingPipeline(args)
    results = pipeline.run_complete_pipeline()
    
    if results and args.train_model:
        print(f"\n🚀 Pipeline completed! You can now proceed with training using:")
        print(f"   Model: {results['model']}")
        print(f"   Train Loader: {len(results['train_loader'])} batches")
        print(f"   Val Loader: {len(results['val_loader'])} batches")
        
        # Example of how to start training
        print(f"\n💡 To start training, you can use the existing main.py with:")
        print(f"   python main.py --use_synthetic_dataset --synthetic_data_dir {args.synthetic_output_dir}")


if __name__ == "__main__":
    main()