#!/usr/bin/env python3
"""
Script to analyze the data distribution in ICBHI dataset for the 4-class problem.
This will help us understand which classes need augmentation.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add the current directory to the path to import from util
sys.path.append('.')

from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_annotations
import torch

def analyze_distribution(data_folder):
    """Analyze the class distribution in the ICBHI dataset."""
    
    # Create a mock args object with necessary parameters
    class Args:
        def __init__(self):
            self.data_folder = data_folder
            self.test_fold = 'official'
            self.class_split = 'lungsound'
            self.n_cls = 4
            self.stetho_id = -1
            self.sample_rate = 16000
            self.desired_length = 8
            self.pad_types = 'repeat'
            self.nfft = 1024
            self.n_mels = 128
            self.raw_augment = 0
            self.specaug_policy = 'icbhi_ast_sup'
            self.specaug_mask = 'mean'
            self.cls_list = ['normal', 'crackle', 'wheeze', 'both']
    
    args = Args()
    
    print("Loading training dataset...")
    # Create training dataset
    train_dataset = ICBHIDataset(
        train_flag=True,
        transform=None,
        args=args,
        print_flag=True,
        mean_std=False
    )
    
    print("\nLoading test dataset...")
    # Create test dataset
    test_dataset = ICBHIDataset(
        train_flag=False,
        transform=None,
        args=args,
        print_flag=True,
        mean_std=False
    )
    
    # Analyze training data distribution
    print("\n" + "="*50)
    print("TRAINING DATA ANALYSIS")
    print("="*50)
    
    train_labels = train_dataset.labels
    train_class_counts = Counter(train_labels)
    total_train_samples = len(train_labels)
    
    print(f"Total training samples: {total_train_samples}")
    print("\nClass distribution:")
    for class_idx in range(4):
        class_name = args.cls_list[class_idx]
        count = train_class_counts[class_idx]
        percentage = (count / total_train_samples) * 100
        print(f"  {class_idx} ({class_name}): {count:4d} samples ({percentage:5.1f}%)")
    
    # Analyze test data distribution
    print("\n" + "="*50)
    print("TEST DATA ANALYSIS")
    print("="*50)
    
    test_labels = test_dataset.labels
    test_class_counts = Counter(test_labels)
    total_test_samples = len(test_labels)
    
    print(f"Total test samples: {total_test_samples}")
    print("\nClass distribution:")
    for class_idx in range(4):
        class_name = args.cls_list[class_idx]
        count = test_class_counts[class_idx]
        percentage = (count / total_test_samples) * 100
        print(f"  {class_idx} ({class_name}): {count:4d} samples ({percentage:5.1f}%)")
    
    # Calculate imbalance ratios
    print("\n" + "="*50)
    print("IMBALANCE ANALYSIS")
    print("="*50)
    
    # Find the majority class in training data
    max_count = max(train_class_counts.values())
    majority_class = max(train_class_counts, key=train_class_counts.get)
    
    print(f"Majority class in training: {majority_class} ({args.cls_list[majority_class]}) with {max_count} samples")
    print("\nImbalance ratios (majority_class_count / class_count):")
    
    augmentation_needed = {}
    
    for class_idx in range(4):
        class_name = args.cls_list[class_idx]
        count = train_class_counts[class_idx]
        if count > 0:
            ratio = max_count / count
            print(f"  {class_idx} ({class_name}): {ratio:.2f}x")
            
            # Classes that need significant augmentation (ratio > 1.5)
            if ratio > 1.5 and class_idx != majority_class:
                samples_needed = max_count - count
                augmentation_needed[class_idx] = {
                    'class_name': class_name,
                    'current_count': count,
                    'target_count': max_count,
                    'samples_needed': samples_needed,
                    'ratio': ratio
                }
        else:
            print(f"  {class_idx} ({class_name}): No samples!")
    
    # Recommendations for augmentation
    print("\n" + "="*50)
    print("AUGMENTATION RECOMMENDATIONS")
    print("="*50)
    
    if augmentation_needed:
        print("Classes that need augmentation (focusing on Crackles, Wheezes, Both):")
        for class_idx, info in augmentation_needed.items():
            if class_idx in [1, 2, 3]:  # Crackles, Wheezes, Both
                print(f"\n  Class {class_idx} ({info['class_name']}):")
                print(f"    Current samples: {info['current_count']}")
                print(f"    Target samples: {info['target_count']}")
                print(f"    Need to generate: {info['samples_needed']} samples")
                print(f"    Imbalance ratio: {info['ratio']:.2f}x")
    else:
        print("No significant class imbalance detected (all ratios <= 1.5)")
    
    # Create visualization
    create_distribution_plot(train_class_counts, test_class_counts, args.cls_list)
    
    return {
        'train_counts': train_class_counts,
        'test_counts': test_class_counts,
        'augmentation_needed': augmentation_needed,
        'class_names': args.cls_list
    }

def create_distribution_plot(train_counts, test_counts, class_names):
    """Create a bar plot showing the class distribution."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training data plot
    classes = list(range(4))
    train_values = [train_counts[i] for i in classes]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
    
    bars1 = ax1.bar(classes, train_values, color=colors, alpha=0.8)
    ax1.set_title('Training Data Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xticks(classes)
    ax1.set_xticklabels([f'{i}\n({class_names[i]})' for i in classes])
    
    # Add value labels on bars
    for bar, value in zip(bars1, train_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Test data plot
    test_values = [test_counts[i] for i in classes]
    bars2 = ax2.bar(classes, test_values, color=colors, alpha=0.8)
    ax2.set_title('Test Data Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xticks(classes)
    ax2.set_xticklabels([f'{i}\n({class_names[i]})' for i in classes])
    
    # Add value labels on bars
    for bar, value in zip(bars2, test_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/workspace/data_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: /workspace/data_distribution.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze ICBHI dataset class distribution')
    parser.add_argument('--data_folder', type=str, default='./ICBHI/ICBHI_final_database',
                       help='Path to ICBHI dataset folder')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' does not exist.")
        print("Please download the ICBHI dataset or specify the correct path.")
        sys.exit(1)
    
    print("Starting ICBHI dataset analysis...")
    results = analyze_distribution(args.data_folder)
    print("\nAnalysis complete!")