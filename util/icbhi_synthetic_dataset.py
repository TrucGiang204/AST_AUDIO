import os
import random
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio
from .icbhi_dataset import ICBHIDataset
import librosa
import soundfile as sf


class ICBHISyntheticDataset(ICBHIDataset):
    """
    Enhanced ICBHI Dataset that incorporates AudioLDM-generated synthetic data
    for addressing class imbalance in respiratory sound classification.
    """
    
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False, 
                 synthetic_data_dir=None, augmentation_ratio=0.5):
        """
        Initialize the enhanced dataset with synthetic data integration.
        
        Args:
            train_flag: Whether this is training data
            transform: Data transforms to apply
            args: Arguments object with dataset parameters
            print_flag: Whether to print dataset information
            mean_std: Whether to compute mean/std
            synthetic_data_dir: Directory containing synthetic audio data
            augmentation_ratio: Ratio of synthetic to original samples (0.0-1.0)
        """
        # Initialize the parent class first
        super().__init__(train_flag, transform, args, print_flag, mean_std)
        
        self.synthetic_data_dir = synthetic_data_dir
        self.augmentation_ratio = augmentation_ratio
        self.synthetic_samples = []
        
        # Load synthetic data if available and this is training
        if train_flag and synthetic_data_dir and os.path.exists(synthetic_data_dir):
            self._load_synthetic_data()
            self._integrate_synthetic_data()
            
            if print_flag:
                print(f"Integrated {len(self.synthetic_samples)} synthetic samples")
                self._print_enhanced_statistics()
    
    def _load_synthetic_data(self):
        """Load synthetic audio data from the specified directory."""
        print("Loading synthetic respiratory sound data...")
        
        # Check for generation summary file
        summary_file = os.path.join(self.synthetic_data_dir, "generation_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                generation_summary = json.load(f)
            self._load_from_summary(generation_summary)
        else:
            # Fallback: scan directory structure
            self._load_from_directory()
    
    def _load_from_summary(self, generation_summary):
        """Load synthetic data using the generation summary file."""
        class_names = {1: "crackles", 2: "wheezes", 3: "both"}
        
        for class_idx_str, class_data in generation_summary.items():
            class_idx = int(class_idx_str)
            
            if class_idx in [1, 2, 3]:  # Only load target classes
                generated_files = class_data.get('generated_files', [])
                
                for filepath in generated_files:
                    if os.path.exists(filepath):
                        # Load audio data
                        try:
                            audio_data, sr = sf.read(filepath)
                            
                            # Ensure correct sample rate
                            if sr != self.sample_rate:
                                audio_data = librosa.resample(
                                    audio_data, orig_sr=sr, target_sr=self.sample_rate
                                )
                            
                            # Convert to tensor format like original data
                            audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                            
                            # Apply the same processing as original data
                            audio_tensor = cut_pad_sample_torchaudio(audio_tensor, self.args)
                            
                            # Store synthetic sample
                            synthetic_sample = {
                                'audio_data': audio_tensor,
                                'label': class_idx,
                                'filepath': filepath,
                                'is_synthetic': True
                            }
                            
                            self.synthetic_samples.append(synthetic_sample)
                            
                        except Exception as e:
                            print(f"Error loading synthetic audio {filepath}: {e}")
    
    def _load_from_directory(self):
        """Load synthetic data by scanning directory structure."""
        class_mapping = {"crackles": 1, "wheezes": 2, "both": 3}
        
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(self.synthetic_data_dir, f"class_{class_idx}_{class_name}")
            
            if os.path.exists(class_dir):
                wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                
                for wav_file in wav_files:
                    filepath = os.path.join(class_dir, wav_file)
                    
                    try:
                        # Load audio data
                        audio_data, sr = sf.read(filepath)
                        
                        # Ensure correct sample rate
                        if sr != self.sample_rate:
                            audio_data = librosa.resample(
                                audio_data, orig_sr=sr, target_sr=self.sample_rate
                            )
                        
                        # Convert to tensor format
                        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                        audio_tensor = cut_pad_sample_torchaudio(audio_tensor, self.args)
                        
                        # Store synthetic sample
                        synthetic_sample = {
                            'audio_data': audio_tensor,
                            'label': class_idx,
                            'filepath': filepath,
                            'is_synthetic': True
                        }
                        
                        self.synthetic_samples.append(synthetic_sample)
                        
                    except Exception as e:
                        print(f"Error loading synthetic audio {filepath}: {e}")
    
    def _integrate_synthetic_data(self):
        """Integrate synthetic samples with original data based on augmentation ratio."""
        if not self.synthetic_samples:
            return
        
        # Group synthetic samples by class
        synthetic_by_class = {1: [], 2: [], 3: []}
        for sample in self.synthetic_samples:
            class_idx = sample['label']
            if class_idx in synthetic_by_class:
                synthetic_by_class[class_idx].append(sample)
        
        # Calculate target augmentation for each class
        original_class_counts = np.zeros(self.args.n_cls)
        for sample in self.audio_data:
            original_class_counts[sample[1]] += 1
        
        # Determine how many synthetic samples to add for each class
        target_classes = [1, 2, 3]  # Crackles, Wheezes, Both
        max_count = max(original_class_counts)
        
        synthetic_audio_data = []
        
        for class_idx in target_classes:
            current_count = original_class_counts[class_idx]
            available_synthetic = len(synthetic_by_class[class_idx])
            
            if current_count > 0 and available_synthetic > 0:
                # Calculate target count based on augmentation ratio
                target_count = int(current_count * (1 + self.augmentation_ratio))
                samples_needed = min(target_count - int(current_count), available_synthetic)
                
                if samples_needed > 0:
                    # Randomly select synthetic samples
                    selected_samples = random.sample(
                        synthetic_by_class[class_idx], 
                        samples_needed
                    )
                    
                    # Convert to format compatible with original data
                    for sample in selected_samples:
                        # Create compatible format: (audio_tensor, label)
                        synthetic_audio_data.append((sample['audio_data'], sample['label']))
                    
                    print(f"Added {samples_needed} synthetic samples for class {class_idx}")
        
        # Add synthetic data to audio_data
        self.audio_data.extend(synthetic_audio_data)
        
        # Update labels list
        for sample in synthetic_audio_data:
            self.labels.append(sample[1])
        
        # Update class counts and ratios
        self.class_nums = np.zeros(self.args.n_cls)
        for sample in self.audio_data:
            self.class_nums[sample[1]] += 1
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        # Regenerate audio images for new data
        self._regenerate_audio_images()
    
    def _regenerate_audio_images(self):
        """Regenerate audio images including synthetic data."""
        print("Regenerating mel-spectrograms with synthetic data...")
        
        self.audio_images = []
        for index in range(len(self.audio_data)):
            audio, label = self.audio_data[index][0], self.audio_data[index][1]
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment + 1):
                if aug_idx > 0:
                    if self.train_flag and not self.mean_std:
                        # Apply augmentation to original data only
                        if index < len(self.audio_data) - len(self.synthetic_samples):
                            audio = augment_raw_audio(audio, self.sample_rate, self.args)
                            audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)
                        else:
                            # For synthetic data, skip augmentation or add minimal variation
                            audio_image.append(None)
                            continue
                    else:
                        audio_image.append(None)
                        continue
                
                # Generate mel-spectrogram
                image = generate_fbank(audio, self.sample_rate, n_mels=self.n_mels)
                audio_image.append(image)
            
            self.audio_images.append((audio_image, label))
    
    def _print_enhanced_statistics(self):
        """Print enhanced dataset statistics including synthetic data information."""
        print(f"\n{'='*60}")
        print(f"ENHANCED {self.split.upper()} DATASET WITH SYNTHETIC DATA")
        print(f"{'='*60}")
        
        print(f"Total audio samples: {len(self.audio_data)}")
        print(f"Synthetic samples: {len(self.synthetic_samples)}")
        print(f"Augmentation ratio: {self.augmentation_ratio}")
        
        print(f"\nClass distribution (with synthetic data):")
        for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
            print(f'Class {i} {self.args.cls_list[i]:<9}: {int(n):<4} ({p:.1f}%)')
        
        # Calculate improvement in class balance
        if hasattr(self, 'original_class_nums'):
            print(f"\nClass balance improvement:")
            for i in range(self.args.n_cls):
                if self.original_class_nums[i] > 0:
                    improvement = (self.class_nums[i] / self.original_class_nums[i] - 1) * 100
                    print(f'Class {i} {self.args.cls_list[i]:<9}: +{improvement:.1f}% samples')
    
    def get_synthetic_info(self):
        """Return information about synthetic data integration."""
        return {
            'num_synthetic_samples': len(self.synthetic_samples),
            'augmentation_ratio': self.augmentation_ratio,
            'synthetic_data_dir': self.synthetic_data_dir,
            'class_distribution': self.class_nums.tolist(),
            'class_ratio': self.class_ratio.tolist()
        }


def create_enhanced_dataset(train_flag, transform, args, synthetic_data_dir=None, 
                          augmentation_ratio=0.5, print_flag=True):
    """
    Factory function to create enhanced ICBHI dataset with synthetic data.
    
    Args:
        train_flag: Whether this is training data
        transform: Data transforms to apply
        args: Arguments object with dataset parameters
        synthetic_data_dir: Directory containing synthetic audio data
        augmentation_ratio: Ratio of synthetic to original samples
        print_flag: Whether to print dataset information
        
    Returns:
        ICBHISyntheticDataset instance
    """
    return ICBHISyntheticDataset(
        train_flag=train_flag,
        transform=transform,
        args=args,
        print_flag=print_flag,
        synthetic_data_dir=synthetic_data_dir,
        augmentation_ratio=augmentation_ratio
    )