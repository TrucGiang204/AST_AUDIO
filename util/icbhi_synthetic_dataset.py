import os
import random
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image
import subprocess
import tempfile
import warnings
warnings.filterwarnings("ignore")

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio
from .icbhi_dataset import ICBHIDataset
import librosa
import soundfile as sf

# AudioLDM integration
try:
    from diffusers import AudioLDMPipeline
    AUDIOLDM_AVAILABLE = True
except ImportError:
    AUDIOLDM_AVAILABLE = False
    print("Warning: AudioLDM not available. Install with: pip install diffusers")


class AudioLDMGenerator:
    """
    AudioLDM-based synthetic respiratory sound generator for addressing class imbalance.
    Generates high-quality audio samples for Crackles, Wheezes, and Both classes.
    """
    
    def __init__(self, model_id="cvssp/audioldm-s-full-v2", device="auto"):
        """
        Initialize AudioLDM generator.
        
        Args:
            model_id: Hugging Face model ID for AudioLDM
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        if not AUDIOLDM_AVAILABLE:
            raise ImportError("AudioLDM not available. Install with: pip install diffusers")
        
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading AudioLDM model {model_id} on {self.device}...")
        self.pipe = AudioLDMPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Enable memory efficient attention if using CUDA
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
                self.pipe.enable_memory_efficient_attention()
            except:
                pass
        
        # Define prompts for each target class
        self.class_prompts = {
            1: [  # Crackles
                "respiratory crackles, lung crackles, fine crackles breathing sound",
                "coarse crackles, wet crackles, inspiratory crackles lung sound", 
                "pathological lung crackles, rales, crackling breathing sounds",
                "abnormal lung sounds with crackles, respiratory crackles pattern",
                "fine inspiratory crackles, wet lung sounds, crackling respiration"
            ],
            2: [  # Wheezes  
                "respiratory wheeze, lung wheeze, wheezing breathing sound",
                "expiratory wheeze, high-pitched wheeze, whistling lung sound",
                "pathological lung wheeze, wheezing respiration, bronchial wheeze", 
                "abnormal lung sounds with wheeze, respiratory wheeze pattern",
                "musical wheeze, continuous wheeze, wheezing lung sounds"
            ],
            3: [  # Both
                "respiratory crackles and wheeze, mixed lung sounds, crackles with wheeze",
                "pathological lung sounds with crackles and wheeze, mixed respiratory sounds",
                "abnormal breathing with crackles and wheezing, combined lung pathology",
                "inspiratory crackles with expiratory wheeze, mixed respiratory pathology",
                "lung sounds with both crackles and wheeze, complex respiratory abnormality"
            ]
        }
        
        print("AudioLDM generator initialized successfully!")
    
    def generate_samples(self, class_idx, num_samples, duration=5.0, output_dir=None, 
                        num_inference_steps=20, guidance_scale=7.5):
        """
        Generate synthetic audio samples for a specific class.
        
        Args:
            class_idx: Class index (1=Crackles, 2=Wheezes, 3=Both)
            num_samples: Number of samples to generate
            duration: Duration of each sample in seconds
            output_dir: Directory to save generated audio files
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            List of generated audio file paths
        """
        if class_idx not in self.class_prompts:
            raise ValueError(f"Invalid class_idx {class_idx}. Must be 1, 2, or 3.")
        
        class_names = {1: "crackles", 2: "wheezes", 3: "both"}
        class_name = class_names[class_idx]
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix=f"audioldm_{class_name}_")
        
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        prompts = self.class_prompts[class_idx]
        
        print(f"Generating {num_samples} samples for class {class_idx} ({class_name})...")
        
        for i in range(num_samples):
            # Randomly select a prompt
            prompt = random.choice(prompts)
            
            try:
                # Generate audio
                audio = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    audio_length_in_s=duration
                ).audios[0]
                
                # Save to file
                filename = f"synthetic_{class_name}_{i:04d}.wav"
                filepath = os.path.join(output_dir, filename)
                sf.write(filepath, audio, self.pipe.vocoder.config.sampling_rate)
                
                generated_files.append(filepath)
                
                if (i + 1) % 5 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples for {class_name}")
                    
            except Exception as e:
                print(f"Error generating sample {i} for class {class_idx}: {e}")
                continue
        
        print(f"Successfully generated {len(generated_files)} samples for {class_name}")
        return generated_files
    
    def generate_balanced_dataset(self, target_counts, output_dir, sample_duration=5.0):
        """
        Generate a balanced dataset for all target classes.
        
        Args:
            target_counts: Dict mapping class_idx to number of samples needed
            output_dir: Base directory for output
            sample_duration: Duration of each sample in seconds
            
        Returns:
            Dict with generation summary
        """
        generation_summary = {}
        
        for class_idx, count in target_counts.items():
            if class_idx in [1, 2, 3] and count > 0:
                class_output_dir = os.path.join(output_dir, f"class_{class_idx}")
                generated_files = self.generate_samples(
                    class_idx=class_idx,
                    num_samples=count,
                    duration=sample_duration,
                    output_dir=class_output_dir
                )
                
                generation_summary[class_idx] = {
                    'class_name': {1: 'crackles', 2: 'wheezes', 3: 'both'}[class_idx],
                    'requested_samples': count,
                    'generated_samples': len(generated_files),
                    'generated_files': generated_files,
                    'output_directory': class_output_dir
                }
        
        # Save generation summary
        summary_file = os.path.join(output_dir, "generation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(generation_summary, f, indent=2)
        
        print(f"Generation summary saved to {summary_file}")
        return generation_summary


def calculate_augmentation_needs(dataset, target_classes=[1, 2, 3], balance_ratio=0.8):
    """
    Calculate how many synthetic samples are needed for each class to achieve balance.
    
    Args:
        dataset: ICBHIDataset instance
        target_classes: List of class indices to augment
        balance_ratio: Target balance ratio relative to max class
        
    Returns:
        Dict mapping class_idx to number of samples needed
    """
    class_counts = dataset.class_nums
    max_count = max(class_counts)
    target_count = int(max_count * balance_ratio)
    
    augmentation_needs = {}
    for class_idx in target_classes:
        current_count = int(class_counts[class_idx])
        if current_count < target_count:
            needed = target_count - current_count
            augmentation_needs[class_idx] = needed
            print(f"Class {class_idx}: {current_count} -> {target_count} (need {needed} samples)")
        else:
            augmentation_needs[class_idx] = 0
    
    return augmentation_needs


def create_synthetic_annotations(synthetic_data_dir, sample_duration=5.0, sample_rate=16000):
    """
    Create annotation files for synthetic respiratory sound data.
    
    Args:
        synthetic_data_dir: Directory containing synthetic audio files
        sample_duration: Duration of each sample in seconds
        sample_rate: Sample rate of audio files
        
    Returns:
        Dict mapping filenames to annotation data
    """
    if not os.path.exists(synthetic_data_dir):
        return {}
    
    annotations = {}
    class_mapping = {1: "crackles", 2: "wheezes", 3: "both"}
    
    # Load generation summary if available
    summary_file = os.path.join(synthetic_data_dir, "generation_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            generation_summary = json.load(f)
        
        for class_idx_str, class_data in generation_summary.items():
            class_idx = int(class_idx_str)
            if class_idx in class_mapping:
                class_name = class_mapping[class_idx]
                
                for filepath in class_data.get('generated_files', []):
                    if os.path.exists(filepath):
                        filename = os.path.basename(filepath).replace('.wav', '')
                        
                        # Create annotation entry compatible with ICBHI format
                        annotations[filename] = [
                            {
                                'start': 0.0,
                                'end': sample_duration,
                                'crackles': 1 if class_idx in [1, 3] else 0,
                                'wheezes': 1 if class_idx in [2, 3] else 0,
                                'class_label': class_idx,
                                'synthetic': True
                            }
                        ]
    
    else:
        # Scan directory structure if no summary available
        for class_idx, class_name in class_mapping.items():
            class_dir = os.path.join(synthetic_data_dir, f"class_{class_idx}")
            if os.path.exists(class_dir):
                wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                
                for wav_file in wav_files:
                    filename = wav_file.replace('.wav', '')
                    annotations[filename] = [
                        {
                            'start': 0.0,
                            'end': sample_duration,
                            'crackles': 1 if class_idx in [1, 3] else 0,
                            'wheezes': 1 if class_idx in [2, 3] else 0,
                            'class_label': class_idx,
                            'synthetic': True
                        }
                    ]
    
    # Save annotations to file
    annotations_file = os.path.join(synthetic_data_dir, "synthetic_annotations.json")
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created annotations for {len(annotations)} synthetic samples")
    print(f"Annotations saved to {annotations_file}")
    
    return annotations


class ICBHISyntheticDataset(ICBHIDataset):
    """
    Enhanced ICBHI Dataset that incorporates AudioLDM-generated synthetic data
    for addressing class imbalance in respiratory sound classification.
    """
    
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False, 
                 synthetic_data_dir=None, augmentation_ratio=0.5, 
                 generate_on_demand=False, audioldm_model_id="cvssp/audioldm-s-full-v2"):
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
            generate_on_demand: Whether to generate synthetic data if not found
            audioldm_model_id: AudioLDM model ID for generation
        """
        # Initialize the parent class first
        super().__init__(train_flag, transform, args, print_flag, mean_std)
        
        self.synthetic_data_dir = synthetic_data_dir
        self.augmentation_ratio = augmentation_ratio
        self.generate_on_demand = generate_on_demand
        self.audioldm_model_id = audioldm_model_id
        self.synthetic_samples = []
        self.audioldm_generator = None
        
        # Store original class counts for comparison
        self.original_class_nums = self.class_nums.copy()
        
        # Load or generate synthetic data if this is training
        if train_flag:
            self._handle_synthetic_data(print_flag)
            
            if print_flag and len(self.synthetic_samples) > 0:
                print(f"Integrated {len(self.synthetic_samples)} synthetic samples")
                self._print_enhanced_statistics()
    
    def _handle_synthetic_data(self, print_flag):
        """Handle loading or generating synthetic data based on configuration."""
        # Try to load existing synthetic data
        if self.synthetic_data_dir and os.path.exists(self.synthetic_data_dir):
            self._load_synthetic_data()
            self._integrate_synthetic_data()
        
        # Generate new synthetic data if enabled and needed
        elif self.generate_on_demand and AUDIOLDM_AVAILABLE:
            if print_flag:
                print("Generating synthetic data using AudioLDM...")
            
            # Calculate augmentation needs
            augmentation_needs = calculate_augmentation_needs(
                self, 
                target_classes=[1, 2, 3], 
                balance_ratio=1.0 + self.augmentation_ratio
            )
            
            if any(count > 0 for count in augmentation_needs.values()):
                # Set up output directory
                if self.synthetic_data_dir is None:
                    self.synthetic_data_dir = os.path.join(
                        os.getcwd(), 
                        "synthetic_respiratory_data"
                    )
                
                self._generate_and_integrate_data(augmentation_needs)
            else:
                if print_flag:
                    print("No synthetic data generation needed - classes already balanced")
        
        elif self.generate_on_demand and not AUDIOLDM_AVAILABLE:
            print("Warning: AudioLDM generation requested but not available. Install diffusers.")
    
    def _generate_and_integrate_data(self, augmentation_needs):
        """Generate synthetic data using AudioLDM and integrate into dataset."""
        try:
            # Initialize AudioLDM generator
            self.audioldm_generator = AudioLDMGenerator(
                model_id=self.audioldm_model_id
            )
            
            # Generate balanced dataset
            generation_summary = self.audioldm_generator.generate_balanced_dataset(
                target_counts=augmentation_needs,
                output_dir=self.synthetic_data_dir,
                sample_duration=5.0
            )
            
            # Create annotations for generated data
            create_synthetic_annotations(
                synthetic_data_dir=self.synthetic_data_dir,
                sample_duration=5.0,
                sample_rate=self.sample_rate
            )
            
            # Load the generated data
            self._load_synthetic_data()
            self._integrate_synthetic_data()
            
            print("AudioLDM generation and integration completed successfully!")
            
        except Exception as e:
            print(f"Error during AudioLDM generation: {e}")
            print("Continuing without synthetic data generation...")
    
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
        
        # Use optimized balancing strategy for target classes only
        target_classes = [1, 2, 3]  # Crackles, Wheezes, Both
        synthetic_audio_data = self._optimize_class_balance(
            synthetic_by_class, 
            original_class_counts, 
            target_classes
        )
        
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
    
    def _optimize_class_balance(self, synthetic_by_class, original_class_counts, target_classes):
        """
        Optimize class balance using intelligent sampling strategy for target classes only.
        
        Args:
            synthetic_by_class: Dict of synthetic samples grouped by class
            original_class_counts: Original class distribution
            target_classes: Classes to augment (1=Crackles, 2=Wheezes, 3=Both)
            
        Returns:
            List of selected synthetic samples
        """
        synthetic_audio_data = []
        
        # Calculate the maximum count among target classes for balancing
        target_class_counts = [original_class_counts[i] for i in target_classes if original_class_counts[i] > 0]
        if not target_class_counts:
            return synthetic_audio_data
        
        max_target_count = max(target_class_counts)
        
        for class_idx in target_classes:
            current_count = original_class_counts[class_idx]
            available_synthetic = len(synthetic_by_class[class_idx])
            
            if current_count > 0 and available_synthetic > 0:
                # Strategy 1: Basic augmentation using augmentation_ratio
                basic_target = int(current_count * (1 + self.augmentation_ratio))
                
                # Strategy 2: Balance towards maximum target class
                balance_target = int(max_target_count * 0.8)  # 80% of max class
                
                # Use the more conservative approach
                target_count = min(basic_target, balance_target)
                samples_needed = min(target_count - int(current_count), available_synthetic)
                
                if samples_needed > 0:
                    # Intelligent sampling: prioritize diversity
                    if available_synthetic <= samples_needed:
                        # Use all available samples
                        selected_samples = synthetic_by_class[class_idx]
                    else:
                        # Randomly sample but ensure diversity
                        random.shuffle(synthetic_by_class[class_idx])
                        selected_samples = synthetic_by_class[class_idx][:samples_needed]
                    
                    # Convert to format compatible with original data
                    for sample in selected_samples:
                        synthetic_audio_data.append((sample['audio_data'], sample['label']))
                    
                    print(f"Added {len(selected_samples)} synthetic samples for class {class_idx} "
                          f"(from {int(current_count)} to {int(current_count) + len(selected_samples)})")
        
        return synthetic_audio_data
    
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
                          augmentation_ratio=0.5, print_flag=True, 
                          generate_on_demand=False, audioldm_model_id="cvssp/audioldm-s-full-v2"):
    """
    Factory function to create enhanced ICBHI dataset with synthetic data.
    
    Args:
        train_flag: Whether this is training data
        transform: Data transforms to apply
        args: Arguments object with dataset parameters
        synthetic_data_dir: Directory containing synthetic audio data
        augmentation_ratio: Ratio of synthetic to original samples
        print_flag: Whether to print dataset information
        generate_on_demand: Whether to generate synthetic data using AudioLDM
        audioldm_model_id: AudioLDM model ID for generation
        
    Returns:
        ICBHISyntheticDataset instance
    """
    return ICBHISyntheticDataset(
        train_flag=train_flag,
        transform=transform,
        args=args,
        print_flag=print_flag,
        synthetic_data_dir=synthetic_data_dir,
        augmentation_ratio=augmentation_ratio,
        generate_on_demand=generate_on_demand,
        audioldm_model_id=audioldm_model_id
    )


def run_audioldm_generation_standalone(data_folder, output_dir, target_classes=[1, 2, 3], 
                                     samples_per_class=50, balance_ratio=0.8):
    """
    Standalone function to run AudioLDM generation for respiratory sounds.
    
    Args:
        data_folder: Path to original ICBHI data
        output_dir: Directory to save generated synthetic data
        target_classes: List of class indices to augment
        samples_per_class: Number of samples to generate per class
        balance_ratio: Target balance ratio
        
    Returns:
        Generation summary dict
    """
    print("="*60)
    print("STANDALONE AUDIOLDM GENERATION FOR RESPIRATORY SOUNDS")
    print("="*60)
    
    try:
        # Initialize AudioLDM generator
        generator = AudioLDMGenerator()
        
        # Calculate generation targets
        target_counts = {class_idx: samples_per_class for class_idx in target_classes}
        
        # Generate synthetic data
        generation_summary = generator.generate_balanced_dataset(
            target_counts=target_counts,
            output_dir=output_dir,
            sample_duration=5.0
        )
        
        # Create annotations
        create_synthetic_annotations(
            synthetic_data_dir=output_dir,
            sample_duration=5.0,
            sample_rate=16000
        )
        
        print("="*60)
        print("AUDIOLDM GENERATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return generation_summary
        
    except Exception as e:
        print(f"Error during standalone AudioLDM generation: {e}")
        return None


if __name__ == "__main__":
    # Example usage for standalone generation
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic respiratory sounds using AudioLDM')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to ICBHI dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for synthetic data')
    parser.add_argument('--samples_per_class', type=int, default=50, help='Samples to generate per class')
    parser.add_argument('--target_classes', type=str, default='1,2,3', help='Classes to augment (1=crackles, 2=wheezes, 3=both)')
    
    args = parser.parse_args()
    target_classes = [int(x.strip()) for x in args.target_classes.split(',')]
    
    run_audioldm_generation_standalone(
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        target_classes=target_classes,
        samples_per_class=args.samples_per_class
    )