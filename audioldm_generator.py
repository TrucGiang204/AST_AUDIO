#!/usr/bin/env python3
"""
AudioLDM-based Audio Generation for Respiratory Sound Augmentation

This script uses AudioLDM to generate synthetic respiratory sounds for the three minority classes:
- Crackles (Class 1)
- Wheezes (Class 2) 
- Both (Class 3)

The script includes:
1. Text prompt engineering for respiratory sounds
2. AudioLDM model loading and generation
3. Audio post-processing and quality control
4. Automatic annotation generation
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Dict, List, Tuple
from datetime import datetime
import random

# AudioLDM imports
from diffusers import AudioLDMPipeline
import warnings
warnings.filterwarnings("ignore")

class RespiratoryAudioGenerator:
    """AudioLDM-based generator for respiratory sounds."""
    
    def __init__(self, model_id="cvssp/audioldm-s-full-v2", device="auto"):
        """
        Initialize the AudioLDM generator.
        
        Args:
            model_id: HuggingFace model ID for AudioLDM
            device: Device to run the model on ('auto', 'cpu', 'cuda')
        """
        self.model_id = model_id
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing AudioLDM on device: {self.device}")
        
        # Load the AudioLDM pipeline
        try:
            self.pipe = AudioLDMPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipe = self.pipe.to(self.device)
            print("AudioLDM model loaded successfully!")
        except Exception as e:
            print(f"Error loading AudioLDM model: {e}")
            raise
        
        # Audio generation parameters
        self.sample_rate = 16000  # Target sample rate for ICBHI
        self.target_length = 8.0  # Target length in seconds
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        
        # Define text prompts for each class
        self.class_prompts = self._create_class_prompts()
        
    def _create_class_prompts(self) -> Dict[int, List[str]]:
        """Create diverse text prompts for each respiratory sound class."""
        
        prompts = {
            1: [  # Crackles
                "lung sounds with fine crackles, respiratory crackling noise, wet rales",
                "breathing with fine crackles, pulmonary crackles, wet lung sounds",
                "respiratory sounds with crackling, fine wet crackles in lungs",
                "lung crackling sounds, fine rales, wet respiratory noise",
                "breathing with fine wet crackles, pulmonary crackling",
                "respiratory crackling, wet lung rales, fine crackle sounds",
                "lung sounds with fine wet rales, respiratory crackling noise",
                "breathing crackling sounds, fine pulmonary crackles",
                "wet lung crackles, fine respiratory crackling",
                "pulmonary crackles, wet rales, fine lung crackling sounds"
            ],
            2: [  # Wheezes
                "lung sounds with wheezing, respiratory wheeze, high pitched wheeze",
                "breathing with wheeze, pulmonary wheezing sounds, musical wheeze",
                "respiratory wheezing, high frequency wheeze, lung wheeze sounds",
                "wheeze sounds, high pitched respiratory wheeze, pulmonary wheeze",
                "breathing with high pitched wheeze, respiratory wheezing noise",
                "lung wheezing, musical wheeze sounds, high frequency respiratory wheeze",
                "pulmonary wheeze, high pitched lung wheeze, respiratory wheezing",
                "wheeze breathing sounds, musical respiratory wheeze",
                "high pitched lung wheeze, respiratory wheezing sounds",
                "breathing wheeze, pulmonary wheezing, musical lung wheeze"
            ],
            3: [  # Both (Crackles + Wheezes)
                "lung sounds with both crackles and wheezes, mixed respiratory sounds",
                "breathing with crackles and wheeze, complex pulmonary sounds",
                "respiratory sounds with crackling and wheezing, mixed lung sounds",
                "lung crackles and wheeze, complex respiratory noise",
                "breathing with both wet crackles and wheeze, mixed pulmonary sounds",
                "respiratory crackling and wheezing, complex lung sounds",
                "lung sounds with fine crackles and high pitched wheeze",
                "breathing with wet rales and wheeze, complex respiratory sounds",
                "pulmonary crackles and wheezing, mixed lung abnormalities",
                "respiratory sounds with both crackling and musical wheeze"
            ]
        }
        
        return prompts
    
    def generate_single_audio(self, prompt: str, duration: float = None) -> Tuple[np.ndarray, int]:
        """
        Generate a single audio sample from a text prompt.
        
        Args:
            prompt: Text description of the desired audio
            duration: Duration in seconds (default: self.target_length)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if duration is None:
            duration = self.target_length
            
        try:
            # Generate audio using AudioLDM
            audio = self.pipe(
                prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                audio_length_in_s=duration
            ).audios[0]
            
            # Ensure audio is the right sample rate
            if hasattr(self.pipe, 'mel_spectrogram'):
                original_sr = self.pipe.mel_spectrogram.sampling_rate
            else:
                original_sr = 16000  # Default assumption
                
            if original_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                
            return audio, self.sample_rate
            
        except Exception as e:
            print(f"Error generating audio for prompt '{prompt}': {e}")
            return None, None
    
    def post_process_audio(self, audio: np.ndarray, target_length: float = None) -> np.ndarray:
        """
        Post-process generated audio to match ICBHI dataset characteristics.
        
        Args:
            audio: Input audio array
            target_length: Target length in seconds
            
        Returns:
            Processed audio array
        """
        if target_length is None:
            target_length = self.target_length
            
        target_samples = int(target_length * self.sample_rate)
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Trim or pad to target length
        if len(audio) > target_samples:
            # Random crop
            start_idx = random.randint(0, len(audio) - target_samples)
            audio = audio[start_idx:start_idx + target_samples]
        elif len(audio) < target_samples:
            # Pad with repetition (similar to ICBHI dataset approach)
            repeats = int(np.ceil(target_samples / len(audio)))
            audio = np.tile(audio, repeats)[:target_samples]
            
        # Apply fade in/out to avoid clicks
        fade_samples = int(0.05 * self.sample_rate)  # 50ms fade
        if len(audio) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
            
        return audio
    
    def generate_class_samples(self, class_idx: int, num_samples: int, output_dir: str) -> List[str]:
        """
        Generate multiple samples for a specific class.
        
        Args:
            class_idx: Class index (1=crackles, 2=wheezes, 3=both)
            num_samples: Number of samples to generate
            output_dir: Output directory for generated audio files
            
        Returns:
            List of generated audio file paths
        """
        if class_idx not in self.class_prompts:
            raise ValueError(f"Invalid class index: {class_idx}")
            
        print(f"Generating {num_samples} samples for class {class_idx}...")
        
        class_names = {1: "crackles", 2: "wheezes", 3: "both"}
        class_name = class_names[class_idx]
        
        # Create output directory
        class_output_dir = os.path.join(output_dir, f"class_{class_idx}_{class_name}")
        os.makedirs(class_output_dir, exist_ok=True)
        
        generated_files = []
        prompts = self.class_prompts[class_idx]
        
        for i in range(num_samples):
            # Select prompt (cycle through available prompts)
            prompt = prompts[i % len(prompts)]
            
            # Add variation to prompt
            variations = [
                f"{prompt}",
                f"{prompt}, clear audio recording",
                f"{prompt}, medical lung examination",
                f"{prompt}, stethoscope recording",
                f"{prompt}, clinical respiratory sounds"
            ]
            selected_prompt = variations[i % len(variations)]
            
            print(f"  Sample {i+1}/{num_samples}: '{selected_prompt[:50]}...'")
            
            # Generate audio
            audio, sr = self.generate_single_audio(selected_prompt)
            
            if audio is not None:
                # Post-process audio
                audio = self.post_process_audio(audio)
                
                # Save audio file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"synthetic_{class_name}_{timestamp}_{i:04d}.wav"
                filepath = os.path.join(class_output_dir, filename)
                
                sf.write(filepath, audio, sr)
                generated_files.append(filepath)
                
                print(f"    Saved: {filename}")
            else:
                print(f"    Failed to generate sample {i+1}")
                
        print(f"Generated {len(generated_files)} samples for class {class_idx}\n")
        return generated_files
    
    def create_annotations(self, generated_files: List[str], class_idx: int, output_dir: str) -> str:
        """
        Create annotation files for generated audio samples.
        
        Args:
            generated_files: List of generated audio file paths
            class_idx: Class index
            output_dir: Output directory for annotations
            
        Returns:
            Path to the annotation file
        """
        class_names = {1: "crackles", 2: "wheezes", 3: "both"}
        class_name = class_names[class_idx]
        
        # Create annotations directory
        ann_dir = os.path.join(output_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        
        # Create annotation file
        ann_file = os.path.join(ann_dir, f"synthetic_{class_name}_annotations.txt")
        
        annotations = []
        for filepath in generated_files:
            filename = os.path.basename(filepath).replace('.wav', '')
            
            # Create annotation entry (start_time, end_time, crackles, wheezes)
            # For synthetic data, we assume the entire duration contains the target sound
            if class_idx == 1:  # Crackles
                crackles, wheezes = 1, 0
            elif class_idx == 2:  # Wheezes  
                crackles, wheezes = 0, 1
            else:  # Both
                crackles, wheezes = 1, 1
                
            # Annotation format: start_time \t end_time \t crackles \t wheezes
            annotation = f"0.0\t{self.target_length:.1f}\t{crackles}\t{wheezes}"
            annotations.append(f"{filename}: {annotation}")
            
            # Also create individual .txt file for compatibility
            txt_file = filepath.replace('.wav', '.txt')
            with open(txt_file, 'w') as f:
                f.write(annotation)
        
        # Save summary annotation file
        with open(ann_file, 'w') as f:
            f.write('\n'.join(annotations))
            
        print(f"Created annotations: {ann_file}")
        return ann_file
    
    def generate_augmentation_dataset(self, augmentation_plan: Dict, output_dir: str) -> Dict:
        """
        Generate augmentation dataset based on class imbalance analysis.
        
        Args:
            augmentation_plan: Dictionary with class_idx -> num_samples_needed
            output_dir: Output directory for generated data
            
        Returns:
            Dictionary with generation results
        """
        print("Starting AudioLDM-based data augmentation...")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for class_idx, num_samples in augmentation_plan.items():
            if class_idx in [1, 2, 3]:  # Only generate for target classes
                print(f"\n{'='*50}")
                print(f"Generating augmentation for Class {class_idx}")
                print(f"{'='*50}")
                
                generated_files = self.generate_class_samples(
                    class_idx=class_idx,
                    num_samples=num_samples,
                    output_dir=output_dir
                )
                
                annotation_file = self.create_annotations(
                    generated_files=generated_files,
                    class_idx=class_idx,
                    output_dir=output_dir
                )
                
                results[class_idx] = {
                    'num_generated': len(generated_files),
                    'generated_files': generated_files,
                    'annotation_file': annotation_file
                }
        
        # Save generation summary
        summary_file = os.path.join(output_dir, "generation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n{'='*50}")
        print("AUGMENTATION COMPLETE")
        print(f"{'='*50}")
        print(f"Summary saved to: {summary_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate respiratory sound augmentation using AudioLDM')
    parser.add_argument('--output_dir', type=str, default='./synthetic_respiratory_sounds',
                       help='Output directory for generated audio files')
    parser.add_argument('--model_id', type=str, default='cvssp/audioldm-s-full-v2',
                       help='AudioLDM model ID from HuggingFace')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--samples_per_class', type=int, default=50,
                       help='Number of samples to generate per class')
    parser.add_argument('--target_classes', nargs='+', type=int, default=[1, 2, 3],
                       help='Class indices to generate (1=crackles, 2=wheezes, 3=both)')
    
    args = parser.parse_args()
    
    # Initialize generator
    print("Initializing AudioLDM Respiratory Sound Generator...")
    generator = RespiratoryAudioGenerator(
        model_id=args.model_id,
        device=args.device
    )
    
    # Create augmentation plan
    augmentation_plan = {}
    for class_idx in args.target_classes:
        augmentation_plan[class_idx] = args.samples_per_class
    
    print(f"\nAugmentation plan: {augmentation_plan}")
    
    # Generate augmentation dataset
    results = generator.generate_augmentation_dataset(
        augmentation_plan=augmentation_plan,
        output_dir=args.output_dir
    )
    
    # Print results summary
    print(f"\n{'='*50}")
    print("GENERATION RESULTS")
    print(f"{'='*50}")
    
    total_generated = 0
    for class_idx, result in results.items():
        class_names = {1: "crackles", 2: "wheezes", 3: "both"}
        print(f"Class {class_idx} ({class_names[class_idx]}): {result['num_generated']} samples")
        total_generated += result['num_generated']
        
    print(f"\nTotal generated samples: {total_generated}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()