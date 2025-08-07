# AudioLDM Integration for ADD-RSC: Synthetic Data Augmentation

This document describes the complete integration of AudioLDM for generating synthetic respiratory sounds to address class imbalance in the ICBHI dataset for the ADD-RSC model.

## 🔥 Overview

The AudioLDM integration provides a sophisticated solution for handling class imbalance in respiratory sound classification by:

1. **Analyzing data distribution** to identify minority classes
2. **Generating high-quality synthetic audio** using AudioLDM text-to-audio generation
3. **Automatically creating annotations** for synthetic data
4. **Seamlessly integrating** synthetic samples with the original dataset
5. **Training with balanced data** to improve model performance

## 🏗️ Architecture

```
Original ICBHI Dataset
         ↓
   Data Analysis
         ↓
   AudioLDM Generation ← Text Prompts (Crackles, Wheezes, Both)
         ↓
   Synthetic Samples
         ↓
   Enhanced Dataset (Original + Synthetic)
         ↓
   ADD-RSC Training
```

## 📋 Requirements

### Core Dependencies
```bash
# Install with break-system-packages flag (if needed)
pip install --break-system-packages \
    diffusers transformers accelerate scipy \
    librosa soundfile matplotlib
```

### Hardware Requirements
- **GPU**: Recommended for AudioLDM generation (CUDA-compatible)
- **RAM**: At least 8GB for model loading and data processing
- **Storage**: Additional space for synthetic audio files (~1GB per 1000 samples)

## 🚀 Quick Start

### Step 1: Analyze Data Distribution
```bash
python analyze_data_distribution.py --data_folder /path/to/ICBHI_final_database
```

### Step 2: Generate Synthetic Data
```bash
python audioldm_generator.py \
    --output_dir ./synthetic_respiratory_sounds \
    --samples_per_class 100 \
    --target_classes 1 2 3
```

### Step 3: Complete Pipeline
```bash
python train_with_audioldm.py \
    --data_folder /path/to/ICBHI_final_database \
    --analyze_data \
    --generate_synthetic \
    --train_model \
    --augmentation_ratio 0.5
```

## 📖 Detailed Usage

### 1. Data Analysis (`analyze_data_distribution.py`)

Analyzes the current ICBHI dataset to understand class imbalance:

```bash
python analyze_data_distribution.py [OPTIONS]

Options:
  --data_folder PATH    Path to ICBHI dataset folder
```

**Output:**
- Class distribution statistics
- Imbalance ratios
- Augmentation recommendations
- Visualization plot (`data_distribution.png`)

### 2. AudioLDM Generation (`audioldm_generator.py`)

Generates synthetic respiratory sounds using AudioLDM:

```bash
python audioldm_generator.py [OPTIONS]

Key Options:
  --output_dir PATH           Output directory for synthetic audio
  --model_id MODEL           AudioLDM model ID (default: cvssp/audioldm-s-full-v2)
  --samples_per_class N      Number of samples per class (default: 50)
  --target_classes LIST      Classes to generate [1 2 3]
  --device DEVICE           Device to use (auto/cpu/cuda)
```

**Generated Structure:**
```
synthetic_respiratory_sounds/
├── class_1_crackles/
│   ├── synthetic_crackles_*.wav
│   └── synthetic_crackles_*.txt
├── class_2_wheezes/
│   ├── synthetic_wheezes_*.wav
│   └── synthetic_wheezes_*.txt
├── class_3_both/
│   ├── synthetic_both_*.wav
│   └── synthetic_both_*.txt
├── annotations/
│   ├── synthetic_crackles_annotations.txt
│   ├── synthetic_wheezes_annotations.txt
│   └── synthetic_both_annotations.txt
└── generation_summary.json
```

### 3. Complete Pipeline (`train_with_audioldm.py`)

Runs the end-to-end pipeline with multiple stages:

```bash
python train_with_audioldm.py [OPTIONS]

Pipeline Control:
  --analyze_data            Analyze data distribution
  --generate_synthetic      Generate synthetic data
  --train_model            Train ADD-RSC model

Synthetic Data Options:
  --augmentation_ratio R    Target augmentation ratio (default: 0.5)
  --synthetic_ratio R       Ratio of synthetic to original in training (default: 0.3)
  --max_synthetic_per_class N  Max synthetic samples per class (default: 200)

Model Options:
  --model MODEL            Model architecture (ast/resnet50)
  --batch_size N           Batch size (default: 8)
  --epochs N               Training epochs (default: 50)
```

## 🎯 AudioLDM Text Prompts

The system uses carefully crafted text prompts for each respiratory sound class:

### Crackles (Class 1)
- "lung sounds with fine crackles, respiratory crackling noise, wet rales"
- "breathing with fine crackles, pulmonary crackles, wet lung sounds"
- "respiratory sounds with crackling, fine wet crackles in lungs"

### Wheezes (Class 2)
- "lung sounds with wheezing, respiratory wheeze, high pitched wheeze"
- "breathing with wheeze, pulmonary wheezing sounds, musical wheeze"
- "respiratory wheezing, high frequency wheeze, lung wheeze sounds"

### Both (Class 3)
- "lung sounds with both crackles and wheezes, mixed respiratory sounds"
- "breathing with crackles and wheeze, complex pulmonary sounds"
- "respiratory sounds with crackling and wheezing, mixed lung sounds"

## 🔧 Enhanced Dataset Class

The `ICBHISyntheticDataset` extends the original `ICBHIDataset` with:

```python
from util.icbhi_synthetic_dataset import create_enhanced_dataset

# Create enhanced training dataset
train_dataset = create_enhanced_dataset(
    train_flag=True,
    transform=train_transform,
    args=args,
    synthetic_data_dir="./synthetic_respiratory_sounds",
    augmentation_ratio=0.3,
    print_flag=True
)
```

**Features:**
- Automatic loading of synthetic data
- Balanced integration with original samples
- Compatible with existing preprocessing pipeline
- Maintains annotation format consistency

## 📊 Expected Results

### Before Augmentation (Example)
```
Class 0 (normal):   1000 samples (60.0%)
Class 1 (crackle):   300 samples (18.0%)
Class 2 (wheeze):    200 samples (12.0%)
Class 3 (both):      167 samples (10.0%)
```

### After AudioLDM Augmentation
```
Class 0 (normal):   1000 samples (50.0%)
Class 1 (crackle):   450 samples (22.5%)  (+50% synthetic)
Class 2 (wheeze):    350 samples (17.5%)  (+75% synthetic)
Class 3 (both):      200 samples (10.0%)  (+20% synthetic)
```

## 🎛️ Configuration Options

### AudioLDM Parameters
```python
# In audioldm_generator.py
self.sample_rate = 16000        # Target sample rate for ICBHI
self.target_length = 8.0        # Target length in seconds
self.num_inference_steps = 50   # Quality vs speed tradeoff
self.guidance_scale = 7.5       # Text adherence strength
```

### Augmentation Parameters
```python
# In train_with_audioldm.py
--augmentation_ratio 0.5        # How much to augment minority classes
--synthetic_ratio 0.3           # Ratio of synthetic to original in training
--max_synthetic_per_class 200   # Cap on synthetic samples per class
```

## 🔍 Quality Control

The pipeline includes several quality control measures:

1. **Audio Post-processing:**
   - Normalization to prevent clipping
   - Fade in/out to avoid clicks
   - Length standardization (8 seconds)
   - Sample rate conversion (16kHz)

2. **Annotation Validation:**
   - Automatic annotation generation
   - Format consistency with ICBHI
   - Individual and batch annotation files

3. **Integration Checks:**
   - Balanced sampling across classes
   - Compatible data format
   - Error handling for failed generations

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use CPU for generation
python audioldm_generator.py --device cpu
```

**2. AudioLDM Model Loading Fails**
```bash
# Try smaller model
python audioldm_generator.py --model_id cvssp/audioldm-s-full
```

**3. Dataset Path Issues**
```bash
# Verify ICBHI dataset structure
ls -la /path/to/ICBHI_final_database/
# Should contain .wav and .txt files
```

**4. Synthetic Data Not Loading**
```bash
# Check generation summary exists
cat ./synthetic_respiratory_sounds/generation_summary.json
```

### Performance Optimization

**1. Faster Generation:**
```bash
# Reduce inference steps (lower quality)
python audioldm_generator.py --num_inference_steps 20
```

**2. Memory Optimization:**
```bash
# Reduce batch size
python train_with_audioldm.py --batch_size 4
```

**3. Storage Optimization:**
```bash
# Limit synthetic samples
python audioldm_generator.py --samples_per_class 50
```

## 📈 Evaluation

### Metrics to Monitor
- **Class Balance:** Check distribution after augmentation
- **Model Performance:** Compare accuracy on validation set
- **Synthetic Quality:** Listen to generated samples manually
- **Training Stability:** Monitor loss curves

### Expected Improvements
- **Better Recall** for minority classes (crackles, wheezes, both)
- **More Balanced Precision** across all classes
- **Higher Overall Score** (harmonic mean of Sp and Se)
- **Reduced Overfitting** to majority class

## 🔮 Advanced Usage

### Custom Text Prompts
```python
# Modify prompts in audioldm_generator.py
custom_prompts = {
    1: ["your custom crackle prompts"],
    2: ["your custom wheeze prompts"],
    3: ["your custom both prompts"]
}
```

### Integration with Existing Training
```python
# In your existing training script
from util.icbhi_synthetic_dataset import ICBHISyntheticDataset

# Replace ICBHIDataset with enhanced version
train_dataset = ICBHISyntheticDataset(
    train_flag=True,
    transform=transform,
    args=args,
    synthetic_data_dir="./synthetic_respiratory_sounds",
    augmentation_ratio=0.3
)
```

### Batch Generation
```bash
# Generate multiple sets with different parameters
for ratio in 0.3 0.5 0.7; do
    python audioldm_generator.py \
        --output_dir "./synthetic_ratio_${ratio}" \
        --samples_per_class $((100 * ratio))
done
```

## 📝 Citation

If you use this AudioLDM integration in your research, please cite:

```bibtex
@article{dong2025adaptive,
  title={Adaptive Differential Denoising for Respiratory Sounds Classification with AudioLDM Augmentation},
  author={Dong, Gaoyang and Zhang, Zhicheng and Sun, Ping and Zhang, Minghui},
  journal={arXiv preprint arXiv:2506.02505},
  year={2025}
}
```

## 🤝 Contributing

Feel free to contribute improvements to the AudioLDM integration:

1. **Better Text Prompts:** More diverse and medical-accurate prompts
2. **Quality Metrics:** Automated quality assessment for synthetic audio
3. **Model Variants:** Support for other text-to-audio models
4. **Optimization:** Memory and speed improvements

## 📞 Support

For issues specific to the AudioLDM integration:

1. Check the troubleshooting section above
2. Verify your AudioLDM model and dependencies
3. Test with smaller datasets first
4. Monitor GPU memory usage during generation

---

**🫁 Happy training with balanced respiratory sound data!**