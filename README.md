# Solaris: A Foundation Model of the Sun

This repository contains the implementation code for the paper:

**[Solaris: A Foundation Model of the Sun](https://arxiv.org/abs/2411.16339v1)**

## Abstract

Foundation models have demonstrated remarkable success across various scientific domains, motivating our exploration of their potential in solar physics. In this paper, we present Solaris, the first foundation model for forecasting the Sun's atmosphere. We leverage 13 years of full-disk, multi-wavelength solar imagery from the Solar Dynamics Observatory, spanning a complete solar cycle, to pre-train Solaris for 12-hour interval forecasting. Solaris is built on a large-scale 3D Swin Transformer architecture with 109 million parameters. We demonstrate Solaris' ability to generalize by fine-tuning on a low-data regime using a single wavelength (1700 Å), that was not included in pre-training, outperforming models trained from scratch on this specific wavelength. Our results indicate that Solaris can effectively capture the complex dynamics of the solar atmosphere and transform solar forecasting.

## Repository Structure

```
solaris/                                    # Main model package
├── model/
│   ├── __init__.py                        # Package initialization
│   ├── solaris.py                         # Main Solaris model implementation
│   ├── encoder.py                         # Encoder components
│   ├── decoder.py                         # Decoder components
│   ├── perceiver.py                       # Perceiver architecture
│   ├── swin3d.py                          # 3D Swin Transformer implementation
│   ├── patchembed.py                      # Patch embedding utilities
│   ├── posencoding.py                     # Positional encoding
│   ├── fourier.py                         # Fourier transform utilities
│   ├── film.py                            # Feature-wise Linear Modulation
│   ├── lora.py                            # LoRA (Low-Rank Adaptation) implementation
│   └── util.py                            # Utility functions
├── __init__.py                            # Package initialization
├── train.py                               # Main training script
├── train_old.py                           # Legacy training script
├── load_data.py                           # Data loading utilities
├── load_data_prov.py                      # Provisional data loading
├── utils_data.py                          # Data utility functions
├── clean_data.py                          # Data cleaning utilities
├── download_data.py                       # Data downloading utilities
├── normalization.py                       # Data normalization
├── optimizer.py                           # Optimizer configurations
├── pretrain_metrics.py                    # Pre-training metrics
└── downstreamtask_metric.py              # Downstream task metrics

scripts/                                   # Data processing utilities
├── process_aia_synoptic_files.py          # Process AIA synoptic data files
├── generate_aia_synoptic_urls.py          # Generate URLs for AIA data download
├── upload_to_huggingface.py               # Upload processed data to HuggingFace
├── how_to_download_aia_synoptic_data.md   # Guide for downloading AIA data
└── wavelength_statistics.csv             # Wavelength-specific statistics

.gitignore                                 # Git ignore file
README.md                                  # This file
```

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{abdulmajid2024solaris,
  title={Solaris: A Foundation Model of the Sun},
  author={Abdul Majid, Harris and Sittoni, Pietro and Tudisco, Francesco},
  journal={arXiv preprint arXiv:2411.16339},
  year={2024}
}
```
