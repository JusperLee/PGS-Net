______________________________________________________________________

<div align="center">

# A fast and lightweight model for Causal Audio-Visual Speech Separation

[![arXiv](https://img.shields.io/badge/arXiv-2506.06689-brightgreen.svg)](https://arxiv.org/abs/2506.06689)
[![Samples](https://img.shields.io/badge/Website-Demo_Samples-blue.svg)](https://cslikai.cn/Swift-Net)

</div>

# Introduction

Here is the code for our proposed Swift-Net and some causal variants of the AVSS method. 

Ours: Swift-Net

Variants of the AVSS: av_convtasnet, av_dprnn, av_sepformer, av_tfgridenet, ctcnet, rtfsnet.


# Installation

Before you begin, ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system. Then, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/spkgyk/SwiftNet.git
   cd SwiftNet
   ```

2. Create a Conda environment with the required dependencies using the provided script:
   ```bash
   conda env create -f environment.yaml
   ```

**Note:** AVSS is a GPU-intensive task, so make sure you have access to a GPU for both installation and training.

# Datasets & Pretrained Video Model

1. Please download the pretrained video module used in our experiments by visiting [CTCNet](https://github.com/JusperLee/CTCNet)'s official repository. Instructions for downloading the various datasets used in our paper are also there, along with some additional scripts and utils for running the models in various real-world situations. 

2. Once the datasets are downloaded and formatted, please run the relevent scripts in the `data-preprocess` folder, i.e.
   ```bash
   python process_lrs23.py --in_audio_dir audio/wav16k/min --in_mouth_dir mouths --out_dir LRS2
   python process_lrs23.py --in_audio_dir audio/wav16k/min --in_mouth_dir mouths --out_dir LRS3
   python process_vox2.py --in_audio_dir audio/wav16k/min --in_mouth_dir mouths --out_dir VOX2
   ```

# Training & Evaluation

Training the AVSS model is a straightforward process using the `audio_train.py` script. You can customize the training by specifying a configuration file and, if necessary, a checkpoint for resuming training. Here's how to get started:

1. Run the training script with a default configuration file:
   ```bash
   python train.py --conf-dir config/lrs2_SwiftNet.yml
   ```

2. If you encounter unexpected interruptions during training and wish to resume from a checkpoint, use the following command (replace the checkpoint path with your specific checkpoint):
   ```bash
   python train.py --conf-dir config/lrs2_SwiftNet.yml \
   --checkpoint ../experiments/audio-visual/SwiftNet.yml/LRS2/4_layers/checkpoints/epoch=220-val_loss=-13.34.ckpt
   ```

Feel free to explore and fine-tune the configuration files in the `config` directory to suit your specific requirements and dataset.

Use the `audio_visual_test.py` script for evaluating your trained model on the test set. 

# Contact

If you have any questions, suggestions, or encounter any issues, please don't hesitate to reach out to us. You can contact us via email at `tsinghua.kaili [AT] gmail.com`.

## Citations

If you find this code useful in your research, please cite our work:
```
@misc{sang2025fastlightweightmodelcausal,
      title={A Fast and Lightweight Model for Causal Audio-Visual Speech Separation}, 
      author={Wendi Sang and Kai Li and Runxuan Yang and Jianqiang Huang and Xiaolin Hu},
      year={2025},
      eprint={2506.06689},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2506.06689}, 
}
```