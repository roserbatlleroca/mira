# MiRA

## clone MiRA repository 

Clone GitHub repository:
```
git clone https://github.com/roserbatlleroca/mira.git
```

## environment settings
If you want to deep in MiRA tool instead of directly using the pip library, you need to install the appropiate environment. Please, run the following commands: 

```
# clone github repository:
git clone https://github.com/roserbatlleroca/mira.git

# create and install conda environment 
conda create --name mira python=3.10
conda activate mira
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**to run KL divergence**
```
pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19'
```

**to run CLAP score**
```
# install pythorch
# note that you can also install pytorch by following the official instruction (https://pytorch.org/get-started/locally/)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html 

## for H100 GPU: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

It is necessary to download the model and specify it's location at model.load_ckpt: 
https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt

