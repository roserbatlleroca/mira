# MiRA :eyes:

MiRA (<p style="color:orange">**M**</p>us<p style="color:orange">**i**</p>c <p style="color:orange">**R**</p>eplication <p style="color:orange">**A**</p>ssessment) tool is a model-independent open evaluation method based on four diverse audio music similarity metrics to assess exact data replication of the training set. 

For detailed description of the MiRA tool, check out our article [Towards Assessing Data Replication in Music Generation with Music Similarity Metrics on Raw Audio](url-missing). 


## quick start (todo!)

Install MiRA PyPi package: 
```
pip install mira
```

**how to use MiRA? (todo)**

< instructions >

< examples >


## environment settings
If you want to deep in MiRA tool instead of directly using the pip library, you need to install the appropiate environment. Please, run the following commands: 

**clone github repository:**
```
git clone https://github.com/roserbatlleroca/mira.git
cd mira
```

**create and install conda environment**
```
conda create --name mira python=3.10
conda activate mira
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**to run KL divergence download [PaSST](https://github.com/kkoutini/PaSST?tab=readme-ov-file#passt-efficient-training-of-audio-transformers-with-patchout) classifier**
```
pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19'
```

**to run CLAP and DEfNet scores install pythorch:**

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html 

# note that you can also install pytorch by following the official instruction (https://pytorch.org/get-started/locally/)
### for H100 GPU: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CLAP-LAION, it is necessary to download the model and specify it's location at model.load_ckpt: 
https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt


Download Discogs-EffNet track scores and specify it's location: 
https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_track_embeddings-effnet-bs64-1.pb

## contact & citation 

Contact Roser Batlle Roca at <roser.batlle@upf.edu>

```
@article{batlleroca2024towards,
  title={Towards Assessing Data Replication in Music Generation with Music Similarity Metrics on Raw Audio},
  author={Batlle-Roca, Roser and Liao, Wei-Hsiang and Serra, Xavier
  and Mitsufuji, Yuki and Gómez, Emilia},
  journal={arXiv preprint arXiv:tbd},
  year={2024}
}
```

