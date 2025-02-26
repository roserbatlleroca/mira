# MiRA 👀

MiRA (**M**us**i**c **R**eplication **A**ssessment) tool is a model-independent open evaluation method based on four diverse audio music similarity metrics to assess exact data replication of the training set. 

For a detailed description of the MiRA tool, check out our article [Towards Assessing Data Replication in Music Generation with Music Similarity Metrics on Raw Audio](https://zenodo.org/records/14877501). 


## 🚀 quick start 

**create and install conda environment**
```
conda create --name mira python=3.10
conda activate mira
python -m pip install --upgrade pip
```

**install [mira package](https://pypi.org/project/mira-sim/)**
```
pip install mira-sim
```

**to run KL divergence download [PaSST](https://github.com/kkoutini/PaSST?tab=readme-ov-file#passt-efficient-training-of-audio-transformers-with-patchout) classifier**

```
pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19'
```

**to run CLAP and DEfNet scores install pythorch...**

```
pip3 install torch torchvision torchaudio 

# note that you can also install pytorch by following the official instructions (https://pytorch.org/get-started/locally/)
```

**... and download corresponding models**

```
mkdir misc/ 
wget -O misc/music_audioset_epoch_15_esc_90.14.pt https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt?download=true 
wget -O misc/discogs_track_embeddings-effnet-bs64-1.pb https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_track_embeddings-effnet-bs64-1.pb
```

🚧 **Attention!** MiRA expects to find weights in `misc` folder in the directory you run mira. Note that if you would like to store the models elsewhere, you MUST change the location directory `model_path` at files [clap.py](mira/metrics/clap.py) and [defnet.py](mira/metrics/defnet.py). 


## 🧪 how to use MiRA?

Run an evaluation by calling `mira` and indicating
the directory of the reference folder (`reference_foldr`), the target folder (`target_folder`) and name of the evaluation or test (`eval_name`). 

Registering results (`log`) is active by default. You can deactivate storing the results by setting log to `no` or you can specify your preferred directory (`log_directory`). If you do not specify any `log` folder where results should be stored, MiRA will automatically create a `log` folder in the current directory.  

MiRA will run the evaluation between the samples in the reference and target folder for four music similarity metrics: CLAP score, DEfNet score, Cover Identification (CoverID) and KL divergence. However, you can specify a metric with `-m` argument. 

```
mira <reference_folder> <target_folder> --eval_name <eval_name> {--log <no/log_directory> -m <clap,defnet,coverid,kld>}
```

:warning: **Important!** Note that MiRA is prepared to interpret `wav` files.  


## 💻 directly running evaluation metrics
If you want to deep in MiRA tool instead of directly using the pip library, you need to install the appropriate environment. Please, run the following commands: 

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

**download [PaSST](https://github.com/kkoutini/PaSST?tab=readme-ov-file#passt-efficient-training-of-audio-transformers-with-patchout) classifier**
```
pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19'
```

**install pythorch and models' weights**

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html 

# note that you can also install pytorch by following the official instructions (https://pytorch.org/get-started/locally/)

cd misc/ 
wget https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt
wget https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs_track_embeddings-effnet-bs64-1.pb
```
🚧 **Attention!** Note that if you would like to store the models elsewhere, you MUST change the location directory `model_path` at files [clap.py](mira/metrics/clap.py) and [defnet.py](mira/metrics/defnet.py). 

### running evaluation metrics

To execute the evaluation metrics, run the following command specifying the metric (`coverid`, `clap`, `defnet` or `kld`), the directory of the reference folder (`a_dir`), the target folder (`b_dir`) and name of the evaluation or test (`eval_name`). Log is active by default. You can deactivate storing the results by setting `log` to `no`.  

```
cd src/mira_sim/
python metrics/<metric_name>.py -a <a_dir> -b <b_dir> --eval_name <eval_name> {--log <no/folder_directory>}
```

For KL divergence, you can also specify pretraining length by adding `--prelen <10,20,30>`. By default, it is set to 10 seconds as the original PaSST was trained on AudioSet with 10s-long audio samples. 

## 📚 citation 

```
@inproceedings{batlleroca2024towards,
  author       = {Roser Batlle{-}Roca and Wei{-}Hsiang Liao and Xavier Serra and Yuki Mitsufuji and Emilia G{\'{o}}mez},
  title        = {Towards Assessing Data Replication in Music Generation With Music Similarity Metrics on Raw Audio},
  booktitle    = {Proceedings of the 25th International Society for Music Information Retrieval Conference, {ISMIR} 2024, San Francisco, California, {USA}, November 10-14, 2024},
  pages        = {1004--1011},
  year         = {2024},
  url          = {https://doi.org/10.5281/zenodo.14877501},
  doi          = {10.5281/ZENODO.14877501}
}
```

