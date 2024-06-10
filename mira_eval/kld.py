#########################################
# MIRA TOOL: KL DIV #####################
#########################################

# KL Divergence: Kullback-Leibler Divergence

# v1.0
# last update: june 2024

#########################################
# Import modules ########################
#########################################

import csv
import os
import numpy as np
import glob
import argparse
import torch
import torchaudio
from datetime import datetime
import librosa
import pandas as pd
from kldiv import kld_metric as kld

#######################################
# Similarity Evaluation ###############
#######################################

# Import information 
parser = argparse.ArgumentParser(description='No help available.')

# Set time count
now = datetime.now()

# Print computing status
print("\nCalculating KL divergence...")

##############################################################################################################################################################

# Input music 
    # Where are the songs located? 
    # groupA 
parser.add_argument('--a_samples', '-a', help='Indicate A samples directory.', required=True)

    # groupB 
parser.add_argument('--b_samples', '-b', help='Indicate B samples directory.', required=True)

    # What was the code name assigned for this evaluation? 
parser.add_argument('--eval_name', help='Indicate eval name.', required=True)

    # What combination are you testing? 
parser.add_argument('--prelen', help='Indicate the pretraining length')

# Do you want to register this results into the log folder?  
parser.add_argument('--log', help='Indicate if you do not want to register the results in the log folder.')

args = parser.parse_args()

folder_A = args.a_samples
folder_B = args.b_samples
eval_name = args.eval_name 
log = args.log
prelen = args.prelen

# Set if log is active by defining boolean LOG_ACTIVE
LOG_ACTIVE = False if log == 'no' else True

##############################################################################################################################################################

# Getting songs in the directory 
song_list_A = glob.glob(folder_A + '*.wav')[0:2]
song_list_B = glob.glob(folder_B + '*.wav')[0:2]

# Prepare variables
data_a = []
data_b = []

al_a = []
al_b  = []

sr_a = []
sr_b = []

kldiv_ab = []
all_kldiv_ab = []

kldiv_ba = []
all_kldiv_ba = []

# Define model 
model = kld.PasstKLDivergenceMetric(pretrained_length=prelen)

# Get A songs waveform characteristics
print("\nLoading baseline group (A)...")
for s in song_list_A: 
    # Load audio 
    y, sr = librosa.load(s, sr=None, mono=False, duration=None)
    # Convert to tensor
    audio_tensor = np.expand_dims(y, axis=0)
    audio_length = torch.tensor([audio_tensor.shape[-1]])
    data_a.append(torch.from_numpy(audio_tensor))
    al_a.append(audio_length)
    sr_a.append(sr)

target_probs_list = []
# Get B songs waveform characteristics
time_tp = datetime.now()
print("\nLoading target group (B) & computing probabilities...")
c = 0
for s in song_list_B: 
    print('  {}/{}'.format(c, len(song_list_B)))
    c += 1
    # Load audio 
    y, sr = librosa.load(s, sr=None, mono=False, duration=None)
    # Convert to tensor
    audio_tensor = np.expand_dims(y, axis=0)
    audio_length = torch.tensor([audio_tensor.shape[-1]])
    data_tensor = torch.from_numpy(audio_tensor)
    data_b.append(data_tensor)
    al_b.append(audio_length)
    sr_b.append(sr)
    target_probs = model._get_label_distribution(x=data_tensor, sizes=audio_length, sample_rates=sr)
    target_probs_list.append(target_probs)

print("Time elapsed to compute target probs: ", datetime.now()-time_tp)


# Calculate KL div for A - B
print("\nComputing KL divergence...") 
for a in range(len(data_a)): 
    print("{}/{}".format(a, len(song_list_A)))
    time_compute_begin = datetime.now()
    # Get reference probabilities from lable distribution of the PaSST classifier
    pred_probs = model._get_label_distribution(x=data_a[a], sizes=al_a[a], sample_rates=sr_a[a])
    for b in range(len(data_b)): 
            print("  {}/{}".format(b, len(song_list_B)))
            
            # Get target probabilities from label distribution of the PaSST classifier
            arget_probs = model._get_label_distribution(x=data_b[b], sizes=al_b[b], sample_rates=sr_b[b])
            target_probs = target_probs_list[b]
            
            # Compute KL divergence A-B, B-A and symmetric 
            time_kld = datetime.now()
            kl_div_ab = kld.kl_divergence(pred_probs, target_probs)
            time_kld_2 = datetime.now()
            kl_div_ba = kld.kl_divergence(target_probs, pred_probs)
            kl_div_symm = (kl_div_ab+kl_div_ba)/2

            kldiv_ab.append(kl_div_ab.mean().item())
            kldiv_ba.append(kl_div_ba.mean().item())
            all_kldiv_ab.append([song_list_A[a].split('/')[-1].split('.')[0], song_list_B[b].split('/')[-1].split('.')[0], kl_div_ab.mean().item(), kl_div_ba.mean().item(), kl_div_symm.mean().item()])
            

df = pd.DataFrame(all_kldiv_ab, columns = ['songA', 'songB', 'kl-div_AB', 'kl-div_BA', 'kl-div'])

# # Obtain KL div mean
# kl_div_ab = np.mean(kldiv_ab)
# kl_div_ba = np.mean(kldiv_ba)
# kldiv_mean = np.mean((kldiv_ab + kldiv_ba))
# kldiv_median = 


print('Results: ')
print('Mean KLDiv-AB:', df['kl-div_AB'].mean())
print('Mean KLDiv-BA:', df['kl-div_BA'].mean())
print('Mean KLDiv:', df['kl-div'].mean())
print('Median KLDiv:', df['kl-div'].median())

if LOG_ACTIVE is True: 
    # independent file with results
    with open('log/{}_kld.csv'.format(eval_name), 'w') as f: 
        writer = csv.writer(f)
        writer.writerow(['songA', 'songB', 'kl_div_ab', 'kl_div_ba', 'kl_div'])
        for r in range(len(all_kldiv_ab)): 
            writer.writerow([all_kldiv_ab[r][0], all_kldiv_ab[r][1], all_kldiv_ab[r][2], all_kldiv_ab[r][3], all_kldiv_ab[r][4]])

    # global file with all results 
    if os.path.exists('log/{}_allresults.csv'.format(eval_name)): 
        print('\nfile exists: adding KL divergence results')
        data = pd.read_csv('log/{}_allresults.csv'.format(eval_name)) 
        for r in all_kldiv_ab: 
            mask = (data['songA'] == r[0]) & (data['songB'] == r[1])
            data.loc[mask, 'kl-div_AB'] = r[2]
            data.loc[mask, 'kl-div_BA'] = r[3]
            data.loc[mask, 'kl-div'] = r[4]

    else: 
        print('\nmissing file: creating new file with KL divergence score results')
        data = df
    
    data.to_csv('log/{}_allresults.csv'.format(eval_name), index=False)


end = datetime.now()
time = end - now

print('Elapsed time: ', time)

print('\n** Done! **')