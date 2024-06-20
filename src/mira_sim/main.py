import argparse

from .metrics.clap import clap_eval as clap
from .metrics.kld import kld_eval as kld
from .metrics.coverid import coverid_eval as coverid
from .metrics.defnet import defnet_eval as defnet

def main():
    # Import information 
    parser = argparse.ArgumentParser(description='No help available.')

    # Input music 
        # Where are the songs located? 
        # groupA: reference group 
    parser.add_argument('a_samples', help='Indicate A samples directory.')

        # groupB: target group 
    parser.add_argument('b_samples', help='Indicate B samples directory.')

    # Evaluation metrics 
    m = ['clap', 'coverid', 'defnet', 'kld']
    parser.add_argument('--metrics', '-m', type=str, choices=m, help='Indicate the metric to run. If no argument, all of them will be concatenated.') 

    # What was the code name assigned for this evaluation? 
    parser.add_argument('--eval_name', help='Indicate eval name.', required=True)

    # Do you want to register this results into the log folder?  
    parser.add_argument('--log', help='Indicate if you do not want to register the results in the log folder.')

    # Only for KL divergence 
     # What combination are you testing? 
    parser.add_argument('--prelen', help='Indicate the pretraining length')
        

    args = parser.parse_args()

    folder_A = args.a_samples
    folder_B = args.b_samples
    eval_name = args.eval_name 
    log = args.log


    metric = args.metrics
    prelen = args.prelen

    if metric == 'clap': clap(folder_A, folder_B, eval_name, log)
    elif metric == 'defnet': defnet(folder_A, folder_B, eval_name, log)
    elif metric == 'coverid': coverid(folder_A, folder_B, eval_name, log)
    elif metric == 'kld': kld(folder_A, folder_B, eval_name, log, prelen)
        
    else: 
        # No metric was selected, run them all 
        coverid(folder_A, folder_B, eval_name, log)
        clap(folder_A, folder_B, eval_name, log)
        defnet(folder_A, folder_B, eval_name, log)
        kld(folder_A, folder_B, eval_name, log, prelen)

if __name__ == "__main__":
    main()

