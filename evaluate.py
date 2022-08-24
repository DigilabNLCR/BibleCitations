"""
This python file serves to run evaluation functions on results from file 'batch_results.csv' 

The evaluation process can be explored (with further description) and run step by step in the jupyther notebook 'evaluation.ipynb'

The evaluation scheme consists of the following steps:
1) Drop duplicate results and transform the initial 'batch_results.csv' file into CSV file with different structure (more info needed for evaluation).
    --> creates 'UNFILTERED_batch_results.csv' file
2) Connecting consequential results and calculting approximate match probabilities.
    --> creates 'FILTERED_UNFILTERED_batch_results.csv' file
3) Filtering stop-subverses.
    --> creates 'ST_SUBS_FILTERED_UNFILTERED_batch_results.csv' file
4) Drop 'hidden' duplicates.
    --> creates 'DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv' file
5) Resolving multiple attributions.
    --> creates 'MA_DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv' file
6) Marking 'sure' citations:
    --> creates 'FINAL_MA_DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv' file
7) Check the results by yourselves ;-)

NOTE: All of the files are created in the process so you can explore the development of the evaluation scheme.
"""
import biblical_intertextuality_package as bip

# TODO: in preparation