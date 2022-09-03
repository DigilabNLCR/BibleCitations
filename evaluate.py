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
    --> + creates 'FILTERED_BY_STOP_SUBS.csv' file (so you can check these results, too)
4) Drop 'hidden' duplicates. These are the duplicate results that have formally different query string, but actually one of the query strings contains the other.
    --> creates 'DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv' file
5) Marking multiple attributions. In this case the multiple attributions are not dropped, but kept with a column that suggest if the result should be dropped or not.
    --> creates 'MA_DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv' file
6) Marking 'sure' citations:
    --> creates 'FINAL_MA_DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv' file

7) Check the results by yourselves ;-)

NOTE: All of the files are created in the process so you can explore the development of the evaluation scheme.
"""
import biblical_intertextuality_package as bip

# # step 1) - drop duplicates
# bip.make_unfiltered_search_dataframe(results_filename='batch_results.csv')
# # step 2) - initial filter
# bip.make_filtered_search_dataframe(results_filename='UNFILTERED_batch_results.csv')
# step 3) - drop stop-subverses
bip.filter_stop_subs(results_filename='FILTERED_UNFILTERED_batch_results.csv')
# step 4) - drop hidden duplicates
bip.filter_duplicates_by_overlap(results_filename='ST_SUBS_FILTERED_UNFILTERED_batch_results.csv')
# step 5) - marking multiple attributions
bip.mark_multiple_attributions(results_filename='DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv')
# step 6) - marking "sure" citations
bip.mark_sure_citations(results_filename='MA_DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv')
# step 7) - check it by yourselves
print('The evaluation has been run, now chcek the results by yourselves ;-)')
print('You can find the results file in "results/FINAL_MA_DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv"')
print('\t- All other steps of the filtering process are also save in the "results" directory')