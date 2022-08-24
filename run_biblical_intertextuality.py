"""
This script executes the search for biblical citations using functions of 'biblical_intertextuality_package.py'

- prior to running this script, you must run following scripts:
1) prepare_environment.py
2) prapare_bible.py (unless using the already prepared 'completeBible' files (in datasets, corpuses and dictionaries))
3) prepare_query_documents.py
4) prapare_batches.py

bip.search_by_batches() saves the results to 'results/batch_results.csv' after each batch is searched (by default max 40 documents)
"""

import biblical_intertextuality_package as bip

batches = range(0, 4186)
bip.search_by_batches(batches, bible_dataset_filename='fullBibleDataset', query_window_len=6, query_overlap=1)
