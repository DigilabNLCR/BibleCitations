""" This python script serves to prepare the folders necessary for the right functioning of the Biblical intertextuality script. """
import os

""" defining paths: """

ROOT_PATH = os.getcwd()

BIBLES_PATH = os.path.join(ROOT_PATH, 'Bible_files')
QUERY_DOC_PATH = os.path.join(ROOT_PATH, 'query_documents')
DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')
DICTS_PATH = os.path.join(ROOT_PATH, 'dictionaries')
CORPUS_PATH = os.path.join(ROOT_PATH, 'corpuses')
RESULTS_PATH = os.path.join(ROOT_PATH, 'results')
ALL_JSONS_PATH = os.path.join(ROOT_PATH, 'query_jsons_archive')

BATCHES_FILE_PATH = os.path.join(ROOT_PATH, 'batches.csv')
BATCH_RESULTS_FILE_PATH = os.path.join(RESULTS_PATH, 'batch_results.csv')

STOP_WORDS_PATH = os.path.join(ROOT_PATH, 'stop_words.txt')
STOP_SUBVERSES_PATH = os.path.join(ROOT_PATH, 'stop_subverses_21.txt')
EXCLUSIVES_PATH = os.path.join(ROOT_PATH, 'exclusives.txt')


def prepare_folders():
    os.makedirs(QUERY_DOC_PATH, exist_ok=True)
    os.makedirs(DATASETS_PATH, exist_ok=True)
    os.makedirs(DICTS_PATH, exist_ok=True)
    os.makedirs(CORPUS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(ALL_JSONS_PATH, exist_ok=True)

prepare_folders()