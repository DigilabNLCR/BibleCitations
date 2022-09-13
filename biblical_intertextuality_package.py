""" This python script includes all functions that are used in the Biblical Intertextuality projetc. """

import pandas as pd
import os
import joblib

from os import listdir as os_listdir
from os.path import isdir as os_path_isdir
from os.path import exists as os_exists
from os import remove as os_remove
from json import load as json_load
from re import sub as re_sub
from re import split as re_split
from os.path import join as join_path
from unidecode import unidecode
from time import time
from Levenshtein import distance
from gensim import corpora
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize
from math import ceil, isnan

__author__ = 'František Válek'
__version__ = '1.0.0'

""" DEFINING_PATHS------------------------------------------------------------------------------------------------- """
ROOT_PATH = os.getcwd()

BIBLES_PATH = join_path(ROOT_PATH, 'Bible_files')
DATASETS_PATH = join_path(ROOT_PATH, 'datasets')
DICTS_PATH = join_path(ROOT_PATH, 'dictionaries')
CORPUS_PATH = join_path(ROOT_PATH, 'corpuses')
ALL_JSONS_PATH = join_path(ROOT_PATH, 'query_jsons')

JOURNAL_FULLDATA_PATH = join_path(ROOT_PATH, 'journals_fulldata.joblib')

# RESULTS_PATH = join_path(ROOT_PATH, 'results')
RESULTS_PATH = join_path(ROOT_PATH, 'PUBLIC_RESULTS')
BATCHES_FILE_PATH = join_path(ROOT_PATH, 'batches.csv')
BATCH_RESULTS_FILE_PATH = join_path(RESULTS_PATH, 'batch_results.csv')

STOP_WORDS_PATH = join_path(ROOT_PATH, 'stop_words.txt')
STOP_SUBVERSES_PATH = join_path(ROOT_PATH, 'stop_subverses_21.txt')
EXCLUSIVES_PATH = join_path(ROOT_PATH, 'exclusives.txt')



""" GENERAL FUNCTIONS --------------------------------------------------------------------------------------------- """


def word_tokenize_no_punctuation(text):
    """ This function serves to return tokenized text to words and removes punctuation form it. To count only words for the split_verse function. """
    punctuation = '\!|\"|\„|\#|\$|\%|\&|\'|\(|\)|\*|\+|\-|\–|\.|\/|\:|\;|\<|\=|\>|\?|\[|\\|\]|\^|\_|\`|\{|\}|\~|\...|\>>|\<<|\»|\«|\||\,|\’|\‘'
    text = re_sub(punctuation, '', text)
    tokenized_text = word_tokenize(text)

    return tokenized_text


def normalize_string(string_to_be_normalized:str) -> str:
    """ This function normalizes a string. It removes punctuation and diacritics. """
    string_to_be_normalized = unidecode(string_to_be_normalized)
    punctuation = '\!|\"|\„|\#|\$|\%|\&|\'|\(|\)|\*|\+|\-|\–|\.|\/|\:|\;|\<|\=|\>|\?|\[|\\|\]|\^|\_|\`|\{|\}|\~|\...|\>>|\<<|\»|\«|\||\,|\’|\‘'
    string_to_be_normalized = re_sub(punctuation, '', string_to_be_normalized)

    return string_to_be_normalized.lower().strip()


""" GENERAL OBJECTS ----------------------------------------------------------------------------------------------- """

""" Define stop words in the file stop_words.txt """
with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as sw_f:
    stop_words = sw_f.read()
    stop_words = unidecode(stop_words)
    stop_words = stop_words.replace('\n', ' ')
    stop_words = stop_words.split(', ')
    stop_words.append(',')

""" Define stop subverses in the file stop_subverses_21.txt (for subverses of tolerance length 21 characters).
Should you wish different subverse lenght tolerance, you have to change the stop verses, too. 
And rename path to it (above). """
with open(STOP_SUBVERSES_PATH, 'r', encoding='utf-8') as ssv_f:
    stop_subverses = ssv_f.read()
    stop_subverses = stop_subverses.split('\n')

""" List of all translations. """
all_translations = ['BKR', 'BSV', 'HEJCL', 'SYK', 'ZP']
# NOTE: translation Bible svatováclavská is unfortunately not public. See README
# all_translations = ['BKR', 'HEJCL', 'SYK', 'ZP']


""" FUNCTIONS THAT ENSURE VERSE SPLITTING ------------------------------------------------------------------------- """


def split_text_by_delimiters(input_text: str):
    """ This function splits input text (sentence) by following delimiters: ',', ';', ':'. Used in split_verse(). """
    delimiters = ',|;|:'
    parts = re_split(delimiters, input_text)
    parts = [item.strip() for item in parts]

    return parts


def join_short_part(list_of_parts: list, current_position: int, tole_len: int):
    """ This function connects shorter passages to its neighbouring parts.
    Preference is set to connect with following parts, then to previous. Used in split_verse(). """
    output = list_of_parts[current_position]

    if current_position == -1:
        # if the part is at input at the end (used in join_short_sents()), we have to proceed differently:
        minus_pos = 1
        while len(output) < tole_len:
            try:
                output = f'{list_of_parts[current_position - minus_pos]} {output}'
                minus_pos += 1
            except IndexError:
                # This should not happen, but if there is no next nor previous position long enough, return joined original list
                return ' '.join(list_of_parts)

    plus_pos = 1
    minus_pos = 1
    while len(output) < tole_len:
        try:
            output += f' {list_of_parts[current_position + plus_pos]}'
            plus_pos += 1
        except IndexError:
            # If there is no next part, connect the previous one.
            try:
                output = f'{list_of_parts[current_position - minus_pos]} {output}'
                minus_pos += 1
            except IndexError:
                # This should not happen, but if there is no next nor previous position long enough, return joined original list
                return ' '.join(list_of_parts)

    return output


def join_short_sents(list_of_sents: list, current_position: int, tole_len: int):
    """ This function connects short sentences to passages of neigbouring sentences.
    Preference is set for the following sentence. Used in split_verse(). """
    output = list_of_sents[current_position]

    plus_sents = 1
    minus_sents = 1
    while len(output) < tole_len:
        try:
            parts_of_the_following = split_text_by_delimiters(list_of_sents[current_position + plus_sents])
            all_parts = [output]
            all_parts.extend(parts_of_the_following)
            output = join_short_part(all_parts, 0, tole_len=tole_len)
            plus_sents += 1
        except IndexError:
            try:
                parts_of_the_previous = split_text_by_delimiters(list_of_sents[current_position - minus_sents])
                parts_of_the_previous.append(output)
                output = join_short_part(parts_of_the_previous, -1, tole_len=tole_len)
                minus_sents += 1
            except IndexError:
                # If there is no more previous or following sentence long enough, return False.
                return False

    return output


def split_long_sent(input_sent_text: str, tole_len: int):
    """ This function splits a long string into two or three parts with a bit of overlap. Used in split_verse(). """
    subverses = []
    if len(input_sent_text) <= 3.5 * tole_len:
        words_in_sub = word_tokenize_no_punctuation(input_sent_text)
        sub_w_len = ceil(len(words_in_sub) / 2) + 1
        subverses.append(' '.join(words_in_sub[0:sub_w_len]))
        subverses.append(' '.join(words_in_sub[-sub_w_len:]))
        return subverses
    else:
        words_in_sub = word_tokenize_no_punctuation(input_sent_text)
        sub_w_len = ceil(len(words_in_sub) / 3) + 1
        subverses.append(' '.join(words_in_sub[0:sub_w_len]))
        subverses.append(
            ' '.join(words_in_sub[ceil(len(words_in_sub) / 3 - 1):ceil(len(words_in_sub) / 3 + sub_w_len)]))
        subverses.append(' '.join(words_in_sub[-sub_w_len:]))
        return subverses


def split_verse(input_text:str, tole_len=21, return_shorts=True, short_limit=9) -> list:
    """ This function ensures verse splitting into smaller subverses.
    :param input_text: text of verse that is to be split.
    :param tole_len: minimal length of subverse in characters.
    """
    if len(input_text) < tole_len:
        if return_shorts:
            if len(input_text) >= short_limit:
                return [input_text]
            else:
                return []
        else:
            return []
    
    else:
        subverses = []    
        sentences = sent_tokenize(input_text)

        for s, sent in enumerate(sentences):
            sent_len = len(sent)
            if sent_len >= tole_len and sent_len <= 2.5*tole_len:
                # If the sentence is in tolerance but not too long, append it
                subverses.append(sent)
            elif sent_len > 2.5*tole_len:
                sent_parts = split_text_by_delimiters(sent)
                if len(sent_parts) == 1:
                    # If there is only one part (there are no delimiters), split it to two or three parts with overlap.
                    subverses.extend(split_long_sent(sent, tole_len=tole_len))
                else:
                    for sp, sent_part in enumerate(sent_parts):
                        subverses.append(join_short_part(sent_parts, sp, tole_len=tole_len))
            else:
                # If the sentence is too short, we need to append it to neighbour parts.
                subverses.append(join_short_sents(sentences, s, tole_len=tole_len))
        
    out_subverses = []
    for sub in subverses:
        # strip the blank spaces in subverse strings
        out_subverses.append(sub.strip())
    
    # return the subverses, set() to remove any duplicates.
    return list(set(out_subverses))


""" FUNCTIONS AND CLASSES THAT WORK WITH BIBLE FILES -------------------------------------------------------------- """


def get_book_id(verse_id:str) -> str:
    """ Gets book's ID from a verse_id (e.g. "Gn 1:1"). """
    book_id = verse_id.split(' ')[0]
    return book_id


def get_verse_text(translation:str, verse_id:str, print_exceptions=True) -> str:
    """ This function gets text selected verse.
    :param translation: ID of Bible translation (select from ['BKR', 'BSV', 'CEP', 'COL', 'JB', 'KLP', 'SYK'])
    :param verse_id: ID of verse, e.g. "Gn 1:1"
    """
    book = get_book_id(verse_id)
    bible_filename = f'bible_{translation}_{book}.txt'
    try:
        with open(join_path(BIBLES_PATH, bible_filename), 'r', encoding='utf-8') as book_file:
            dict_of_verses = eval(book_file.read())
            verse_text = dict_of_verses[verse_id]
    except KeyError:
        if print_exceptions:
            print(f'Verse does not exist in {translation} translation.')
        return ''
    except FileNotFoundError:
        if print_exceptions:
            print(f'Book {book} does not exist in {translation} translation.')
        return ''

    return verse_text


def save_dataset(dataset, dataset_name='fullBibleDataset'):
    """ Saving dataset has default name, because there is supposedly only one version of it. """
    joblib.dump(dataset, join_path(DATASETS_PATH, f'{dataset_name}.joblib'))


def load_dataset(dataset_name='fullBibleDataset'):
    """ Loading dataset has default name, because there is supposedly only one version of it. """
    dataset = joblib.load(join_path(DATASETS_PATH, f'{dataset_name}.joblib'))
    return dataset


def save_dictionary(dictionary, dictionary_name:str):
    joblib.dump(dictionary, join_path(DICTS_PATH, f'{dictionary_name}.joblib'))


def load_dictionary(dictionary_name:str):
    dictionary = joblib.load(join_path(DICTS_PATH, f'{dictionary_name}.joblib'))
    return dictionary


def save_corpus(bongrammed_corpus, corpus_name:str):
    corpora.MmCorpus.serialize(join_path(CORPUS_PATH, f'{corpus_name}.mm'), bongrammed_corpus)


def load_corpus(corpus_name:str):
    corpus = corpora.MmCorpus(join_path(CORPUS_PATH, f'{corpus_name}.mm'))
    return corpus


class bibleDataset:
    def __init__(self, data, target):
        # NOTE: data = texts of a subverse
        self.data = data
        # NOTE: target = verse ID
        self.target = target

        self.target_names = sorted(set(target))

    def __len__(self):
        return len(self.data)

    def check_valid(self):
        return (len(self.data) == len(self.target))


def bible_to_dataset(save=True, ignore_stop_subs=False, dataset_prefix='fullBible', return_shorts=True,  tole_len=21, short_limit=9):
    """ This function prepares dataset into bibleDataset class.

    :param ignore_stop_subs: if True, the stop subverses defined in stop_subverses_21 are ignored.
    :param save: set False if you do not want to save the dataset.
    :param dataset_prefix: prefix for the filename of the saved dataset (prefixDataset.joblib).
    :param return_shorts: should verses shorter than tole_len characters be returned?
    :param tole_len: minimal length for verse split
    :param short_limit: minimal characters needed for the verse to be included in the dataset.
    """
    data = []
    targets = []

    bible_files = os_listdir(BIBLES_PATH)

    for bible_file in bible_files:
        with open(join_path(BIBLES_PATH, bible_file), 'r', encoding='utf-8') as bible_f:
            bible_data = bible_f.read()
            try:
                verses_dict = eval(bible_data)
            except:
                print('There is some error in:', bible_file)
            for verse_id in verses_dict:
                subverses = split_verse(verses_dict[verse_id], return_shorts=return_shorts, tole_len=tole_len, short_limit=short_limit)
                for sub in subverses:
                    if ignore_stop_subs:
                        if sub in stop_subverses:
                            continue
                        else:
                            data.append(sub)
                            targets.append(verse_id)
                    else:
                        data.append(sub)
                        targets.append(verse_id)

    bible_dataset = bibleDataset(data, targets)

    if save:
        save_dataset(bible_dataset, dataset_name=f'{dataset_prefix}Dataset')

    return bible_dataset


""" N-GRAM TOKENIZING FUNCTIONS ------------------------------------------------------------------------------------ """


def ngramming(tokens: list, ngram_size=4, use_stop_words=True) -> list:
    """ Core of nrgamming functions. Used in text_to_ngrams(). """
    ngrammed_text = []
    if use_stop_words:
        for token in tokens:
            if token not in stop_words:
                if len(token) > ngram_size:
                    for i in range(len(token) - (ngram_size - 1)):
                        ngram = token[i:i + ngram_size]
                        if ngram not in stop_words:
                            ngrammed_text.append(ngram)
                        else:
                            continue
                else:
                    ngrammed_text.append(token)
            else:
                continue

    else:
        for token in tokens:
            if len(token) > ngram_size:
                for i in range(len(token) - (ngram_size - 1)):
                    ngram = token[i:i + ngram_size]
                    ngrammed_text.append(ngram)
            else:
                ngrammed_text.append(token)

    return ngrammed_text


def text_to_ngrams(input_text: str, ngram_size=4, use_stop_words=True) -> list:
    """ This function ngrams input string. Used in tokenize_to_ngrams(). """
    tokens = word_tokenize_no_punctuation(normalize_string(input_text))
    ngrammed_text = ngramming(tokens, ngram_size=ngram_size, use_stop_words=use_stop_words)

    return ngrammed_text


def tokenize_to_ngrams(input_documents: list, ngram_size=4, use_stop_words=True) -> list:
    """ This function serves to ngram documents for bow (or rather "bag of n-grams") creation. """
    ngrammed_docs = []

    for doc in input_documents:
        ngrams = text_to_ngrams(doc, ngram_size=ngram_size, use_stop_words=use_stop_words)
        ngrammed_docs.append(ngrams)

    return ngrammed_docs


""" PROCESSING CORPUS ---------------------------------------------------------------------------------------------- """


def process_corpus(dataset: bibleDataset, cut_off_value=0, ngram_size=4):
    """
    This function creates gensim dictionary from input dataset (class bibleDataset).

    :param dataset: input dataset (class bibleDataset)
    :param cut_off_value: int; sets minimal token appearance
    :param ngram_size: int; size of ngrams (in characters)
    :return dictionary, processed_corpus
    """
    ngrammed_verses = tokenize_to_ngrams(dataset.data, ngram_size=ngram_size)

    frequency = defaultdict(int)
    for verse in ngrammed_verses:
        for ngram in verse:
            frequency[ngram] += 1

    processed_corpus = [[token for token in text if frequency[token] > cut_off_value] for text in ngrammed_verses]

    dictionary = corpora.Dictionary(processed_corpus)

    return dictionary, processed_corpus


def create_corpus(dictionary, processed_corpus):
    """ This function creates corpus from input dictionary and processed corpus. """
    return [dictionary.doc2bow(text) for text in processed_corpus]


def transfer_corpus_to_simple_token_vectors(corpus):
    """ This function extracts from corpus simple vectors consisting of only token IDs"""
    return [[token for token, token_count in subverse] for subverse in corpus]


""" WORKING WITH QUERY DOCUMENTS ---------------------------------------------------------------------------------- """


def query_to_bongrams(query_doc:str, dictionary, ngram_size=4):
    """ Transfers text into bag of ngrams vector according to Bible dictionary (defined above!)"""
    query_ngrams = text_to_ngrams(query_doc, ngram_size=ngram_size)
    query_bon = dictionary.doc2bow(query_ngrams)

    return query_bon


def load_json_data(json_path: str) -> dict:
    """ This function loads JSON data so that we do not have to write the open statement over and over again. """
    with open(json_path, encoding='utf-8') as json_file:
        data = json_load(json_file)

    return data


def split_query(input_text:str, window_len=4, overlap=1):
    """
    This function is used to split query document (text) into smaller parts for comparison with Biblical verses.
    :param window_len: int; how many sentences are to be connected
    :param overlap: int; how many sentences are set to overlap in the mooving window; must be lower than window_len
    """
    query_sentences = sent_tokenize(input_text)

    query_docs = []

    constant = window_len-overlap
    for i in range(int(ceil(len(query_sentences)/constant))):
        query_part = query_sentences[i*constant:i*constant+window_len]
        query_docs.append(' '.join(query_part))

    return query_docs


def create_all_jsons_metadata_file(return_dict=False):
    """ This function is used to create 'journals_metadata.joblib' file, i.e. a dictionary object that stores metadata of all JSON files according to their filename. It is used because it is much faster than load JSON files and extract the matadata all over (for evaluation processes). This function is used in prepare_query_documents.py """
    files_as_keys = {}

    journals_folders = os_listdir(ALL_JSONS_PATH)

    for folder in journals_folders:
        print('Working on', folder)
        files_in_folder = os_listdir(join_path(ALL_JSONS_PATH, folder))
        print('\tFiles in folder', len(files_in_folder))
        files_done = 0
        print_progress = 0
        for file_ in files_in_folder:
            if print_progress >= 500:
                print('\tProgress:', files_done, '/', len(files_in_folder))
                print_progress = 0

            with open(join_path(ALL_JSONS_PATH, folder, file_), 'r', encoding='utf-8') as js_file:
                data = json_load(js_file)
                journal = data['journal']
                issue_date = data['date']
                issue_page = data['page_num']
                issue_uuid = data['issue_uuid']
                kramerius_url = f'https://kramerius5.nkp.cz/view/uuid:{issue_uuid}'
                
                files_as_keys[file_] = {'journal': journal, 'issue_date': issue_date, 'issue_page': issue_page, 'issue_uuid': issue_uuid, 'kramerius_url': kramerius_url}
            
            print_progress += 1
            files_done += 1

    joblib.dump(files_as_keys, join_path(ROOT_PATH, 'journals_metadata.joblib'))

    if return_dict:
        return (files_as_keys)


def create_all_jsons_fulldata_file(return_dict=False):
    """ This function is used to create 'journals_fulldata.joblib' file, i.e. a dictionary object that stores metadata of all JSON files according to their filename. It is used because it is much faster than load JSON files and extract the matadata and text all over. However, the file is quite large (ca. 2 GB) so some computers may have problems with RAM using this file. """
    files_as_keys = {}

    journals_folders = os.listdir(ALL_JSONS_PATH)

    for folder in journals_folders:
        print('Working on', folder)
        files_in_folder = os.listdir(join_path(ALL_JSONS_PATH, folder))
        print('\tFiles in folder', len(files_in_folder))
        files_done = 0
        print_progress = 0
        for file_ in files_in_folder:
            if print_progress >= 500:
                print('\tProgress:', files_done, '/', len(files_in_folder))
                print_progress = 0

            with open(join_path(ALL_JSONS_PATH, folder, file_), 'r', encoding='utf-8') as js_file:
                data = json_load(js_file)
                journal = data['journal']
                issue_date = data['date']
                issue_page = data['page_num']
                issue_uuid = data['issue_uuid']
                kramerius_url = f'https://kramerius5.nkp.cz/view/uuid:{issue_uuid}'
                full_text = data['text']
                
                files_as_keys[file_] = {'journal': journal, 'issue_date': issue_date, 'issue_page': issue_page, 'issue_uuid': issue_uuid, 'kramerius_url': kramerius_url, 'text': full_text}
            
            print_progress += 1
            files_done += 1

    joblib.dump(files_as_keys, JOURNAL_FULLDATA_PATH)

    if return_dict:
        return (files_as_keys)


""" CREATING AND LOADING NECESSARY OBJECTS ------------------------------------------------------------------------- """


def create_necessary_objects(ngram_size=4, skip_dataset=True, dataset=None, save_objects=True, objects_name='fullBible'):
    """
    This function creates all necessary objects for the search. Run this function if you are starting the process or if
    you have changed the dataset or functions that create it, otherwise it is not necessary.

    :param ngram_size: size of ngrams (in characters) to which everything is parsed; According to it,
        other objects for search are loaded.
    :param skip_dataset: if the dataset exists, it is loaded instead of created.
    :param dataset: dataset can be also loaded externaly, so it does not have to be loaded for every iteration.
    :param save_objects: if True, objects are saved, if False, objects are only returned.
    """

    if dataset:
        bible_dataset = dataset
    else:
        if skip_dataset:
            print('Dataset already exists --> loaded.')
            bible_dataset = load_dataset(f'{objects_name}Dataset')
        else:
            start_ = time()
            print('Creating bible dataset...')
            bible_dataset = bible_to_dataset()
            save_dataset(bible_dataset, f'{objects_name}Dataset')
            end_ = time()
            print(f'Dataset has been created in {round((end_-start_)/60, 2)} minutes. Saved as {objects_name}Dataset.joblib')

    start_ = time()
    print('Processing corpus and creating dictionary...')
    dictionary, processed_corpus = process_corpus(bible_dataset, ngram_size=ngram_size)
    if save_objects:
        save_dictionary(dictionary, f'n{ngram_size}_{objects_name}Dict')
    end_ = time()
    print(f'Dictionary has been created in {round((end_-start_), 2)} seconds. Saved as n{ngram_size}_{objects_name}Dict.joblib')

    start_ = time()
    print('Creating corpus...')
    corpus = create_corpus(dictionary, processed_corpus)
    if save_objects:
        save_corpus(corpus, f'n{ngram_size}_{objects_name}Corpus')
    end_ = time()
    print(f'Corpus has been created in {round((end_-start_), 2)} seconds. Saved as n{ngram_size}_{objects_name}Corpus.mm')

    return corpus, dictionary


def load_necessary_objects(ngram_size=4, objects_name='fullBible'):
    """
    This function loads all necessary objects for the search.

    :param ngram_size: int; size of ngrams to which everything is parsed; According to it,
        other objects for search are loaded.
    """
    dataset = load_dataset(f'{objects_name}Dataset')
    corpus = load_corpus(f'n{ngram_size}_{objects_name}Corpus')
    dictionary = load_dictionary(f'n{ngram_size}_{objects_name}Dict')

    subverses = transfer_corpus_to_simple_token_vectors(corpus)

    return dataset, corpus, dictionary, subverses


""" SEARCH FUNCTIONS AND CLASSES ----------------------------------------------------------------------------------- """


class bibleObject:
    def __init__(self, dataset:bibleDataset, create_anew_other_necessary_objects=False, ngram_size=4, objects_prefix='fullBible'):
        """
        :param dataset: prepared dataset object of bibleDataset class
        :param create_anew_necessary_objects: Bool, set True if you want to create and save corpus and dictionary. These may be already created so the default is set to False.
        :param ngram_size: (maximum) size of n-grams (in characters)
        """
        if create_anew_other_necessary_objects:
            print('Creating necessary objects for bibleObject...')
            corpus, dictionary = create_necessary_objects(ngram_size=ngram_size)
            subverse_vectors = transfer_corpus_to_simple_token_vectors(corpus)
            # NOTE: corpus is then discarded, it is not used anymore, it can be romoved from memory
            del corpus
            
        else:
            print('Loading necessary objects for bibleObject...')
            corpus = load_corpus(f'n{ngram_size}_{objects_prefix}Corpus')
            dictionary = load_dictionary(f'n{ngram_size}_{objects_prefix}Dict')
            subverse_vectors = transfer_corpus_to_simple_token_vectors(corpus)
            # NOTE: corpus is then discarded, it is not used anymore, it can be romoved from memory
            del corpus

        attributed_subverses = defaultdict(list)
        subverse_lens = defaultdict(int)

        for i, sub in enumerate(subverse_vectors):
            subverse_lens[i] = len(sub)
            for elem in sub:
                attributed_subverses[elem].append(i)

        self.attr_subs = attributed_subverses
        self.sub_lens = subverse_lens

        self.data = dataset.data
        self.verse_id = dataset.target
        self.dictionary = dictionary
        self.subverse_vectors = subverse_vectors
        self.dataset = dataset

        self.verse_ids = sorted(set(dataset.target))

    def __len__(self):
        return len(self.data)


def compare_vector(query_string:str, attributed_subverses:dict, subverse_lens:dict, dictionary:corpora.Dictionary, tolerance=0.85, ngram_size=4):
    """ 
    This function implements vector comparison based on preprepared dictionaries of token references to subverses. 
    
    :param query_string:
    :param attributed_subverses:
    :param subverse_lens:
    :param dictionary:
    :param tolerance:
    :param ngram_size:

    """
    vector_q_bon = query_to_bongrams(query_string, dictionary, ngram_size=ngram_size)
    tokens_in_query = [token for token, token_count in vector_q_bon]

    subverse_scores = defaultdict(int)
    # Get scores of subverses by every token.
    for token in tokens_in_query:
        for sft in attributed_subverses[token]:
            subverse_scores[sft] += 1

    # Select those subverses that have scored high enough.    
    possible_subverses = []
    for sub in subverse_scores:
        if subverse_scores[sub] >= subverse_lens[sub]*tolerance:
            possible_subverses.append(sub)

    return possible_subverses


def fuzzy_string_matching_for_implementation(subverse_string:str, query_string:str, tolerance=0.85):
    """ 
    This function is for implementation of typo similarity detection applied to two strings. It returns bool value of match.

    :param subverse_string: string of the biblical subverse we are searching for.
    :param query_string: string in which we are searching for the seubverse_string.
    :param tolerance: how large proportion of the subverse_string must be present in query_string to consider it a match.
    """
    subverse_string = normalize_string(subverse_string)
    subverse_len = len(subverse_string)

    query_string = normalize_string(query_string)
    query_len = len(query_string)

    tolerance = subverse_len * (1-tolerance)

    match = False
 
    if subverse_len-tolerance > query_len:
        # If subverse is longer than query string, it is not a match by default
        return match
    elif subverse_len-tolerance <= query_len <= subverse_len+tolerance:
        # If subverse is more or les of the same length as query string, just compare them.
        if distance(subverse_string, query_string) <= tolerance:
            match = True
    else:
        # Oherwise, compare parts of the query string (always staring with word, so it is quicker; however, some mistakes may be made here.
        # NOTE: change in split - skipping by words but len by characters...
        char_len_sub = len(subverse_string)
        word_len_subv = len(word_tokenize(subverse_string))
        words_in_query_string = word_tokenize(query_string)
        word_len_query_string = len(words_in_query_string)

        for i, cycle in enumerate(range(word_len_subv, word_len_query_string+1)):
            gram_str = ' '.join(words_in_query_string[i:])[:char_len_sub]
            if distance(subverse_string, gram_str) <= tolerance:
                match = True
                return match
            else:
                continue
    
    return match


def search_for_bible_for_batches_implementation(bible_object:bibleObject, journals_fulldata:dict, batch_id:int, ngram_tolerance=0.7, edit_distance_tolerance=0.85, ngram_size=4, query_window_len=4, query_overlap=1):
    """    
    This function is appropriated for implementation within search  by batches (in run_search_by_batch() function)
    This function executes search for Bible quotations within all JSON documents assigned to selected batch using dictionary from 'journals_fulldata.joblib'.

    :param bible_object: an input object of bibleObject class (created from bibleDataset).
    :param ngram_tolerance: what portion of ngrams of Bible subverse must match to be considered as a match.
    :param edit_distance_tolerance: what portion of characters of a subverse must match a sequence from a "query document".
    :param ngram_size: size of ngrams (in characters) to which everything is parsed; According to it, other objects for search are loaded.
    :param query_window_len: How many sentences are put together as smaller "query documets". This parameter can influence the speed of the process, depending on the nature of the document.
    :param query_overlap: Overlap of the sentences among "query documents". Must be higher than query_window_len.
    :return: Detected citations, time of search.
    """
    function_start = time()

    attributed_subverses = bible_object.attr_subs
    subverse_lens = bible_object.sub_lens
    dictionary = bible_object.dictionary
    dataset = bible_object.dataset
    
    query_files = get_query_files_from_batch(batch_id)
    num_of_query_files = len(query_files)

    discovered_citations = defaultdict(list)
    
    print('Initiating search in query documents...')
    for qi, query_file in enumerate(query_files):
        query_time_start = time()
        print(f'\tBatch {batch_id} ... Analysing document ({qi+1}/{num_of_query_files}) {query_file}')
        
        full_query_text = journals_fulldata[query_file]['text']
        query_documents = split_query(full_query_text, window_len=query_window_len, overlap=query_overlap)

        for i, query_doc in enumerate(query_documents):
            # First stage - compare vectors (token = n-gram of ngram_size)
            results_by_ngrams = compare_vector(query_doc, attributed_subverses=attributed_subverses, subverse_lens=subverse_lens, dictionary=dictionary, tolerance=ngram_tolerance, ngram_size=ngram_size)

            # Second stage - compare by fuzzy string matching
            for subverse_id in results_by_ngrams:
                try:
                    subverse_text = dataset.data[subverse_id]
                except IndexError:
                    # TODO: tohle zčeknout a vyřadit!
                    print('ERRROOROROROR', subverse_id)
                match = fuzzy_string_matching_for_implementation(subverse_string=subverse_text, query_string=query_doc, tolerance=edit_distance_tolerance)
                if match:
                    discovered_citations[subverse_id].append((query_file, i))

        query_time_end = time()
        print(f'\t\tDocument analysed in {round((query_time_end-query_time_start), 2)} seconds.')

    function_end = time()
    average_per_page = round(((function_end-function_start)/num_of_query_files), 2)
    
    return discovered_citations, average_per_page


""" PREPARING SEARCH BATCHES --------------------------------------------------------------------------------------- """


def collect_query_jsons():
    """ This function list all json files as stored in folder "query_jsons"."""
    folders_in_extracted_query_jsons = os_listdir(ALL_JSONS_PATH)

    journals_et_folders = {}

    for journal_folder in folders_in_extracted_query_jsons:
        journal_folder_path = join_path(ALL_JSONS_PATH, journal_folder)
        if not os_path_isdir(journal_folder_path):
            continue

        jsons_in_journal = os_listdir(journal_folder_path)
        journals_et_folders[journal_folder] = jsons_in_journal

    return journals_et_folders


batches_columns = ['journal', 'json_file', 'batch_id', 'run', 'runtime']


def clear_batches_csv(clear: bool):
    if clear:
        # Delete old batches.csv
        os_remove(BATCHES_FILE_PATH)
        # create clear batches.csv
        empty_df = pd.DataFrame(columns=batches_columns)
        pd.DataFrame.to_csv(empty_df, BATCHES_FILE_PATH)


def create_batches_csv():
    """ create batches.csv if not exists """
    if not os_exists(BATCHES_FILE_PATH):
        empty_df = pd.DataFrame(columns=batches_columns)
        pd.DataFrame.to_csv(empty_df, BATCHES_FILE_PATH)


def get_last_batch_id():
    """ This function is used in 'run_biblical_intertextuality.py' to detect the last batch in order to set the right range. """
    batches_df = pd.read_csv(BATCHES_FILE_PATH)
    last_assigned_batch = batches_df['batch_id'].max()

    return last_assigned_batch


def update_batches_csv(clear=False, max_batch_size=40):
    """ This function creates or updates batches file. Use this e.g. if you have added some new journals to dir. extracted_query_jsons

    :param clear: bool; set true if you want to reastart the batches.csv (delete information on runs).
    """
    # create batches.csv if not exists
    create_batches_csv()

    # clear batches.csv if set
    clear_batches_csv(clear)

    # create dictionary of all json files for each journal to be ignored while updating
    all_jsons_by_journal = collect_query_jsons()

    # load batches.csv as pandas dataframe
    batches_df = pd.read_csv(BATCHES_FILE_PATH)

    existing_jsons = batches_df['json_file'].to_list()
    last_assigned_batch = batches_df['batch_id'].max()
    if isnan(last_assigned_batch):
        last_assigned_batch = -1

    # creat dictionary dataframe by individual uuids
    batch_id = last_assigned_batch + 1
    data = defaultdict(list)
    for journal in all_jsons_by_journal:
        print(f"working on {journal}")
        batch_size = 0
        for json_f in all_jsons_by_journal[journal]:
            if batch_size == max_batch_size:
                batch_size = 0
                batch_id += 1

            if json_f in existing_jsons:
                continue

            data['journal'].append(journal)
            data['json_file'].append(json_f)
            data['batch_id'].append(batch_id)
            data['run'].append(False)
            data['runtime'].append(0.0)

            batch_size += 1

    # merge old and new dataframes
    append_df = pd.DataFrame(data)
    batches_df = pd.concat([batches_df, append_df])
    batches_df.to_csv(BATCHES_FILE_PATH, index=False, columns=batches_columns)

    print(f'The last batch id is: {batch_id}')


def change_run_log(batch_id:int, avereage_pre_page:float):
    """ Changes run log in batches.csv. """
    batches_df = pd.read_csv(BATCHES_FILE_PATH)
    batches_df.loc[batches_df['batch_id'] == batch_id, "run"] = True
    batches_df.loc[batches_df['batch_id'] == batch_id, "runtime"] = avereage_pre_page
    batches_df.to_csv(BATCHES_FILE_PATH, index=False, columns=batches_columns)


def get_query_files_from_batch(batch_id:int):
    """ This fnction gets the list of query files assigned to a batch. """
    batches_df = pd.read_csv(BATCHES_FILE_PATH)
    relevant_data = batches_df.loc[batches_df['batch_id'] == batch_id]
    query_files_in_batch = relevant_data['json_file'].to_list()

    return query_files_in_batch


""" WORKING WITH RESULTS CSV FILE -------------------------------------------------------------------------------------------------------- """


results_columns_names = ['verse_id', 'query_file', 'index_query_part', 'batch_id', 'ngram_size', 'query_window_len', 'query_overlap', 'ngram_tolerance', 'edit_distance_tolerance']


def create_batches_results_csv():
    """ Create batch_results.csv if not exists. """
    if not os_exists(BATCH_RESULTS_FILE_PATH):
        empty_df = pd.DataFrame(columns=results_columns_names)
        pd.DataFrame.to_csv(empty_df, BATCH_RESULTS_FILE_PATH)


def save_batch_results(results:dict, dataset:bibleDataset, batch_id:int, ngram_size=4, query_window_len=6, query_overlap=1, ngram_tolerance=0.7, edit_distance_tolerance=0.85):
    """
    This function saves results as generated by search_for_bible() function. Results are append to previously created results.

    :param results: results as generated by search_for_bible() function. The structure of results dict is: key = subverse_id, key_content: [(query_file, index_query_part), ...].
    :param dataset: bibleDataset object to which current results are associated. When the dataset is changed, the subverse_id will no longer be valid, therefore, we also save verse_id right away.
    :param batch_id: id of batch to match with batches.csv.
    :param ngram_size: size of n-grams on which the search was based.
    :param query_window_len: size of query split by which the search was done.
    :param query_overlap: size of query split overlap by which the search was done.
    """
    # If result CSV do not exist, create it:
    create_batches_results_csv()

    # Load results dataframe:
    results_df = pd.read_csv(BATCH_RESULTS_FILE_PATH)

    # Create new df from current results:
    new_data = defaultdict(list)

    for subverse_id in results:
            for attr in results[subverse_id]:
                new_data['verse_id'].append(dataset.target[subverse_id])
                new_data['query_file'].append(attr[0])
                new_data['index_query_part'].append(attr[1])
                new_data['batch_id'].append(batch_id)
                new_data['ngram_size'].append(ngram_size)
                new_data['query_window_len'].append(query_window_len)
                new_data['query_overlap'].append(query_overlap)
                new_data['ngram_tolerance'].append(ngram_tolerance)
                new_data['edit_distance_tolerance'].append(edit_distance_tolerance)

    # merge old and new dataframes
    append_df = pd.DataFrame(new_data)
    results_df = pd.concat([results_df, append_df])
    results_df.to_csv(BATCH_RESULTS_FILE_PATH, index=False, columns=results_columns_names)


""" RUNNING SEARCH BY BATHES --------------------------------------------------------------------------------------- """


def run_search_by_batch(batch_id:int, journals_fulldata:dict, bible_object:bibleObject, ngram_tolerance=0.7, edit_distance_tolerance=0.85, ngram_size=4, query_window_len=4, query_overlap=1):
    """ This function executes search by a batch_id (as linked to json files in batches.csv) and saves it results to batch_results.csv. """
    bible_dataset = bible_object.dataset
    
    # Run search:
    print(f'... Initiating search of batch {batch_id}')
    batch_results, avg_time_per_page = search_for_bible_for_batches_implementation(bible_object=bible_object, journals_fulldata=journals_fulldata, batch_id=batch_id, ngram_tolerance=ngram_tolerance, edit_distance_tolerance=edit_distance_tolerance, ngram_size=ngram_size, query_window_len=query_window_len, query_overlap=query_overlap)

    # Save results:
    print(f'... Saving results of batch {batch_id}')
    save_batch_results(results=batch_results, dataset=bible_dataset, batch_id=batch_id, ngram_size=ngram_size, query_window_len=query_window_len, query_overlap=query_overlap, ngram_tolerance=ngram_tolerance, edit_distance_tolerance=edit_distance_tolerance)

    # Change run log in batches.csv:
    print(f'... Changing search log for {batch_id}')
    change_run_log(batch_id=batch_id, avereage_pre_page=avg_time_per_page)


def search_by_batches(batches_to_run:list, bible_dataset_filename='fullBibleDataset', skip_done=True, ngram_tolerance=0.7, edit_distance_tolerance=0.85, ngram_size=4, query_window_len=4, query_overlap=1):
    """ 
    NOTE: CHENGE, this function uses journals_fulldata:dict, to possibly make it faster...

    This function executes search across a number of batches.
    Batches must be prepared in batches.csv (with update_batches_csv() function).
    
    :param batches_to_run: list of batches IDs that are to be run (batch IDs are int)
    :param bible_dataset_filename: filename (without ".joblib"!) of a dataset that is to be loaded and with which the search is run.
    :param skip_done: bool, if True, batches that have already been run are skipped.
    :param ngram_tolerance: what portion of ngrams of Bible subverse must match to be considered as a match.
    :param edit_distance_tolerance: what portion of characters of a subverse must match a sequence from a "query document".
    :param ngram_size: size of ngrams (in characters) to which everything is parsed; According to it, other objects for search are loaded.
    :param query_window_len: How many sentences are put together as smaller "query documets". This parameter can influence the speed of the process, depending on the nature of the document.
    :param query_overlap: Overlap of the sentences among "query documents". Must be higher than query_window_len.
    """
    bible_dataset = load_dataset(bible_dataset_filename)
    object_prefix = bible_dataset_filename.replace('Dataset', '')
    bible_object = bibleObject(bible_dataset, ngram_size=ngram_size, objects_prefix=object_prefix)

    print('Loading journals_fulldata.joblib')
    journals_fulldata = joblib.load(JOURNAL_FULLDATA_PATH)

    batches_to_skip = []
    if skip_done:
        batches_df = pd.read_csv(BATCHES_FILE_PATH)
        relevant_data = batches_df.loc[batches_df['run'] == True]
        batches_to_skip = list(set(relevant_data['batch_id'].to_list()))

    for batch_id in batches_to_run:
        if batch_id in batches_to_skip:
            print(f'Batch {batch_id} has already been run.')
            continue
        else:
            run_search_by_batch(batch_id=batch_id, journals_fulldata=journals_fulldata, bible_object=bible_object, ngram_tolerance=ngram_tolerance, edit_distance_tolerance=edit_distance_tolerance, ngram_size=ngram_size, query_window_len=query_window_len, query_overlap=query_overlap)

    print('SEARCH FINISHED')


""" EVALUATION OF RESULTS --------------------------------------------------------------------------------------- """


""" GENERAL FUNCTIONS FOR EVALUATION """


def load_results(results_filename='batch_results.csv', delimiter=',') -> pd.core.frame.DataFrame:
    """ This function loads selected results from the results folder. It is returned as pandas dataframe
    
    :param results_filename: filename of results; 'batch_results.csv' is the default parameter, as this is the default filename of results from the search functions.
    """
    return pd.read_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', delimiter=delimiter, encoding='utf-8')


def get_verseid_queryfile(dataframe:pd.core.frame.DataFrame, row_id:int):
    """ This function returns search properties of a given row in the results dataframe. """

    verse_id = dataframe.loc[row_id]['verse_id']
    query_file = dataframe.loc[row_id]['query_file']

    return verse_id, query_file


""" 'UNFILTERED' REDUCTION OF RESULTS """


def make_unfiltered_search_dataframe(results_filename='batch_results.csv', save=True, return_df=False):
    """ This functions converts the preliminary results to structure same as all of the other results (filtered and improved). This is for purely statistical reasons. It only drops duplicates. """
    # Load results:
    results_dataframe = load_results(results_filename)

    # Load metadata from json_metadata.joblib (created with prepare_query_documents.py)
    jsons_metadata = joblib.load(join_path(ROOT_PATH, 'journals_metadata.joblib'))

    # Remove duplicate rows from the result dataframe
    print('Original size of the results dataframe:', len(results_dataframe))
    results_dataframe.drop_duplicates(subset=['verse_id', 'query_file', 'index_query_part'], keep='first', inplace=True)
    print('Size of the results dataframe after droping duplicates:', len(results_dataframe))

    # Create (empty) final results dataframe:
    final_results = {}
    res_id = 0
    print_progress = 0
    iter_ = 0

    print('Dropping duplicates...')
    for row_id in results_dataframe.index:
        iter_ += 1
        if print_progress == 500:
            print('\t', iter_, 'of', len(results_dataframe))
            print_progress = 0
      
        verse_id, query_file = get_verseid_queryfile(dataframe=results_dataframe, row_id=row_id)

        # NOTE: repair Syr verses to Sir (there has been a mistake in my dataset, now it is repaired but not in the initial batch_results.csv file in PUBLIC_RESULTS)
        if 'Syr' in verse_id:
            verse_id = verse_id.replace('Syr', 'Sir')

        row_dict = results_dataframe.loc[row_id].to_dict()

        # NOTE: 334149b0-877c-11e6-8aeb-5ef3fc9ae867 has wrong date --> it is repaired here in the process:
        if '334149b0-877c-11e6-8aeb-5ef3fc9ae867' in row_dict['query_file']:
            row_dict['date'] = '30.06.1935'
        else:
            row_dict['date'] = jsons_metadata[query_file]['issue_date']

        row_dict['verse_id'] = verse_id
        row_dict['book'] = get_book_id(verse_id)
        row_dict['journal'] = jsons_metadata[query_file]['journal']
        row_dict['page_num'] = jsons_metadata[query_file]['issue_page']

        # NOTE: filtering out year out of the scope of 1925-1939:
        issue_year = row_dict['date'].split('.')[-1]
        years_to_consider = ['1925', '1926', '1927', '1928', '1929', '1930', '1931', '1932', '1933', '1934', '1935', '1936', '1937', '1938', '1939', '1937-1938']
        if issue_year not in years_to_consider:
            continue
        else:
            final_results[res_id] = row_dict
            res_id += 1
        
        print_progress += 1

    final_results_df = pd.DataFrame.from_dict(final_results)
    final_results_df = final_results_df.transpose()
    
    if save:
        final_results_df.to_csv(join_path(RESULTS_PATH, f'UNFILTERED_{results_filename}'), encoding='utf-8', quotechar='"', sep=';')

    if return_df:
        return final_results_df


""" FUNCTIONS AND OBJECTS FOR INITIAL FILTER AND SCORINGS """

""" Define mutually exclusive words in exclusives.txt """
with open(EXCLUSIVES_PATH, 'r', encoding='utf-8') as exclusives_file:
    data = exclusives_file.read()
    words_lines = data.split('\n')

    exclusives_dict = defaultdict(list)
    list_of_exclusives = []

    for line in words_lines:
        word_list = line.split(', ')
        for word_from in word_list:
            for word_to in word_list:
                if word_from == word_to:
                    continue
                if word_from == 'je' and word_to == 'jest':
                    continue
                if word_from == 'jest' and word_to == 'je':
                    continue
                else:
                    exclusives_dict[normalize_string(word_from)].append(normalize_string(word_to))
            list_of_exclusives.append(normalize_string(word_from))


def exclusiveness_test(subverse_string:str, query_string:str) -> bool:
    """
    This function serves to check if the detected string is not false positive based on mutually exclusive words. E.g. naše vs. vaše;, je vs, není etc.
    """
    subverse_string = normalize_string(subverse_string)
    query_string = normalize_string(query_string)

    subverse_words = word_tokenize_no_punctuation(subverse_string)
    query_words = word_tokenize_no_punctuation(query_string)

    for i, word in enumerate(subverse_words):
        if word in list_of_exclusives:
            list_to_ex = exclusives_dict[word]
            try:
                if query_words[i] in list_to_ex:
                    return False
            except:
                continue

    return True


def get_row_data_for_initial_filter(dataframe:pd.core.frame.DataFrame, row_id:int):
    """ This function returns search properties of a given row in the results dataframe to be used in check_results function. """

    query_file = dataframe.loc[row_id]['query_file']
    query_window_len = dataframe.loc[row_id]['query_window_len']
    query_overlap = dataframe.loc[row_id]['query_overlap']

    return query_file, query_window_len, query_overlap


def get_verse_et_idx(dataframe:pd.core.frame.DataFrame, row_id:int):
    """ This function returns search properties of a given row in the results dataframe. """

    verse_id = dataframe.loc[row_id, 'verse_id']
    index_query_part = dataframe.loc[row_id, 'index_query_part']

    return verse_id, index_query_part


def select_attributions_to_json(dataframe:pd.core.frame.DataFrame, query_file:str):
    """ This function selects all attributions to a given JSON file. 
    
    It returns: dataframe of all of the results, row_ids to skip
    """
    subset_dataframe = dataframe[dataframe['query_file'] == query_file]

    # If the subset dataframe contains only one result, return it and empty skips.
    if len(subset_dataframe) == 1:
        verse_id, index_query_part = get_verse_et_idx(subset_dataframe, subset_dataframe.index[0])
        attributed_verses = {verse_id: [index_query_part]}
        return attributed_verses, []

    # If the subset dataframe contains more rows, check if further.
    else:
        row_ids_to_skip = subset_dataframe.index
        attributed_verses = defaultdict(list)
        for row_id in row_ids_to_skip:
            verse_id, index_query_part = get_verse_et_idx(dataframe=subset_dataframe, row_id=row_id)
            attributed_verses[verse_id].append(index_query_part)

        return attributed_verses, row_ids_to_skip


def join_overlap(list_of_parts:list, query_index:int) -> str:
    """ This function serves to join two parts of a query into one string (when the citation has been discovered in two consecutive parts of the query document). """
    output = ''

    sentences_in_1 = sent_tokenize(list_of_parts[query_index])
    try:
        sentences_in_2 = sent_tokenize(list_of_parts[query_index+1])
    except IndexError:
        print(sentences_in_1)
        print(list_of_parts[-1])

    for sent_1 in sentences_in_1:
        if sent_1 not in sentences_in_2:
            output += sent_1 + ' '
        else:
            break

    for sent_2 in sentences_in_2:
        output += sent_2 + ' '

    return output.strip()


def fuzzy_string_matching_for_implementation_with_text(subverse_string:str, query_string:str, tolerance=0.85):
    """ 
    Contrary to fuzzy_string_matching_for_implementation(), this function also returns the matched part of the query string and the edit distance of the compared strings. The function is duplicated so as not to speed down the function in the broad search. However, the speed difference has not been tested yet.

    This function is for implementation of typo similarity detection applied to two strings. It returns bool value of match.

    :param subverse_string: string of the biblical subverse we are searching for.
    :param query_string: string in which we are searching for the seubverse_string.
    :param tolerance: how large proportion of the subverse_string must be present in query_string to consider it a match.
    """
    subverse_string = normalize_string(subverse_string)
    subverse_len = len(subverse_string)

    query_string = normalize_string(query_string)
    query_len = len(query_string)

    tolerance = subverse_len * (1-tolerance)

    if subverse_len-tolerance > query_len:
        # If subverse is longer than query string, it is not a match by default
        return False, '', 0
    elif subverse_len-tolerance <= query_len <= subverse_len+tolerance:
        # If subverse is more or les of the same length as query string, just compare them.
        edit_distance = distance(subverse_string, query_string)
        if edit_distance <= tolerance:
            return True, query_string, edit_distance
    else:
        char_len_sub = len(subverse_string)
        word_len_subv = len(word_tokenize(subverse_string))
        words_in_query_string = word_tokenize(query_string)
        word_len_query_string = len(words_in_query_string)

        for i, cycle in enumerate(range(word_len_subv, word_len_query_string+1)):
            gram_str = ' '.join(words_in_query_string[i:])[:char_len_sub]
            edit_distance = distance(subverse_string, gram_str)
            if edit_distance <= tolerance:
                return True, gram_str, edit_distance
            else:
                continue
    
    return False, '', 0


def check_for_verse(verse_id:str, string_to_check:str) -> dict:
    """ This function performs the inner check for a verse in all availiable translations. It is implemented in the check_results() function. """
    possible_citations = []

    for trsl in all_translations:
        verse_text = get_verse_text(trsl, verse_id, print_exceptions=False)
        
        if verse_text:
            subverses = split_verse(verse_text, tole_len=21, return_shorts=True, short_limit=9)

            fuzzy_matched_subs_num = 0
            fuzzy_matched_subs = []
            matched_subs_edit_distance = 0
            matched_subs_chars = 0
            exclusive_matched_subs_num = 0

            for subverse in subverses:
                # check for every subverse in edit distance
                fuzzy_match, query_match, edit_distance = fuzzy_string_matching_for_implementation_with_text(subverse, query_string=string_to_check, tolerance=0.85)
                if fuzzy_match:
                    fuzzy_matched_subs_num += 1
                    fuzzy_matched_subs.append(subverse)
                    matched_subs_edit_distance += edit_distance
                    matched_subs_chars += len(subverse)

                    # run the exclussiveness test
                    if exclusiveness_test(subverse, query_match):
                        exclusive_matched_subs_num += 1

                else:
                    continue

            if fuzzy_matched_subs_num == 0:
                continue
            else:
                matched_characters = (matched_subs_chars-matched_subs_edit_distance)/matched_subs_chars
                matched_subverses_score = fuzzy_matched_subs_num/len(subverses)

                match_probability = matched_characters*matched_subverses_score

                result_for_trsl = {'verse_id': verse_id,
                                    'verse_text': verse_text, 
                                    'matched_subverses': fuzzy_matched_subs, 
                                    'query_string': string_to_check, 
                                    'matched_characters': (matched_subs_chars-matched_subs_edit_distance)/matched_subs_chars, 
                                    'matched_subverses_score': fuzzy_matched_subs_num/len(subverses),
                                    'exclusives_match': exclusive_matched_subs_num/fuzzy_matched_subs_num,
                                    'match_probability': match_probability}
            
                possible_citations.append(result_for_trsl)

    # If there are none possible citation (which is weird and it should not happen and it probably means that there are differently split verses in the originally used BibleDataset) return result that are basically False:
    if not possible_citations:
        result = {'verse_id': verse_id,
                    'verse_text': verse_text, 
                    'matched_subverses': [], 
                    'query_string': string_to_check, 
                    'matched_characters': 0, 
                    'matched_subverses_score': 0,
                    'exclusives_match': 0,
                    'match_probability': 'FALSE'}
        
        return result
    
    # Now, if the results seem OK, select the best match (translations as such are not evaluated, just select the best result of all possible results)... in this evaluation, we consider the result with most detected subverses as a match, if same then based on the characters, and finally on the exclusiveness test results.
    matched_subverses_scores = [pc['matched_subverses'] for pc in possible_citations]
    matched_characters_scores = [pc['matched_characters'] for pc in possible_citations]
    exclusiveness_test_scores = [pc['exclusives_match'] for pc in possible_citations]

    # Check subverses score results:
    best_subverses_match = max(matched_subverses_scores)
    if matched_subverses_scores.count(best_subverses_match) == 1:
        best_pc_idx = matched_subverses_scores.index(best_subverses_match)
        return possible_citations[best_pc_idx]
    else:
        # check the character scores results:
        idxs = [i for i, score in enumerate(matched_subverses_scores) if score == best_subverses_match]
        best_chars_match = max([matched_characters_scores[i] for i in idxs])
        if matched_characters_scores.count(best_chars_match) == 1:
            best_pc_idx = matched_characters_scores.index(best_chars_match)
            return possible_citations[best_pc_idx]
        else:
            # check exclusiveness test results:
            idxs = [i for i, score in enumerate(matched_characters_scores) if score == best_chars_match]
            best_excl_res = max([exclusiveness_test_scores[i] for i in idxs])
            best_pc_idx = exclusiveness_test_scores.index(best_excl_res)
            return possible_citations[best_pc_idx]


def load_data_from_journals_fulldata(journals_fulldata:dict, query_file:str):
    journal = journals_fulldata[query_file]['journal']
    issue_date = journals_fulldata[query_file]['issue_date']
    issue_page = journals_fulldata[query_file]['issue_page']
    issue_uuid = journals_fulldata[query_file]['issue_uuid']
    kramerius_url = journals_fulldata[query_file]['kramerius_url']
    full_query_string = journals_fulldata[query_file]['text']

    return journal, issue_date, issue_page, issue_uuid, kramerius_url, full_query_string


def evaluate_attributions_in_doc(attributed_verses:dict, query_file:str, query_window_len:int, query_overlap:int, journals_fulldata:dict) -> list:
    """ This function evaluates attributed verses, supposedly detected in a single JSON file. """
    # Load data from journals_fulldata dictionary:
    journal, issue_date, issue_page, issue_uuid, kramerius_url, full_query_string = load_data_from_journals_fulldata(journals_fulldata=journals_fulldata, query_file=query_file)
    
    # NOTE: repair wrong date with 334149b0-877c-11e6-8aeb-5ef3fc9ae867
    if '334149b0-877c-11e6-8aeb-5ef3fc9ae867' in issue_uuid:
        issue_date = '30.06.1935'

    query_parts = split_query(full_query_string, window_len=query_window_len, overlap=query_overlap)
    
    results_of_attributions = []
  
    for verse_id in attributed_verses:
        attributed_idxs = attributed_verses[verse_id]
        if len(attributed_idxs) == 1:
            string_to_check = query_parts[attributed_idxs[0]]
            possible_citation = check_for_verse(verse_id=verse_id, string_to_check=string_to_check)
            results_of_attributions.append(possible_citation)
            
        else:
            skip = False
            for i, q_idx in enumerate(attributed_idxs):
                if not skip:
                    try:
                        if attributed_idxs[i+1] == q_idx+1:
                            # checking if the next part is a joined sequence
                            skip = True
                            string_to_check = join_overlap(query_parts, q_idx)
                            possible_citation = check_for_verse(verse_id=verse_id, string_to_check=string_to_check)
                            results_of_attributions.append(possible_citation)
                        else:
                            string_to_check = query_parts[attributed_idxs[0]]
                            possible_citation = check_for_verse(verse_id=verse_id, string_to_check=string_to_check)
                            results_of_attributions.append(possible_citation)
                    except IndexError:
                        string_to_check = query_parts[attributed_idxs[i]]
                        possible_citation = check_for_verse(verse_id=verse_id, string_to_check=string_to_check)
                        results_of_attributions.append(possible_citation)
                else:
                    skip = False
                    continue

    # TODO: zde pak přidat všechny další parametry nalezené citace --> pak se to vrátí a přidá do výsledného DF.
    if len(results_of_attributions) == 1:
        results_of_attributions[0]['multiple_attribution'] = False
        results_of_attributions[0]['journal'] = journal
        results_of_attributions[0]['date'] = issue_date
        results_of_attributions[0]['page_num'] = issue_page
        results_of_attributions[0]['uuid'] = issue_uuid
        results_of_attributions[0]['kramerius_url'] = kramerius_url
    else:
        for res in results_of_attributions:
            res['multiple_attribution'] = True
            res['journal'] = journal
            res['date'] = issue_date
            res['page_num'] = issue_page
            res['uuid'] = issue_uuid
            res['kramerius_url'] = kramerius_url

    return results_of_attributions


def make_filtered_search_dataframe(results_filename='UNFILTERED_batch_results.csv', save=True, return_df=False):
    """ This functions applies initial checks on the preliminary results. """
    # Load results:
    print('Loading UNFILTERED results...')
    results_dataframe = load_results(results_filename, delimiter=';')

    # Load journals_fulldata:
    print('Loading journals_fulldata.joblib...')
    journals_full_data = joblib.load(join_path(ROOT_PATH, 'journals_fulldata.joblib'))

    # Create (empty) final results dataframe:
    final_results = {}
    res_id = 0

    print('Running initial filtering...')
    rows_to_skip = []
    print_progress = 0
    iter_ = 0
    for row_id in results_dataframe.index:
        print_progress += 1
        iter_ += 1
        if print_progress >= 500:
            print('\t', iter_, '/', len(results_dataframe))
            print_progress = 0

        if row_id in rows_to_skip:
            continue
        else:
            query_file, query_window_len, query_overlap = get_row_data_for_initial_filter(dataframe=results_dataframe, row_id=row_id)
            attributed_verses, add_to_skip = select_attributions_to_json(dataframe=results_dataframe, query_file=query_file)
            rows_to_skip.extend(add_to_skip)

            results = evaluate_attributions_in_doc(attributed_verses=attributed_verses, query_file=query_file, query_window_len=query_window_len, query_overlap=query_overlap, journals_fulldata=journals_full_data)

            for res in results:
                final_results[res_id] = res
                res_id += 1

    final_results_df = pd.DataFrame.from_dict(final_results)
    final_results_df = final_results_df.transpose()
    
    if save:
        final_results_df.to_csv(join_path(RESULTS_PATH, f'FILTERED_{results_filename}'), encoding='utf-8', quotechar='"', sep=';')

    if return_df:
        return final_results_df


""" FILTERING STOP-SUBVERSES """
# This function filter stop subverses if these are the only one detected (they are kept if there are more subverses detected in the citation)
# NOTE: define/change stop-subverses in evaluation_stop_subverses_21.txt


def filter_stop_subs(results_filename='FILTERED_UNFILTERED_batch_results.csv', input_df=False, subverse_len=21, rewrite_original_csv=False, save=True, return_df=False, save_filtered_out_file=True, filtered_out_filename='FILTERED_BY_STOP_SUBS.csv'):
    """
    This function filters those results that are detected based on solely one subverse that is listed in file evaluation_stop_subverses_{subverse_len}.txt
    
    Also, there are subverses in evaluation_stop_subverses_{subverse_len}.txt' that need 100% hit in characters in order to be taken seriously...
    """
    if input_df is not False:
        original_df = input_df
    else:
        print('Loading results dataframe ...')
        original_df = pd.read_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', delimiter=';', encoding='utf-8', index_col=0)

    print('Length of the original dataframe is:', len(original_df))

    print(f'Loading stop-subverses from evaluation_stop_subverses_{subverse_len}.txt ...')
    with open(join_path(ROOT_PATH, f'evaluation_stop_subverses_{subverse_len}.txt'), 'r', encoding='utf-8') as stops_f:
        data = stops_f.read()
        stop_subs = data.split('\n')

    print(f'Loading subverses that need 100 % hit from 100_hit_needed_subs_{subverse_len}.txt ...')
    with open(join_path(ROOT_PATH, f'evaluation_stop_subverses_{subverse_len}.txt'), 'r', encoding='utf-8') as stops_f:
        data = stops_f.read()
        full_hit_subs = data.split('\n')

    print('Number of stop subverses to filter:', len(set(stop_subs)))

    filtered_df_dict = {}
    filtered_out_dict = {}
    fil_id = 0
    fil_out_id = 0  

    print('Filtering rows ...')
    for row_id in original_df.index:
        # If the hit sontains only the stop-subverse, filter it
        if original_df.loc[row_id]['matched_subverses'] in stop_subs:
            row_as_dict = original_df.loc[row_id].to_dict()
            filtered_out_dict[fil_out_id] = row_as_dict
            fil_out_id += 1
        
        # If the hit contains only the subverse that need 100% hit, check it
        elif original_df.loc[row_id]['matched_subverses'] in full_hit_subs:
            matched_chars = eval(original_df.loc[row_id]['matched_characters'])
            if matched_chars == 1:
                row_as_dict = original_df.loc[row_id].to_dict()            
                filtered_df_dict[fil_id] = row_as_dict
                fil_id += 1
            else:
                row_as_dict = original_df.loc[row_id].to_dict()
                filtered_out_dict[fil_out_id] = row_as_dict
                fil_out_id += 1

        # Otherwise, take the citation as OK
        else:
            row_as_dict = original_df.loc[row_id].to_dict()            
            filtered_df_dict[fil_id] = row_as_dict
            fil_id += 1

    filtered_df = pd.DataFrame.from_dict(filtered_df_dict)
    filtered_df = filtered_df.transpose()

    print('Length of the filtered dataframe is:', len(filtered_df))
    print('Number of filtered rows:', len(original_df)-len(filtered_df))

    if rewrite_original_csv:
        filtered_df.to_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', sep=';', encoding='utf-8')
    
    if save:
        filtered_df.to_csv(join_path(RESULTS_PATH, f'ST_SUBS_{results_filename}'), quotechar='"', sep=';', encoding='utf-8')

    if save_filtered_out_file:
        filtered_out_df = pd.DataFrame.from_dict(filtered_out_dict)
        filtered_out_df = filtered_out_df.transpose()

        filtered_out_df.to_csv(join_path(RESULTS_PATH, filtered_out_filename), quotechar='"', sep=';', encoding='utf-8')
    
    if return_df:
        return filtered_df


""" FILTERING 'HIDDEN DUPLICATES' """


def get_row_data_for_overlap_dups(dataframe:pd.core.frame.DataFrame, row_id:int):
    """ This function returns search properties of a given row in the results dataframe to be used in check_results function. """

    verse_id = dataframe.loc[row_id]['verse_id']
    uuid = dataframe.loc[row_id]['uuid']
    page_num = dataframe.loc[row_id]['page_num']

    return verse_id, uuid, page_num


def select_attributions_to_same_page_et_verse(dataframe:pd.core.frame.DataFrame, uuid:str, page_num:int, verse_id:str):
    """ This function selects all attributions to a given uuid, page nuber and verse ID
    
    It returns: dataframe of all of the results, row_ids to skip
    """
    subset_dataframe = dataframe[dataframe['uuid'] == uuid]
    subset_dataframe = subset_dataframe[subset_dataframe['page_num'] == page_num]
    subset_dataframe = subset_dataframe[subset_dataframe['verse_id'] == verse_id]

    return subset_dataframe, subset_dataframe.index


def is_string_in_other_string(str_0:str, str_1:str):
    """ This function checks if one of two string contain the other. It returns bool and what string to discard """
    if str_0 in str_1:
        return True, 0
    elif str_1 in str_0:
        return True, 1
    else:
        return False, None


def compare_overlapped_query_string(subset_dataframe:pd.core.frame.DataFrame):
    """ This function compares query strings of a subset dataframe (as filtered by select_attributions_to_same_page_et_verse function). """
    query_strings = {}
    for row_id in subset_dataframe.index:
        query_string = subset_dataframe.loc[row_id]['query_string']
        query_strings[row_id] = query_string

    output_rows = []
    rows_dropped = 0
    for qs_a in query_strings:
        for qs_b in query_strings:
            if qs_a == qs_b:
                continue
            else:
                is_overlap, qs_to_drop = is_string_in_other_string(query_strings[qs_a], query_strings[qs_b])
                if is_overlap:
                    if qs_to_drop == 1:
                        row_dict = subset_dataframe.loc[qs_a].to_dict()
                        output_rows.append(row_dict)
                        rows_dropped += 1
                else:
                    row_dict = subset_dataframe.loc[qs_a].to_dict()
                    output_rows.append(row_dict)

    return output_rows, rows_dropped


def filter_duplicates_by_overlap(results_filename='ST_SUBS_FILTERED_UNFILTERED_batch_results.csv', input_df=False, subverse_len=21, rewrite_original_csv=False, save=True, return_df=False):
    """ This function filters those duplicates that do not seem as full duplicates, because the query string is different. However, sometimes (due to overlaps), there are some matches that include full query string of other match. """
    if input_df is not False:
        results_dataframe = input_df
    else:
        print('Loading results dataframe ...')
        results_dataframe = pd.read_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', delimiter=';', encoding='utf-8', index_col=0)

    print('Length of the original dataframe is:', len(results_dataframe))

    # Create (empty) final results dataframe:
    final_results = {}
    res_id = 0

    print('Filtering "hidden" duplicates...')
    rows_to_skip = []
    rows_dropped = 0
    res_id = 0
    for row_id in results_dataframe.index:
        if row_id in rows_to_skip:
            continue
        else:
            verse_id, uuid, page_num = get_row_data_for_overlap_dups(dataframe=results_dataframe, row_id=row_id)
            
            subset_dataframe, add_to_skip = select_attributions_to_same_page_et_verse(dataframe=results_dataframe, uuid=uuid, page_num=page_num, verse_id=verse_id)
            rows_to_skip.extend(add_to_skip)

            if len(subset_dataframe) == 1:
                row_dict = results_dataframe.loc[row_id].to_dict()
                final_results[res_id] = row_dict
                res_id += 1
            else:
                row_dicts, num_of_rows_dropped = compare_overlapped_query_string(subset_dataframe=subset_dataframe)
                rows_dropped += num_of_rows_dropped
                for rd in row_dicts:
                    final_results[res_id] = rd
                    res_id += 1

    final_results_df = pd.DataFrame.from_dict(final_results)
    final_results_df = final_results_df.transpose()

    print('Number of dropped rows:', rows_dropped)

    if rewrite_original_csv:
        final_results_df.to_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', sep=';', encoding='utf-8')
    
    if save:
        final_results_df.to_csv(join_path(RESULTS_PATH, f'DUPS_{results_filename}'), quotechar='"', sep=';', encoding='utf-8')
    
    if return_df:
        return final_results_df


""" RESOLVING MULTIPLE ATTRIBUTIONS """


def select_multiply_attributed_rows(dataframe:pd.core.frame.DataFrame, row_id):
    """ This finction selects all rows that share same multiple attribution. """
    uuid = dataframe.loc[row_id]['uuid']
    query_string = dataframe.loc[row_id]['query_string']

    other_attributions_df = dataframe[dataframe['uuid'] == uuid]
    other_attributions_df = other_attributions_df[other_attributions_df['multiple_attribution'] == True]
    other_attributions_df = other_attributions_df[other_attributions_df['query_string'] == query_string]

    row_ids_to_skip = other_attributions_df.index

    return other_attributions_df, row_ids_to_skip


def fuzzy_string_matching_for_multiple_attributions(subverse_string:str, query_string:str, tolerance=0.85):
    """ 
    This function is used to evaluate multiple attributions within the same query string. The function returns parts of query string thatare the supposed match - if these overlap then it is multiple attribution, if not, there are probably just more verses cited within one passage.

    :param subverse_string: string of the biblical subverse we are searching for.
    :param query_string: string in which we are searching for the seubverse_string.
    :param tolerance: how large proportion of the subverse_string must be present in query_string to consider it a match.
    """
    subverse_string = normalize_string(subverse_string)
    subverse_len = len(subverse_string)

    query_string = normalize_string(query_string)
    query_len = len(query_string)

    tolerance = subverse_len * (1-tolerance)

    if subverse_len-tolerance > query_len:
        # If subverse is longer than query string, it is not a match by default
        return ()
    elif subverse_len-tolerance <= query_len <= subverse_len+tolerance:
        # If subverse is more or les of the same length as query string, just compare them.
        edit_distance = distance(subverse_string, query_string)
        if edit_distance <= tolerance:
            return (0, len(query_string))
    else:
        char_len_sub = len(subverse_string)
        word_len_subv = len(word_tokenize(subverse_string))
        words_in_query_string = word_tokenize(query_string)
        word_len_query_string = len(words_in_query_string)

        for i, cycle in enumerate(range(word_len_subv, word_len_query_string+1)):
            gram_str = ' '.join(words_in_query_string[i:])[:char_len_sub]
            edit_distance = distance(subverse_string, gram_str)
            if edit_distance <= tolerance:
                return (i, (char_len_sub+i))
            else:
                continue
    
    return ()


def check_consecutive(input_list:list):
    return sorted(input_list) == list(range(min(input_list), max(input_list)+1))


def return_ranges_if_not_consecutives(full_range:list):
    """ This function returns ranges of non-consecutive match. """
    ranges = []
    range_start = full_range[0]
    for i, value in enumerate(full_range):
        try:
            if value+1 == full_range[i+1]:
                continue
            else:
                ranges.append((range_start, value))
                range_start = full_range[i+1]
        except IndexError:
            ranges.append((range_start, value))
            continue
    
    return ranges


def make_full_range(subverses:list, query_string:str):
    """ This function returns the full range of matched subverses in the query string and bool whether the matched subverses are consecutively present in the query string. """
    subs_range = []
    for sub in subverses:
        gram_str_range = fuzzy_string_matching_for_multiple_attributions(sub, query_string)
        for i in range(gram_str_range[0], gram_str_range[1]+1):
            if i not in subs_range:
                subs_range.append(i)
    
    if check_consecutive(subs_range):
        return [(min(subs_range), max(subs_range))]
    
    else:
        subs_range.sort()
        return return_ranges_if_not_consecutives(subs_range)


def make_all_ranges_into_all_other_ranges(all_ranges_list:list, list_to_remove:list):
    """ This function removes matched ranges of a selected subverse from the list of all ranges, so it is compared only with other ranges, not with itself. Theoretically this should not be really a problem, but I find it better to deal with it anyhow.  """
    reduced_list = all_ranges_list.copy()
    for item_to_remove in list_to_remove:
        reduced_list.remove(item_to_remove)

    return reduced_list


def check_overlap_of_subs_ranges(subs_ranges:list):
    """
    This function checks if there are overlaps between ranges of subverses within the query string.
    
    :param subs_ranges: list of lists of ranges (e.g., [[(1,3)], [(5,6), (8,15)]])
    """
    overlap_stats = []
    for a, s_range_a in enumerate(subs_ranges):
        overlaps_of_sub_a = {}
        for b, s_range_b in enumerate(subs_ranges):
            if a == b:
                continue
            else:
                current_stat = False
                for a_range in s_range_a:
                    for b_range in s_range_b:
                        try:
                            for i in range(a_range[0], a_range[1]+1):
                                if i in range(b_range[0], b_range[1]+1):
                                    current_stat = True
                                    break

                        except IndexError:
                            continue
                overlaps_of_sub_a[b] = current_stat

        overlap_stats.append(overlaps_of_sub_a)
    
    return overlap_stats


def evaluate_multiple_attributions(subset_dataframe:pd.core.frame.DataFrame):
    """ This function evaluates the DROP value of respective rows. """
    match_probability_values = []
    verse_ids = []
    matched_subverses = []
    query_strings = []
    for row_id in subset_dataframe.index:
        match_probability_values.append(subset_dataframe.loc[row_id]['match_probability'])
        matched_subverses.append(eval(subset_dataframe.loc[row_id]['matched_subverses']))
        verse_ids.append(subset_dataframe.loc[row_id]['verse_id'])
        query_strings.append(subset_dataframe.loc[row_id]['query_string'])
    
    matched_ranges = []
    for matched_subs in matched_subverses:
        # NOTE: the query_strings should be all the same, so we can just select the first one
        sub_ranges = make_full_range(matched_subs, query_string=query_strings[0])
        matched_ranges.append(sub_ranges)
    
    # Get overlap states of each mach to all other matches.
    overlap_stats = check_overlap_of_subs_ranges(matched_ranges)

    output_dicts = []

    num_of_rows_to_drop = 0
    
    for i, mpv in enumerate(match_probability_values):
        # get overlap states of the matchon position "i":
        i_overlaps = overlap_stats[i]
        overlapped_positions = [i]
        for i_match in i_overlaps:
            # If there is some other match that overlaps with this specific match, get its id
            if i_overlaps[i_match]:
                overlapped_positions.append(i_match)

        # If there are some overlaps, we will drop the current match if it does not have the highest probability value of all overlapping matches
        if len(overlapped_positions) > 1:
            scores_of_overlapped_matches = []
            for op in overlapped_positions:
                scores_of_overlapped_matches.append(match_probability_values[op])
            if mpv == max(scores_of_overlapped_matches):
                to_drop = False
            else:
                to_drop = True
                num_of_rows_to_drop += 1

        # But if there are no overlaps, we will not drop this match:
        else:
            to_drop = False

        df_dict = subset_dataframe.loc[subset_dataframe.index[i]].to_dict()
        df_dict['drop?'] = to_drop
        output_dicts.append(df_dict)

    return output_dicts, num_of_rows_to_drop


def mark_multiple_attributions(results_filename='DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv', input_df=False, rewrite_original_csv=False, save=True, return_df=False):
    """ This function suggest which of the multiple attribution is the right one. """
    if input_df is not False:
        original_df = input_df
    else:
        print('Loading results dataframe ...')
        original_df = pd.read_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', delimiter=';', encoding='utf-8', index_col=0)

    output_df_dict = {}
    out_idx = 0

    num_of_rows_to_drop = 0

    rows_to_skip = []

    print('Evaluating multiple attributions ...')
    for row_id in original_df.index:
        if row_id in rows_to_skip:
            continue
        else:
            if original_df.loc[row_id]['multiple_attribution']:
                other_attributions_df, add_to_skip = select_multiply_attributed_rows(dataframe=original_df, row_id=row_id)
                rows_to_skip.extend(add_to_skip)
                if len(other_attributions_df) == 1:
                    row_as_dict = original_df.loc[row_id].to_dict()
                    row_as_dict['drop?'] = False
                    output_df_dict[out_idx] = row_as_dict
                    out_idx += 1
                else:
                    rows_to_add, rows_to_drop_count = evaluate_multiple_attributions(subset_dataframe=other_attributions_df)
                    num_of_rows_to_drop += rows_to_drop_count
                    for rta in rows_to_add:
                        output_df_dict[out_idx] = rta
                        out_idx += 1
            else:
                row_as_dict = original_df.loc[row_id].to_dict()
                row_as_dict['drop?'] = False
                output_df_dict[out_idx] = row_as_dict
                out_idx += 1

    filtered_df = pd.DataFrame.from_dict(output_df_dict)
    filtered_df = filtered_df.transpose()

    print('Number of rows selected for drop:', num_of_rows_to_drop, 'out of', len(original_df))

    if rewrite_original_csv:
        filtered_df.to_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', sep=';', encoding='utf-8')
    
    if save:
        filtered_df.to_csv(join_path(RESULTS_PATH, f'MA_{results_filename}'), quotechar='"', sep=';', encoding='utf-8')
    
    if return_df:
        return filtered_df


""" MARKING "SURE" CITATIONS """


def mark_sure_citations(results_filename='MA_DUPS_ST_SUBS_FILTERED_UNFILTERED_batch_results.csv', input_df=False, rewrite_original_csv=False, save=True, return_df=False):
    """ This function marks some of the citations as sure citations while other unsure. """
    if input_df is not False:
        original_df = input_df
    else:
        print('Loading results dataframe ...')
        original_df = pd.read_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', delimiter=';', encoding='utf-8', index_col=0)

    output_df_dict = {}
    out_idx = 0

    num_of_sure_citations = 0

    print('Marking "sure" citations ...')
    for row_id in original_df.index:
        row_as_dict = original_df.loc[row_id].to_dict()

        num_of_subverses_in_verse = len(split_verse(row_as_dict['verse_text'], tole_len=21))
        match_subs_score = row_as_dict['matched_subverses_score']

        if num_of_subverses_in_verse <= 2:
            if match_subs_score == 1 and row_as_dict['exclusives_match'] == 1:
                row_as_dict['CITATION'] = True
                num_of_sure_citations += 1
            else:
                row_as_dict['CITATION'] = False
        elif num_of_subverses_in_verse <= 4:
            if match_subs_score >= 0.8 and row_as_dict['exclusives_match'] == 1:
                row_as_dict['CITATION'] = True
                num_of_sure_citations += 1
            else:
                row_as_dict['CITATION'] = False

        else:
            if match_subs_score >= 0.5 and row_as_dict['exclusives_match'] == 1:
                row_as_dict['CITATION'] = True
                num_of_sure_citations += 1
            else:
                row_as_dict['CITATION'] = False

        output_df_dict[out_idx] = row_as_dict
        out_idx += 1
    
    filtered_df = pd.DataFrame.from_dict(output_df_dict)
    filtered_df = filtered_df.transpose()

    print('Number of rows selected as sure citations:', num_of_sure_citations, 'out of', len(original_df))

    if rewrite_original_csv:
        filtered_df.to_csv(join_path(RESULTS_PATH, results_filename), quotechar='"', sep=';', encoding='utf-8')
    
    if save:
        filtered_df.to_csv(join_path(RESULTS_PATH, f'FINAL_{results_filename}'), quotechar='"', sep=';', encoding='utf-8')
    
    if return_df:
        return filtered_df


""" RESOLVING "SAME" VERSES - WORK IN PROGRESS ... """
# TODO: The script that resolves if there are some verses that are actually "the same".

mutual_verses = {
    'L 11:3/Mt 6:11': ['L 11:3', 'Mt 6:11'],
    'Mk 13:31/Mt 24:35/L 21:33': ['Mk 13:31', 'Mt 24:35', 'L 21:33'],
    'Ex 20:16/Dt 5:20': ['Ex 20:16', 'Dt 5:20'],
    '2K 1:2/Fp 1:2/2Te 1:2/1K 1:3/Ef 1:2/Ga 1:3': ['2K 1:2', 'Fp 1:2', '2Te 1:2', '1K 1:3', 'Ef 1:2', 'Ga 1:3'],
    'Mt 11:15/Mt 13:9': ['Mt 11:15', 'Mt 13:9']   
}