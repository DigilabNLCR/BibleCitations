""" This python script includes all functions that are used in the Biblical Intertextuality projetc. """
import pandas as pd
import os
import joblib

# using these from xy import z improves the performace quite well...
from os import listdir as os_listdir
from os.path import isdir as os_path_isdir
from os.path import exists as os_exists
from os import remove as os_remove
from shutil import copyfile as shutil_copyfile
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
__version__ = '0.0.2'

""" DEFINING_PATHS------------------------------------------------------------------------------------------------- """
ROOT_PATH = os.getcwd()

BIBLES_PATH = join_path(ROOT_PATH, 'Bible_files')
QUERY_DOC_PATH = join_path(ROOT_PATH, 'query_documents')
DATASETS_PATH = join_path(ROOT_PATH, 'datasets')
DICTS_PATH = join_path(ROOT_PATH, 'dictionaries')
CORPUS_PATH = join_path(ROOT_PATH, 'corpuses')
RESULTS_PATH = join_path(ROOT_PATH, 'results')
ALL_JSONS_PATH = os.path.join(ROOT_PATH, 'query_jsons_archive')

BATCHES_FILE_PATH = join_path(ROOT_PATH, 'batches.csv')
BATCH_RESULTS_FILE_PATH = join_path(RESULTS_PATH, 'batch_results.csv')

STOP_WORDS_PATH = join_path(ROOT_PATH, 'stop_words.txt')
STOP_SUBVERSES_PATH = join_path(ROOT_PATH, 'stop_subverses_21.txt')


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


def split_verse(input_text:str, tole_len=21, return_shorts=False, short_limit=9) -> list:
    """ This function ensures verse splitting into smaller subverses.
    :param input_text: text of verse that is to be split.
    :param tole_len: minimal length of subverse in characters.
    """
    if len(input_text) < tole_len:
        if return_shorts:
            if len(input_text) <= short_limit:
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


def save_dataset(dataset, dataset_name='completeBibleDataset'):
    """ Saving dataset has default name, because there is supposedly only one version of it. """
    joblib.dump(dataset, join_path(DATASETS_PATH, f'{dataset_name}.joblib'))


def load_dataset(dataset_name='completeBibleDataset'):
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


def bible_to_dataset(save=True, ignore_stop_subs=True, return_shorts=True,  dataset_name='completeBibleDataset'):
    """ This function prepares dataset into bibleDataset class.

    :param ignore_stop_subs: if True, the stop subverses defined in stop_subverses_21 are ignored.
    :param save: set False f you do not want to save the dataset.
    :param dataset_name: filename of the saved dataset.
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
                subverses = split_verse(verses_dict[verse_id], return_shorts=return_shorts)
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
        save_dataset(bible_dataset, dataset_name=dataset_name)

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


def list_query_docs_json():
    """ This function returns a list of referential docs in folder query_documents. """
    list_of_query_docs = os_listdir(QUERY_DOC_PATH)
    return list_of_query_docs


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


""" CREATING AND LOADING NECESSARY OBJECTS ------------------------------------------------------------------------- """


def create_necessary_objects(ngram_size=4, skip_dataset=True, dataset=None, save_objects=True, out_prefix='completeBible'):
    """
    This function creates all necessary objects for the search. Run this function if you are starting the process or if you have changed the dataset or functions that create it, otherwise it is not necessary.

    :param ngram_size: size of ngrams (in characters) to which everything is parsed; According to it, other objects for search are loaded.
    :param skip_dataset: if the dataset exists, it is loaded instead of created.
    :param dataset: dataset can be also loaded externaly, so it does not have to be loaded for every iteration.
    :param save_objects: if True, objects are saved, if False, objects are only returned.
    """

    if dataset:
        bible_dataset = dataset
    else:    
        if skip_dataset:
            print('Dataset already exists --> loaded.')
            bible_dataset = load_dataset(f'{out_prefix}Dataset')
        else:
            start_ = time()
            print('Creating bible dataset...')
            bible_dataset = bible_to_dataset()
            save_dataset(bible_dataset, f'{out_prefix}Dataset')
            end_ = time()
            print(f'Dataset has been created in {round((end_-start_)/60, 2)} minutes. Saved as {out_prefix}Datset.joblib')

    start_ = time()
    print('Processing corpus and creating dictionary...')
    dictionary, processed_corpus = process_corpus(bible_dataset, ngram_size=ngram_size)
    if save_objects:
        save_dictionary(dictionary, f'n{ngram_size}_{out_prefix}Dict')
    end_ = time()
    print(f'Dictionary has been created in {round((end_-start_), 2)} seconds. Saved as n{ngram_size}_{out_prefix}Dict.joblib')

    start_ = time()
    print('Creating corpus...')
    corpus = create_corpus(dictionary, processed_corpus)
    if save_objects:
        save_corpus(corpus, f'n{ngram_size}_{out_prefix}Corpus')
    end_ = time()
    print(f'Corpus has been created in {round((end_-start_), 2)} seconds. Saved as n{ngram_size}_{out_prefix}Corpus.mm')

    return corpus, dictionary


def load_necessary_objects(ngram_size=4):
    """
    This function loads all necessary objects for the search.

    :param ngram_size: int; size of ngrams to which everything is parsed; According to it,
        other objects for search are loaded.
    """
    dataset = load_dataset('completeBibleDataset')
    corpus = load_corpus(f'n{ngram_size}_completeBibleCorpus')
    dictionary = load_dictionary(f'n{ngram_size}_completeBibleDict')

    subverses = transfer_corpus_to_simple_token_vectors(corpus)

    return dataset, corpus, dictionary, subverses


""" SEARCH FUNCTIONS AND CLASSES ----------------------------------------------------------------------------------- """


class bibleObject:
    def __init__(self, dataset:bibleDataset, create_anew_other_necessary_objects=False, ngram_size=4, objects_prefix='completeBible'):
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


def search_for_bible(dataset: bibleDataset, ngram_tolerance=0.7, edit_distance_tolerance=0.85, ngram_size=4,
                     query_window_len=4, query_overlap=1) -> dict:
    """
    This function executes search for Bible quotations within all JSON documents placed in QUERY_DOC_PATH.

    .param dataset: an input dataset of bibleDataset class.
    :param ngram_tolerance: what portion of ngrams of Bible subverse must match to be considered as a match.
    :param edit_distance_tolerance: what portion of characters of a subverse must match a sequence from a
        "query document".
    :param ngram_size: size of ngrams (in characters) to which everything is parsed; According to it,
        other objects for search are loaded.
    :param query_window_len: How many sentences are put together as smaller "query documets".
        This parameter can influence the speed of the process, depending on the nature of the document.
    :param query_overlap: Overlap of the sentences among "query documents". Must be higher than query_window_len.
    :return: Detected citations.
    """
    print('Getting necesarry objects from the dataset...')
    bible_object = bibleObject(dataset, ngram_size=ngram_size)

    attributed_subverses = bible_object.attr_subs
    subverse_lens = bible_object.sub_lens

    dictionary = bible_object.dictionary

    query_files = os_listdir(QUERY_DOC_PATH)
    num_of_query_files = len(query_files)

    discovered_citations = defaultdict(list)

    print('Initiating search in query documents...')
    for qi, query_file in enumerate(query_files):
        query_time_start = time()
        print(f'Analysing document ({qi + 1}/{num_of_query_files}) {query_file}')

        data = load_json_data(join_path(QUERY_DOC_PATH, query_file))
        query_documents = split_query(data['text'], window_len=query_window_len, overlap=query_overlap)

        for i, query_doc in enumerate(query_documents):
            # First stage - compare vectors (token = n-gram of ngram_size)
            results_by_ngrams = compare_vector(query_doc, attributed_subverses=attributed_subverses,
                                               subverse_lens=subverse_lens, dictionary=dictionary,
                                               tolerance=ngram_tolerance, ngram_size=ngram_size)

            # Second stage - compare by fuzzy string matching
            for subverse_id in results_by_ngrams:
                try:
                    subverse_text = dataset.data[subverse_id]
                except IndexError:
                    # TODO: tohle zčeknout a vyřadit!
                    print('ERRROOROROROR', subverse_id)
                match = fuzzy_string_matching_for_implementation(subverse_string=subverse_text, query_string=query_doc,
                                                                 tolerance=edit_distance_tolerance)
                if match:
                    discovered_citations[subverse_id].append((query_file, i))

                    print('POSSIBLE MATCH: ')
                    print(f'\tsubverse text: {subverse_text} ({dataset.target[subverse_id]})' +
                          f'\n\tquery text: {query_doc}' + f'\n\tdiscovered in: {query_file} {i}')

        query_time_end = time()
        print(f'\tDocument analysed in {round((query_time_end - query_time_start), 2)} seconds.')

    return discovered_citations


def search_for_bible_for_batches_implementation(bible_object:bibleObject, batch_id:int, ngram_tolerance=0.7, edit_distance_tolerance=0.85, ngram_size=4, query_window_len=4, query_overlap=1):
    """
    This function is appropriated for implementation within search  by batches (in run_search_by_batch() function)
    This function executes search for Bible quotations within all JSON documents placed in QUERY_DOC_PATH.

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
    
    query_files = os_listdir(QUERY_DOC_PATH)
    num_of_query_files = len(query_files)

    discovered_citations = defaultdict(list)
    
    print('Initiating search in query documents...')
    for qi, query_file in enumerate(query_files):
        query_time_start = time()
        print(f'\tBatch {batch_id} ... Analysing document ({qi+1}/{num_of_query_files}) {query_file}')
        
        data = load_json_data(join_path(QUERY_DOC_PATH, query_file))
        query_documents = split_query(data['text'], window_len=query_window_len, overlap=query_overlap)

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
    """ This function list all json files as stored in folder "extracted_query_jsons"."""
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
    # create batches.csv if not exists
    if not os_exists(BATCHES_FILE_PATH):
        empty_df = pd.DataFrame(columns=batches_columns)
        pd.DataFrame.to_csv(empty_df, BATCHES_FILE_PATH)


def update_batches_csv(clear=False, max_batch_size=500):
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


def extract_batch(batch_id: int):
    """
    This function extracts json files from dir. extracted_query_path to dir. query_documents based on theor batch_id.
    """
    # first, delete files in the query_documents directory:
    files_in_query = os_listdir(QUERY_DOC_PATH)
    print(f'Deleting previous batch. Number of files in previous batch: {len(files_in_query)}')
    for q_f in files_in_query:
        os_remove(join_path(QUERY_DOC_PATH, q_f))

    # load batches_csv and extract data for the relevant batch:
    batches_df = pd.read_csv(BATCHES_FILE_PATH)
    relevant_data = batches_df.loc[batches_df['batch_id'] == batch_id]
    journals_to_extract = relevant_data['journal'].to_list()
    jsons_to_extract = relevant_data['json_file'].to_list()

    # copy the batch files to query_documents dir.:
    print(f'... copying files in batch {batch_id}')
    for i, json_f in enumerate(jsons_to_extract):
        original = join_path(ALL_JSONS_PATH, journals_to_extract[i], json_f)
        target = join_path(QUERY_DOC_PATH, json_f)
        shutil_copyfile(original, target)

    print(f'Batch {batch_id} moved to query_documents; number of files in current batch: {len(os_listdir(QUERY_DOC_PATH))}')


def create_batches_results_csv(results_columns_names:list):
    """ Create batch_results.csv if not exists. """
    if not os_exists(BATCH_RESULTS_FILE_PATH):
        empty_df = pd.DataFrame(columns=results_columns_names)
        pd.DataFrame.to_csv(empty_df, BATCH_RESULTS_FILE_PATH)


def change_run_log(batch_id:int, avereage_pre_page:float):
    """ Changes run log in batches.csv. """
    batches_df = pd.read_csv(BATCHES_FILE_PATH)
    batches_df.loc[batches_df['batch_id'] == batch_id, "run"] = True
    batches_df.loc[batches_df['batch_id'] == batch_id, "runtime"] = avereage_pre_page
    batches_df.to_csv(BATCHES_FILE_PATH, index=False, columns=batches_columns)


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
    results_columns_names = ['verse_id', 'query_file', 'index_query_part', 'batch_id', 'ngram_size', 'query_window_len', 'query_overlap', 'ngram_tolerance', 'edit_distance_tolerance']
    # If result CSV do not exist, create it:
    create_batches_results_csv(results_columns_names)

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


def run_search_by_batch(batch_id:int, bible_object:bibleObject, ngram_tolerance=0.7, edit_distance_tolerance=0.85, ngram_size=4, query_window_len=4, query_overlap=1):
    """ This function executes search by a batch_id (as linked to json files in batches.csv) and saves it results to batch_results.csv. """
    bible_dataset = bible_object.dataset
    
    # First, prepare query files:
    extract_batch(batch_id=batch_id)

    # Then, run search:
    print(f'... Initiating search of batch {batch_id}')
    batch_results, avg_time_per_page = search_for_bible_for_batches_implementation(bible_object=bible_object, batch_id=batch_id, ngram_tolerance=ngram_tolerance, edit_distance_tolerance=edit_distance_tolerance, ngram_size=ngram_size, query_window_len=query_window_len, query_overlap=query_overlap)

    # Save results:
    print(f'... Saving results of batch {batch_id}')
    save_batch_results(results=batch_results, dataset=bible_dataset, batch_id=batch_id, ngram_size=ngram_size, query_window_len=query_window_len, query_overlap=query_overlap, ngram_tolerance=ngram_tolerance, edit_distance_tolerance=edit_distance_tolerance)

    # Change run log in batches.csv:
    print(f'... Changing search log for {batch_id}')
    change_run_log(batch_id=batch_id, avereage_pre_page=avg_time_per_page)


def search_by_batches(batches_to_run:list, bible_dataset_filename='completeBibleDataset', skip_done=True, ngram_tolerance=0.7, edit_distance_tolerance=0.85, ngram_size=4, query_window_len=4, query_overlap=1):
    """ This function executes search across a number of batches.
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

    batches_to_skip = []
    if skip_done:
        # TODO: tohle upravit, protože už tam nemám True, ale mám tam čas!!! --> to musím rozdělit na odlišné sloupce!!!
        batches_df = pd.read_csv(BATCHES_FILE_PATH)
        relevant_data = batches_df.loc[batches_df['run'] == True]
        batches_to_skip = list(set(relevant_data['batch_id'].to_list()))

    for batch_id in batches_to_run:
        if batch_id in batches_to_skip:
            print(f'Batch {batch_id} has already been run.')
            continue
        else:
            run_search_by_batch(batch_id=batch_id, bible_object=bible_object, ngram_tolerance=ngram_tolerance, edit_distance_tolerance=edit_distance_tolerance, ngram_size=ngram_size, query_window_len=query_window_len, query_overlap=query_overlap)

    print('SEARCH FINISHED')


if __name__ == "__main__":
    low_batch = input('Set the lowest batch_id to be searched in')
    high_batch = input('Set the highest batch_id to be searched in')
    batches = range(eval(low_batch), eval(high_batch)+1)
    search_by_batches(batches, bible_dataset_filename='completeBibleDataset', query_window_len=6, query_overlap=1)