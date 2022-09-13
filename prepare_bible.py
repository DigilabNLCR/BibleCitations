""" This script serves to create bible dataset and other necessary objects like dictionaries etc. """
import biblical_intertextuality_package as bip

bible_dataset = bip.bible_to_dataset(save=True, ignore_stop_subs=False, dataset_prefix='fullBible', return_shorts=True,  tole_len=21, short_limit=9)
bip.create_necessary_objects(ngram_size=4, dataset=bible_dataset, objects_name='fullBible')