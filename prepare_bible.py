""" This script serves to create bible dataset and other necessary objects like dictionaries etc. """
import run_biblical_intertextuality as rbi

bible_dataset = rbi.bible_to_dataset(return_shorts=True, ignore_stop_subs=False, dataset_name='completeBibleDataset')
rbi.create_necessary_objects(ngram_size=4, dataset=bible_dataset, out_prefix='completeBible')