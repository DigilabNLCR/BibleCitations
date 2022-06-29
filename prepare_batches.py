""" This script prepares batches to be runned by the run_biblical_intertextuality.py. """
import run_biblical_intertextuality as rbi

rbi.update_batches_csv(clear=True, max_batch_size=40)