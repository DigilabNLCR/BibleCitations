""" This script prepares batches to be runned by the biblical_intertextuality_package.py. """
import biblical_intertextuality_package as bip

bip.update_batches_csv(clear=True, max_batch_size=40)