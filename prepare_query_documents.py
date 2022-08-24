""" This python script prepares necessary files from JSON files in 'query_jsons' directory. """

import biblical_intertextuality_package as bip

bip.create_all_jsons_metadata_file()
bip.create_all_jsons_fulldata_file()