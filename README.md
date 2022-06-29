# BibleCitations
Welcome to the Git repository of a case study on searching Bible citations within czech press of 1925-1939. This is a part of DL4DH project.
See [web of DL4DH on nkp.cz](https://dl4dh.nkp.cz) for more information and results of the project (to be published by the end of September 2022)

- The script that is published here does not include all of the data available to replicate the case study. This is due to copyright both on journals material and some of the Biblical tranlsations.
- Nonetheless, the script can be used on your own material. It can be modified in any way and you can also add your own Biblical translations should you have access to them.

## Available tranlsations of Bible
- Following translations are freely available in this repository:
    - Bible Kralická (BKR) - Old and New Testament, see [Bible Kralická on obohu.cz](https://obohu.cz/bible/index.php?k=Gn&kap=1&styl=BKR)
    - Translation of Jan Hejčl (HEJCL) - Old Testament + Deuterocanonical books, see [translation of Jan Hejčl on obohu.cz](https://obohu.cz/bible/index.php?styl=HEJCL&k=Gn&kap=1)
    - Translation of Jan Ladislav Sýkora - New Testament, see [translation of Jan Ladislav Sýkora on obohu.cz](https://obohu.cz/bible/index.php?k=Mt&kap=1&styl=SYK)
    - Translation of František Žilka (ZP) - New Testament, see [translation of František Žilka on obohu.cz](https://obohu.cz/bible/index.php?styl=ZP&k=Mt&kap=1)
- We have also worked with an incomplete translation of "Bible Svatováclavská" that was provided us by prof. [Pavel Kosek from the Masaryk University](https://www.muni.cz/lide/4755-pavel-kosek) but we do not have a permission to make this one public.

# How to use this script

## 1) Preparation
- install required packages from the [requirements.txt](https://github.com/DigilabNLCR/BibleCitations/blob/main/requirements.txt) file
    - The virtual environment is yet in preparation...
- create necessary folders by running [prepare_environment.py](https://github.com/DigilabNLCR/BibleCitations/blob/main/prepare_environment.py)
- create biblical dataset objects by running [prepare_bible.py](https://github.com/DigilabNLCR/BibleCitations/blob/main/prepare_bible.py)
    - here, you can influence the process and performance by changing some of the parameters - for example by the n-gramming length, or by defining stop-subverses in [stop_subverses_21.txt](https://github.com/DigilabNLCR/BibleCitations/blob/main/stop_subverses_21.txt), or by ignoring the short subverses, or by changing stop words in [stop_words.txt](https://github.com/DigilabNLCR/BibleCitations/blob/main/stop_words.txt)
    - some more deep changes, like the length of subverse, need to be implemented in [run_biblical_intertextuality.py](https://github.com/DigilabNLCR/BibleCitations/blob/main/run_biblical_intertextuality.py)

## 2) Prepare your data
- prepare your data in folder "query_jsons_archive"
- each subset of your data must be in a separate folder (e.g. by a journal name), if you have only one subset, place it still in the subdiretory within "query_jsons_archive"
- each file must have a unique name (e.g. uuid)
- the data must be in a JSON format with following fields (in string!):
    - "journal"
    - "date"
    - "issue_uuid"
    - "page_num"
    - "text"
- should you wish to change the fields, you need to make changes to [run_biblical_intertextuality.py](https://github.com/DigilabNLCR/BibleCitations/blob/main/run_biblical_intertextuality.py), too.

## 3) Prepare batches to run
- Batch approach to search is used for convenience of searching through large dataset - without the need to save the results after each page which slows the process, and on the other hand providing the researcher with a possibility to stop tu search without loosing much of the process.
- do this by running [prepare_batches.py](https://github.com/DigilabNLCR/BibleCitations/blob/main/prepare_batches.py)
- this cleares the previous batches, so be careful.
- to set parameters (like batch size), change the setting in the python file.

## 4) Run search
- [run_biblical_intertextuality.py](https://github.com/DigilabNLCR/BibleCitations/blob/main/run_biblical_intertextuality.py)

## 5) Apply evaluation functions
- IN PROGRESS, this process serves to improve and filter the discovered citations.