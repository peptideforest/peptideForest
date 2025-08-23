Peptide Forest
=============

Peptide Forest is a machine learning based tool for semisupervised integration of multiple peptide identification search engines.


Contributors
------------

* T. Ranff
* M. Dennison
* J. Bédorf
* S. Schulze
* N. Zinn
* M. Bantscheff
* J.J.R.M. van Heugten
* C. Fufezan

Usage
-----

We included an executeable `peptide_forest_3.py` which takes the following parameters:

```
   -c: the path to the config json
   -o: the path to the output
```

The config json has the following structure, e.g.:

```
{
    "input_files": {
        "tests/_data/mascot_dat2csv_1_0_0.csv": {
            "engine": "mascot",
            "score_col": "mascot:score"
        },
        "tests/_data/omssa_2_1_9.csv": {
            "engine": "omssa",
            "score_col": "omssa:pvalue"
        }
    },
    "initial_engine": "omssa"
}
```

The most important section is the input_files, where each key is the path to a input json, and the value capturing the meta data to the input file, e.g. engine name `engine` and `score_col` which column should be used for scoring.

The input jsons are generate by pyiohat (https://github.com/computational-ms/pyiohat), a converter for many proteomic DDA search engines.

In principle any csv could be suplpied as long it contains the standard columns, such as,

  * spectrum_id
  * protein_id
  * <score columns, e.g. mascot:score>
  * sequence
  * modifications
  * charge
  * rank
  * ucalc_mz
  * accuracy_ppm
  * is_decoy
  * sequence_pre_aa
  * sequence_post_aa
  * exp_mz
  * search_engine

Find more examples in the exmaple_data folder.
