# MeasureDiversity
Measure the diversity of image descriptions

# Requirements
* Python 3.6.3
  * SpaCy 2.0.4
    * Model: `en_core_web_sm`
  * NLTK 3.2.2
    * WordNet 3.0
  * Matplotlib 2.1.1
  * Seaborn 0.7.1
  * Tabulate 0.7.7
  * Numpy  1.13.1
* Pdfcrop 1.38 (only to crop the graphs for the paper)

# How to use

Instructions to either:

* Reproduce our results
* Analyze your own system

## Reproducing our results

First run `bash get_data.sh` in the `Data/COCO/Raw/` folder. This downloads the 2014 MS COCO annotation data.
Then the following commands in order:

* `python 1. annotate_coco.py` to annotate the MS COCO training and val data.
* `python 2. annotate_generated.py` to annotate the generated descriptions.
* `python 3. coco_stats.py` to generate statistics about the MS COCO data.
* `python 4. system_stats.py` to generate statistics about the systems.
* `python 5. plot_ttr_curve.py` to plot the type-token curve for MS COCO and all systems.
* `python 6. global_recall.py` to compute global recall.
* `python 7. local_recall.py` to compute local recall.
* `python 8. generate_main_table.py` to generate the main table.
* `python 9. generate_ranking_table.py` to generate the rankings.
* `python 10. wordnet.py` to generate the WordNet specificity results.
* `python 11. nouns_pps.py` to generate the compound noun and PP results.

If you are interested to reproduce our exact figures, run `pdfcrop FILENAME.pdf`
on the relevant files in `Data/Output/`. This tool is provided with the TeXLive
LaTeX distribution.

## Analyzing your own system

If you don't care about other systems, you can also just run `python 1. annotate_coco.py`
and then run `python analyze_my_system.py`. This will generate all statistics for a single system.
Make sure your system output is in the standard JSON format. See the Systems folder for examples.

TODO: write `analyze_my_system.py`
