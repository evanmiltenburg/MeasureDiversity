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

* Inspect our data
* Reproduce our results
* Analyze your own system

## Inspecting our data

Please find all output in `Data/Output/`.

## Reproducing our results

First run `bash get_data.sh` in the `Data/COCO/Raw/` folder. This downloads the 2014 MS COCO annotation data
and unpacks the zip file. Please ensure that both `JSON` files are unpacked in `Data/COCO/Raw/` (and not in some subfolder).

Then the following commands in order:

* `python annotate_coco.py` to annotate the MS COCO training and val data.
* `python annotate_generated.py` to annotate the generated descriptions.
* `python coco_stats.py` to generate statistics about the MS COCO data.
* `python system_stats.py` to generate statistics about the systems.
* `python plot_ttr_curve.py` to plot the type-token curve for MS COCO and all systems.
* `python global_recall.py` to compute global recall.
* `python local_recall.py` to compute local recall.
* `python generate_main_table.py` to generate the main table.
* `python generate_ranking_table.py` to generate the rankings.
* `python nouns_pps.py` to generate the compound noun and PP results.
* `python plot_compound_length.py` to generate a histogram of compound length for different systems.
* `python plot_pp_length.py` to generate a histogram of PP depth for different systems.

If you modify any of the Python files, you can rerun the analysis using `bash run_experiment.sh`.
We commented out the first two commands, because annotating all the data takes a long time.

If you are interested to reproduce our exact figures, run `pdfcrop FILENAME.pdf`
on the relevant files in `Data/Output/`. This tool is provided with the TeXLive
LaTeX distribution.

## Analyzing your own system

If you don't care about other systems, you can also just run `python annotate_coco.py`
and then run `python analyze_my_system.py descriptions.json`. This will generate all statistics for a single system.
Make sure your system output is in the standard JSON format. See the Systems folder for examples.

If you don't care about other systems, you can also just run the following commands (assuming you stored your system output in `descriptions.json`).

* `python annotate_coco.py`
* `python analyze_my_system.py descriptions.json`

This will first generate the basis statistics for MS COCO (the standard of comparison), and then generate all statistics for a single system. Make sure your system output is in the standard JSON format. See the Systems folder for examples.
