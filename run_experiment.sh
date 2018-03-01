# python annotate_coco.py      # This runs the parser. (Warning: this takes quite long.)
# python annotate_generated.py # This runs the parser. (Warning: this takes quite long.)
echo "coco stats"
python coco_stats.py
echo "system stats"
python system_stats.py
echo "ttr curve"
python plot_ttr_curve.py
echo "global recall"
python global_recall.py
echo "local recall"
python local_recall.py
echo "main table"
python generate_main_table.py
echo "ranking table"
python generate_ranking_table.py
echo "wordnet"
python wordnet.py
echo "nouns and pps"
python nouns_pps.py
echo "make compound plot"
python plot_compound_length.py
echo "make pp plot"
python plot_pp_length.py
echo "Done."
