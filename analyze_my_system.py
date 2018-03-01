"""
This script combines the exact code from the following files:

* annotate_generated.py
* system_stats.py
* global_recall.py
* local_recall.py
* nouns_pps.py

We did not streamline anything so as to prevent any discrepancies with the original code.

This script does not:
* Plot the TTR curve. It does compute the curve, with all points stored in stats.json.
* Produce any tables. All results are stored in JSON format.
"""

import json
import spacy
from collections import defaultdict
import glob
import argparse

from methods import sentences_from_file, system_stats, load_json, save_json, sentence_stats, index_from_file
from global_recall import most_frequent_omissions, get_count_list, percentiles
from local_recall import local_recall_counts, local_recall_scores
from nouns_pps import pp_stats, compound_stats

nlp = spacy.load('en_core_web_sm')

def compounds_from_doc(doc):
    "Return a list of compounds from the document."
    compounds = []
    current = []
    for token in doc:
        if token.tag_.startswith('NN'):
            current.append(token.orth_)
        elif len(current) == 1:
            current = []
        elif len(current) > 1:
            compounds.append(current)
            current = []
    if len(current) > 1:
        compounds.append(current)
    return compounds


def annotate_data(source_file, annotations_file, tag=False, compounds=False):
    "Function to annotate existing coco data"
    data = load_json(source_file)
    for entry in data:
        raw_description = entry['caption']
        doc = nlp.tokenizer(raw_description)
        entry['tokenized'] = [tok.orth_ for tok in doc]
        if tag:
            # Call the tagger on the document.
            nlp.tagger(doc)
            entry['tagged'] = [(tok.orth_,tok.tag_) for tok in doc]
        if compounds:
            list_of_compounds = compounds_from_doc(doc)
            entry['compounds'] = list_of_compounds
    save_json(data, annotations_file)
    return data


def run_all(args):
    "Run all metrics on the data and save JSON files with the results."
    # Annotate generated data.
    annotated = annotate_data(args.source_file,
                              args.annotations_file,
                              tag=True,
                              compounds=True)
    
    # Load training data. (For computing novelty.)
    train_data = load_json('./Data/COCO/Processed/tokenized_train2014.json')
    train_descriptions = [entry['caption'] for entry in train_data['annotations']]
    
    # Load annotated data.
    sentences = sentences_from_file(args.annotations_file)
    
    # Analyze the data.
    stats = system_stats(sentences)
    
    # Get raw descriptions.
    gen_descriptions = [entry['caption'] for entry in load_json(args.source_file)]
    extra_stats = sentence_stats(train_descriptions, gen_descriptions)
    stats.update(extra_stats)
    
    # Save statistics data.
    save_json(stats, args.stats_file)
    
    ################################
    # Global recall
    
    train_stats = load_json('./Data/COCO/Processed/train_stats.json')
    val_stats = load_json('./Data/COCO/Processed/val_stats.json')

    train     = set(train_stats['types'])
    val       = set(val_stats['types'])
    learnable = train & val
    
    gen = set(stats['types'])
    recalled = gen & val
    
    coverage = {"recalled": recalled,
                "score": len(recalled)/len(learnable),
                "not_in_val": gen - learnable}
    
    coverage['omissions'] = most_frequent_omissions(coverage['recalled'],
                                                    val_stats,         # Use validation set as reference.
                                                    n=None)
    val_count_list = get_count_list(val_stats)
    coverage['percentiles'] = percentiles(val_count_list, recalled)
    save_json(coverage, args.global_coverage_file)
    
    ####################################
    # Local recall
    
    val_index = index_from_file('./Data/COCO/Processed/tagged_val2014.json', tagged=True, lower=True)
    generated = {entry['image_id']: entry['tokenized'] for entry in annotated}
    local_recall_res = dict(scores = local_recall_scores(generated, val_index),
                            counts = local_recall_counts(generated, val_index))
    save_json(local_recall_res, args.local_coverage_file)
    
    ##################################
    # Nouns pps
    npdata = {'pp_data': pp_stats(annotated), 'compound_data': compound_stats(annotated)}
    save_json(npdata, args.noun_pp_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze diversity of your image description system.')
    parser.add_argument('source_file',
                        help="The file containing your system output in MS COCO format.")
    parser.add_argument('--annotations_file',
                        help="Where to store the annotated output. Should end in .json.",
                        default="annotations.json")
    parser.add_argument('--stats_file',
                        help="Where to store the statistics. Should end in .json.",
                        default="stats.json")
    parser.add_argument('--global_coverage_file',
                        help="Where to store the global coverage results. Should end in .json.",
                        default="global_recall.json")
    parser.add_argument('--local_coverage_file',
                        help="Where to store the local coverage results. Should end in .json.",
                        default="local_recall.json")
    parser.add_argument('--noun_pp_file',
                        help="Where to store the noun & pp results. Should end in .json.",
                        default="noun_pp_data.json")
    args = parser.parse_args()
    run_all(args)
