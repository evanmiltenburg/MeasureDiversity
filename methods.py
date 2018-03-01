import json
import random
import numpy as np
from collections import defaultdict, Counter
from nltk import ngrams

random.seed(1234)

################################################################################
# Basic functions

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    
    From: https://stackoverflow.com/a/312464/2899924
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class SetEncoder(json.JSONEncoder):
    "Encoder that saves sets as lists in JSON."
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def load_json(filename):
    "Wrapper function to load JSON data."
    with open(filename) as f:
        data = json.load(f)
    return data


def save_json(data, filename):
    "Wrapper function to save the data as JSON."
    with open(filename, 'w') as f:
        json.dump(data, f, cls=SetEncoder)


def write_csv(rows, header, filename):
    "Write rows to a CSV file."
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

################################################################################
# Functions to help compute novel descriptions

def normalize_string(raw_description):
    "Normalize string by lowercasing it and removing final punctuation."
    return raw_description.lower().strip('.?!')


def sentence_stats(train_descriptions, gen_descriptions):
    "Compute stats about the uniqueness and novelty of generated descriptions."
    train_normalized = [normalize_string(desc) for desc in train_descriptions]
    gen_normalized   = [normalize_string(desc) for desc in gen_descriptions]
    
    train_unique = set(train_normalized)
    gen_unique = set(gen_normalized)
    
    novel_gen = gen_unique - train_unique
    
    num_novel_descriptions = len([d for d in gen_normalized if d in novel_gen])
    percentage_novel = (num_novel_descriptions/len(gen_descriptions)) * 100
    
    return {"unique_descriptions": gen_unique,
            "num_unique_descriptions": len(gen_unique),
            "novel_descriptions": novel_gen,
            "num_novel_description_types": len(novel_gen),
            "total_num_novel_descriptions": num_novel_descriptions,
            "percentage_novel": percentage_novel}

################################################################################
# Building an index for MS COCO descriptions

def lower_sent(sentence, tagged=False):
    "Make sentence lowercase."
    if tagged:
        return [(word.lower(), pos) for word,pos in sentence]
    else:
        return [word.lower() for word in sentence]


def build_index(data, tagged=False, lower=True):
    """
    Build index of image descriptions for further processing.
    """
    key = 'tagged' if tagged else 'tokenized'
    index = defaultdict(list)
    for entry in data['annotations']:
        imgid = entry['image_id']
        if lower:
            description = lower_sent(entry[key], tagged)
        else:
            description = entry[key]
        index[imgid].append(description)
    return index


def index_from_file(filename, tagged=False, lower=True):
    "Wrapper function to get index directly from file."
    data = load_json(filename)
    index = build_index(data, tagged=tagged, lower=lower)
    return index


def parallel_sentences_from_index(index):
    "Get a list of lists of sentences from the index."
    return list(zip(*index.values()))


def parallel_sentences_from_file(filename, tagged=False, lower=True):
    "Wrapper function to load parallel sentences directly from a file."
    data            = load_json(filename)
    index           = build_index(data, tagged=tagged, lower=lower)
    parallel_sents  = parallel_sentences_from_index(index)
    return parallel_sents

################################################################################
# Creating a list of sentences.

def mapping_from_file(filename, tagged=False):
    "Load system output and map image ID to descriptions."
    data = load_json(filename)
    mapping = {entry['image_id']: entry['tagged' if tagged else 'tokenized']
                for entry in data}
    return mapping


def get_sentences(data, lower=True, tagged=False):
    "Get a list of tokenized sentences from generated output."
    key = 'tagged' if tagged else 'tokenized'
    sentences = [entry[key] for entry in data]
    if lower:
        return [lower_sent(sent, tagged) for sent in sentences]
    else:
        return sentences


def sentences_from_file(filename, lower=True, tagged=False):
    "Get sentences from a file containing system output."
    data = load_json(filename)
    sentences = get_sentences(data, lower, tagged)
    return sentences

################################################################################
# Metrics

# General function to be used with:
# - average sentence length
# - type-token-ratio


def average_function(function, parallel_sentences):
    "Compute average function for a list of lists of tokenized sentences."
    results = [function(sentences) for sentences in parallel_sentences]
    return float(sum(results))/len(results)

###########################################

def average_sentence_length(sentences):
    "Compute average sentence length for a list of tokenized sentences."
    lengths = [len(sentence) for sentence in sentences]
    return float(sum(lengths))/len(lengths)

def std_sentence_length(sentences):
    "Compute standard deviation of sentence lengths."
    lengths = [len(sentence) for sentence in sentences]
    return np.std(lengths)

###########################################

def type_token_ratio(sentences, n=1000):
    """
    Compute average type-token ratio (normalized over n tokens)
    with a repeated sample of n words.
    """
    all_words = [word for sentence in sentences for word in sentence]
    ttrs = []
    for chunk in chunks(all_words, n):
        if len(chunk) == n:
            types = set(chunk)
            ttr = float(len(types))/n
            ttrs.append(ttr)
    final_ttr = float(sum(ttrs))/len(ttrs)
    return final_ttr


def ngram_ttr(sentences, n=2, window_size=1000):
    """
    Compute average ngram type-token ratio (normalized over window_size ngrams)
    with a repeated sample of n words.
    """
    all_ngrams = list(ngrams([word for sentence in sentences for word in sentence], n))
    ttrs = []
    for chunk in chunks(all_ngrams, window_size):
        if len(chunk) == window_size:
            types = set(chunk)
            ttr = float(len(types))/window_size
            ttrs.append(ttr)
    final_ttr = float(sum(ttrs))/len(ttrs)
    return final_ttr


def bigram_ttr(sentences):
    "Compute bigram TTR"
    return ngram_ttr(sentences, n=2)


def trigram_ttr(sentences):
    "Compute trigram TTR"
    return ngram_ttr(sentences, n=3)

###########################################

def type_token_curve(sentences):
    """
    Compute the type-token curve for a given list of sentences.
    
    See: Youmans, G. (1990) Measuring lexical style and competence: the type-token vocabulary curve’. Style 24(Winter): 584-599.
    """
    all_words = [word for sentence in sentences for word in sentence]
    types = set()
    curve = dict()
    for i, word in enumerate(all_words,start=1):
        types.add(word)
        curve[i] = len(types)
    return curve


def average_curves(curves):
    """
    Helper function to average curves.
    """
    avg_curve = defaultdict(list)
    for d in curves:
        for x,y in d.items():
            avg_curve[x].append(y)
    avg_curve = {x: float(sum(vals))/len(vals) for x,vals in avg_curve.items()}
    return avg_curve


def cut_curve(curve, n):
    "Cut all values above n."
    for i in range(n + 1,               # Cut values above n.
                   max(curve) + 1):     # Including the maximal value.
        del curve[i]


def curve_to_coords(curve):
    """
    Convert curve to X and Y coordinates.
    Usage: x,y = curve_to_coords(curve)
    """
    return list(zip(*curve.items()))


def repeated_random_type_token_curve(sentences, n=10):
    """
    Perform type token curve analysis N times, randomizing the sentence order.
    
    This makes the curve more reliable than a single TTC evaluation.
    """
    curves = []
    for i in range(n):
        shuffled = random.sample(sentences, len(sentences))
        curve = type_token_curve(shuffled)
        curves.append(curve)
    return average_curves(curves)


def curve_for_parallel_sents(parallel_sentences, randomize=True, n=10):
    "Average curves for all parallel lists of sentences."
    if randomize:
        curves = [repeated_random_type_token_curve(sentences, n) for sentences in parallel_sentences]
    else:
        curves = [type_token_curve(sentences) for sentences in parallel_sentences]
    return average_curves(curves)

###########################################

def count_words(sentences):
    "Create a dictionary with counts for all words in the provided sentences."
    return Counter((word for sent in sentences for word in sent))


def get_types_tokens(sentences):
    "Return the total number of types and tokens."
    counts = count_words(sentences)
    return {"types": set(counts.keys()),
            "counts": counts,
            "num_types": len(counts),
            "num_tokens": sum(counts.values())}


def parallel_types_tokens(parallel_sentences):
    "Get type and token counts for parallel sentences."
    results = [get_types_tokens(sentences) for sentences in parallel_sentences]
    avg_types = sum(result["num_types"] for result in results)/len(results)
    
    all_counts = Counter()
    for result in results:
        all_counts.update(result['counts'])
    
    total_tokens = sum(all_counts.values())
    avg_tokens = total_tokens/len(results)
    
    all_types = set(all_counts.keys())
    total_types = len(all_types)
    return {"avg_types": avg_types,
            "avg_tokens": avg_tokens,
            "total_types": total_types,
            "total_tokens": total_tokens,
            "separate_counts": [result['counts'] for result in results],
            "total_counts": all_counts,
            "types": all_types}

################################################################################
# Functions to compute general stats for MS COCO and for individual systems.

def ttr10k(sentences):
    "ttr with 10K tokens."
    return type_token_ratio(sentences, n=10000)

def ttr100k(sentences):
    "ttr with 100K tokens."
    return type_token_ratio(sentences, n=100000)


def parallel_stats(parallel_sentences):
    "Compute all stats for the parallel sentences."
    data = parallel_types_tokens(parallel_sentences)
    data['ttr_curve']               = curve_for_parallel_sents(parallel_sentences)
    data['average_sentence_length'] = average_function(average_sentence_length, parallel_sentences)
    data['std_sentence_length']     = average_function(std_sentence_length, parallel_sentences)
    data['type_token_ratio']        = average_function(type_token_ratio, parallel_sentences)
    data['bittr']                   = average_function(bigram_ttr, parallel_sentences)
    data['trittr']                  = average_function(trigram_ttr, parallel_sentences)
    data['ttr10k']                  = average_function(ttr10k, parallel_sentences)
    data['ttr100k']                 = average_function(ttr100k, parallel_sentences)
    return data


def system_stats(sentences):
    "Compute all stats for the different systems."
    data = get_types_tokens(sentences)
    data["ttr_curve"]               = repeated_random_type_token_curve(sentences)
    data['average_sentence_length'] = average_sentence_length(sentences)
    data['std_sentence_length']     = std_sentence_length(sentences)
    data['type_token_ratio']        = type_token_ratio(sentences)
    data['bittr']                   = bigram_ttr(sentences)
    data['trittr']                  = trigram_ttr(sentences)
    data['ttr10k']                  = ttr10k(sentences)
    data['ttr100k']                 = ttr100k(sentences)
    return data
