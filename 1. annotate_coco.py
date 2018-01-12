import json
import spacy
from collections import defaultdict

nlp = spacy.load('en_core_web_sm')


def load_json(filename):
    "Wrapper function to load JSON data."
    with open(filename) as f:
        data = json.load(f)
    return data


def save_json(data, filename):
    "Wrapper function to save the data as JSON."
    with open(filename, 'w') as f:
        json.dump(data, f)


def compounds_from_doc(doc):
    compounds = []
    current = []
    for token in doc:
        if token.tag_.startswith('NN'):
            current.append(token.orth_.lower())
        elif len(current) == 1:
            current = []
        elif len(current) > 1:
            compounds.append(current)
            current = []
    if len(current) > 1:
        compounds.append(current)
    return compounds


def annotate_coco(filename, tag=False, compounds=False):
    "Function to annotate existing coco data"
    data = load_json(filename)
    for entry in data['annotations']:
        raw_description = entry['caption']
        doc = nlp.tokenizer(raw_description)
        entry['tokenized'] = [tok.orth_ for tok in doc]
        if tag:
            # Call the tagger on the document.
            nlp.tagger(doc)
            entry['tagged'] = [(tok.orth_.lower(),tok.tag_) for tok in doc]
        if compounds:
            list_of_compounds = compounds_from_doc(doc)
            entry['compounds'] = list_of_compounds
    return data


tokenized_train = annotate_coco('./Data/COCO/Raw/captions_train2014.json', tag=True, compounds=True)
save_json(tokenized_train, './Data/COCO/Processed/tokenized_train2014.json')

tagged_val = annotate_coco('./Data/COCO/Raw/captions_val2014.json', tag=True, compounds=True)
save_json(tagged_val, './Data/COCO/Processed/tagged_val2014.json')
