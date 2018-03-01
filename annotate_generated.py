import json
import spacy
from collections import defaultdict
import glob

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


def annotate_data(filename, tag=False, compounds=False):
    "Function to annotate existing coco data"
    data = load_json(filename)
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
    return data


def main(source_file, target_file):
    "Annotate data and save to file."
    data = annotate_data(source_file, tag=True, compounds=True)
    save_json(data, target_file)


for folder in ['Dai-et-al-2017',
               'Liu-et-al-2017',
               'Mun-et-al-2017',
               'Shetty-et-al-2016',
               'Shetty-et-al-2017',
               'Tavakoli-et-al-2017',
               'Vinyals-et-al-2017',
               'Wu-et-al-2016',
               'Zhou-et-al-2017']:
    print('Processing:', folder)
    
    # Define source and target.
    base = './Data/Systems/'
    pattern = base + folder + '/Val/*.json'
    files = glob.glob(pattern)
    source = [path for path in glob.glob(pattern) if (not path.endswith('stats.json'))
                                                  and (not path.endswith('annotated.json'))][0]
    target = base + folder + '/Val/annotated.json'
    main(source,target)
    
# main('./Data/Systems/Dai-et-al-2017/Val/gan_val2014.json',
#      './Data/Systems/Dai-et-al-2017/Val/annotated.json')
#
# main('./Data/Systems/Liu-et-al-2017a/Val/captions_val2014_MAT_results.json',
#      './Data/Systems/Liu-et-al-2017a/Val/annotated.json')
#
# main('./Data/Systems/Mun-et-al-2017/Val/captions_val2014_senAttKnn-kCC_results.json',
#      './Data/Systems/Mun-et-al-2017/Val/annotated.json')
#
# main('./Data/Systems/Shetty-et-al-2016/Val/captions_val2014_r-dep3-frcnn80detP3+3SpatGaussScaleP6grRBFsun397-gaA3cA3-per9.72-b5_results.json',
#      './Data/Systems/Shetty-et-al-2016/Val/annotated.json')
#
# main('./Data/Systems/Shetty-et-al-2017/Val/captions_val2014_MLE-20Wrd-Smth3-randInpFeatMatch-ResnetMean-56k-beamsearch5_results.json',
#      './Data/Systems/Shetty-et-al-2017/Val/annotated.json')
#
# main('./Data/Systems/Tavakoli-et-al-2017/Val/captions_val2014_PayingAttention-ICCV2017_results.json',
#      './Data/Systems/Tavakoli-et-al-2017/Val/annotated.json')
#
# main('./Data/Systems/Vinyals-et-al-2017/Val/captions_val2014_googlstm_results.json',
#      './Data/Systems/Vinyals-et-al-2017/Val/annotated.json')
#
# main('./Data/Systems/Wu-et-al-2016/Val/captions_val2014_Attributes_results.json',
#      './Data/Systems/Wu-et-al-2016/Val/annotated.json')
#
# main('./Data/Systems/Zhou-et-al-2017/Val/captions_val2014_e2e_results.json',
#      './Data/Systems/Zhou-et-al-2017/Val/annotated.json')
