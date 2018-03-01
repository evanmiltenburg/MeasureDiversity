from methods import load_json, save_json
from collections import defaultdict, Counter
from tabulate import tabulate
import spacy

################################################################################
# PP stats

nlp = spacy.load('en_core_web_sm', entity=False)

def pp_stats(entries):
    "Function to annotate existing coco data"
    print('PPs')
    data = dict()
    data['pp_counter'] = Counter()
    data['level_counter'] = Counter()
    data['pp_counts_by_length'] = defaultdict(Counter)
    data['total_prepositions'] = 0
    raw_captions = (entry['caption'] for entry in entries)
    # Let's run this thing in parallel!
    for doc in nlp.pipe(raw_captions, batch_size=1000, n_threads=4):
        prepositions = [tok for tok in doc if tok.tag_=='IN']
        num_prepositions = len(prepositions)
        data['total_prepositions'] += num_prepositions
        for head in prepositions:
            # Get the actual PP.
            pp = ' '.join([token.orth_.lower() for token in head.subtree])
            # Count the level PP embedding.
            levels = len([token for token in head.subtree if token.tag_=='IN'])
            # Count the PP and level.
            data['pp_counter'][pp] += 1
            data['level_counter'][levels] += 1
            data['pp_counts_by_length'][levels][pp] += 1
    data['prep_ratio'] = data['total_prepositions']/len(entries)
    return data

################################################################################
# Compound stats

def compound_stats(entries):
    "Count the total number of compounds in the data, and their lengths"
    print('Compounds')
    data = dict()
    data['compound_lengths'] = Counter()
    data['compound_counts']  = Counter()
    data['counts_by_length'] = defaultdict(Counter)
    data['total_compounds'] = 0
    for entry in entries:
        for compound in entry['compounds']:
            length = len(compound)
            compound_string = ' '.join(compound)
            # Count everything
            data['compound_lengths'][length] += 1
            data['compound_counts'][compound_string] += 1
            data['counts_by_length'][length][compound_string] += 1
            data['total_compounds'] += 1
    data['compound_ratio'] = data['total_compounds']/ len(entries)
    return data

################################################################################
# Helpers

def parallel_entries(val_tagged):
    "Get parallel lists of entries."
    d = defaultdict(list)
    for entry in val_tagged['annotations']:
        img_id = entry['image_id']
        d[img_id].append(entry)
    return list(zip(*d.values()))


def average_counters(counters):
    "Function to produce an average counter for a list of counters."
    main_counter = Counter()
    for c in counters:
        main_counter.update(c)
    for i in main_counter:
        main_counter[i] = main_counter[i]/len(counters)
    return c

################################################################################
# Computing the results.

def get_compound_lengths(data):
    return [data['compound_lengths'][i] for i in range(2,5)]

def get_compound_ratio(data):
    return [data['compound_ratio']]

def get_pp_levels(data):
    return [data['level_counter'][i] for i in range(1,6)]

def get_pp_ratio(data):
    return [data['prep_ratio']]

def get_compound_types(data, length=2):
    return [len(data['counts_by_length'][length])]

def get_num_pp_types(data, level=1):
    return [len(data['pp_counts_by_length'][level])]

def average_rows(rows):
    return [sum(items)/len(items) for items in zip(*rows)]

def get_system_row(compound_data, pp_data):
    compound_length_counts  = get_compound_lengths(compound_data)
    compound_ratio          = get_compound_ratio(compound_data)
    compound_types          = get_compound_types(compound_data, length=2)
    
    pp_levels               = get_pp_levels(pp_data)
    prep_ratio              = get_pp_ratio(pp_data)
    pp_types                = get_num_pp_types(pp_data, level=1)
    
    compound_ratio          = ["{:.2f}".format(compound_ratio[0])]
    prep_ratio              = ["{:.2f}".format(prep_ratio[0])]
    
    row = compound_length_counts + compound_ratio + compound_types + pp_levels + prep_ratio + pp_types
    return row

def get_reference_row(all_compound_data, all_pp_data):
    compound_length_counts  = average_rows([get_compound_lengths(data) for data in all_compound_data])
    compound_ratio          = average_rows([get_compound_ratio(data) for data in all_compound_data])
    compound_types          = average_rows([get_compound_types(data, length=2) for data in all_compound_data])
    
    pp_levels               = average_rows([get_pp_levels(data) for data in all_pp_data])
    prep_ratio              = average_rows([get_pp_ratio(data) for data in all_pp_data])
    pp_types                = average_rows([get_num_pp_types(data, level=1) for data in all_pp_data])
    
    compound_length_counts  = ["{:.0f}".format(i) for i in compound_length_counts]
    compound_ratio          = ["{:.2f}".format(compound_ratio[0])]
    compound_types          = ["{:.0f}".format(compound_types[0])]
    
    pp_levels               = ["{:.0f}".format(i) for i in pp_levels]
    prep_ratio              = ["{:.2f}".format(prep_ratio[0])]
    pp_types                = ["{:.0f}".format(pp_types[0])]
    
    row = compound_length_counts + compound_ratio + compound_types + pp_levels + prep_ratio + pp_types
    return row

#########################################
# Systems..

if __name__ == "__main__":
    systems = ['Dai-et-al-2017',
               'Liu-et-al-2017',
               'Mun-et-al-2017',
               'Shetty-et-al-2016',
               'Shetty-et-al-2017',
               'Tavakoli-et-al-2017',
               'Vinyals-et-al-2017',
               'Wu-et-al-2016',
               'Zhou-et-al-2017']

    all_data = dict()
    system_rows = []
    def load_system_data(name):
        base = './Data/Systems/'
        path = base + name + '/Val/annotated.json'
        return load_json(path)

    loaded_systems = {system: load_system_data(system) for system in systems}
    for name, entries in loaded_systems.items():
        print(name)
        compound_data = compound_stats(entries)
        pp_data = pp_stats(entries)
        row = [name] + get_system_row(compound_data, pp_data)
        system_rows.append(row)
        all_data[name] = {'pp_data': pp_data, 'compound_data': compound_data}

    #########################################
    # Val..

    val_tagged = load_json('./Data/COCO/Processed/tagged_val2014.json')
    parallel_entries    = parallel_entries(val_tagged)
    print('Val')
    all_compound_data   = list(map(compound_stats, parallel_entries))
    all_pp_data         = list(map(pp_stats, parallel_entries))

    val_row             = ['Val'] + get_reference_row(all_compound_data, all_pp_data)
    all_data['val'] = {'pp_data': all_pp_data, 'compound_data': all_compound_data}

    #########################################
    # Create table

    systems = {'Dai-et-al-2017': "\citeauthor{Dai_2017_ICCV} (\citeyear{Dai_2017_ICCV})",
               'Liu-et-al-2017': "\citeauthor{liu2017mat} (\citeyear{liu2017mat})",
               'Mun-et-al-2017': "\citeauthor{mun2017text} (\citeyear{mun2017text})",
               'Shetty-et-al-2016': '\citeauthor{Shetty:2016:ESC:2983563.2983571} (\citeyear{Shetty:2016:ESC:2983563.2983571})',
               'Shetty-et-al-2017': '\citeauthor{Shetty_2017_ICCV} (\citeyear{Shetty_2017_ICCV})',
               'Tavakoli-et-al-2017': '\citeauthor{tavakoli2017paying} (\citeyear{tavakoli2017paying})',
               'Vinyals-et-al-2017': '\citeauthor{vinyals2017show} (\citeyear{vinyals2017show})',
               'Wu-et-al-2016': '\citeauthor{wu2017image} (\citeyear{wu2017image})',
               'Zhou-et-al-2017': '\citeauthor{zhou2017watch} (\citeyear{zhou2017watch})'}

    data = [val_row] + system_rows
    headers = ['','2','3','4','Ratio', 'Types-2', '1','2','3','4','5', 'Ratio', 'Types-1']
    table = tabulate(data, headers, tablefmt='latex_booktabs')
    # Modify table
    # table = table.replace('{lrrrrrrrrrr}','{lrrrcrrrrrc}')
    table = table.replace('&    0.3  &','&    0.30  &')
    table = table.replace('\\toprule', '\\toprule \n & \multicolumn{3}{c}{Compound length} & \multicolumn{2}{c}{Compound stats} & \multicolumn{5}{c}{Prepositional phrase depth} & \multicolumn{2}{c}{PP stats}\\\\\n \cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-11} \cmidrule(lr){12-13}\n')
    for system, cite in systems.items():
        table = table.replace(system, cite)

    # Print and save.
    print(table)

    with open('./Data/Output/nouns_pps_table.txt', 'w') as f:
        f.write(table)

    save_json(all_data, './Data/Output/nouns_pps.json')
