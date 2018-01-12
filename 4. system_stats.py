from methods import sentences_from_file, system_stats, load_json, save_json, sentence_stats

train_data = load_json('./Data/COCO/Processed/tokenized_train2014.json')
train_descriptions = [entry['caption'] for entry in train_data['annotations']]

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
    source = base + folder + '/Val/annotated.json'
    target = base + folder + '/Val/stats.json'
    
    # Load data.
    sentences = sentences_from_file(source)
    
    # Process data.
    stats = system_stats(sentences)
    
    # Get raw descriptions.
    gen_descriptions = [entry['caption'] for entry in load_json(source)]
    extra_stats = sentence_stats(train_descriptions, gen_descriptions)
    
    stats.update(extra_stats)
    
    # Save data.
    save_json(stats, target)
