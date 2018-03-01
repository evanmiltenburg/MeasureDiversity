from methods import parallel_sentences_from_file, parallel_stats, load_json, save_json, sentence_stats

train = parallel_sentences_from_file('./Data/COCO/Processed/tokenized_train2014.json',
                                     tagged=False,  # Don't load tags.
                                     lower=True)    # Lowercase all descriptions.

val   = parallel_sentences_from_file('./Data/COCO/Processed/tagged_val2014.json',
                                     tagged=False,  # Don't load tags.
                                     lower=True)    # Lowercase all descriptions.

# Compute stats for train and val data.
train_stats = parallel_stats(train)
val_stats   = parallel_stats(val)

# Extra stats.
train_data = load_json('./Data/COCO/Processed/tokenized_train2014.json')
train_descriptions = [entry['caption'] for entry in train_data['annotations']]

val_data = load_json('./Data/COCO/Processed/tagged_val2014.json')
val_descriptions = [entry['caption'] for entry in val_data['annotations']]

extra_stats = sentence_stats(train_descriptions, val_descriptions)

val_stats.update(extra_stats)

# Save data to file.
save_json(train_stats, './Data/COCO/Processed/train_stats.json')
save_json(val_stats, './Data/COCO/Processed/val_stats.json')
