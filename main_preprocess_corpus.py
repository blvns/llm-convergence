import argparse
import os
import json
import csv
import pandas as pd
from helper_functions_corpora import CorpusConverter, PrepareModelInputs, DialogStatistics

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Preprocess conversation corpus.")
    parser.add_argument("--corpus", 
                        type=str, 
                        #["movie", "dialogsum", "dailydialog", "bsd", "npr", "friends", "tennis", "empathetic", "mantis", "persuasionforgood", "diplomacy", "wiki", "switchboard", "reddit", "casino", "maia"],
                        choices=["movie", "dailydialog", "npr"], 
                        required=True,
                        help="Choose the corpus to preprocess: 'movie', 'dailydialog', or 'npr'.")
    args = parser.parse_args()

    with open('corpora_paths.json', 'r') as f:
        corpora_paths = json.load(f)
    
    split_choices = ['train', 'dev', 'test']

    paths = corpora_paths[args.corpus]
    dir_corpus = f'data/corpora/{args.corpus}_corpus'
    
    if args.corpus == 'movie':
        # Check if conversations file exists
        if not os.path.exists(paths["conversations"]):
            raise FileNotFoundError(f"Required file '{paths['conversations']}' not found.")

        # # Process the Corpus
        converter = CorpusConverter(dir_corpus=dir_corpus, train_split=0.3, dev_split=0.4, test_split=0.3)
        converter.convert_movie(conversations_path=paths["conversations"])

    elif args.corpus == "dailydialog":
        # Check if all conversations files exist (skip output files)
        for key in ["train_data", "dev_data", "test_data"]:
            if not os.path.exists(paths[key]):
                raise FileNotFoundError(f"Required file '{paths[key]}' not found.")

        # Process the corpus
        converter = CorpusConverter(dir_corpus=dir_corpus)
        converter.convert_dailydialog(train_path=paths["train_data"], dev_path=paths["dev_data"], test_path=paths["test_data"])


    elif args.corpus == 'npr': #or args.corpus == 'switchboard' or args.corpus == 'casino' or args.corpus == 'friends'
        # Check if conversations file exists
        if not os.path.exists(paths["conversations"]):
            raise FileNotFoundError(f"Required file '{paths['conversations']}' not found.")

        # # Process the corpus
        converter = CorpusConverter(dir_corpus=dir_corpus, train_split=0.3, dev_split=0.4, test_split=0.3)
        converter.convert_npr(conversations_path=paths["conversations"])

    
    for split in split_choices:
        preprocessor = PrepareModelInputs(file_path=os.path.join(dir_corpus, f'{split}/{split}.json'))
        masked_df = preprocessor.replace_one_speaker_utterances()
        model_inputs = preprocessor.prepare_model_inputs(masked_df, output_file=os.path.join(dir_corpus, f'{split}/model_inputs.json'))
        
    print('\n\nAdd test set statistics to data/corpora_statistics_test_set.csv\n\n')
    
    test_set = f'data/corpora/{args.corpus}_corpus/test/test.json'

    d_stat = DialogStatistics(input_file=test_set)
    d_stat.calculate_statistics()

    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Table file path
    table_file = os.path.join('data', "corpora_statistics_test_set.csv")
    all_data = []

    # Check if the CSV file exists and read its content
    if os.path.isfile(table_file):
        with open(table_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            all_data = list(reader)  # Read all rows into a list

        # Update existing row if the corpus is already present
        for i, row in enumerate(all_data):
            if i == 0:
                continue  # Skip the header row
            if row and row[0] == args.corpus:
                all_data[i] = [args.corpus] + [round(value, 4) for value in d_stat.statistics.values()]
                break
        else:
            # Add a new row if the corpus was not found
            new_row = [args.corpus] + [round(value, 4) for value in d_stat.statistics.values()]
            all_data.append(new_row)
    else:
        # Create a new CSV file with a header and the first row
        header = ["Corpus"] + list(d_stat.statistics.keys())
        new_row = [args.corpus] + [round(value, 4) for value in d_stat.statistics.values()]
        all_data = [header, new_row]

    # Write all data back to the CSV file
    with open(table_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_data)

    print(f"Statistics for {args.corpus} successfully updated in {table_file}")


    # Load the CSV file into a Pandas DataFrame for visualization
    df = pd.read_csv(table_file)
    print("\nCurrent Statistics:\n")
    print(df)

if __name__ == "__main__":
    main()
