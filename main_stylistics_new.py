import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd



def process_original_random():
    
    #first, we reduce the OG corpus to the utterances that we would like to use for comparison

    from helper_functions_stylistics import reduce_corpus
    # Define input and output file paths relative to your project structure
    input_file = 'data/corpora/movie_corpus/test/test.json'  
    output_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG.json' 

    # Call the helper function to reduce the corpus
    reduce_corpus(input_file, output_file)

#then, we create random pairs from user_y and user_x as a comparison baseline (because in the random matches, we would expect no accommodation to occur)

    from helper_functions_stylistics import build_random_pairs

    # Define the file paths for the input and output
    input_file = "data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG.json"  # The reduced corpus
    output_file = "data/stylometrics/movie_corpus/llama3_1_8b_it/random_utterance_pairs_OG.json"  # The file where random pairs will be stored
    
    # Load the reduced corpus from the input file
    with open(input_file, "r", encoding="utf-8") as json_file:
        reduced_corpus = json.load(json_file)
    
    # Call the function to build random pairs
    build_random_pairs(reduced_corpus, output_file)
    
    from helper_functions_stylistics import process_and_save_corpus
    file_paths = [
        ("original", 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG.json'),
        ("random", 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_utterance_pairs_OG.json')
    ]

    output_paths = {
        "original": 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG_pos_tagged.json',
        "random": 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_corpus_pos_tagged.json'
    }

    for corpus_type, input_path in file_paths:
        print(f"Processing the '{corpus_type}' corpus...")
        output_path = output_paths[corpus_type]
        process_and_save_corpus(input_path, output_path)

    #Calculate POS percentages
    from helper_functions_stylistics import calculate_pos_percentages
    print("Starting original POS calculation...")
    input_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG_pos_tagged.json'
    output_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/OG_POS_percentagesperuser.csv'
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    print("Calling calculate_pos_percentages for original...")
    calculate_pos_percentages(input_file, output_file)
    print(f"POS percentages calculation complete for original. Results saved to {output_file}")


if __name__ == "__main__":
        process_original_random()


def process_model():

#Now, we shorten the corpus that the model created
    from helper_functions_stylistics import reduce_model_corpus
    # Define input and output file paths
    input_file = 'LLMs/corpora/movie_corpus/test/outputs/instruction_tuned/llama3_1_8b_it/conversations/conversations.json'  
    output_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL.json'  

    # Call the reduce_model_corpus function
    reduce_model_corpus(input_file, output_file)

    from helper_functions_stylistics import process_and_save_corpus
    file_paths = [
        ("model", 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL.json')
    ]

    output_paths = {
        "model": 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL_tagged.json',
    }

    for corpus_type, input_path in file_paths:
        print(f"Processing the '{corpus_type}' corpus...")
        output_path = output_paths[corpus_type]
        process_and_save_corpus(input_path, output_path)

    #Calculate POS percentages
    #for original
    from helper_functions_stylistics import calculate_pos_percentages

    print("Starting model POS calculation...")
    input_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL_tagged.json'
    output_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/MODEL_POS_percentagesperuser.csv'
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    print("Calling calculate_pos_percentages for model...")
    calculate_pos_percentages(input_file, output_file)
    print(f"POS percentages calculation complete for model. Results saved to {output_file}")

    #for random
    print("Starting random POS calculation...")
    input_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_corpus_pos_tagged.json'
    output_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/RANDOM_POS_percentagesperuser.csv'

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    print("Calling calculate_pos_percentages for random...")
    calculate_pos_percentages(input_file, output_file)
    print(f"POS percentages calculation complete for random. Results saved to {output_file}")

if __name__ == "__main__":
        process_model()



#Functions to analyse stylistic features

#Count utterance length and differences between users
from helper_functions_stylistics import count_utterance_length_and_differences
def main_countutt():
    # Load corpus data from three different files (user_y vs model, user_y vs user_x)
    with open("data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL.json", "r", encoding="utf-8") as f:
        model_data = json.load(f)
    with open("data/stylometrics/movie_corpus/llama3_1_8b_it/random_utterance_pairs_OG.json", "r", encoding="utf-8") as f:
        random_data = json.load(f)

    with open("data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG.json", "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Specify the output files for each comparison type
    output_file_model = "results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_length_model.json"
    output_file_random = "results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_length_random.json"
    output_file_original = "results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_length_original.json"
    
    # Call the function for user_y vs model comparison
    count_utterance_length_and_differences(model_data, output_file_model, "user_y_model")
    
    # Call the function for random comparison
    count_utterance_length_and_differences(random_data, output_file_random, "user_y_user_x")

    
    # Call the function for original comparison
    count_utterance_length_and_differences(original_data, output_file_original, "user_y_user_x")


if __name__ == "__main__":
        main_countutt()  

# Merge the results
from helper_functions_stylistics import merge_and_calculate_differences

def main_uttmerge():
    # Define file paths for each dataset
    original_file = "results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_length_original.json"
    random_file = "results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_length_random.json"
    model_file = "results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_length_model.json"
    output_csv = "results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_lengths_merged.csv"


     # Load data
    with open(original_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    with open(random_file, "r", encoding="utf-8") as f:
        random_data = json.load(f)

    with open(model_file, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    # Call the merging function
    merge_and_calculate_differences(original_data, random_data, model_data, output_csv)

if __name__ == "__main__":
        main_uttmerge()  
        

#Calculating the averages of these results
from helper_functions_stylistics import calculate_averages
def main_calcavg():
    merged_df = pd.read_csv("results/movie_corpus/stylometrics/llama3_1_8b_it/utterance_lengths_merged.csv")
    
    # Calculate averages
    averages_df = calculate_averages(merged_df)
    
    # Save the averages to a new CSV file
    averages_df.to_csv("results/movie_corpus/stylometrics/llama3_1_8b_it/average_length_diffs_comparison.csv", index=False)
    print("Averages saved to results/movie_corpus/stylometrics/average_length_diffs_comparison.csv")

if __name__ == "__main__":
        main_calcavg() 


#Let's visualise it!
from helper_functions_stylistics import plot_comparison_avglength

def main_plotlength():
    # Load the data (e.g., the averages CSV you generated earlier)
    averages_df = pd.read_csv('results/movie_corpus/stylometrics/llama3_1_8b_it/average_length_diffs_comparison.csv')
    
    # Call the plot function and save the plot
    averages_df = pd.read_csv('results/movie_corpus/stylometrics/llama3_1_8b_it/average_length_diffs_comparison.csv')
    plot_comparison_avglength(averages_df, 'results/movie_corpus/stylometrics/llama3_1_8b_it/average_length_diffs_comparison.png')
if __name__ == "__main__":
        main_plotlength() 

#Count the parts of speech 
from helper_functions_stylistics import calculate_pos_tag_differences, process_and_save_pos_tag_differences, merge_csv, load_data, save_merged_data
def main_countpos():
    
    output_paths = {
        "original": 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG_pos_tagged.json',
        "random": 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_corpus_pos_tagged.json',
        "model": 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL_tagged.json'
    }

    # Load the tagged data for each corpus
    original_tagged_data = load_data(output_paths["original"])
    random_tagged_data = load_data(output_paths["random"])
    model_tagged_data = load_data(output_paths["model"])

    # Calculate POS tag differences for each corpus
    original_pos_tag_differences = calculate_pos_tag_differences(original_tagged_data)
    random_pos_tag_differences = calculate_pos_tag_differences(random_tagged_data)
    model_pos_tag_differences = calculate_pos_tag_differences(model_tagged_data)

    # Save the POS tag differences for each corpus
    process_and_save_pos_tag_differences(original_pos_tag_differences, 'results/movie_corpus/stylometrics/llama3_1_8b_it/OG_pos_tag_differences.json')
    process_and_save_pos_tag_differences(random_pos_tag_differences, 'results/movie_corpus/stylometrics/llama3_1_8b_it/random_pos_tag_differences.json')
    process_and_save_pos_tag_differences(model_pos_tag_differences, 'results/movie_corpus/stylometrics/llama3_1_8b_it/model_pos_tag_differences.json')

    # Merge the POS tag differences into one table
    df = merge_csv(original_pos_tag_differences, random_pos_tag_differences, model_pos_tag_differences)

    # Save the merged table to a CSV file
    save_merged_data(df, 'results/movie_corpus/stylometrics/llama3_1_8b_it/merged_pos_tag_differences.csv')

if __name__ == "__main__":
    main_countpos()

#Now we plot the data
from helper_functions_stylistics import POS_plot_absolute
def main_plotPOSabs():
    # Path to the CSV file with comparison data
    input_file_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/merged_pos_tag_differences.csv' 
    output_plot_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/merged_pos_tag_differences.png' 
    
    # Load the data from CSV file
    data = pd.read_csv(input_file_path)
    
    # Call the helper function to plot the comparison of POS tag differences
    POS_plot_absolute(data, output_plot_path)

if __name__ == "__main__":
    main_plotPOSabs()

#Now, merge percentages for all users
from helper_functions_stylistics import merge_and_save_as_csv

def main_percmerge():
    file1 = 'results/movie_corpus/stylometrics/llama3_1_8b_it/MODEL_POS_percentagesperuser.csv'
    file2 = 'results/movie_corpus/stylometrics/llama3_1_8b_it/RANDOM_POS_percentagesperuser.csv'
    file3 = 'results/movie_corpus/stylometrics/llama3_1_8b_it/OG_POS_percentagesperuser.csv'
    output_csv_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/merged_POS_percentages.csv'
    
    merge_and_save_as_csv(file1, file2, file3, output_csv_file)

if __name__ == "__main__":
    main_percmerge()

#Visualise it!
from helper_functions_stylistics import visualize_merged_results
def main_vizpos():
    # Define the path for the merged CSV file and output plot file
    input_csv_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/merged_POS_percentages.csv'  
    output_plot_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/merged_POS_percentages.png'
    
    # Call the function to visualize the data
    visualize_merged_results(input_csv_file, output_plot_file)

if __name__ == "__main__":
    main_vizpos()

#Compare proper noun usage
from helper_functions_stylistics import calculate_shared_proper_nouns_per_conversation
def main_propnoun():
    # File paths for the different corpus files
    file_paths = {
        'original': 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG_pos_tagged.json',
        'random': 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_corpus_pos_tagged.json',
        'model': 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL_tagged.json'
    }

    # Process each file
    for corpus, file_path in file_paths.items():
        # Load the corpus data from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)

        # Construct the output file path
        output_file_path = f"results/shared_proper_nouns_{corpus}.csv"

        # Calculate and save shared proper nouns per conversation
        print(f"Processing {corpus} corpus...")
        calculate_shared_proper_nouns_per_conversation(corpus_data, output_file_path)
        print(f"Results saved to {output_file_path}")
    
if __name__ == "__main__":
    main_propnoun()

#Calculate sums and merge
from helper_functions_stylistics import merge_sums_into_final_csv, save_unique_proper_names_to_csv
def main_sumpropnouns():
    # Path for the final CSV file with merged sums
    final_output_file_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/shared_proper_nouns_sums.csv'
    
    # Merge sums into the final CSV file
    merge_sums_into_final_csv(final_output_file_path)
    print(f"Final results saved to {final_output_file_path}")#

if __name__ == "__main__":
    main_sumpropnouns()

#Look who introduces unique proper nouns

def load_corpus_data(file_path):
    """Load corpus data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

if __name__ == "__main__":
    load_corpus_data()

def main_uniquepropnouns():
    # File paths for the tagged corpora
    original_corpus_path = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG_pos_tagged.json'
    random_corpus_path = 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_corpus_pos_tagged.json'
    model_corpus_path = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL_tagged.json'
    
    # Output file paths for the results
    original_output_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_original.csv'
    random_output_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_random.csv'
    model_output_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_model.csv'

    # Load the corpus data
    original_corpus_data = load_corpus_data(original_corpus_path)
    random_corpus_data = load_corpus_data(random_corpus_path)
    model_corpus_data = load_corpus_data(model_corpus_path)

    # Call the function for each corpus to save unique proper names to CSV
    save_unique_proper_names_to_csv(original_corpus_data, original_output_path)
    save_unique_proper_names_to_csv(random_corpus_data, random_output_path)
    save_unique_proper_names_to_csv(model_corpus_data, model_output_path)

    print("Unique proper names have been saved to CSV files.")

if __name__ == "__main__":
    main_uniquepropnouns()


#Sum up and merge
from helper_functions_stylistics import calculate_and_save_sums
def main_sumuniquenouns():

    # File paths for the input CSV files
    corpus_files = {
        "unique_proper_names_original": 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_original.csv',
        "unique_proper_names_random": 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_random.csv',
        "unique_proper_names_model": 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_model.csv'
    }
    
    # Output file path for the summarized results
    output_file_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/summed_proper_names.csv'
    
    # Call the calculate_and_save_sums function
    calculate_and_save_sums(corpus_files, output_file_path)

if __name__ == "__main__":
    main_sumuniquenouns()

#Visualisation
from helper_functions_stylistics import visualize_summed_proper_names
def main_vizpropnouns():

    # File paths for input and output
    corpus_files = {
        "unique_proper_names_original": 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_original.csv',
        "unique_proper_names_random": 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_random.csv',
        "unique_proper_names_model": 'results/movie_corpus/stylometrics/llama3_1_8b_it/unique_proper_names_model.csv'
    }
    summed_output_csv = 'results/movie_corpus/stylometrics/llama3_1_8b_it/summed_proper_names.csv'
    plot_output_path = 'results/movie_corpus/stylometrics/llama3_1_8b_it/summed_proper_names.png'

    # Ensure output directories exist
    os.makedirs(os.path.dirname(summed_output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

    # Calculate and save sums to CSV
    calculate_and_save_sums(corpus_files, summed_output_csv)

    # Visualize the results and save the plot
    visualize_summed_proper_names(summed_output_csv, plot_output_path)

if __name__ == "__main__":
    main_vizpropnouns()

#Hedge word analysis
#for original
from helper_functions_stylistics import load_hedge_words
from helper_functions_stylistics import count_hedge_words

def main_hedge():
    # Define file paths
    hedge_word_files = [
        'data/stylometrics/hedge_words_list_1.txt',
        'data/stylometrics/hedge_words_list_2.txt'
    ]

    input_files = {
        "original": 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG.json',
        "random": 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_utterance_pairs_OG.json',
        "model": 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL.json'
    }

    output_paths = {
        "original": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_percentages_OG.csv',
        "random": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_percentages_RANDOM.csv',
        "model": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_percentages_MODEL.csv'
    }

    # Load hedge words
    hedge_list = load_hedge_words(hedge_word_files)

    # Process each input file
    for label, input_file in input_files.items():
        output_file = output_paths[label]
        count_hedge_words(input_file, hedge_list, output_file)
        print(f"Processed {label}: Results saved to {output_file}")

if __name__ == "__main__":
    main_hedge()

from helper_functions_stylistics import calculate_hedge_differences
def main_hedgediff():
    # Define input and output file paths
    hedge_percent_files = {
        "original": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_percentages_OG.csv',
        "random": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_percentages_RANDOM.csv',
        "model": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_percentages_MODEL.csv'
    }

    hedge_diff_output = {
        "original": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_differences_OG.csv',
        "random": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_differences_RANDOM.csv',
        "model": 'results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_differences_MODEL.csv'
    }

    # Process each dataset
    for label, input_file in hedge_percent_files.items():
        output_file = hedge_diff_output[label]
        calculate_hedge_differences(input_file, output_file)

if __name__ == "__main__":
    main_hedgediff()

#Calculate and merge averages
from helper_functions_stylistics import calculate_average_hedge_differences
def main_avghedge():

    output_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/avg_hedge_differences.csv'

    calculate_average_hedge_differences(
        output_file=output_file,
        original_path='results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_differences_OG.csv',
        random_path='results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_differences_RANDOM.csv',
        model_path='results/movie_corpus/stylometrics/llama3_1_8b_it/hedge_differences_MODEL.csv'
    )

    print(f"Average hedge word differences saved to: {output_file}")

if __name__ == "__main__":
    main_avghedge()

#Visualize the results
from helper_functions_stylistics import visualize_hedge_differences
def main_hedgeviz():
    # Define the input and output file paths
    differences_filepath = 'results/movie_corpus/stylometrics/llama3_1_8b_it/avg_hedge_differences.csv'
    plot_filepath = 'results/movie_corpus/stylometrics/llama3_1_8b_it/avg_hedge_differences.png'

    # Visualize the average differences
    visualize_hedge_differences(differences_filepath, plot_filepath)

    print(f"Visualization saved to {plot_filepath}")

if __name__ == "__main__":
    main_hedgeviz()

#Calculating accommodation for POS
from helper_functions_stylistics import process_acc_corpus

def main_acc():
    # Define file paths for the corpora
    original_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG_pos_tagged.json'
    random_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_corpus_pos_tagged.json'
    model_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL_tagged.json'

    # Define output file paths for the CSV results
    original_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/accommodation_OG.csv'
    random_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/accommodation_RANDOM.csv'
    model_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/accommodation_MODEL.csv'

    # Process the original corpus
    process_acc_corpus(original_file, 'user_x', 'user_y', original_output)
    
    # Process the random corpus
    process_acc_corpus(random_file, 'user_x', 'user_y', random_output)
    
    # Process the model corpus
    process_acc_corpus(model_file, 'model', 'user_y', model_output)

if __name__ == "__main__":
    main_acc()

#Merge the results and visualize
from helper_functions_stylistics import merge_POS_accommodation_files, visualize_POS_accommodation

def main_mergeacc():
    # Define file paths for the input and output
    original_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/accommodation_OG.csv'
    random_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/accommodation_RANDOM.csv'
    model_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/accommodation_MODEL.csv'
    merged_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/POS_accommodation_merged.csv'
    plot_output = 'results/movie_corpus/stylometrics/llama3_1_8b_it/POS_accommodation_visualization.png'

    # Call the helper function to merge the files
    merge_POS_accommodation_files(original_output, random_output, model_output, merged_output)

    # Visualize the accommodation values and save the plot
    visualize_POS_accommodation(merged_output, plot_output)

if __name__ == "__main__":
    main_mergeacc()

#Novelty tokens
from helper_functions_stylistics import calculate_novelty_tokens
def load_corpus_novelty(file_path):
    """Load a JSON corpus from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

if __name__ == "__main__":
    load_corpus_novelty()

def main_novelty():
    # File paths
    original_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_OG.json'
    random_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/random_utterance_pairs_OG.json'
    model_file = 'data/stylometrics/movie_corpus/llama3_1_8b_it/reduced_corpus_MODEL.json'

    # Load corpora
    original_corpus = load_corpus_novelty(original_file)
    random_corpus = load_corpus_novelty(random_file)
    model_corpus = load_corpus_novelty(model_file)

    # Calculate novelty tokens
    novelty_original = calculate_novelty_tokens(original_corpus, "user_x", "user_y")
    novelty_random = calculate_novelty_tokens(random_corpus, "user_x", "user_y")
    novelty_model = calculate_novelty_tokens(model_corpus, "model", "user_y")

    # Save results
    output_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/novelty_results.json'
    results = {
        "novelty_original": novelty_original,
        "novelty_random": novelty_random,
        "novelty_model": novelty_model
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Novelty token analysis completed. Results saved to {output_file}")

if __name__ == "__main__":
    main_novelty()

#Calculate the average and visualize results
from helper_functions_stylistics import calculate_average_novelty, save_novelty_results_to_csv, visualize_novelty_averages

def main_avgnovelty():
    # Path to the novelty results JSON file
    novelty_results_file = 'results/movie_corpus/stylometrics/llama3_1_8b_it/novelty_results.json'
    output_csv = 'results/movie_corpus/stylometrics/llama3_1_8b_it/novelty_averages.csv'
    output_png = 'results/movie_corpus/stylometrics/llama3_1_8b_it/novelty_averages.png'

    # Calculate the average novelty values for each corpus
    averages = calculate_average_novelty(novelty_results_file)

    # Print the results
    print(f"Average Novelty - Original: {averages['original']:.2f}%")
    print(f"Average Novelty - Random: {averages['random']:.2f}%")
    print(f"Average Novelty - Model: {averages['model']:.2f}%")

    # Save the novelty results to CSV
    save_novelty_results_to_csv(novelty_results_file, output_csv)

    # Visualize the averages in a bar chart and save as PNG
    visualize_novelty_averages(averages, save_path=output_png)

if __name__ == "__main__":
    main_avgnovelty()


