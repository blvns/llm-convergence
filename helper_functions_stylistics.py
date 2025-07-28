import json
import random
import pandas as pd
import os
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import csv
import re

#reduce the OG corpus to the last utterances of user _x + the corresponding utterance of user_y
def reduce_corpus(input_file, output_file):
    """
    Reduces a corpus to the last utterance of user_x and 
    the utterance of user_y that comes directly before it for each conversation.
    The results are stored in a JSON file.

    Parameters:
        input_file (str): The path to the JSON file containing the original corpus.
        output_file (str): The path to the JSON file where the reduced corpus will be saved.

    Returns:
        None
    """
    # Load the original corpus
    with open(input_file, "r", encoding="utf-8") as json_file:
        original_corpus = json.load(json_file)

    reduced_corpus = {}

    # Iterate through the conversations
    for conversation_id, turns in original_corpus.items():
        last_user_x = None
        last_user_y_before_x = None

        # Iterate through the turns in reverse order to find the last user_x and preceding user_y
        for i in range(len(turns) - 1, -1, -1):  
            if turns[i]["role"] == "user_x":  
                last_user_x = turns[i]
                # Find the user_y utterance that directly precedes it
                for j in range(i - 1, -1, -1):
                    if turns[j]["role"] == "user_y": 
                        last_user_y_before_x = turns[j]
                        break
                break  

        # Only include the conversation in the reduced corpus if both utterances are found
        if last_user_x and last_user_y_before_x:
            reduced_corpus[conversation_id] = [last_user_y_before_x, last_user_x]

    # Save the reduced corpus to the specified output file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(reduced_corpus, json_file, indent=4, ensure_ascii=False)

    print(f"Reduced corpus saved to {output_file}")

#Creating a corpus of random conversations 
def build_random_pairs(reduced_corpus, output_file):
    """
    Matches every user_y utterance with a random user_x utterance from a different conversation.
    Saves the results in the same format as the input file.

    Parameters:
        reduced_corpus (dict): A dictionary where each key is a conversation ID and the value is a list of utterances.
        output_file (str): The path to the JSON file where the random utterance pairs will be saved.

    Returns:
        None
    """
    # Extract all user_x and user_y utterances along with their conversation IDs
    user_x_utterances = []
    user_y_utterances = []

    for conversation_id, utterances in reduced_corpus.items():
        for turn in utterances:
            if turn["role"] == "user_x":
                user_x_utterances.append({
                    "conversation_id": conversation_id,
                    "content": turn["content"]
                })
            elif turn["role"] == "user_y":
                user_y_utterances.append({
                    "conversation_id": conversation_id,
                    "content": turn["content"]
                })

    # Create a new dictionary to store the random pairs
    random_pairs = {}

    # Match user_y utterances with random user_x utterances from different conversations
    for user_y in user_y_utterances:
        random_user_x = random.choice([x for x in user_x_utterances if x["conversation_id"] != user_y["conversation_id"]])
        
        conversation_id = user_y["conversation_id"]
        if conversation_id not in random_pairs:
            random_pairs[conversation_id] = []
        
        random_pairs[conversation_id].append({
            "role": "user_x",
            "content": random_user_x["content"]
        })
        random_pairs[conversation_id].append({
            "role": "user_y",
            "content": user_y["content"]
        })

    # Save the random pairs to the specified output file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(random_pairs, json_file, indent=4, ensure_ascii=False)

    print(f"Random utterance pairs saved to {output_file}")

#Now we define a function to shorten the generated corpus to one turn of model and user_y
def reduce_model_corpus(input_file, output_file):
    """
    Reduces the generated corpus to the last utterance of the model and the user_y utterance
    that comes directly before it for each conversation.
    The respective conversation ID remains the same, and all other utterances are removed.
    The results are stored in a new JSON file.

    Parameters:
        input_file (str): The path to the JSON file containing the original generated corpus.
        output_file (str): The path to the JSON file where the reduced corpus will be saved.

    Returns:
        None
    """
    # Load the original corpus
    with open(input_file, "r", encoding="utf-8") as json_file:
        original_corpus = json.load(json_file)

    reduced_corpus = {}

    # Iterate over the conversation IDs and their respective turns
    for conversation_id, turns in original_corpus.items():
        last_model_utterance = None
        user_y_before_model = None

        # Iterate through the conversation to find the last utterance of the model and the preceding user_y
        for i in range(len(turns) - 1, -1, -1):  # Iterate in reverse order
            if turns[i]["role"] == "model":
                last_model_utterance = turns[i]
                # Find the user_y utterance that directly precedes it
                for j in range(i - 1, -1, -1):
                    if turns[j]["role"] == "user_y":
                        user_y_before_model = turns[j]
                        break
                break  # Stop searching once the last model utterance is found

        # Only include the conversation if both the model's and user_y's utterances are found
        if last_model_utterance and user_y_before_model:
            reduced_corpus[conversation_id] = [
                {
                    "role": "user_y",
                    "content": user_y_before_model["content"]
                },
                {
                    "role": "model",
                    "content": last_model_utterance["content"]
                }
            ]

    # Save the reduced corpus to the specified output file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(reduced_corpus, json_file, indent=4, ensure_ascii=False)

    print(f"Reduced model corpus saved to {output_file}")

#Calculating utterance length and differences between users
def count_utterance_length_and_differences(corpus_data, output_file, comparison_type):
    """
    Counts the utterance lengths in characters for each user in each conversation, 
    calculates the absolute differences between users, and stores the results.

    Parameters:
        corpus_data (dict): The corpus data (dict with conversation IDs as keys and lists of utterances).
        output_file (str): The path to the output file where the results will be stored.
        comparison_type (str): The comparison type ('user_y_model' or 'user_y_user_x') to determine which roles to compare.

    Returns:
        None
    """
    results = []


    for conversation_id, conversation in corpus_data.items():
        user_y_utterances = []
        comparison_utterances = []
        
        # Iterate through the conversation and count the lengths of each utterance
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            utterance_length = len(content)
            
            if role == "user_y":
                user_y_utterances.append({"utterance": content, "length": utterance_length})
            elif (role == "model" and comparison_type == "user_y_model") or (role == "user_x" and comparison_type == "user_y_user_x"):
                comparison_utterances.append({"utterance": content, "length": utterance_length})
        
        # Calculate absolute differences between user_y and the comparison group (either model or user_x)
        user_y_comparison_diff = 0
        
        # Calculate absolute difference for each utterance between user_y and comparison group
        if user_y_utterances and comparison_utterances:
            for uy, comparison in zip(user_y_utterances, comparison_utterances):
                user_y_comparison_diff += abs(uy["length"] - comparison["length"])

        # Add the result to the list for this conversation
        results.append({
            "conversation_id": conversation_id,
            "user_y_comparison_diff": user_y_comparison_diff,
            "user_y_utterances": user_y_utterances,
            "comparison_utterances": comparison_utterances
        })
    
    # Save the results to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

    print(f"Utterance length differences saved to {output_file}")


#Now, we want to merge the results
import pandas as pd

def merge_and_calculate_differences(original_data, random_data, model_data, output_file):
    """
    Merges differences from three datasets (original, random, model) into a single dataframe 
    and saves it as a CSV file.

    Parameters:
        original_data (list): List of dictionaries, each representing a conversation.
        random_data (list): List of dictionaries, each representing a conversation.
        model_data (list): List of dictionaries, each representing a conversation.
        output_file (str): Path where the merged results should be saved.

    Returns:
        None
    """
    # Helper: Convert list of dictionaries into a dictionary with conversation_id as the key
    def convert_to_dict(data):
        return {entry["conversation_id"]: entry["user_y_comparison_diff"] for entry in data}

    # Convert all datasets to dictionaries for easy lookup
    original_dict = convert_to_dict(original_data)
    random_dict = convert_to_dict(random_data)
    model_dict = convert_to_dict(model_data)

    # Find intersection of conversation IDs
    all_conversation_ids = set(original_dict.keys()) & set(random_dict.keys()) & set(model_dict.keys())

    # Prepare merged data
    merged_data = []
    for conversation_id in all_conversation_ids:
        merged_data.append({
            "conversation_id": conversation_id,
            "usery_vs_userx_original": original_dict[conversation_id],
            "usery_vs_userx_random": random_dict[conversation_id],
            "usery_vs_model": model_dict[conversation_id]
        })

    # Create a DataFrame
    df = pd.DataFrame(merged_data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

    print(f"Data merged and saved to {output_file}.")


#Now, we are calculating averages: 
def calculate_averages(data):
    """
    Calculate the average of each comparison pair for all conversation IDs.

    Parameters:
        data (DataFrame): A DataFrame containing conversation data with columns for the different comparison pairs.

    Returns:
        DataFrame: A DataFrame with one row showing the average values for each comparison pair.
    """
    # Check if the data has the necessary columns
    if "conversation_id" not in data.columns:
        raise ValueError("The DataFrame must contain a 'conversation_id' column.")

    # Calculate averages for all columns except 'conversation_id'
    comparison_columns = [col for col in data.columns if col != 'conversation_id']
    
    # Calculate the mean for each comparison column
    averages = data[comparison_columns].mean()
    
    # Create a new DataFrame to store averages with the conversation_id as 'Average'
    averages_df = pd.DataFrame(averages).transpose()
    averages_df["conversation_id"] = "Average"
    
    # Reorder columns so that 'conversation_id' is the first column
    averages_df = averages_df[["conversation_id"] + comparison_columns]
    
    return averages_df

#Merging the averages
def merge_results(original_file, random_file, model_file, output_file):
    """
    Merges differences from three JSON datasets (original, random, model) into a single dataframe
    based on a common identifier (e.g., conversation_id) and compares the "Average" value.

    Parameters:
        original_file (str): Path to the JSON file containing the original dataset.
        random_file (str): Path to the JSON file containing the random dataset.
        model_file (str): Path to the JSON file containing the model dataset.
        output_file (str): Path where the merged results should be saved.

    Returns:
        None
    """
    # Load the JSON files into dictionaries
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    with open(random_file, 'r', encoding='utf-8') as f:
        random_data = json.load(f)

    with open(model_file, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    # Convert dictionaries to dataframes for easier manipulation
    df_original = pd.DataFrame(original_data)
    df_random = pd.DataFrame(random_data)
    df_model = pd.DataFrame(model_data)

    # Merge the dataframes on a common identifier (e.g., conversation_id)
    merged_df = pd.merge(df_original, df_random, on='conversation_id', suffixes=('_original', '_random'))
    merged_df = pd.merge(merged_df, df_model, on='conversation_id')

    # Now we have columns for the "Average" value 
    merged_df['average_original_random'] = merged_df['Average_original'] - merged_df['Average_random']
    merged_df['average_random_model'] = merged_df['Average_random'] - merged_df['Average_model']
    merged_df['average_original_model'] = merged_df['Average_original'] - merged_df['Average_model']

    # Convert the merged dataframe back to a dictionary
    merged_data = merged_df.to_dict(orient='records')

    # Save the merged data as a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Results successfully merged and saved to {output_file}.")


#Here, I am defining a function for a barplot to compare the results

def plot_comparison_avglength(data, output_path):
    """
    Creates and saves a bar plot comparing the differences for user_y vs user_x, user_y vs model, and user_x vs user_y random.

    Parameters:
        data (DataFrame): A DataFrame with the comparison values.
        output_path (str): Path where the plot image will be saved.
    
    Returns:
        None
    """
    # Ensure that the columns are correctly referenced
    comparison_columns = ['usery_vs_userx_original', 'usery_vs_userx_random', 'usery_vs_model']
    
    # Extract the averages
    averages = data[comparison_columns].iloc[0]  # Only one row for averages
    
    # Set colors for each bar
    colors = {
        'usery_vs_userx_original': 'palegreen',
        'usery_vs_userx_random': 'peachpuff',
        'usery_vs_model': 'cornflowerblue'
    }
    
    # Plotting the bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set bar positions
    positions = [0, 1, 2]  # Left, middle, right
    
    # Create bars
    ax.bar(positions[0], averages['usery_vs_userx_random'], color=colors['usery_vs_userx_random'], width=0.4, label='user_y vs user_x random', align='center')
    ax.bar(positions[1], averages['usery_vs_model'], color=colors['usery_vs_model'], width=0.4, label='user_y vs model', align='center')
    ax.bar(positions[2], averages['usery_vs_userx_original'], color=colors['usery_vs_userx_original'], width=0.4, label='user_y vs user_x original', align='center')
    
    # Labels and title
    ax.set_ylabel('Difference')
    ax.set_title('Comparison of Utterance Length Differences')
    ax.set_xticks(positions)
    ax.set_xticklabels(['user_y vs user_x random', 'user_y vs model', 'user_y vs user_x original'])
    
    # Show legend
    ax.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Bar plot saved to {output_path}")

#Now we define a function for POS-Tagging 
def process_and_save_corpus(file_path, output_path):
    """
    Loads a corpus from a file, performs POS tagging, and saves the results to a new file.
    
    Parameters:
        file_path (str): The path to the input corpus file.
        output_path (str): The path to save the POS-tagged corpus.
    
    Returns:
        None
    """
    import json
    import spacy

    # Load the corpus data
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    # Perform POS tagging
    tagged_data = pos_tag_corpus(corpus_data)

    # Save the POS-tagged data to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tagged_data, f, indent=4, ensure_ascii=False)
    
    print(f"POS-tagged data saved to {output_path}")


def pos_tag_corpus(corpus_data):
    """
    Tags the corpus data with POS using a spaCy model.
    
    Parameters:
        corpus_data (dict): A dictionary where keys are conversation IDs and values are lists of utterances.
        
    Returns:
        tagged_data (dict): The tagged corpus data, with POS tags added to each utterance.
    """
    # Load the spaCy model
    nlp = spacy.load("en_core_web_trf")

    tagged_data = {}

    for conversation_id, turns in corpus_data.items():
        tagged_turns = []

        for turn in turns:
            content = turn["content"]
            doc = nlp(content)

            # Generate tagged tokens for each utterance
            tagged_turn = [{"word": token.text, "pos": token.pos_} for token in doc]

            tagged_turns.append({
                "role": turn["role"],
                "content": content,
                "tagged_utterance": tagged_turn
            })

        tagged_data[conversation_id] = tagged_turns

    return tagged_data



#Now, we count the POS categories

def calculate_pos_tag_differences(corpus_data):
    """
    Calculate the average difference in POS tag usage for each role (user_x, user_y, model)
    per utterance for each POS tag.

    Parameters:
    - corpus_data (dict): A dictionary where keys are conversation IDs and values are lists of conversation turns.

    Returns:
    - avg_pos_tag_differences (dict): The calculated average differences for each POS tag.
    """

    # Initialize containers for storing POS tag differences
    pos_tag_differences = defaultdict(list)

    # Loop through the dictionary of conversation data
    for conversation_id, conversation_turns in corpus_data.items():
        # Ensure conversation_turns is a list of dictionaries (conversation turns)
        if isinstance(conversation_turns, list):
            user_x_pos_counts = defaultdict(int)  # Initialize for each conversation
            user_y_pos_counts = defaultdict(int)  # Initialize for each conversation
            model_pos_counts = defaultdict(int)  # Initialize for each conversation

            for turn in conversation_turns:
                # Make sure the turn has the expected structure
                if isinstance(turn, dict) and 'role' in turn and 'tagged_utterance' in turn:
                    # Get the role and the tagged utterance
                    role = turn['role']
                    tagged_utterance = turn['tagged_utterance']
                    
                    # Create a dictionary to store POS counts for the current utterance
                    pos_counts = defaultdict(int)
                    
                    # Count the POS tags in the current utterance
                    for word in tagged_utterance:
                        pos_counts[word['pos']] += 1
                    
                    # Store the counts based on the role
                    if role == 'user_x':
                        user_x_pos_counts.update(pos_counts)
                    elif role == 'user_y':
                        user_y_pos_counts.update(pos_counts)
                    elif role == 'model':
                        model_pos_counts.update(pos_counts)

            # Now calculate the difference for each POS tag
            for pos in set(user_x_pos_counts.keys()).union(set(user_y_pos_counts.keys())):
                diff = abs(user_x_pos_counts.get(pos, 0) - user_y_pos_counts.get(pos, 0))
                pos_tag_differences[pos].append(diff)

    # Calculate the average difference for each POS tag across all utterances
    avg_pos_tag_differences = {pos: sum(diff) / len(diff) if len(diff) > 0 else 0 for pos, diff in pos_tag_differences.items()}

    return avg_pos_tag_differences

def process_and_save_pos_tag_differences(corpus_data, file_path):
    """
    Process the corpus data, calculate the POS tag differences, and save the results to a file.
    """
    avg_pos_tag_differences = calculate_pos_tag_differences(corpus_data)

    # Save the results to a JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(avg_pos_tag_differences, f, indent=4, ensure_ascii=False)

    print(f"POS tag differences saved to {file_path}")

# Merging the POS average data
def load_data(file_path):
    """Load the JSON data from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_csv(user_y_vs_userx_original, user_y_vs_userx_random, user_y_vs_model):
    """Merge the results of the three comparison files into one table."""

    # Ensure all POS tags are included (union of all keys)
    all_pos_tags = set(user_y_vs_userx_original.keys()).union(user_y_vs_userx_random.keys(), user_y_vs_model.keys())

    # Prepare data for the table
    data = {}
    for tag in all_pos_tags:
        data[tag] = {
            'user_y_vs_userx_original': user_y_vs_userx_original.get(tag, 0),
            'user_y_vs_userx_random': user_y_vs_userx_random.get(tag, 0),
            'user_y_vs_model': user_y_vs_model.get(tag, 0)
        }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data).T
    df.reset_index(inplace=True)
    df.columns = ['POS_tag', 'user_y_vs_userx_original', 'user_y_vs_userx_random', 'user_y_vs_model']

    return df

def save_merged_data(df, output_path):
    """Save the merged DataFrame to a CSV or JSON file."""
    df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

#Plot the results
import matplotlib.pyplot as plt
import pandas as pd

def POS_plot_absolute(data, output_path):
    """
    Creates and saves a bar plot visualizing the average POS tag differences.

    Parameters:
        data (DataFrame): A DataFrame containing POS tags and their average differences for various comparisons.
        output_path (str): Path where the plot image will be saved.

    Returns:
        None
    """
    # Debugging: Print the data to ensure it matches expectations
    print("Data for plotting:\n", data)

    # Ensure required columns are present 
    required_columns = ['POS_tag', 'user_y_vs_userx_original', 'user_y_vs_userx_random', 'user_y_vs_model']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return

    # Set colors for the comparisons
    colors = {
        'user_y_vs_userx_original': 'palegreen',
        'user_y_vs_userx_random': 'peachpuff',
        'user_y_vs_model': 'cornflowerblue'
    }

    # Prepare the plot
    plt.figure(figsize=(10, 8))

    # Define the positions for the bars
    bar_width = 0.25
    index = range(len(data))

    # Plot each comparison column against POS_tag
    plt.bar(index, data['user_y_vs_userx_original'], bar_width, color=colors['user_y_vs_userx_original'], label='User Y vs User X (Original)')
    plt.bar([i + bar_width for i in index], data['user_y_vs_userx_random'], bar_width, color=colors['user_y_vs_userx_random'], label='User Y vs User X (Random)')
    plt.bar([i + 2 * bar_width for i in index], data['user_y_vs_model'], bar_width, color=colors['user_y_vs_model'], label='User Y vs Model')

    # Add labels and title
    plt.ylabel('Average Difference')
    plt.title('Average POS Tag Differences')
    plt.xticks([i + bar_width for i in index], data['POS_tag'], rotation=45, ha='right')

    # Add a legend
    plt.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Bar plot saved to {output_path}")

#Now, we analyse the relative differences in POS
# Function to calculate the percentual usage of POS tags for each user in a given conversation
def calculate_pos_percentages(input_file, output_file):
    """
    Calculate the percentual POS usage for each user (user_x, user_y, or model),
    compare the differences in percentages between user_x and user_y, and save the results to a CSV file.
    """
    # Load the data from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if data is empty
    if not data:
        print(f"Error: No data found in {input_file}")
        return

    # Initialize dictionaries to store POS usage for user_x and user_y separately
    pos_user_x_counts = defaultdict(list)
    pos_user_y_counts = defaultdict(list)
    
    # Iterate over each conversation
    for conversation_id, utterances in data.items():
        # Initialize dictionaries to count POS for each user
        user_x_pos_counts = defaultdict(int)
        user_y_pos_counts = defaultdict(int)
        total_user_x_pos_count = 0
        total_user_y_pos_count = 0

        # Count POS tags for user_x and user_y in the current conversation
        for utterance in utterances:
            role = utterance['role']
            for word in utterance['tagged_utterance']:
                pos = word['pos']
                if role == "user_x":
                    user_x_pos_counts[pos] += 1
                    total_user_x_pos_count += 1
                elif role == "user_y":
                    user_y_pos_counts[pos] += 1
                    total_user_y_pos_count += 1

        # Calculate relative usage in percentages for user_x
        user_x_usage = {pos: (count / total_user_x_pos_count) * 100 if total_user_x_pos_count > 0 else 0
                        for pos, count in user_x_pos_counts.items()}
        
        # Calculate relative usage in percentages for user_y
        user_y_usage = {pos: (count / total_user_y_pos_count) * 100 if total_user_y_pos_count > 0 else 0
                        for pos, count in user_y_pos_counts.items()}
        
        # Store the calculated POS usage for each user
        pos_user_x_counts[conversation_id] = user_x_usage
        pos_user_y_counts[conversation_id] = user_y_usage

    # Now calculate the average differences in percentages between user_x and user_y
    pos_diff_avg = defaultdict(float)
    total_comparisons = 0

    # Compare the POS usage between user_x and user_y for each conversation
    for conversation_id in pos_user_x_counts.keys():
        user_x_usage = pos_user_x_counts[conversation_id]
        user_y_usage = pos_user_y_counts[conversation_id]

        # Compare each POS tag's usage between user_x and user_y
        for pos in set(user_x_usage.keys()).union(user_y_usage.keys()):
            user_x_percent = user_x_usage.get(pos, 0)
            user_y_percent = user_y_usage.get(pos, 0)
            pos_diff_avg[pos] += abs(user_x_percent - user_y_percent)  # Calculate the absolute difference
            total_comparisons += 1

    # Calculate average difference for each POS tag
    pos_diff_avg = {pos: diff / total_comparisons for pos, diff in pos_diff_avg.items()}

    # Prepare DataFrame for CSV
    pos_diff_df = pd.DataFrame(list(pos_diff_avg.items()), columns=["POS_tag", "Average_difference_percentage"])

    # Ensure the output directory exists
    output_directory = os.path.dirname(output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save results to the output CSV file
    pos_diff_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


#Compare percentages for all users
def merge_and_save_as_csv(file1, file2, file3, output_csv_file):
    """
    Merge three CSV files based on POS_tag and rename the 'Average_difference_percentage' 
    column based on the file type (model, random, original) for each file.
    Save the merged result into a new CSV file.
    """
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file1)  # This file contains model_avg_percentage
    df2 = pd.read_csv(file2)  # This file contains user_x_percentage and user_y_percentage
    df3 = pd.read_csv(file3)  # This file contains additional percentages or data

    # Rename the 'Average_difference_percentage' column based on the file type
    if 'MODEL' in file1.upper(): 
        df1 = df1.rename(columns={'Average_difference_percentage': 'model'})
    elif 'RANDOM' in file1.upper():
        df1 = df1.rename(columns={'Average_difference_percentage': 'random'})
    else:
        df1 = df1.rename(columns={'Average_difference_percentage': 'original'})

    if 'MODEL' in file2.upper(): 
        df2 = df2.rename(columns={'Average_difference_percentage': 'model'})
    elif 'RANDOM' in file2.upper():
        df2 = df2.rename(columns={'Average_difference_percentage': 'random'})
    else:
        df2 = df2.rename(columns={'Average_difference_percentage': 'original'})

    if 'MODEL' in file3.upper(): 
        df3 = df3.rename(columns={'Average_difference_percentage': 'model'})
    elif 'RANDOM' in file3.upper():
        df3 = df3.rename(columns={'Average_difference_percentage': 'random'})
    else:
        df3 = df3.rename(columns={'Average_difference_percentage': 'original'})

    # Extract the relevant columns from each DataFrame
    df1 = df1[['POS_tag', 'model']]  # After renaming, keep POS_tag and the new column
    df2 = df2[['POS_tag', 'random']]  
    df3 = df3[['POS_tag', 'original']]  

    # Merge the three DataFrames on 'POS_tag'
    merged_df = pd.merge(df1, df2, on='POS_tag', how='outer')
    merged_df = pd.merge(merged_df, df3, on='POS_tag', how='outer')

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_csv_file, index=False)

    print(f"Merged data saved to {output_csv_file}")


#Visualize the results
def visualize_merged_results(input_csv_file, output_plot_file):
    """
    Visualize the merged POS tag percentage data (model, random, original) from a CSV file
    and save the plot to an output file.
    """
    # Load the merged CSV data into a DataFrame
    df = pd.read_csv(input_csv_file)
    
    # Set the 'POS_tag' column as the index for better plotting
    df.set_index('POS_tag', inplace=True)
    
    # Define the colors for each category
    colors = {'model': 'palegreen', 'random': 'peachpuff', 'original': 'cornflowerblue'}
    
    # Plot the data (using a bar plot)
    ax = df.plot(kind='bar', figsize=(12, 6), color=[colors['model'], colors['random'], colors['original']], width=0.8)
    
    # Set plot labels and title
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('POS Tag')
    ax.set_title('POS Tag Percentage Avg. Differences per Dataset')
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    
    # Save the plot to the output file
    plt.tight_layout()
    plt.savefig(output_plot_file)
    
    # Show the plot (optional)
    plt.show()

    # Print confirmation message
    print(f"Visualization saved to {output_plot_file}")


#Compare proper noun usage: 
def visualize_proper_noun_usage(input_csv_file, output_plot_file):
    """
    Visualize the average difference in the usage of proper nouns (PROPN) 
    between model, random, and original, and save the plot to an output file.
    """
    # Load the merged CSV data into a DataFrame
    df = pd.read_csv(input_csv_file)
    
    # Filter the DataFrame to get only the row corresponding to 'PROPN' (proper noun)
    proper_noun_data = df[df['POS_tag'] == 'PROPN']
    
    # Check if 'PROPN' exists in the data
    if proper_noun_data.empty:
        print("No data found for proper nouns (PROPN).")
        return
    
    # Plot the comparison for proper noun usage between model, random, and original
    ax = proper_noun_data[['model', 'random', 'original']].plot(kind='bar', figsize=(8, 6), 
                                                                color=['palegreen', 'peachpuff', 'cornflowerblue'], width=0.8)
    
    # Set plot labels and title
    ax.set_ylabel('Percentage Usage (%)')
    ax.set_xlabel('Proper Noun (PROPN)')
    ax.set_title('Average Difference in Proper Noun Usage: Model, Random, and Original')
    
    # Display the percentage values on top of the bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    fontsize=10, color='black', 
                    xytext=(0, 8), textcoords='offset points')
    
    # Save the plot to the output file
    plt.tight_layout()
    plt.savefig(output_plot_file)
    
    # Show the plot (optional)
    plt.show()
    
    # Print confirmation message
    print(f"Proper noun usage visualization saved to {output_plot_file}")

#More thorough analysis
#Shared proper nouns
def find_shared_proper_nouns_per_conversation(corpus_data):
    """
    This function finds the shared proper nouns per conversation between two roles in the corpus.

    :param corpus_data: The conversation data from the corpus.
    :return: A dictionary with conversation_id as keys and shared proper nouns count as values.
    """
    shared_proper_nouns_per_conversation = {}

    # Iterate over each conversation in the corpus
    for conversation_id, conversation in corpus_data.items():
        # Initialize sets for proper nouns for each role
        user_proper_nouns = set()
        model_proper_nouns = set()

        # Iterate over each utterance in the conversation
        roles_in_conversation = set()
        for utterance in conversation:
            role = utterance['role']
            tagged_utterance = utterance['tagged_utterance']

            # Track which roles are present in the conversation
            roles_in_conversation.add(role)

            # Extract proper nouns from the tagged utterance
            for token in tagged_utterance:
                word = token['word']
                pos = token['pos']

                # Check if the word is a proper noun (POS = 'PROPN')
                if pos == 'PROPN':
                    if role == 'user_x' or role == 'user_y':
                        user_proper_nouns.add(word)
                    elif role == 'model':
                        model_proper_nouns.add(word)

        # If both roles are present (user_x/user_y and model)
        if len(roles_in_conversation) == 2:
            # Find the intersection of proper nouns between the two roles
            shared_proper_nouns = user_proper_nouns.intersection(model_proper_nouns)
            shared_proper_nouns_per_conversation[conversation_id] = len(shared_proper_nouns)

    return shared_proper_nouns_per_conversation

def write_shared_nouns_to_csv(shared_nouns, output_file_path):
    """
    Writes the shared proper nouns count per conversation to a CSV file.

    :param shared_nouns: A dictionary with conversation_id as keys and shared proper nouns count as values.
    :param output_file_path: Path to the output CSV file.
    """
    # Write the shared nouns count to CSV
    with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['conversation_id', 'shared_proper_nouns'])  # Write header

        # Write each conversation's shared proper noun count
        for conversation_id, shared_count in shared_nouns.items():
            writer.writerow([conversation_id, shared_count])

def calculate_shared_proper_nouns_per_conversation(corpus_data, output_file_path):
    """
    This function calculates the shared proper nouns for each conversation in the corpus
    and writes the results to a CSV file.

    :param corpus_data: The conversation data from the corpus.
    :param output_file_path: The output CSV file to save the results.
    """
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['conversation_id', 'shared_proper_nouns']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each conversation in the corpus
        for conversation_id, conversation in corpus_data.items():
            # Initialize sets for proper nouns for each role
            user_proper_nouns = set()
            model_proper_nouns = set()
            roles_in_conversation = set()

            # Iterate over each utterance in the conversation
            for utterance in conversation:
                role = utterance['role']
                tagged_utterance = utterance['tagged_utterance']

                # Track which roles are present in the conversation
                roles_in_conversation.add(role)

                # Extract proper nouns from the tagged utterance
                for token in tagged_utterance:
                    word = token['word']
                    pos = token['pos']

                    # Check if the word is a proper noun (POS = 'PROPN')
                    if pos == 'PROPN':
                        if role == 'user_x' or role == 'user_y':
                            user_proper_nouns.add(word)
                        elif role == 'model':
                            model_proper_nouns.add(word)

            # Check if both roles are present (only two roles should appear in the conversation)
            if len(roles_in_conversation) == 2:
                # Find the intersection of proper nouns between the two roles
                shared_proper_nouns = user_proper_nouns.intersection(model_proper_nouns)
                shared_proper_nouns_count = len(shared_proper_nouns)

                # Write the result to the CSV file
                writer.writerow({'conversation_id': conversation_id, 'shared_proper_nouns': shared_proper_nouns_count})

#calculate the sums and merge the results
def calculate_sum_of_shared_proper_nouns(file_path):
    """
    This function calculates the sum of shared proper nouns from a given CSV file.

    :param file_path: Path to the CSV file with shared proper nouns data.
    :return: The sum of shared proper nouns in the CSV file.
    """
    total_shared_nouns = 0
    
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_shared_nouns += int(row['shared_proper_nouns'])
    
    return total_shared_nouns


def merge_sums_into_final_csv(output_file_path):
    """
    This function merges the sums of shared proper nouns from the three corpora
    into one final CSV file.

    :param output_file_path: Path to the final CSV file where the results will be saved.
    """
    # File paths for the resulting output files from each corpus
    sum_files = {
        'original': 'results/shared_proper_nouns_original.csv',
        'random': 'results/shared_proper_nouns_random.csv',
        'model': 'results/shared_proper_nouns_model.csv'
    }

    # Calculate the sum of shared proper nouns for each file
    sums = {}
    for corpus, file_path in sum_files.items():
        sums[corpus] = calculate_sum_of_shared_proper_nouns(file_path)

    # Write the sums into the final CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['corpus', 'sum_of_shared_proper_nouns']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write the sum for each corpus
        for corpus, total in sums.items():
            writer.writerow({'corpus': corpus, 'sum_of_shared_proper_nouns': total})

#Unique proper nouns
def find_unique_proper_names(corpus_data):
    """
    This function finds how many unique proper names each role (user_x, user_y, model) uses in each conversation.
    
    A unique proper name is defined as a proper noun that appears only once within a conversation.
    
    :param corpus_data: The conversation data from the corpus.
    :return: A dictionary with conversation_id as the key and the count of unique proper names for each role.
    """
    unique_proper_names = {}

    # Iterate over each conversation in the corpus
    for conversation_id, conversation in corpus_data.items():
        # Initialize counters for proper names for each role
        role_proper_names = {
            'user_x': Counter(),
            'user_y': Counter(),
            'model': Counter()
        }

        # Iterate over each utterance in the conversation
        for utterance in conversation:
            role = utterance['role']
            tagged_utterance = utterance['tagged_utterance']

            # Extract proper nouns from the tagged utterance
            for token in tagged_utterance:
                word = token['word']
                pos = token['pos']

                # Check if the word is a proper noun (POS = 'PROPN') and count it for the correct role
                if pos == 'PROPN' and role in role_proper_names:
                    role_proper_names[role][word] += 1

        # Identify unique proper names (those that appear only once) for each role
        conversation_unique_proper_names = {
            role: sum(1 for count in counter.values() if count == 1)
            for role, counter in role_proper_names.items()
        }

        # Store the result for the conversation
        unique_proper_names[conversation_id] = conversation_unique_proper_names

    return unique_proper_names


def save_unique_proper_names_to_csv(corpus_data, output_file_path):
    """
    This function calculates the number of unique proper names each role uses in each conversation
    and saves the results to a CSV file.
    """
    unique_proper_names = find_unique_proper_names(corpus_data)
    
    # Debugging output: print the structure of the result
    print("Unique Proper Names Structure:")
    print(unique_proper_names)

    # Write the results to a CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['conversation_id', 'user_x_unique', 'user_y_unique', 'model_unique']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write each conversation's result
        for conversation_id, counts in unique_proper_names.items():
            writer.writerow({
                'conversation_id': conversation_id,
                'user_x_unique': counts.get('user_x', 0),
                'user_y_unique': counts.get('user_y', 0),
                'model_unique': counts.get('model', 0)
            })

#Sum up and merge

def calculate_and_save_sums(corpus_files, output_file_path):
    """
    This function calculates the sum of unique proper names for each user across conversations in each corpus
    and saves the results to a CSV file.

    :param corpus_files: A dictionary where keys are the document names and values are the file paths for the corpora CSVs.
    :param output_file_path: Path to save the resulting CSV file.
    """
    # Initialize a dictionary to store sums of unique proper names for each user across documents
    sum_unique_proper_names = {
        'user_x': {},
        'user_y': {},
        'model': {}
    }

    # Process each corpus file
    for doc_name, file_path in corpus_files.items():
        # Initialize sums for the current document
        user_x_sum = 0
        user_y_sum = 0
        model_sum = 0

        # Read the CSV file
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Sum up the counts for each role
            for row in reader:
                user_x_sum += int(row['user_x_unique'])
                user_y_sum += int(row['user_y_unique'])
                model_sum += int(row['model_unique'])

        # Store the sums in the dictionary
        sum_unique_proper_names['user_x'][doc_name] = user_x_sum
        sum_unique_proper_names['user_y'][doc_name] = user_y_sum
        sum_unique_proper_names['model'][doc_name] = model_sum

    # Write the results to a CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['user', *corpus_files.keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write the sums for each user
        for user, sums in sum_unique_proper_names.items():
            row = {'user': user, **sums}
            writer.writerow(row)

    print(f"Summed unique proper names saved to {output_file_path}")

#Visualisation
def visualize_summed_proper_names(input_csv, output_image_path="results/plots/summed_proper_names.png"):
    """
    Visualizes the summed unique proper names across corpora and saves the plot.

    :param input_csv: Path to the CSV file containing the summed data.
    :param output_image_path: Path to save the plot image.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the data
    data = pd.read_csv(input_csv)
    print("Loaded data:\n", data)  # Debugging: Print the data to ensure correctness

    # Define colors for the bars
    colors = {
        "unique_proper_names_original": "palegreen",
        "unique_proper_names_random": "peachpuff",
        "unique_proper_names_model": "cornflowerblue"
    }

    # Prepare the data for visualization
    available_columns = ["unique_proper_names_original", "unique_proper_names_random", "unique_proper_names_model"]
    column_colors = [colors[col] for col in available_columns]
    
    data.set_index("user", inplace=True)  # Set "user" as the index for plotting
    data = data[available_columns]  # Filter to ensure only the expected columns are visualized

    # Create the bar plot
    ax = data.plot(kind="bar", color=column_colors, figsize=(10, 6))

    # Add labels, title, and legend
    plt.title("Summed Unique Proper Names Across Corpora", fontsize=14)
    plt.xlabel("Role", fontsize=12)
    plt.ylabel("Sum of Unique Proper Names", fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Corpus", fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_image_path, dpi=300)
    plt.close()  # Close the plot to prevent display in non-interactive environments

    print(f"Plot successfully saved to {output_image_path}")

#Hegde word analysis

def count_hedge_words(json_path, hedge_list, output_csv):
    """
    Computes the percentage of hedge words per user in each conversation and saves the results to a CSV file.

    Args:
        json_path (str): Path to the JSON file containing conversation data.
        hedge_list (set): A set of hedge words to check occurrences.
        output_csv (str): Path to save the output CSV file.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as file:
        conversation_data = json.load(file)

    # Dictionary to store results
    hedge_counts = {}

    for conversation_id, utterances in conversation_data.items():
        user_word_counts = {}
        user_hedge_counts = {}

        for utterance in utterances:
            role = utterance["role"]
            words = utterance["content"].lower().split()  # Simple tokenization

            # Initialize user count storage
            if role not in user_word_counts:
                user_word_counts[role] = 0
                user_hedge_counts[role] = 0

            # Count words and hedge words
            user_word_counts[role] += len(words)
            user_hedge_counts[role] += sum(1 for word in words if word in hedge_list)

        # Compute hedge word percentages
        hedge_counts[conversation_id] = {
            role: (user_hedge_counts[role] / user_word_counts[role] * 100) if user_word_counts[role] > 0 else 0
            for role in user_word_counts
        }

    # Get unique user roles for CSV header
    all_roles = set(role for conv in hedge_counts.values() for role in conv.keys())

    # Save results as CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["conversation_id"] + list(all_roles))

        for conversation_id, counts in hedge_counts.items():
            row = [conversation_id] + [counts.get(role, 0) for role in all_roles]
            writer.writerow(row)

def load_hedge_words(file_paths):
    """
    Loads hedge words from multiple text files and merges them into a single set.

    Args:
        file_paths (list of str): List of file paths to hedge word lists.

    Returns:
        set: A set of unique hedge words.
    """
    hedge_words = set()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                hedge_words.add(line.strip().lower())  # Normalize case and strip spaces
    return hedge_words

#Calculate the differences
import pandas as pd

def calculate_hedge_differences(input_csv, output_csv):
    """
    Reads a CSV file containing hedge word percentages per user per conversation,
    calculates absolute differences between each pair of users, and saves the result.

    Args:
        input_csv (str): Path to the input CSV file with hedge word percentages.
        output_csv (str): Path to save the output CSV file with hedge differences.
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Ensure role columns exist (excluding 'conversation_id')
    role_columns = [col for col in df.columns if col != "conversation_id"]

    # If fewer than 2 roles exist, skip computation
    if len(role_columns) < 2:
        print(f"Skipping {input_csv}: Not enough roles to compare.")
        return

    # Compute absolute differences between all user pairs
    diff_data = {"conversation_id": df["conversation_id"]}
    
    for i in range(len(role_columns)):
        for j in range(i + 1, len(role_columns)):  # Avoid duplicate pairs
            role1, role2 = role_columns[i], role_columns[j]
            column_name = f"{role1}_vs_{role2}_diff"
            diff_data[column_name] = abs(df[role1] - df[role2])  # Absolute difference

    # Convert to DataFrame and save
    diff_df = pd.DataFrame(diff_data)
    diff_df.to_csv(output_csv, index=False)

    print(f"Processed {input_csv}: Hedge differences saved to {output_csv}")

#Calculate and merge averages

def calculate_average_hedge_differences(output_file, original_path, random_path, model_path):
    """
    Calculates the average hedge word difference for each dataset and stores the results in a CSV file.

    Args:
        output_file (str): Path to save the average hedge differences CSV.
        original_path (str): Path to the hedge word difference CSV for original data.
        random_path (str): Path to the hedge word difference CSV for random data.
        model_path (str): Path to the hedge word difference CSV for model data.
    """
    # Load the CSV files
    original_df = pd.read_csv(original_path)
    random_df = pd.read_csv(random_path)
    model_df = pd.read_csv(model_path)

    # Compute the average difference for each dataset
    avg_original = original_df.iloc[:, 1].mean()  # Assuming column index 1 contains the differences
    avg_random = random_df.iloc[:, 1].mean()
    avg_model = model_df.iloc[:, 1].mean()

    # Create a DataFrame with the results
    avg_diff_df = pd.DataFrame({
        "dataset": ["original", "random", "model"],
        "avg_hedge_diff": [avg_original, avg_random, avg_model]
    })

    # Save the results
    avg_diff_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Average hedge differences saved to {output_file}")

#Visualize the results
def visualize_hedge_differences(differences_filepath, plot_filepath):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(differences_filepath)
    
    # Print the column names to ensure they match
    print(df.columns)

    # Retrieve the average hedge differences for each dataset type
    original_avg_diff = df.loc[df['dataset'] == 'original', 'avg_hedge_diff'].values[0]
    random_avg_diff = df.loc[df['dataset'] == 'random', 'avg_hedge_diff'].values[0]
    model_avg_diff = df.loc[df['dataset'] == 'model', 'avg_hedge_diff'].values[0]
    
    # Define the color palette
    colors = ['palegreen', 'peachpuff', 'cornflowerblue']

    # Visualize the results
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Original', 'Random', 'Model'], [original_avg_diff, random_avg_diff, model_avg_diff], color=colors)
    
    # Add percentage labels to each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Average Difference in Hedge Words (%)')
    ax.set_title('Average Difference in Hedge Words Between Users')
    plt.tight_layout()

    # Save the plot to the specified file path
    plt.savefig(plot_filepath)
    plt.close()

#We start calculating Accommodation for POS

def load_corpus(file_path):
    """
    Load the corpus data from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_accommodation_for_pos_category(user_a_utterances, user_b_utterances):
    """
    This function calculates the accommodation value between user a and user b for each POS category,
    aggregated over the entire corpus.
    """
    # Initialize dictionaries to store counts for each POS category
    total_a_counts = {}
    total_b_counts = {}
    total_shared_counts = {}
    
    total_a_utterances = len(user_a_utterances)
    total_b_utterances = len(user_b_utterances)

    # Iterate through all the utterances for both users
    for ua, ub in zip(user_a_utterances, user_b_utterances):
        # Count POS tags for user A
        for word_data in ua:
            pos = word_data['pos']
            total_a_counts[pos] = total_a_counts.get(pos, 0) + 1
            if pos not in total_shared_counts:
                total_shared_counts[pos] = 0

        # Count POS tags for user B
        for word_data in ub:
            pos = word_data['pos']
            total_b_counts[pos] = total_b_counts.get(pos, 0) + 1
            if pos not in total_shared_counts:
                total_shared_counts[pos] = 0

        # Count the shared POS tags between user A and user B in the same utterance
        for ua_word_data in ua:
            for ub_word_data in ub:
                if ua_word_data['pos'] == ub_word_data['pos']:
                    total_shared_counts[ua_word_data['pos']] += 1

    # Initialize the accommodation result dictionary
    accommodation_results = {}

    # Calculate probabilities and accommodation for each POS category
    for pos in total_a_counts:
        a_pos_count = total_a_counts[pos]
        b_pos_count = total_b_counts.get(pos, 0)
        shared_pos_count = total_shared_counts.get(pos, 0)

        # Calculate probabilities
        p_b_given_a = shared_pos_count / a_pos_count if a_pos_count > 0 else 0
        p_b = b_pos_count / total_b_utterances if total_b_utterances > 0 else 0

        # Calculate accommodation (difference between conditional and independent probabilities)
        accommodation_value = p_b_given_a - p_b

        # Ensure the result stays within the bounds [0, 1]
        accommodation_results[pos] = max(0, min(accommodation_value, 1))

    return accommodation_results

def process_acc_corpus(corpus_file, user_a_role, user_b_role, output_filepath):
    """
    Processes the corpus file to calculate accommodation values for each POS category
    across the entire corpus and saves the results in a CSV file.
    """
    corpus = load_corpus(corpus_file)
    
    # Initialize dictionaries to accumulate utterances for user_a and user_b
    user_a_utterances = []
    user_b_utterances = []

    # Iterate through all conversations in the corpus
    for conversation_id, conversation_data in corpus.items():
        for utterance in conversation_data:
            if utterance['role'] == user_a_role:
                user_a_utterances.append(utterance['tagged_utterance'])
            elif utterance['role'] == user_b_role:
                user_b_utterances.append(utterance['tagged_utterance'])

    # Calculate accommodation for the entire corpus, aggregated by POS category
    accommodation_values = calculate_accommodation_for_pos_category(user_a_utterances, user_b_utterances)

    # Convert the accommodation results to a DataFrame
    df = pd.DataFrame(list(accommodation_values.items()), columns=['POS Category', 'Accommodation Value'])

    # Save the results to the output CSV file
    df.to_csv(output_filepath, index=False)
    
    print(f"Accommodation results saved to {output_filepath}")

#Merge the results
def merge_POS_accommodation_files(original_file, random_file, model_file, output_file):
    """
    Merges the accommodation values from three corpora into a single CSV file.
    
    Parameters:
    - original_file: Path to the accommodation CSV file for the original corpus.
    - random_file: Path to the accommodation CSV file for the random corpus.
    - model_file: Path to the accommodation CSV file for the model corpus.
    - output_file: Path where the merged CSV file will be saved.
    """
    # Load the three CSV files into DataFrames
    original_df = pd.read_csv(original_file)
    random_df = pd.read_csv(random_file)
    model_df = pd.read_csv(model_file)

    # Merge the DataFrames on the 'POS Category' column
    merged_df = pd.merge(original_df, random_df, on='POS Category', suffixes=('_OG', '_RANDOM'))
    merged_df = pd.merge(merged_df, model_df, on='POS Category')
    
    # Rename the columns for clarity
    merged_df.columns = ['POS Category', 'Accommodation Value OG', 'Accommodation Value RANDOM', 'Accommodation Value MODEL']

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged accommodation data saved to {output_file}")

#Visualization
def visualize_POS_accommodation(merged_file, output_plot_file):
    """
    Visualizes the accommodation values from the merged CSV file.
    
    Parameters:
    - merged_file: Path to the merged CSV file containing accommodation values.
    - output_plot_file: Path to save the resulting plot.
    """
    # Load the merged CSV data
    df = pd.read_csv(merged_file)
    
    # Set the color palette based on the colors we always use
    color_palette = {
        'Accommodation Value OG': 'palegreen',
        'Accommodation Value RANDOM': 'peachpuff',
        'Accommodation Value MODEL': 'cornflowerblue'
    }

    # Plot the data as a bar plot
    plt.figure(figsize=(10, 6))
    df.set_index('POS Category')[['Accommodation Value OG', 'Accommodation Value RANDOM', 'Accommodation Value MODEL']].plot(
        kind='bar', 
        color=[color_palette[col] for col in df.columns[1:]], 
        width=0.8
    )
    
    # Set plot labels and title
    plt.title('POS Category Accommodation Values', fontsize=16)
    plt.xlabel('POS Category', fontsize=12)
    plt.ylabel('Accommodation Value', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 1)  # Adjust y-axis to scale the values correctly
    
    # Add a grid for easier reading
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot to the output file
    plt.tight_layout()
    plt.savefig(output_plot_file)
    plt.show()

    print(f"Accommodation visualization saved to {output_plot_file}")

#Calculate novelty tokens
def calculate_novelty_tokens(corpus_data, role_a, role_b):
    """
    Calculate the novelty percentage of tokens used by role_a in comparison to the previous utterance of role_b.
    The novelty percentage is the proportion of unique tokens used by role_a that are not in the previous utterance of role_b.

    Parameters:
        corpus_data (dict): The conversation data in the format {conversation_id: [{...}, {...}, ...]}
        role_a (str): The role of the current speaker (e.g., 'user_x' or 'model')
        role_b (str): The role of the previous speaker (e.g., 'user_y')

    Returns:
        dict: A dictionary of conversation_ids and their corresponding novelty percentages.
    """
    novelty_percentage_by_conversation = {}

    for conversation_id, conversation in corpus_data.items():
        # Get the utterances of role_a and role_b
        utterances_a = [utterance for utterance in conversation if utterance['role'] == role_a]
        utterances_b = [utterance for utterance in conversation if utterance['role'] == role_b]

        # Ensure we have at least one utterance from role_a and one from role_b
        if utterances_a and utterances_b:
            for i in range(1, len(conversation)):
                # Check if the current utterance is from role_a and previous utterance from role_b
                if conversation[i]['role'] == role_a and conversation[i-1]['role'] == role_b:
                    utterance_a = conversation[i]['content']
                    utterance_b = conversation[i-1]['content']
                    
                    # Tokenize using regex (extract words)
                    tokens_a = set(re.findall(r'\b\w+\b', utterance_a.lower()))
                    tokens_b = set(re.findall(r'\b\w+\b', utterance_b.lower()))

                    # Find the novel tokens used by role_a
                    novel_tokens = tokens_a - tokens_b

                    # Calculate the novelty percentage
                    novel_percentage = (len(novel_tokens) / len(tokens_a) * 100) if tokens_a else 0

                    # Store the novelty percentage for this conversation
                    novelty_percentage_by_conversation[conversation_id] = novel_percentage

                # Check if the current utterance is from role_b and previous utterance from role_a
                elif conversation[i]['role'] == role_b and conversation[i-1]['role'] == role_a:
                    utterance_b = conversation[i]['content']
                    utterance_a = conversation[i-1]['content']

                    # Tokenize using regex (extract words)
                    tokens_a = set(re.findall(r'\b\w+\b', utterance_a.lower()))
                    tokens_b = set(re.findall(r'\b\w+\b', utterance_b.lower()))

                    # Find the novel tokens used by role_a
                    novel_tokens = tokens_a - tokens_b

                    # Calculate the novelty percentage
                    novel_percentage = (len(novel_tokens) / len(tokens_a) * 100) if tokens_a else 0

                    # Store the novelty percentage for this conversation
                    novelty_percentage_by_conversation[conversation_id] = novel_percentage

        else:
            print(f"No valid utterances found for conversation {conversation_id}")

    return novelty_percentage_by_conversation


#Calculate the average
import csv

def save_novelty_results_to_csv(novelty_results_file, output_csv):
    """
    Save novelty results from a JSON file to a CSV file.
    
    Parameters:
        novelty_results_file (str): Path to the novelty results JSON file.
        output_csv (str): Path to the output CSV file where results will be saved.
    """
    with open(novelty_results_file, 'r') as file:
        novelty_results = json.load(file)

    # Extract the results
    novelty_original = novelty_results.get("novelty_original", {})
    novelty_random = novelty_results.get("novelty_random", {})
    novelty_model = novelty_results.get("novelty_model", {})

    # Create CSV rows
    rows = []
    conversation_ids = set(novelty_original.keys()) | set(novelty_random.keys()) | set(novelty_model.keys())

    for conversation_id in conversation_ids:
        rows.append({
            "conversation_id": conversation_id,
            "novelty_original": novelty_original.get(conversation_id, None),
            "novelty_random": novelty_random.get(conversation_id, None),
            "novelty_model": novelty_model.get(conversation_id, None)
        })

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['conversation_id', 'novelty_original', 'novelty_random', 'novelty_model']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Novelty results saved to CSV file: {output_csv}")

def calculate_average_novelty(novelty_results_file):
    """
    Calculate the average novelty for the three corpora (original, random, model) from the results file.
    
    Parameters:
        novelty_results_file (str): Path to the novelty results JSON file.
    
    Returns:
        dict: A dictionary containing average novelty percentages for original, random, and model.
    """
    with open(novelty_results_file, 'r') as file:
        novelty_results = json.load(file)

    # Extract novelty values for each corpus
    original_novelty = list(novelty_results.get("novelty_original", {}).values())
    random_novelty = list(novelty_results.get("novelty_random", {}).values())
    model_novelty = list(novelty_results.get("novelty_model", {}).values())

    # Calculate the average novelty for each corpus
    def calculate_average(novelty_list):
        return sum(novelty_list) / len(novelty_list) if novelty_list else 0

    original_avg = calculate_average(original_novelty)
    random_avg = calculate_average(random_novelty)
    model_avg = calculate_average(model_novelty)

    return {
        "original": original_avg,
        "random": random_avg,
        "model": model_avg
    }

def visualize_novelty_averages(averages, save_path="novelty_averages.png"):
    """
    Visualizes the average novelty percentages for original, random, and model corpora in a bar chart.
    
    Parameters:
        averages (dict): Dictionary containing average novelty percentages for original, random, and model.
        save_path (str): Path where the plot will be saved as a PNG file.
    """
    # Define colors as per the user's preference
    colors = {
        'original': 'palegreen',
        'random': 'peachpuff',
        'model': 'cornflowerblue'
    }

    # Data for plotting
    labels = list(averages.keys())
    values = list(averages.values())

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=[colors[label] for label in labels])
    
    # Title and labels
    plt.title('Average Novelty Comparison')
    plt.ylabel('Average Novelty (%)')
    plt.xlabel('Corpus')

    # Save the plot as a PNG file
    plt.savefig(save_path, format='png')

    # Display the plot
    plt.show()
