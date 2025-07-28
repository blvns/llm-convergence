import numpy as np
import os
import pandas as pd
import csv
import re
import string
import random
import json
from collections import Counter



class CorpusConverter:
    """
    A class that takes one of the corpora and converts it:
        - filtering out conversations with more than 2 speakers
        - filtering out conversations with less than five turns
        - dividing the corpus in train/dev/set if not already split
        - converting the corpus in a json file
    """
    def __init__(self, dir_corpus, train_split=None, dev_split=None, test_split=None, random_seed=42):
        """
        args:
            dir_corpus (str): directory of the corpus
            train, dev and test_split (float): percentage of the split (values from 0 to 1)
            random_seed: to ensure reproucibility 
        """
        self.dir_corpus = dir_corpus
        self.train_split = train_split
        self.dev_split = dev_split
        self.test_split = test_split
        self.random_seed = random_seed
    
    def convert_movie(self, conversations_path):
        """
        Function that takes the movie corpus original file and converts it
            - conversations with less than five turns are filtered out
            - conversations with less than two speakers are filtered out
            - each conversation is associated to a conversation_id
            - for each turn in the conversation role and content are specified
            - role: either user (first speaker of the conversation) or assistant
            - content: the text of that turn
            - if the same speaker occupies two consecutive turns, the turns are merged together
            - the conversations are then shuffled, split according to split percentages and saved in the split directory

        Args:
            conversations_path (str): file of the original conversations
        """
        print('Starting Movie Corpus conversion...')
        utterances = pd.read_json(path_or_buf=conversations_path, lines=True)

        utterances['id_numeric'] = utterances['id'].str.extract('(\d+)').astype(int)
        utterances['conv_id_numeric'] = utterances['conversation_id'].str.extract('(\d+)').astype(int)

        # Sort the DataFrame by conversation ID and utterance ID to maintain correct order
        utterances = utterances.sort_values(by=['conv_id_numeric', 'id_numeric']).reset_index(drop=True)

        invalid_start_conversations = []

        for conversation_id, group in utterances.groupby('conversation_id'):
            # Check the reply-to value of the first row in the group
            first_row_reply_to = group.iloc[0]['reply-to']
            if first_row_reply_to is not None:
                invalid_start_conversations.append(conversation_id)

        if invalid_start_conversations:
            print("Conversations that start with a non-None reply-to value:")
            for conv_id in invalid_start_conversations:
                print(f"Conversation ID: {conv_id}")
        else:
            print("All conversations begin with None in the reply-to column: conversations are sorted correctly")


        reformatted_data = []

        for i, (conversation_id, group) in enumerate(utterances[['conversation_id', 'speaker', 'text']].groupby('conversation_id')):

            # Skip conversations with empty text in any turn
            if self.has_empty_text(group, text=True):
                continue
            #assign a new conversation id, but keep also the original id
            reformatted_conversation ={
                'conversation_id': i,
                'original_id': conversation_id,
                'turns': []
            }

            # Assign speakers consistently and create the turns
            speaker_mapping = {}
            for idx, row in group.iterrows():
                # Map speakers to 'user' and 'assistant'
                if row['speaker'] not in speaker_mapping:
                    if len(speaker_mapping) == 0:
                        speaker_mapping[row['speaker']] = "user"
                    elif len(speaker_mapping) == 1:
                        speaker_mapping[row['speaker']] = "assistant"
                
                # Create turn entry
                speaker = speaker_mapping[row['speaker']]
                text = row['text']
                
                reformatted_conversation["turns"].append({
                    "role": speaker,
                    "content": text
                })

            # Append the reformatted conversation to the list
            reformatted_conversation["turns"] = self.merge_turns(reformatted_conversation["turns"])

            reformatted_data.append(reformatted_conversation)

        print(f"Total reformatted conversations: {len(reformatted_data)}")
        reformatted_data = self.filter_n_speaker_conversations(reformatted_data, n=2)
        reformatted_data = self.filter_conversations_by_utterance_count(reformatted_data, min_utterances=6)

        self.shuffle_split_save(reformatted_data)

    def convert_dailydialog(self, train_path, dev_path, test_path):
        """
        For explanations see function convert_movie
        """
        print("Starting DailyDialog corpus conversion...")

        for split, split_path in zip(["train", "dev", "test"], [train_path, dev_path, test_path]):
        
            with open(split_path, "r", encoding="utf-8") as file:
                data = file.read()
    
            dialogs = [dialog.strip() for dialog in data.split('\n') if dialog.strip()]
            print(f"Total dialogs loaded: {len(dialogs)}")
            
            reformatted_data = []
    
            for idx, dialog in enumerate(dialogs):
                # Remove trailing whitespace and split based on "_eou_"
                turns = dialog.strip().split('__eou__')[:-1]
                conversation = {"conversation_id": idx + 1, "turns": []}
                
                for i, turn in enumerate(turns):
                    role = "user" if i % 2 == 0 else "assistant"
    
                    content = turn.strip()
                    
                    if content:  # Ensure content is not empty
                        conversation["turns"].append({"role": role, "content": content})
                
                reformatted_data.append(conversation)
        
            print(f"Total reformatted conversations in {split}: {len(reformatted_data)}")
            reformatted_data = self.filter_n_speaker_conversations(reformatted_data, n=2)
            reformatted_data = self.filter_conversations_by_utterance_count(reformatted_data, min_utterances=6)
            reformatted_data = self.transform_to_dict_by_id(reformatted_data)

            split_directory = os.path.join(self.dir_corpus, split)
            os.makedirs(split_directory, exist_ok=True)

            converted_corpus = os.path.join(split_directory, f'{split}.json')
            with open(converted_corpus, 'w', encoding='utf-8') as file:
                json.dump(reformatted_data, file, indent=4, ensure_ascii=False)

        print(f"DailyDialog reformatted splits saved successfully to {self.dir_corpus}!")

    #convert_switchboard_casino_friends_npr
    def convert_npr(self, conversations_path):
        """
        For explanations see function convert_movie
        """

        print('Starting NPR Corpus conversion...')
        utterances = pd.read_json(path_or_buf=conversations_path, lines=True)

        reformatted_data = []

        for i, (conversation_id, group) in enumerate(utterances[['conversation_id', 'speaker', 'text']].groupby('conversation_id')):
            #assign a new conversation id, but keep also the original id
            reformatted_conversation ={
                'conversation_id': i,
                'original_id': conversation_id,
                'turns': []
            }

            # Assign speakers consistently and create the turns
            speaker_mapping = {}
            for idx, row in group.iterrows():
                if row['speaker'] == "TRANSCRIPT_NOTE":
                    continue
                # Map speakers to 'user_x' and 'user_y'
                if row['speaker'] not in speaker_mapping:
                    if len(speaker_mapping) == 0:
                        speaker_mapping[row['speaker']] = "user"
                    elif len(speaker_mapping) == 1:
                        speaker_mapping[row['speaker']] = "assistant"
                    else:
                        speaker_mapping[row['speaker']] = "other_user"
                
                # Create turn entry
                speaker = speaker_mapping[row['speaker']]
                text = row['text']
                
                reformatted_conversation["turns"].append({
                    "role": speaker,
                    "content": text
                })

            # Append the reformatted conversation to the list
            reformatted_conversation["turns"] = self.merge_turns(reformatted_conversation["turns"])
            reformatted_data.append(reformatted_conversation)

        print(f"Total reformatted conversations: {len(reformatted_data)}")
        reformatted_data = [conv for conv in reformatted_data if not self.has_empty_text(conv, content=True)]
        reformatted_data = self.filter_n_speaker_conversations(reformatted_data, n=2)
        reformatted_data = self.filter_conversations_by_utterance_count(reformatted_data, min_utterances=6)

        self.shuffle_split_save(reformatted_data)

    def has_empty_text(self, group, text=None, content=None):
        """
        Check if any turn in the conversation has empty text.

        Args:
            group (dict or DataFrame): DataFrame or dictionary containing conversation turns.
            text (bool): if True, it allows to acces group['text']
            content (bool): if True, it allows to acces grou['content']

        Returns:
            bool: True if any turn has empty text, False otherwise.
        """
        if isinstance(group, dict) and content:
            # Check for empty 'content' in a list of turns
            return any(turn['content'].strip() == "" for turn in group.get("turns", []))
        elif isinstance(group, pd.DataFrame) and content:
            # Original implementation for DataFrame
            return any(group['content'].str.strip() == "")
        elif text:
            return any(group['text'].str.strip() == "")
        return False
             
    def merge_turns(self, turns):
        """
        Merge consecutive turns by the same speaker into a single turn.

        Args:
            turns (list): List of turns where each turn is a dictionary with 'role' and 'content'.

        Returns:
            list: List of merged turns.
        """
        merged_turns = []
        previous_speaker = None
        merged_content = ""

        for turn in turns:
            if turn["role"] == previous_speaker:
                merged_content += " " + turn["content"]
            else:
                if previous_speaker is not None:
                    merged_turns.append({
                        "role": previous_speaker,
                        "content": merged_content.strip()
                    })

                previous_speaker = turn["role"]
                merged_content = turn["content"]

        # Append the last turn
        if previous_speaker is not None:
            merged_turns.append({
                "role": previous_speaker,
                "content": merged_content.strip()
            })

        return merged_turns

    def replace_masked_data(self, text):
        """
        Replace masked data placeholders with realistic-looking imaginary values.
        The function is applied only to the maia corpus because sensitive data were masked.
        To simulate real conversation we unmask masked sensitive data with imaginary ones

        Args:
            text (str): Text containing masked placeholders.

        Returns:
            str: Text with placeholders replaced by generated values.
        """

        # Define replacement functions for each placeholder
        replacements = {
            "#NAME#": lambda: random.choice(["John", "Emma", "Michael", "Sophia"]),
            "#PRS_ORG#": lambda: random.choice(["TechCorp", "QuickShop", "HealthPro"]),
            "#ADDRESS#": lambda: f"{random.randint(100, 999)} {random.choice(['Main St', 'Elm St', 'Oak Rd'])}",
            "#EMAIL#": lambda: f"{random.choice(['john.levi', 'emma.williams', 'michael.brown'])}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}",
            "#IP#": lambda: ".".join(str(random.randint(0, 255)) for _ in range(4)),
            "#PASSWORD#": lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=10)),
            "#PHONENUMBER#": lambda: f"+{random.randint(1, 99)}-{random.randint(100, 999)}-{random.randint(1000000, 9999999)}",
            "#CREDITCARD#": lambda: ''.join(str(random.randint(0, 9)) for _ in range(16)),
            "#URL#": lambda: f"https://www.{random.choice(['example', 'sample', 'demo'])}.com",
            "#IBAN#": lambda: f"IBAN{''.join(random.choices(string.ascii_uppercase + string.digits, k=16))}",
            "#NUMBER#": lambda: str(random.randint(1, 10000)),
            "#ALPHANUMERIC_ID#": lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=12)),
        }

        # Pattern to match placeholders
        pattern = re.compile("|".join(re.escape(key) for key in replacements.keys()))

        # Replace placeholders with generated values
        def replacer(match):
            placeholder = match.group(0)
            return replacements[placeholder]()

        return pattern.sub(replacer, text)

    def filter_conversations_by_utterance_count(self, conversations, min_utterances):

        """
        Filter the dataframe to keep only conversations with at least a specified number of utterances.
        
        args:
            conversations (list): A list of conversations, where each conversation is a dictionary
        containing "turns" with "role" and "content".
            min_utterances (int): conversations with less turns than min_utterances are filtered out

        returns:
            filtered_conversations (list): list of conversations with at least min_utterances turns
        
        """

        filtered_conversations = [conversation for conversation in conversations if len(conversation['turns']) >= min_utterances]
        

        print(f"Filtered {len(filtered_conversations)} with more than {min_utterances} turns out of {len(conversations)} conversations")

        return filtered_conversations
    
    def filter_n_speaker_conversations(self, conversations, n):
        """
        Filters out conversations with more than 2 unique speakers.

        args:
            conversations (list): A list of conversations, where each conversation is a dictionary
        containing "turns" with "role" and "content".

        Returns:
        - filtered_data (list): A list of conversations with exactly 2 unique speakers.
        """
        filtered_data = []

        for conversation in conversations:
            # Extract the unique speakers (roles) in the conversation
            unique_speakers = set(turn['role'] for turn in conversation['turns'])

            # Keep only conversations with exactly 2 unique speakers
            if len(unique_speakers) == n:
                filtered_data.append(conversation)

        print(f"Removed {len(conversations) - len(filtered_data)} conversations with more than 2 speakers.")
        print(f"Remaining conversations: {len(filtered_data)} with exactly 2 speakers.")
    
        return filtered_data

    def transform_to_dict_by_id(self, data):
        """
        Transforms a list of conversations into a dictionary keyed by conversation_id.
        Each entry contains the turns of the conversation.

        args: 
            data (list): a list of conversation, where each conversation is a dictionary containing the key 'conversation_id'
        returns:
            transformed_data (dict): a dictionary {'conversation_id(int)': 'turns[list of dictionaries]'}

        """
        transformed_data = {}
        for conversation in data:
            conversation_id = conversation.get("conversation_id")
            if conversation_id is not None:
                transformed_data[str(conversation_id)] = conversation.get("turns", [])
        return transformed_data
    
    def shuffle_split_save(self, reformatted_data):
        """
        Function to shuffle, split in train, dev and test set and save the preprocessed conversations

        args:
            data (list): list of conversations
        """
        random.seed(self.random_seed)
        random.shuffle(reformatted_data)

        train_size = int(len(reformatted_data) * self.train_split)
        dev_size = int(len(reformatted_data) * self.dev_split)

        train_data = reformatted_data[:train_size]
        dev_data = reformatted_data[train_size:train_size + dev_size]
        test_data = reformatted_data[train_size + dev_size:]

        # Sort each split by conversation_id
        train_data = sorted(train_data, key=lambda x: int(x['conversation_id']))
        dev_data = sorted(dev_data, key=lambda x: int(x['conversation_id']))
        test_data = sorted(test_data, key=lambda x: int(x['conversation_id']))

        train = os.path.join(self.dir_corpus, 'train')
        dev = os.path.join(self.dir_corpus, 'dev')
        test = os.path.join(self.dir_corpus, 'test')

        for path in [train, dev, test]:
            os.makedirs(path, exist_ok=True)

        train_file = os.path.join(train, "train.json")
        dev_file = os.path.join(dev, "dev.json")
        test_file = os.path.join(test, "test.json")

        for dataset, file_path in zip([train_data, dev_data, test_data], [train_file, dev_file, test_file]):
            # Transform dataset into dictionary format
            transformed_dataset = self.transform_to_dict_by_id(dataset)

            # Save transformed dataset to file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(transformed_dataset, file, indent=4, ensure_ascii=False)

        print(f"Train set saved to {train_file} ({len(train_data)} entries)")
        print(f"Dev set saved to {dev_file} ({len(dev_data)} entries)")
        print(f"Test set saved to {test_file} ({len(test_data)} entries)")

    def filter_csv_first_n_columns(self, file_path, expected_columns):
        """
        Filters rows in a CSV file to include only the first `expected_columns` columns. 
        The functions is only applied for the empathetic corpus: some lines in the csv file have an unaligned number of columns

        args:
            file_path (str): Path to the original CSV file.
            expected_columns (int): Number of columns to include in the output.
        return
            filtered_file_path (str): Path to the filtered CSV file.
        """
        filtered_file_path = f"{file_path}_filtered.csv"
        with open(file_path, "r", encoding="utf-8") as infile, open(filtered_file_path, "w", newline='', encoding="utf-8") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                # Only write rows with at least `expected_columns` columns
                if len(row) >= expected_columns:
                    writer.writerow(row[:expected_columns])

        print(f"Filtered CSV saved to {filtered_file_path}")
        return filtered_file_path


class PrepareModelInputs:
    def __init__(self, file_path):

        '''
        This class takes conversations in json format in which user and assistant alternate each other masking some turns in the conversations and assigning them to the role 'model'.
            1. the fourth turn is masked and assigned to 'model'. Every two turns the process is repeated.
            2. all the even turns are masked and assigned to 'model'. The odd turns are assigned to 'user'
        args:
            file_path (str): The path to the JSON file containing the data to preprocess.
        '''
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if not isinstance(data, dict):
            raise ValueError("The input data must be a dictionary with 'conversation_id' as keys and 'turns' as values.")

        # Convert dictionary to DataFrame for processing
        records = []
        for conversation_id, turns in data.items():
            for turn in turns:
                records.append({
                    "conversation_id": int(conversation_id),
                    "speaker": turn["role"],
                    "text": turn["content"]
                })

        self.df = pd.DataFrame(records)
        
        print('\n\nPhase 2\n\n'
              'Preparation of the inputs for the model\n'
              'Parts of the conversations need to be masked\n'
              'Loaded DataFrame structure:')
        print(self.df.head())
        
    def replace_one_speaker_utterances(self, mask_turn=6):
        """
        Replaces utterances of one speaker in each conversation based on the argument mask_turn with the string '[MISSING]'.The role of these turns is assigned to 'model'

        args:
            mask_after_turn (int): The base turn number after which to start masking. Every (mask_turn + 2n) turn are replaced until the end of the conversation.

        Returns:
            result_df (DataFrame): A modified DataFrame with specified utterances masked.
        """
        modified_conversations = []

        # Process each conversation
        for conversation_id, group in self.df.groupby('conversation_id'):
            group = group.reset_index(drop=True)

            # If the conversation length exceeds the masking threshold
            if len(group) >= mask_turn:
                    idx_mask = mask_turn - 1

                    if idx_mask < len(group):
                        speaker_to_mask = group.loc[idx_mask, 'speaker']
                        modified_texts = []
                        modified_speakers = []

                        # Apply masking based on turn number and speaker
                        for i, row in group.iterrows():
                            if i < idx_mask:
                                # Keep turns before the specified threshold unchanged
                                modified_texts.append(row['text'])
                                modified_speakers.append(row['speaker'])
                            elif row['speaker'] == speaker_to_mask:
                                # Mask the chosen speaker's utterances from the threshold turn onward
                                modified_texts.append("[MISSING]")
                                modified_speakers.append('model')
                            else:
                                # Keep other speaker's text
                                modified_texts.append(row['text'])
                                modified_speakers.append(row['speaker'])

                        # Assign modified texts back to the group's 'text' column
                        group['text'] = modified_texts
                        group['speaker'] = modified_speakers

            # Append modified group
            modified_conversations.append(group)

        # Concatenate all modified groups
        result_df = pd.concat(modified_conversations, ignore_index=True)
        print("Modified DataFrame after replacing one speaker's utterances (first few rows):")
        print(result_df.head())
        return result_df

    def format_for_model_dict(self, group):
        """
        Assigns role 'model' to all turns that have '[MISSING]' text.
        
        args:
            group (DataFrame): DataFrame representing a single conversation.
        
        Returns:
            messages [list]: A list of conversations formatted for the model.
        """
        messages = []

        # Find the speaker who has the first missing utterance
        for _, row in group.iterrows():
            if row['text'] == '[MISSING]':
                messages.append({'role': 'model', 'content': ''})
            else:
                #Keep the original speaker role
                messages.append({'role': row['speaker'], 'content': row['text']})

        return messages
        
    def prepare_model_inputs(self, df, output_file):
        """
        Prepares and saves model inputs from the modified DataFrame with masked utterances.
        
        args:
            df (DataFrame): DataFrame with masked utterances.
        
        Returns:
            dict: A dictionary where each key is a conversation ID and each value is the formatted messages for that conversation.
        """
        model_inputs = {}

        for conversation_id, group in df.groupby('conversation_id'):
            model_inputs[conversation_id] = self.format_for_model_dict(group)


        print("Prepared model inputs (first few conversations):")
        print(dict(list(model_inputs.items())[:2]))
        
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(model_inputs, file, indent=4, ensure_ascii=False)


class DialogStatistics:

    """
    Class for measuring statiscal features of a corpus
    """

    def __init__(self, input_file):

        self.input_file = input_file
  
        self.statistics = None
    
    def calculate_statistics(self):

        with open(self.input_file, "r", encoding="utf-8") as file:
            dialogs = json.load(file)

        num_conversations = len(dialogs)

        # Initialize variables for statistics
        total_turns = 0
        turn_lengths = []
        conversation_lengths = []
        conversation_word_counts = []
        word_counts = Counter()

        # Process each conversation
        for conversation_id, turns in dialogs.items():
            num_turns = len(turns)
            total_turns += num_turns
            conversation_lengths.append(num_turns)

            # Calculate turn and conversation word counts
            conversation_words = []
            for turn in turns:
                words = turn["content"].split()
                turn_lengths.append(len(words))
                conversation_words.extend(words)

            # Update statistics
            conversation_word_counts.append(len(conversation_words))
            word_counts.update(conversation_words)

        # Calculate mean statistics
        mean_turns_per_conversation = np.mean(conversation_lengths)
        mean_turn_length = np.mean(turn_lengths)
        mean_conversation_length = np.mean(conversation_word_counts)

        # Calculate lexical variability (unique words / total words)
        total_words = sum(word_counts.values())
        unique_words = len(word_counts)
        lexical_variability = unique_words / total_words if total_words > 0 else 0

        # Output statistics
        self.statistics = {
            "Num_conversations": num_conversations,
            "Num_turns": total_turns,
            "Mean_turns_x_conv": mean_turns_per_conversation,
            "Mean_len_turn": mean_turn_length,
            "Mean_len_conv": mean_conversation_length,
            "Tot_words": total_words,
            "Unique_words/total": lexical_variability,
            "Unique_words": unique_words
        }
