import json
import re
import os
from tqdm import tqdm
import torch
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json
import pandas as pd
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download

## evaluation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import evaluate
from bert_score import score
import rouge_score
import os

class Model:
    def __init__(self, model_folder, hf_token):
        self.model_folder = model_folder
        self.hf_token = hf_token  # active HF token
        print(f"Initializing model {model_folder.split('/')[-1]}")
        self.device = self.define_device()
        self.model, self.tokenizer = self.initialize_model(self.model_folder, self.device, self.hf_token)
        self.pipeline_text_generation = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def define_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def clear_memory(self):
        del self.tokenizer
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleared.")

    def initialize_model(self, model_folder, device, hf_token):
        # data type for computation
        #compute_dtype = torch.bfloat16

        # The tokenizer with specified device and max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(
            model_folder,
            use_auth_token=hf_token,  
            device_map="auto",
            pad_to_multiple_of=8,
            padding_side="left",
        )

        #use accelerate to do inference with multiple GPUs
        #if model_folder in ["google/gemma-3-27b-pt", "google/gemma-3-27b-it"]:
        #    model = AutoModelForCausalLM.from_pretrained(
        #        model_folder,
        #        use_auth_token=hf_token,  
        #        device_map="auto",
        #        pad_token_id=tokenizer.eos_token_id
        #    )
        if model_folder in ["meta-llama/Llama-3.1-70B","meta-llama/Llama-3.1-70B-Instruct"]:
            # configuration for quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_folder,
                use_auth_token=hf_token,  
                device_map="auto",
                quantization_config=bnb_config,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_folder,
                use_auth_token=hf_token,  
                device_map="auto",
                pad_token_id=tokenizer.eos_token_id
            )

        # initialized model and tokenizer
        return model, tokenizer


class DataLoader:
    def __init__(self, paths, max_samples=-1, corpus_name=None, set_name=None, base_output_path="LLMs/corpora"):
        """
        Args:
            paths (list): List of paths to JSON files to be loaded.
            max_samples (int): Max. Number of examples to load in.
            corpus_name (str): Name of the corpus for saving subsets.
            set_name (str): Name of the set ('train', 'dev', 'test') for saving subsets.
            base_output_path (str): Base output directory for saving subsets.
        """
        self.paths = paths
        self.max_samples = max_samples
        self.corpus_name = corpus_name
        self.set_name = set_name
        self.base_output_path = base_output_path
        self.data = self._load_data()

    def _load_data(self):
        """
        Load data from the provided paths.
        """
        data = []
        for path in self.paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    data.append(json.loads(file_content))
            else:
                print(f"Warning: The file {path} does not exist.")
        return data

    def get_data(self):
        """
        Get the loaded data, downsample to max_samples if needed
        """
        split_results = []
        for d in self.data:
            keys = sorted(list(d.keys()))
            subset_keys = keys[:self.max_samples]
            subset_d = {k: d[k] for k in subset_keys}
            split_results.append(subset_d)
        return split_results

class ConversationProcessor():

    def generate_prompt(
        self, conversation, intro_prompt, tokenizer
    ):

        if intro_prompt!= None and conversation[0]['role']!='system':
            system_message = { "role": "system", "content": intro_prompt}
            conversation.insert(0, system_message)

        context = ''

        for turn in conversation:
            context += f"<{turn['role']}> {turn['content']} </{turn['role']}>\n"
        context+="<assistant>"

        return context


    def clean_model_output(self, generated_text):
        """
        Cleans the generated text, removing unnecessary spaces, emojis, and ensuring the output ends at the last sentence.
        """
        # newline characters, tabs, and extra spaces
        generated_text = " ".join(generated_text.splitlines()).replace('"', '').strip()
        generated_text = re.sub(r"\s{2,}", " ", generated_text)  # replace multiple spaces with a single space

        # emojis
        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
        generated_text = emoji_pattern.sub("", generated_text)

        # text before the first tag (e.g., <assistant>)
        text_up_to_first_tag = re.split(r"<[^>]+>", generated_text, maxsplit=1)[0].strip()

        # remove unwanted patterns (like text between ** or parentheses)
        cleaned_text = re.sub(r"\*.*?\*|\(.*?\)", "", text_up_to_first_tag).strip()

        # only full sentences (up to the last punctuation mark: .!?)
        match = re.search(r'(.*[.!?])', cleaned_text)
        return match.group(1) if match else cleaned_text



class ModelGenerate():

    def __init__(self, model_pipeline, tokenizer):
        self.pipeline_model = model_pipeline
        self.processor = ConversationProcessor()
        self.tokenizer = tokenizer

    def process_single_conversation(
        self, conversation, intro_prompt, max_new_tokens=40, temp=0.6, top_k=20, top_p=0.4
    ):
        """
        process a single conversation to generate missing utterances.
        """
        def modify_roles(conversation):

            for i, turn in enumerate(conversation):
                if turn['role'] in ('user_y','user'):
                    turn["role"] = "user"
                elif turn['role'] =='system':
                    turn['role'] = 'system'
                elif turn['role'] in ('model','user_x'):
                    turn["role"] ='assistant'
            return conversation

        conversation = modify_roles(conversation)

        generated_conversation = []  # store  the conversation turn-by-turn
        context_for_generation = []  # store contexts

        for i,turn in enumerate(conversation):

            if turn["content"] == "":
               # prepare conversation history
                input_text = self.processor.generate_prompt(
                    generated_conversation, intro_prompt, self.tokenizer
                )


                # save context for this turn
                context_for_generation.append(input_text)

                # model response

                response = self.pipeline_model(
                    input_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    return_full_text=False
                )[0]["generated_text"]

                # clean model's response
                response_text = self.processor.clean_model_output(response)

                # append the generated turn

                generated_conversation.append({"role": turn['role'], "content": response_text})



                # add the existing turn to the conversation
            else:
              generated_conversation.append(turn)

        if intro_prompt:
            generated_conversation = generated_conversation[1:]


        return generated_conversation, context_for_generation

    def conversation_history_generate_missing_utterances(
        self, inputs, intro_prompt, max_new_tokens=40, temp=0.6, top_k=20, top_p=0.4
    ):
        """
        Generate missing uterances for all conversations, providing the intro prompt only for the first conversation.
        """
        generated_results = {}
        contexts = {}


        for conv_id, conversation in tqdm(inputs.items()):
            # process the conversation
            generated_conversation, context_for_generation = self.process_single_conversation(
                conversation,
                intro_prompt,
                max_new_tokens=max_new_tokens,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
            )

            #save conversation and contexts
            generated_results[conv_id] = generated_conversation
            contexts[conv_id] = context_for_generation

        return generated_results, contexts


class OutputHandler:

    def prepare_outputs(self, outputs, file_path):
        """
        Save outputs to a JSON file. If the file exists, append the new data.
        """
        if os.path.exists(file_path + ".json"):
            # Load existing JSON data
            with open(file_path + ".json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # Append new outputs
            if isinstance(existing_data, list) and isinstance(outputs, list):
                outputs = existing_data + outputs
            elif isinstance(existing_data, dict) and isinstance(outputs, dict):
                existing_data.update(outputs)
                outputs = existing_data
            else:
                raise ValueError("Incompatible data formats between existing and new outputs.")

        # Save updated or new outputs
        with open(file_path + ".json", "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
        print(f"JSON saved to {file_path}.json")

    def save_context_input(self, context_input, file_path):
        """
        Save context inputs to a CSV file. If the file exists, append the new data.
        """
        # Flatten the JSON structure into lists
        conv_ids = []
        contexts = []

        for conv_id, context_list in context_input.items():
            for context in context_list:
                conv_ids.append(conv_id)
                contexts.append(context)

        # Create a DataFrame
        new_data = pd.DataFrame({
            "conv_id": conv_ids,
            "context": contexts
        })

        if os.path.exists(file_path + ".csv"):
            # Load the existing file and append new data
            existing_data = pd.read_csv(file_path + ".csv", sep=";", encoding="utf-8")
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data

        # Save DataFrame to CSV
        updated_data.to_csv(file_path + ".csv", index=False, encoding='utf-8', sep=';')
        print(f"CSV saved to {file_path}.csv")

    def map_roles(self, original_file, outputs_file, output_mapped_file):
        """
        Maps roles in the outputs JSON file based on the original JSON file and appends to an existing mapped JSON file if it exists.
        """
        # the original JSON data
        with open(original_file, 'r', encoding='utf-8') as orig_f:
            original_data = json.load(orig_f)

        # outputs  data
        outputs_data = outputs_file

        # if the mapped file already exists
        if os.path.exists(output_mapped_file + ".json"):
            with open(output_mapped_file + ".json", 'r', encoding='utf-8') as mapped_f:
                existing_mapped_data = json.load(mapped_f)
        else:
            existing_mapped_data = {}

        # iterate through conversations to map roles
        for conv_id, output_turns in outputs_data.items():
            if conv_id in original_data:
                original_turns = original_data[conv_id]

                # map roles
                for i, turn in enumerate(output_turns):
                    if i < len(original_turns):  # valid index check
                        turn["role"] = original_turns[i]["role"]

            # append the mapped conversation
            existing_mapped_data[conv_id] = output_turns

        # save the updated mapped data
        with open(output_mapped_file + ".json", 'w', encoding='utf-8') as mapped_f:
            json.dump(existing_mapped_data, mapped_f, indent=4, ensure_ascii=False)
        print(f"Mapped roles saved to {output_mapped_file}.json")



## EVALUATION FUNCTIONS

class PrepareTurns:

    def filter_model_turns(self, outputs, originals, filename):

        ## read json files
        with open(outputs+".json", "r", encoding="utf-8") as f:
            outputs = json.load(f)

        with open(originals+".json", "r", encoding="utf-8") as f:
            originals = json.load(f)


        # initialize lists to store content and their conversation IDs
        conversation_ids = []
        model_differences = []
        original_differences = []

        # compare content in each conversation
        for conv_id in outputs:
            if conv_id in originals:
                output_conversation = outputs[conv_id]
                original_conversation = originals[conv_id]
                for output_turn, original_turn in zip(output_conversation, original_conversation):
                    #  if content differs
                    if output_turn["content"] != original_turn["content"]:
                        conversation_ids.append(conv_id)
                        model_differences.append(output_turn["content"])
                        original_differences.append(original_turn["content"])

        # a DataFrame
        df = pd.DataFrame({
            "conv_id": conversation_ids,
            "model": model_differences,
            "original": original_differences
        })

        # Save to CSV
        df.to_csv(filename+".csv", index=False, sep=';')

        return df


## evaluate model outputs
class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path, sep=';')
        else:
            raise FileNotFoundError(f"File {self.file_path} does not exist")

    def extract_reference_and_outputs(self):
        reference_texts = self.data['original']
        model_daily = self.data['model']
        return reference_texts, model_daily

    def prepare_model_outputs(self, model, variant): 
        return {f"{variant}": model.fillna("").tolist()}



class Metrics():
    def __init__(self):
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.google_bleu = evaluate.load("google_bleu")
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")

    def preprocess_series(self, references, candidates):
        references = references.fillna("").tolist() if isinstance(references, pd.Series) else references
        candidates = candidates.fillna("").tolist() if isinstance(candidates, pd.Series) else candidates

        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length.")
        return references, candidates

    def STS(self, references, candidates):
        reference_embeddings = self.sbert_model.encode(references, convert_to_tensor=True).cpu()
        candidate_embeddings = self.sbert_model.encode(candidates, convert_to_tensor=True).cpu()
        similarities = cosine_similarity(reference_embeddings, candidate_embeddings)
        return np.mean(similarities.diagonal())

    def GoogleBLEU(self, references, candidates):
        assert len(references) == len(candidates), "Predictions and references must have the same length."
        results = self.google_bleu.compute(predictions=candidates, references=references)
        return results

    def METEOR(self, references, candidates):
        assert len(references) == len(candidates), "Predictions and references must have the same length."
        results = self.meteor.compute(predictions=candidates, references=references)
        return results["meteor"]

    def ROUGE(self, references, candidates):
        assert len(references) == len(candidates), "Predictions and references must have the same length."
        results = self.rouge.compute(predictions=candidates, references=references)
        return {
            "ROUGE-1": results["rouge1"],
            "ROUGE-2": results["rouge2"],
            "ROUGE-L": results["rougeL"],
            "ROUGE-Lsum": results["rougeLsum"],
        }

    def BERTScore(self, references, candidates, lang="en", model_type="roberta-large"):
        references, candidates = self.preprocess_series(references, candidates)
        precision, recall, f1 = score(candidates, references, lang=lang, model_type=model_type, verbose=False)
        return {
            "average_precision": precision.mean().item(),
            "average_recall": recall.mean().item(),
            "average_f1": f1.mean().item()
        }


class MetricsEvaluator:
    def __init__(self):
        self.metrics = Metrics()

    def evaluate_variants(self, reference, variants):
        """
        Evaluate multiple model output variants against the reference dataset.
        """
        results = []

        for variant_name, outputs in variants.items():
            reference_list = reference  
            outputs_list = outputs  

            # Compute scores for each metric
            sts_score = self.metrics.STS(reference_list, outputs_list)
            google_bleu = self.metrics.GoogleBLEU(reference_list, outputs_list)
            meteor = self.metrics.METEOR(reference_list, outputs_list)
            rouge_scores = self.metrics.ROUGE(reference_list, outputs_list)
            bert_scores = self.metrics.BERTScore(reference_list, outputs_list)

            # Append scores to the results
            results.append({
                "Variant": variant_name,
                "STS": sts_score,
                "Google BLEU": google_bleu["google_bleu"],
                "METEOR": meteor,
                "ROUGE-1": rouge_scores["ROUGE-1"],
                "ROUGE-2": rouge_scores["ROUGE-2"],
                "ROUGE-L": rouge_scores["ROUGE-L"],
                "ROUGE-Lsum": rouge_scores["ROUGE-Lsum"],
                "BERTScore Precision": bert_scores["average_precision"],
                "BERTScore Recall": bert_scores["average_recall"],
                "BERTScore F1": bert_scores["average_f1"]
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def save_scores(self, scores_df, filename="evaluation_scores"):
        """Save scores to a CSV file."""
        scores_df.to_csv(filename+".csv", index=False, sep = ';')
        



### MAIN FUNCTIONS TO RUN LLMs AND EVALUATE OUTPUTS


class Run(): 

    def run_model(self,
        model,
        model_folder,
        hf_token,
        corpus_name,
        set_name,
        paths,
        max_samples=1000,
        intro_prompt="Continue this conversation based on the given context.",
        temperature=0.4,
        top_k=20,
        top_p=0.8,
        max_new_tokens=40,
        base_output_path="LLMs/corpora"
    ):
        """
        Run the text-generation model on the given corpus and set, generating outputs, saving results,
        and mapping roles in outputs based on the original data.

        Args:
            model(str): Name of the HuggingFace model.
            model_folder: Folder corresponding to the model where results connected to it are stored.
            hf_token (str): HuggingFace authentication token.
            corpus_name (str): Name of the corpus (e.g., daily_dialog_corpus).
            set_name (str): Dataset split name (train/dev/test).
            paths (list): Paths to input files.
            split (int, optional): Number of subsets to split the data into.
            intro_prompt (str, optional): System prompt for initializing the conversation.
            temperature (float, optional): Sampling temperature for generation.
            top_k (int, optional): Top-k sampling value.
            top_p (float, optional): Top-p sampling value.
            max_new_tokens (int, optional): Maximum number of new tokens to generate.
            base_output_path (str, optional): Base folder to save outputs.

        Returns:
            None
        """
        # 1) load the data
        data_loader = DataLoader(
            paths=paths,
            corpus_name=corpus_name,
            max_samples=max_samples,
            set_name=set_name,
            base_output_path=base_output_path
        )
        subsets = data_loader.get_data()

        # 2) initialize the model and tokenizer with the HF token
        model_instance = Model(model_folder=model, hf_token=hf_token)
        tokenizer = model_instance.tokenizer
        pipeline_model = model_instance.pipeline_text_generation

        # 3) process each subset
        evaluator = ModelGenerate(pipeline_model, tokenizer)
        output_handler = OutputHandler()

        # output paths
        outputs_folder = os.path.join(
            base_output_path,
            corpus_name,
            set_name,
            "outputs",
            model_folder,
            "evaluation"
        )
        conversations_folder = os.path.join(
            base_output_path,
            corpus_name,
            set_name,
            "outputs",
            model_folder,
            "conversations"
        )
        os.makedirs(outputs_folder, exist_ok=True)
        os.makedirs(conversations_folder, exist_ok=True)

        output_json_path = os.path.join(outputs_folder, "outputs")
        context_csv_path = os.path.join(outputs_folder, "context")
        mapped_json_path = os.path.join(conversations_folder, "conversations")

        # path to the original data for role mapping
        original_path = paths[0]

        for idx, subset in enumerate(subsets):
            print(f"Processing subset {idx + 1} of {len(subsets)}...")

            # generate outputs
            outputs, context = evaluator.conversation_history_generate_missing_utterances(
                inputs=subset,
                intro_prompt=intro_prompt,
                temp=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )

            # append outputs and context
            output_handler.prepare_outputs(outputs, file_path=output_json_path)
            output_handler.save_context_input(context, file_path=context_csv_path)
            
            output_handler.map_roles(
                original_file=original_path,
                outputs_file=outputs,
                output_mapped_file=mapped_json_path
            )

        print("Data processed, outputs saved, and roles mapped.")



    def run_evaluation(self, corpus_name, set_name, model_folder):

         # output paths
        base_path = os.path.join(
            "./LLMs/corpora",
            corpus_name,
            set_name,
            "outputs",
            model_folder,
            "evaluation"
        )
    
        originals_path = os.path.join(
            "./data/corpora",
            corpus_name,
            set_name,
            set_name

        )
        # if the evaluation directory exists
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(originals_path, exist_ok=True)
        

        outputs_path = os.path.join(base_path, "outputs")
        turns_csv_path = os.path.join(base_path, "turns")
        evaluation_csv_path =  os.path.join(base_path, "evaluation_scores")
        # generate turns.csv
        prepare_turns = PrepareTurns()
        prepare_turns.filter_model_turns(outputs_path, originals_path, turns_csv_path)
        
        # load and process data
        processor = DataProcessor(turns_csv_path+ ".csv")
        references, model_outputs = processor.extract_reference_and_outputs()
        model_variants = processor.prepare_model_outputs(model_outputs, model_folder)
        
        # evaluate and save scores
        evaluator = MetricsEvaluator()
        scores_df = evaluator.evaluate_variants(references, model_variants)
        evaluator.save_scores(scores_df, evaluation_csv_path)
        
        print(f"Evaluation complete. Scores saved to: {evaluation_csv_path}.csv")


