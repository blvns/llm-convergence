# Linguistic Convergence in LLMs

This is the codebase and model generations for the paper *Do language models accommodate their users? A Study of Linguistic Convergence*. You can find the paper [here](blvns.github.io/publications), which also contains more details about our experimental setup in this repo.


## Getting started
Large Language Models used in the project:
 - [Gemma3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)-Xb-{it/pt}, for X = {1, 4, 12, 27}
 - [Llama3](https://www.llama.com/models/llama-3/)-Xb(-Instruct), for X = {1, 3, 8, 70}

## Python version

Install **python 3.11.11** to successfully install the packages for this project

## Get the packages installed

Make sure to create a new virtual environment.
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Download spacy model

To download the spacy tokenizer run:

```bash
python -m spacy download en_core_web_trf
```

## Download the data

In `data/corpora`, different folders named `{corpus_name}_corpus` can be created. To create folders, download and unzip data for each corpus, follow the code below.
Every time you download a new corpus remember to run the code from the root folder `DAP_Roth`.

```bash

#dailydialog corpus
mkdir data/corpora/dailydialog_corpus
cd data/corpora/dailydialog_corpus
wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip ijcnlp_dailydialog.zip
cd ijcnlp_dailydialog
unzip train.zip; unzip validation.zip; unzip test.zip

#movie corpus
mkdir data/corpora/movie_corpus
cd data/corpora/movie_corpus
wget https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip
unzip movie-corpus.zip

#npr corpus
mkdir data/corpora/npr_corpus
cd data/corpora/npr_corpus
wget https://zissou.infosci.cornell.edu/convokit/datasets/npr-2p-corpus/npr-2p-corpus.zip
unzip npr-2p-corpus.zip

```

## Preprocess the data

To preprocess the conversations, you should run the `main_preprocess_corpus.py` module. This module takes as argument:

- `corpus`: Select the corpus to preprocess from the following: `'movie', 'dailydialog', 'npr'`.

Example:

```bash
python main_preprocess_corpus.py --corpus <Corpus_Name>
```

The conversations will be processed, and for each folder (`train`, `dev`, and `test`) in the corpus folder, two JSON files will be created:

- `split.json`: Original conversations with two speakers (assigned to `user` and `assistant`) with at least five utterances.
- `model_inputs.json`: Starting from the fourth turn, the model takes the role of `assistant`, and its content is masked.

Statistics about the test set of each corpus are added to the `data/corpora_statistics_test_set.csv` file.

## Run LLMs to get outputs for analysis

To run the models you need to get a permission to use them. As mentioned in the **"Getting started"** section, there are 4 models used in the project two smaller models (Gemma2-2b-it and Llama3.2-3b-it) and two bigger models (Gemma2-9b-it and Llama3.1-8b-it). To run the models, use main_LLMs.py script. The script is structured to perform two tasks: 
- `run_model()` function: run the selected model on chosen corpus to get the outputs for missing parts of the conversations.
- `run_evaluation()` function: run evaluation metrics to compare model outputs with original utterances. The main use of this function is to evaluate the model output for selecting the best way the model is prompted(e.g. if to include system meta-prompt along with conversation history) and deciding the best parameter setup (here: temperature, top_k, top_p).

#### How to?
How to run the model on chosen corpus: 

run_model has few arguments that need to be provided in order to run it:

Required parameters:
*  `--model`: name of the model from Hugging Face: e.g. google/gemma3-1b-it,
*  `--hf_token`: Access Token from Hugging Face account,
*  `--corpus_name`: name of the corpus to process: e.g "npr_corpus" or "dailydialog_corpus", 
*  `--set_name`: which set to process: choose  **dev**, 
*  `--paths`: the path to the file that stores model inputs, path from data/ directory, e.g *./data/corpora/npr_corpus/dev/model_inputs.json*, 

Optional parameters: 
*  `--intro_prompt`: the meta prompt to include, default set to *"Continue this conversation based on the given context."*
*  `--temperature`: default 0.4,
*  `--top_k`: default 20,
*  `--top_p`: default 0.8,
*  `--chat_template`: if to use the default chat_template of the LLM if available, default True,
*  `--max_new_tokens`: default 40.

Example runs: 

```bash
python  main_LLMs.py run_model --model "google/gemma3-1b-it" --hf_token "your_token" --corpus_name npr_corpus --set_name dev --paths ./data/corpora/npr_corpus/test/model_inputs.json --chat_template False
python  main_LLMs.py run_model --model "google/gemma3-1b-pt" --hf_token "your_token" --corpus_name npr_corpus --set_name dev --paths ./data/corpora/npr_corpus/test/model_inputs.json --chat_template False
```

## Stylometric analysis
To analyse linguistic/stylometric features of the generations, you can run the `postprocess_stylometrics.py` module. You will need to set the file paths to access and save data.
For the “original” (as referred to in the main stylometrics function), human-authored data sets, access `data/corpora/<corpus_name>/dev/dev.json`; this is also used in the random baseline setting (with additional sampling of random utterances across conversations). For the “model” data set, access `generations/<corpus_name>/dev/outputs/<model_name>/conversations/conversations`.json.

Example postprocessing run:
```bash
python postprocess_stylometrics.py --data_path "data/path/for/setting" --setting model/human/random --corpus "corpus name" --setting_name "model name" &
```

This generates a new statistics file, `stylometrics/analysis/<corpus_name>/<setting_name>/stats.json`, with statistics for individual utterance generations and aggregate statistics across the dataset (ALL index).

You can also calculate the statistical significance between a reference and a list of target data files across various stylometric features with ``stylometrics_significance.py''. For example, 

```bash
python stylometrics_significance.py  --ref_path ./stylometrics/analysis/dailydialog/human/stats.json --corpus dailydialog --test_paths ./stylometrics/analysis/dailydialog/<model_name./stats.json ... &
```

**Citation and Contact**

If you use this code for your own work, please cite the corresponding paper:
```
@article{blevins2025convergence,
    title={Do language models accommodate their users? A Study of Linguistic Convergence},
    author={Terra Blevins and Susanne Schmalwieser and Benjamin Roth},
    year={2025},
    archivePrefix={arXiv},
    url={https://blvns.github.io/papers/blevins2025convergence.pdf}
}
```

Please address any questions or comments about this codebase to t.blevins@northeastern.edu. 
