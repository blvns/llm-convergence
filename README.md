# Accommodation in LLMs

## Getting started
Large Language Models used in the project:
 - [Gemma3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)-Xb-{it/pt}, for X = {1, 4, 12, 27}
 - Llama3-Xb, for X = {1, 3, 8, 70}

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

## Download pretrained word embeddings

We use [pretrained word embeddings](https://fasttext.cc/). You can prepare them as follows:

```bash
cd data
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
7z x cc.en.300.bin.gz
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
1) How to run the model on chosen corpus: 

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


Examples with settings to replicate Gemma3, LLama-3 results: 

```bash
python  main_LLMs.py run_model --model "google/gemma3-1b-it" --hf_token "your_token" --corpus_name npr_corpus --set_name dev --paths ./data/corpora/npr_corpus/test/model_inputs.json --chat_template False
python  main_LLMs.py run_model --model "google/gemma3-1b-pt" --hf_token "your_token" --corpus_name npr_corpus --set_name dev --paths ./data/corpora/npr_corpus/test/model_inputs.json --chat_template False
```
 
 2) How to run the evaluation on models outputs: 

run_evaluation has the following arguments: 
*  `--corpus_name`: name of the corpus: e.g. npr_corpus,
*  `--set_name`: name of split set, choose **test**, 

Example evaluation run: 

```bash
python main_LLMs.py run_evaluation --corpus_name npr_corpus --set_name test --model_folder gemma2_2b_it --model_type instruction_tuned
```

## Stylometric analysis
To analyse specific linguistic features, you should run the `main_stylometrics.py` module. You will need to set the file paths to access and save data. We recommend changing the relevant file paths via regular expressions. You will not need to change the file names. 
For the “original” data sets (as referred to in the main stylometrics function), access `data/corpora/<corpus_name>/test/test.json`. For the “model” data set, access `LLMs/corpora/<corpus_name>/test/outputs/instruction_tuned/<model_name>/conversations/conversations`.json


For accessing and reducing data generated during the analysis, file paths are structured as `data/stylometrics/<corpus_name>/<model_name>/<file_name>`. 
To save and access results, file paths are structured as `results/<corpus_name>/stylometrics/<model_name>/<file_name>`.
If only selected features are to be analysed, only the following functions need to be run: 
```bash
    main_reduce() to reduce the original corpus to only the relevant utterances
    main_random() to generate random utterance pairs out of the original corpus for comparison 
    main_reducemodel() to reduce the generated results to only the relevant utterances
```
… as well as of course the selected feature(s).

The linguistic features included and corresponding functions are: 
```bash
Utterance length    [main_countutt(), main_uttmerge(), main_calcavg(), main_plotlength()]
Parts of speech     [main_tag(), main_countpos(),  main_plotPOSabs(), main_pos_model(), main_pos_original(), main_pos_random(), main_percmerge(), main_vizpos()] 
                     to calculate the linguistic accommodation metric: [main_acc(), main_mergeacc()]   
Proper nouns        [main_tag(), main_propnoun(), main_sumpropnouns(), main_uniquepropnouns(), main_sumuniquenouns(), main_vizpropnouns()]
Hedge words         [Download hedge_words_list_1.txt hedge_words_list_1.txt from data/stylomatrics! Then use main_hedge(), main_hedgediff(), main_avghedge(), main_hedgeviz()]
 
Token Novelty       [main_novelty(), main_avgnovelty()]
```

**Sources:** 
The hedge word lists were created by Stepanenko (2022). Their work can be accessed [here](https://github.com/alexandrastepanenko/InvestigatingStyleGPT2/tree/main/Jupyter%20notebook).
The Asymmetric Accommodation Metric (as applied for POS) was described by Waissbluth et. al. (2021). Their work can be accessed [here](https://github.com/elliottwaissbluth/LSA-online-arguments/tree/main).
