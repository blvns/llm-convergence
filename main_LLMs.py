import argparse
from helper_functions_LLMs import Run

def main():
    parser = argparse.ArgumentParser(description="Run model generation and evaluation.")
    
    parser.add_argument("action", choices=["run_model", "run_evaluation"], help="Choose whether to run the model or evaluation.")
    
    # Arguments for run_model
    parser.add_argument("--model", choices=[
        #Gemma 3
        "google/gemma-3-1b-pt", #pretrained
        "google/gemma-3-1b-it", #instruction-tuned
        "google/gemma-3-4b-pt", #pretrained
        "google/gemma-3-4b-it", #instruction-tuned
        "google/gemma-3-12b-pt", #pretrained
        "google/gemma-3-12b-it", #instruction-tuned
        "google/gemma-3-27b-pt", #pretrained
        "google/gemma-3-27b-it", #instruction-tuned
        #Llama 3
        "meta-llama/Llama-3.2-1B", 
        "meta-llama/Llama-3.2-1B-Instruct", 
        "meta-llama/Llama-3.2-3B", 
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B", 
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        #"meta-llama/Llama-3.1-405B",
        #"meta-llama/Llama-3.1-405B-Instruct",
    ], help="HuggingFace model name.")
    
    parser.add_argument("--hf_token", type=str, help="HuggingFace authentication token.")
    parser.add_argument("--corpus_name", choices=["npr_corpus", "dailydialog_corpus", "movie_corpus"], help="Corpus name.")
    parser.add_argument("--set_name", choices=["train", "dev", "test"], help="Dataset split name.")
    parser.add_argument("--model_type", choices=["pre-trained", "instruction_tuned"], help="Folder type.")
    parser.add_argument("--paths", nargs='+', help="Paths to input JSON files.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of conversations to fill in.")
    parser.add_argument("--intro_prompt", type=str, default="Continue this conversation based on the given context.", help="System prompt for initializing the conversation.")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature for generation.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling value.")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling value.")
    parser.add_argument("--max_new_tokens", type=int, default=40, help="Max new tokens to generate.")
    
    args = parser.parse_args()
    runner = Run()
    
    if args.action == "run_model":
        required_args = [args.model, args.hf_token, args.corpus_name, args.set_name, args.paths]
        if any(arg is None for arg in required_args):
            parser.error("Missing required arguments for run_model: --model, --hf_token, --corpus_name, --set_name, --paths")
        print("Running model with parameters:", vars(args))
        m_folder = args.model.split('/')[1]
        runner.run_model(
            model=args.model,
            model_folder=m_folder,
            hf_token=args.hf_token,
            corpus_name=args.corpus_name,
            set_name=args.set_name,
            paths=args.paths,
            max_samples=args.max_samples,
            intro_prompt=args.intro_prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens
        )
    elif args.action == "run_evaluation":
        required_args = [args.corpus_name, args.set_name, args.model_folder]
        if any(arg is None for arg in required_args):
            parser.error("Missing required arguments for run_evaluation: --corpus_name, --set_name, --model_folder")
        # print("Running evaluation with parameters:", vars(args))
        runner.run_evaluation(
            corpus_name=args.corpus_name,
            set_name=args.set_name,
            model_folder=args.model_folder,
        )

if __name__ == "__main__":
    main()
    
# Example usage:
# Run model:
# python main_LLMs.py run_model --model "meta-llama/Llama-3.1-8B-Instruct" --model_folder llama3_1_8b_it --hf_token your_token --corpus_name dailydialog_corpus --set_name test --model_type instruction_tuned --paths ./data/corpora/npr_corpus/test/x_y_model_inputs.json

# Run evaluation:
# python main_LLMs.py run_evaluation --corpus_name dailydialog_corpus --set_name test --model_folder gemma2_2b_it --model_type instruction_tuned
