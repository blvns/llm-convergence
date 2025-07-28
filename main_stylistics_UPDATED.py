import os
import json
import pandas as pd

from helper_functions_stylistics_UPDATED import (
    convert_dict_to_list_format,
    reduce_corpus,
    reduce_model_corpus_to_first_model_turn,
    build_random_pairs,
    process_and_save_corpus,
    calculate_pos_percentages,
    visualize_POS_accommodation_grouped_by_pos,
    calculate_pos_tag_differences,
    count_utterance_length_and_differences,
    calculate_accommodation_for_pos_category_b_to_a,
    find_shared_proper_nouns_per_conversation,
    merge_accommodation_jsons_to_csv,
    merge_and_calculate_differences,
    calculate_averages,
    plot_comparison_avglength,
    load_hedge_words,
    count_hedge_words,
    calculate_hedge_differences,
    calculate_novelty_tokens,
    plot_combined_pos_differences,
    plot_propn_sharing,
    plot_hedge_differences,
    plot_novelty_per_role,
    build_instance_table,
    calculate_pairwise_asymmetric_accommodation,
    calculate_true_asymmetric_accommodation_by_corpus
)

# === PATH SETUP ===
SHARED_DATA_PATH = "stylometrics/data/dailydialog_corpus"
MODEL_DATA_PATH = os.path.join(SHARED_DATA_PATH, "Llama-3.2-3B-Instruct")
RESULTS_PATH = "stylometrics/results/dailydialog_corpus/Llama-3.2-3B-Instruct"

os.makedirs(SHARED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_DATA_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# === INPUT FILES ===
original_raw = "LLMs/corpora/dailydialog_corpus/dev/outputs/Llama-3.2-3B-Instruct/evaluation/outputs.json"
model_raw = "LLMs/corpora/dailydialog_corpus/dev/outputs/Llama-3.2-3B-Instruct/conversations/conversations.json"

# === DATA FILES ===
converted_original = os.path.join(SHARED_DATA_PATH, "converted_original.json")
reduced_original = os.path.join(SHARED_DATA_PATH, "reduced_original.json")
random_pairs = os.path.join(SHARED_DATA_PATH, "random_pairs.json")

converted_model = os.path.join(MODEL_DATA_PATH, "converted_model.json")
reduced_model = os.path.join(MODEL_DATA_PATH, "reduced_model.json")

tagged_original = os.path.join(SHARED_DATA_PATH, "tagged_original.json")
tagged_random = os.path.join(SHARED_DATA_PATH, "tagged_random.json")
tagged_model = os.path.join(MODEL_DATA_PATH, "tagged_model.json")

# === MAIN PIPELINE ===
def run_all():
    print("Starting stylistic analysis pipeline for Llama-3.2-3B-Instruct ")

    # 1–2: Convert & Reduce original (use reduced only for random)
    if not os.path.exists(converted_original):
        convert_dict_to_list_format(original_raw, converted_original)
    else:
        print("Converted original already exists.")

    if not os.path.exists(reduced_original):
        reduce_corpus(converted_original, reduced_original)
    else:
        print("Reduced original already exists.")

    # 3: Convert & Reduce model
    convert_dict_to_list_format(model_raw, converted_model)
    reduce_model_corpus_to_first_model_turn(converted_model, reduced_model)

    # 4: Build random pairs from reduced
    if not os.path.exists(random_pairs):
        with open(reduced_original, "r", encoding="utf-8") as f:
            reduced_data = json.load(f)
        build_random_pairs(reduced_data, random_pairs)
    else:
        print("Random pairs already exist.")

    # 5: POS tagging
    if not os.path.exists(tagged_original):
        process_and_save_corpus(converted_original, tagged_original)
    else:
        print("Tagged original already exists.")

    if not os.path.exists(tagged_random):
        process_and_save_corpus(random_pairs, tagged_random)
    else:
        print("Tagged random already exists.")

    if not os.path.exists(tagged_model):
        process_and_save_corpus(reduced_model, tagged_model)
    else:
        print("Tagged model already exists.")

    # 6: POS percentages
    calculate_pos_percentages(tagged_original, os.path.join(RESULTS_PATH, "POS_percentages_original.csv"))
    calculate_pos_percentages(tagged_random, os.path.join(RESULTS_PATH, "POS_percentages_random.csv"))
    calculate_pos_percentages(tagged_model, os.path.join(RESULTS_PATH, "POS_percentages_model.csv"))

    # 7: POS tag differences
    for input_file, name in [
        (tagged_original, "original"),
        (tagged_random, "random"),
        (tagged_model, "model")
    ]:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        calculate_pos_tag_differences(data, os.path.join(RESULTS_PATH, f"POS_differences_{name}.csv"))

    # 8: Utterance length differences
    with open(tagged_original, "r", encoding="utf-8") as f1, \
         open(tagged_random, "r", encoding="utf-8") as f2, \
         open(tagged_model, "r", encoding="utf-8") as f3:
        original_data = json.load(f1)
        random_data = json.load(f2)
        model_data = json.load(f3)

    count_utterance_length_and_differences(original_data, os.path.join(RESULTS_PATH, "utterance_length_original.json"), "user_vs_assistant")
    count_utterance_length_and_differences(random_data, os.path.join(RESULTS_PATH, "utterance_length_random.json"), "user_vs_assistant")
    count_utterance_length_and_differences(model_data, os.path.join(RESULTS_PATH, "utterance_length_model.json"), "user_vs_model")

    # 9: POS accommodation
    with open(tagged_original, "r", encoding="utf-8") as f1, \
         open(tagged_random, "r", encoding="utf-8") as f2, \
         open(tagged_model, "r", encoding="utf-8") as f3:
        t1 = json.load(f1)
        t2 = json.load(f2)
        t3 = json.load(f3)

    # Debug: transitions
    print("Checking transitions in original corpus:")
    transition_count = 0
    for convo in t1:
        for i in range(len(convo["utterances"]) - 1):
            r1 = convo["utterances"][i]["role"]
            r2 = convo["utterances"][i + 1]["role"]
            if r1 == "user" and r2 == "assistant":
                transition_count += 1
    print("user → assistant transitions found:", transition_count)

    # Debug: example marker
    style_marker = "ADV"
    triggered_base = 0
    triggered = 0
    base_total = 0
    base_triggered = 0

    for convo in t1:
        for i in range(len(convo["utterances"]) - 1):
            u1 = convo["utterances"][i]
            u2 = convo["utterances"][i + 1]
            if u1["role"] == "user" and u2["role"] == "assistant":
                prev_has = u1.get("pos_counts", {}).get(style_marker, 0) > 0
                curr_has = u2.get("pos_counts", {}).get(style_marker, 0) > 0
                if prev_has:
                    triggered_base += 1
                    if curr_has:
                        triggered += 1
                else:
                    base_total += 1
                    if curr_has:
                        base_triggered += 1

    print(f"→ {style_marker}:")
    print("triggered_base:", triggered_base)
    print("triggered:", triggered)
    print("base_total:", base_total)
    print("base_triggered:", base_triggered)
    prob_triggered = triggered / triggered_base if triggered_base else 0
    prob_base = base_triggered / base_total if base_total else 0
    print("accommodation score:", round(prob_triggered - prob_base, 4))

    # Now calculate full accommodation
    pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    for tag in pos_tags:
        calculate_accommodation_for_pos_category_b_to_a(t1, tag, "assistant", "user", os.path.join(RESULTS_PATH, f"acc_{tag}_original.json"))
        calculate_accommodation_for_pos_category_b_to_a(t2, tag, "assistant", "user", os.path.join(RESULTS_PATH, f"acc_{tag}_random.json"))
        calculate_accommodation_for_pos_category_b_to_a(t3, tag, "model", "user", os.path.join(RESULTS_PATH, f"acc_{tag}_model.json"))

    # 10: Shared proper nouns
    find_shared_proper_nouns_per_conversation(t1, os.path.join(RESULTS_PATH, "shared_propn_original.csv"))
    find_shared_proper_nouns_per_conversation(t2, os.path.join(RESULTS_PATH, "shared_propn_random.csv"))
    find_shared_proper_nouns_per_conversation(t3, os.path.join(RESULTS_PATH, "shared_propn_model.csv"))

    # 11–12: Hedge words
    hedge_list_1 = load_hedge_words("stylometrics/data/hedge_words_list_1.txt")
    hedge_list_2 = load_hedge_words("stylometrics/data/hedge_words_list_2.txt")
    hedge_list = hedge_list_1 + hedge_list_2
    for label, path, role1, role2 in [
        ("original", converted_original, "user_x", "user_y"),
        ("random", random_pairs, "user_x", "user_y"),
        ("model", reduced_model, "model", "user_y"),
    ]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hedge_csv = os.path.join(RESULTS_PATH, f"hedges_{label}.csv")
        diff_csv = os.path.join(RESULTS_PATH, f"hedge_diff_{label}.csv")
        count_hedge_words(data, hedge_list, hedge_csv)
        df = pd.read_csv(hedge_csv)
        calculate_hedge_differences(df, role1, role2, diff_csv)

    # 13: Novelty
    for label, path in [
        ("original", converted_original),
        ("random", random_pairs),
        ("model", reduced_model)
    ]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calculate_novelty_tokens(data, os.path.join(RESULTS_PATH, f"novelty_{label}.csv"))

    # 14–15: Merge & averages
    with open(os.path.join(RESULTS_PATH, "utterance_length_original.json")) as f1, \
         open(os.path.join(RESULTS_PATH, "utterance_length_random.json")) as f2, \
         open(os.path.join(RESULTS_PATH, "utterance_length_model.json")) as f3:
        uo = json.load(f1)
        ur = json.load(f2)
        um = json.load(f3)

    merged_csv = os.path.join(RESULTS_PATH, "utterance_length_merged.csv")
    merge_and_calculate_differences(uo, ur, um, merged_csv)

    avg_df = calculate_averages(pd.read_csv(merged_csv))
    avg_df.to_csv(os.path.join(RESULTS_PATH, "utterance_length_averages.csv"), index=False)

    # 16–20: Visualizations
    plot_comparison_avglength(avg_df, os.path.join(RESULTS_PATH, "utterance_length_averages.png"))
    plot_combined_pos_differences(
        {
            "original": os.path.join(RESULTS_PATH, "POS_differences_original.csv"),
            "random": os.path.join(RESULTS_PATH, "POS_differences_random.csv"),
            "model": os.path.join(RESULTS_PATH, "POS_differences_model.csv")
        },
        os.path.join(RESULTS_PATH, "POS_differences_grouped_by_pos.png")
    )
    merge_accommodation_jsons_to_csv(
        results_path=RESULTS_PATH,
        output_csv=os.path.join(RESULTS_PATH, "POS_accommodation_merged.csv")
    )
    visualize_POS_accommodation_grouped_by_pos(
        input_csv=os.path.join(RESULTS_PATH, "POS_accommodation_merged.csv"),
        output_path=os.path.join(RESULTS_PATH, "POS_accommodation_plot_grouped_by_POS.png")
    )
    plot_propn_sharing(
        {"original": os.path.join(RESULTS_PATH, "shared_propn_original.csv"),
         "random": os.path.join(RESULTS_PATH, "shared_propn_random.csv"),
         "model": os.path.join(RESULTS_PATH, "shared_propn_model.csv")},
        os.path.join(RESULTS_PATH, "shared_propn_plot.png")
    )
    plot_hedge_differences(
        {"original": os.path.join(RESULTS_PATH, "hedge_diff_original.csv"),
         "random": os.path.join(RESULTS_PATH, "hedge_diff_random.csv"),
         "model": os.path.join(RESULTS_PATH, "hedge_diff_model.csv")},
        os.path.join(RESULTS_PATH, "hedge_differences_plot.png")
    )
    plot_novelty_per_role(
        os.path.join(RESULTS_PATH, "novelty_model.csv"),
        os.path.join(RESULTS_PATH, "novelty_plot.png"),
        "Novelty by Role for Model Corpus"
    )

            # 21: Build unified instance table with relative differences
    df_original = build_instance_table(
        tagged_path=os.path.join(SHARED_DATA_PATH, "tagged_original.json"),
        corpus_type="original",
        hedge_words=hedge_list
    )

    df_random = build_instance_table(
        tagged_path=os.path.join(SHARED_DATA_PATH, "tagged_random.json"),
        corpus_type="random",
        hedge_words=hedge_list
    )

    df_model = build_instance_table(
        tagged_path=os.path.join(MODEL_DATA_PATH, "tagged_model.json"),
        corpus_type="model",
        hedge_words=hedge_list
    )

    df_all = pd.concat([df_original, df_random, df_model], ignore_index=True)

    instance_output_path = os.path.join(RESULTS_PATH, "instance_table_ALL.csv")
    df_all.to_csv(instance_output_path, index=False)
    print(f"Saved unified instance table to: {instance_output_path}")

  
    # 22: Add asymmetric accommodation scores per POS tag (constant per tag over full dataset)
    df_with_acc = calculate_pairwise_asymmetric_accommodation(df_all)
    acc_path = os.path.join(RESULTS_PATH, "instance_table_ALL_with_ACC.csv")
    df_with_acc.to_csv(acc_path, index=False)
    print(f"Saved instance table with acc_<POS>_asym columns to: {acc_path}")

    # 23: Build POS-wise asymmetric accommodation summary by corpus (true formula)
    from helper_functions_stylistics_UPDATED import calculate_true_asymmetric_accommodation_by_corpus

    final_summary_df = calculate_true_asymmetric_accommodation_by_corpus(df_all)
    summary_output_path = os.path.join(RESULTS_PATH, "POS_accommodation_asymmetric_summary.csv")
    final_summary_df.to_csv(summary_output_path, index=False)
    print(f"Saved final POS accommodation summary (C_m) by corpus to: {summary_output_path}")


if __name__ == "__main__":
    run_all()
