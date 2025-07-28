import json
import re
import random
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import os
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_trf")
from tqdm import tqdm


def convert_dict_to_list_format(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level of {input_path}, but got {type(data)}")

    converted = []
    for convo_id, utterances in data.items():
        if isinstance(utterances, list):
            converted.append({
                "conversation_id": convo_id,
                "utterances": utterances
            })
        else:
            print(f"[WARNING] Skipping {convo_id} because utterances are not a list.")

    print(f"Converted {len(converted)} conversations (original: {len(data)})")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

def reduce_corpus(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reduced_data = []
    for convo in data:
        convo_id = convo["conversation_id"]
        utterances = convo["utterances"]
        reduced_data.append({"conversation_id": convo_id, "utterances": utterances})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reduced_data, f, indent=4, ensure_ascii=False)

def reduce_model_corpus_to_first_model_turn(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reduced_data = []

    for convo in data:
        convo_id = convo.get("conversation_id", "unknown")
        utterances = convo["utterances"]

        first_model_index = next((i for i, utt in enumerate(utterances) if utt["role"] == "model"), None)

        if first_model_index is not None and first_model_index > 0:
            trimmed = utterances[first_model_index - 1:]
            reduced_data.append({"conversation_id": convo_id, "utterances": trimmed})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reduced_data, f, indent=4, ensure_ascii=False)

def build_random_pairs(reduced_corpus, output_path):
    all_utterances = []
    for convo in reduced_corpus:
        for utt in convo["utterances"]:
            if utt["role"] in ["user", "assistant"]:
                all_utterances.append(utt)

    random.shuffle(all_utterances)

    random_pairs = []
    for i in range(0, len(all_utterances) - 1, 2):
        random_pairs.append({
            "conversation_id": str(i),
            "utterances": [all_utterances[i], all_utterances[i + 1]]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(random_pairs, f, indent=4, ensure_ascii=False)


def process_and_save_corpus(input_path, output_path):
    print(f"Loading file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} conversations")

    for convo in tqdm(corpus, desc="POS tagging conversations"):
        for utt in convo["utterances"]:
            text = utt.get("content", "")
            doc = nlp(text)
            pos_tags = [(token.text, token.pos_) for token in doc]
            utt["pos_tags"] = pos_tags

            counts = Counter([pos for _, pos in pos_tags])
            utt["pos_counts"] = dict(counts)

    print("Finished tagging. Saving to file...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)
    print(f"Saved POS-tagged file: {output_path}")

def calculate_pos_tag_differences(corpus, output_file):
    conversation_differences = []

    for conversation in corpus:
        convo_id = conversation["conversation_id"]
        utterances = conversation["utterances"]
        diffs_by_tag = defaultdict(list)

        for i in range(len(utterances) - 1):
            current = utterances[i].get("pos_counts", {})
            next_ = utterances[i + 1].get("pos_counts", {})

            all_tags = set(current.keys()).union(set(next_.keys()))

            for tag in all_tags:
                diff = abs(current.get(tag, 0) - next_.get(tag, 0))
                diffs_by_tag[tag].append(diff)

        avg_diffs = {tag: (sum(vals) / len(vals)) if vals else 0 for tag, vals in diffs_by_tag.items()}
        avg_diffs["conversation_id"] = convo_id
        conversation_differences.append(avg_diffs)

    df = pd.DataFrame(conversation_differences)
    df.to_csv(output_file, index=False)


def calculate_pos_percentages(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for convo in data:
        convo_id = convo["conversation_id"]
        for utt in convo["utterances"]:
            row = {"conversation_id": convo_id, "role": utt["role"]}
            total = sum(utt["pos_counts"].values())
            for tag, count in utt["pos_counts"].items():
                row[tag] = round(count / total, 4) if total else 0
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

def count_utterance_length_and_differences(data, output_file, comparison_type):
    results = []

    for convo in data:
        convo_id = convo["conversation_id"]

        if comparison_type == "user_vs_assistant":
            utterances = [utt for utt in convo["utterances"] if utt["role"] in ["user", "assistant"]]
        elif comparison_type == "user_vs_model":
            utterances = [utt for utt in convo["utterances"] if utt["role"] in ["user", "model"]]
        else:
            continue

        diffs = []
        for i in range(len(utterances) - 1):
            len1 = len(utterances[i]["content"].split())
            len2 = len(utterances[i + 1]["content"].split())
            diffs.append(abs(len1 - len2))

        avg_diff = sum(diffs) / len(diffs) if diffs else 0

        results.append({
            "conversation_id": convo_id,
            "total_diff": avg_diff,
            "utterance_pairs_counted": len(diffs)
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def calculate_pos_tag_differences(corpus, output_file):
    conversation_differences = []

    for conversation in corpus:
        convo_id = conversation["conversation_id"]
        utterances = conversation["utterances"]
        diffs_by_tag = defaultdict(list)

        for i in range(len(utterances) - 1):
            current = utterances[i].get("pos_counts", {})
            next_ = utterances[i + 1].get("pos_counts", {})
            all_tags = set(current.keys()).union(set(next_.keys()))
            for tag in all_tags:
                diff = abs(current.get(tag, 0) - next_.get(tag, 0))
                diffs_by_tag[tag].append(diff)

        avg_diffs = {tag: (sum(vals) / len(vals)) if vals else 0 for tag, vals in diffs_by_tag.items()}
        avg_diffs["conversation_id"] = convo_id
        conversation_differences.append(avg_diffs)

    df = pd.DataFrame(conversation_differences)
    df.to_csv(output_file, index=False)
 
def calculate_accommodation_for_pos_category_b_to_a(conversation_data, style_marker, user_b, user_a, output_file=None):
    convo_results = []

    for convo in conversation_data:
        convo_id = convo["conversation_id"]
        utterances = convo.get("utterances", [])

        triggered = 0
        triggered_base = 0
        base_triggered = 0
        base_total = 0

        for i in range(len(utterances) - 1):
            prev_utt = utterances[i]
            curr_utt = utterances[i + 1]

            if prev_utt["role"] == user_a and curr_utt["role"] == user_b:
                prev_has = prev_utt.get("pos_counts", {}).get(style_marker, 0) > 0
                curr_has = curr_utt.get("pos_counts", {}).get(style_marker, 0) > 0

                if prev_has:
                    triggered_base += 1
                    if curr_has:
                        triggered += 1
                else:
                    base_total += 1
                    if curr_has:
                        base_triggered += 1

        prob_triggered = triggered / triggered_base if triggered_base else 0
        prob_base = base_triggered / base_total if base_total else 0
        accommodation_score = prob_triggered - prob_base

        convo_results.append({
            "conversation_id": convo_id,
            "style_marker": style_marker,
            "accommodation_score": round(accommodation_score, 4),
            "triggered": triggered,
            "triggered_base": triggered_base,
            "base_triggered": base_triggered,
            "base_total": base_total
        })

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(convo_results, f, indent=4, ensure_ascii=False)

    return convo_results



def find_shared_proper_nouns_per_conversation(corpus_data, output_file_path):
    shared_counts = []

    for conversation in corpus_data:
        convo_id = conversation["conversation_id"]
        utterances = conversation["utterances"]
        shared = 0

        for i in range(len(utterances) - 1):
            u1 = utterances[i]
            u2 = utterances[i + 1]

            if u1["role"] != u2["role"]:
                u1_nouns = set([t[0].lower() for t in u1.get("pos_tags", []) if t[1] == "PROPN"])
                u2_nouns = set([t[0].lower() for t in u2.get("pos_tags", []) if t[1] == "PROPN"])
                shared += len(u1_nouns.intersection(u2_nouns))

        shared_counts.append({
            "conversation_id": convo_id,
            "shared_proper_nouns": shared
        })

    df = pd.DataFrame(shared_counts)
    df.to_csv(output_file_path, index=False)

def merge_and_calculate_differences(original_data, random_data, model_data, output_path):
    merged = []

    convo_ids = {item["conversation_id"] for item in original_data + random_data + model_data}

    for convo_id in convo_ids:
        orig = next((x["total_diff"] for x in original_data if x["conversation_id"] == convo_id), None)
        rand = next((x["total_diff"] for x in random_data if x["conversation_id"] == convo_id), None)
        model = next((x["total_diff"] for x in model_data if x["conversation_id"] == convo_id), None)

        merged.append({
            "conversation_id": convo_id,
            "user_vs_assistant_original": orig,
            "user_vs_assistant_random": rand,
            "user_vs_model": model
        })

    df = pd.DataFrame(merged)
    df.to_csv(output_path, index=False)

def calculate_averages(df):
    mean_original = df["user_vs_assistant_original"].mean()
    mean_random = df["user_vs_assistant_random"].mean()
    mean_model = df["user_vs_model"].mean()

    result = pd.DataFrame([{
        "user_vs_assistant_original": mean_original,
        "user_vs_assistant_random": mean_random,
        "user_vs_model": mean_model
    }])

    return result

def plot_comparison_avglength(df, output_path):
    categories = ["user_vs_assistant_original", "user_vs_assistant_random", "user_vs_model"]
    means = df.iloc[0][categories].values
    colors = ["palegreen", "peachpuff", "cornflowerblue"]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, means, color=colors)
    plt.ylabel("Average Length Difference")
    plt.title("Comparison of Average Utterance Length Differences")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_hedge_words(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines()]

def count_hedge_words(conversation_data, hedge_list, output_file):
    results = []

    for convo in conversation_data:
        convo_id = convo["conversation_id"]
        for utt in convo["utterances"]:
            content = utt["content"].lower()
            hedge_count = sum(1 for word in hedge_list if word in content)
            results.append({
                "conversation_id": convo_id,
                "role": utt["role"],
                "utterance": utt["content"],
                "hedge_count": hedge_count
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def calculate_hedge_differences(hedge_df, role1, role2, output_file):
    diffs = []
    grouped = hedge_df.groupby("conversation_id")

    for convo_id, group in grouped:
        utts_r1 = group[group["role"] == role1]["hedge_count"]
        utts_r2 = group[group["role"] == role2]["hedge_count"]
        avg_r1 = utts_r1.mean() if not utts_r1.empty else 0
        avg_r2 = utts_r2.mean() if not utts_r2.empty else 0
        diff = abs(avg_r1 - avg_r2)

        diffs.append({
            "conversation_id": convo_id,
            f"{role1}_vs_{role2}_hedge_diff": diff
        })

    df_out = pd.DataFrame(diffs)
    df_out.to_csv(output_file, index=False)

def calculate_novelty_tokens(conversation_data, output_file):
    results = []

    for convo in conversation_data:
        convo_id = convo["conversation_id"]
        seen = set()
        for utt in convo["utterances"]:
            tokens = set(utt["content"].lower().split())
            novelty = len(tokens - seen)
            seen.update(tokens)
            results.append({
                "conversation_id": convo_id,
                "role": utt["role"],
                "novelty_count": novelty
            })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def merge_csv(file1, file2, output_file, how="outer"):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    merged = pd.merge(df1, df2, on="conversation_id", how=how)
    merged.to_csv(output_file, index=False)


def plot_combined_pos_differences(input_paths, output_path):
    color_map = {
        "original": "palegreen",
        "random": "peachpuff",
        "model": "cornflowerblue"
    }

    dfs = []
    for corpus, path in input_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "conversation_id" in df.columns:
                df = df.drop(columns=["conversation_id"])
            mean_diffs = df.mean().reset_index()
            mean_diffs.columns = ["pos_tag", "avg_diff"]
            mean_diffs["corpus"] = corpus
            dfs.append(mean_diffs)
        else:
            print(f"Missing POS diff file: {path}")

    if not dfs:
        print("No data available for POS difference plot.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    pos_tags = sorted(combined["pos_tag"].unique())
    x = range(len(pos_tags))
    width = 0.25

    plt.figure(figsize=(14, 6))

    for i, (corpus, color) in enumerate(color_map.items()):
        subset = combined[combined["corpus"] == corpus]
        heights = [subset[subset["pos_tag"] == tag]["avg_diff"].values[0] if tag in subset["pos_tag"].values else 0 for tag in pos_tags]
        plt.bar([xi + i * width for xi in x], heights, width=width, label=corpus, color=color)

    plt.xticks([r + width for r in x], pos_tags, rotation=45)
    plt.ylabel("Average POS Tag Difference")
    plt.title("POS Tag Differences Across Corpora")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt

def visualize_POS_accommodation_grouped_by_pos(input_csv, output_path):
    df = pd.read_csv(input_csv)

    # Gruppieren nach pos_tag und corpus
    grouped = df.groupby(["pos_tag", "corpus"])["accommodation_score"].mean().reset_index()

    # Pivot für gruppierten Balkenplot
    pivot = grouped.pivot(index="pos_tag", columns="corpus", values="accommodation_score")
    pivot = pivot.fillna(0)

    # Farben zuweisen
    color_map = {
        "original": "palegreen",
        "random": "peachpuff",
        "model": "cornflowerblue"
    }

    pivot = pivot.sort_index()
    pivot.plot(kind="bar", color=[color_map.get(c, "gray") for c in pivot.columns], figsize=(12, 6))

    plt.ylabel("Mean Accommodation Score")
    plt.title("POS Accommodation per Corpus (Grouped by POS Tag)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Accommodation plot saved to: {output_path}")


def plot_propn_sharing(csv_paths_dict, output_path):
    data = {}
    for label, path in csv_paths_dict.items():
        df = pd.read_csv(path)
        total_shared = df["shared_proper_nouns"].sum()
        data[label] = total_shared

    plt.figure(figsize=(6, 5))
    plt.bar(data.keys(), data.values(), color=["palegreen", "peachpuff", "cornflowerblue"])
    plt.ylabel("Total Shared Proper Nouns")
    plt.title("Proper Noun Overlap Across Corpora")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_hedge_differences(csv_paths, output_path):
    diffs = {}
    for label, path in csv_paths.items():
        df = pd.read_csv(path)
        avg = df.iloc[:, 1].mean()
        diffs[label] = avg

    plt.figure(figsize=(8, 5))
    plt.bar(diffs.keys(), diffs.values())
    plt.ylabel("Average Hedgeword Difference")
    plt.title("Hedgeword Differences Across Roles and Corpora")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_novelty_per_role(csv_path, output_path, title):
    df = pd.read_csv(csv_path)
    df_grouped = df.groupby("role")["novelty_count"].mean()

    plt.figure(figsize=(6, 5))
    df_grouped.plot(kind="bar")
    plt.ylabel("Average Novelty (new tokens)")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def merge_accommodation_jsons_to_csv(results_path, output_csv):
    data = []

    corpora = ["original", "random", "model"]
    pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
                "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    for corpus in corpora:
        for tag in pos_tags:
            file_path = os.path.join(results_path, f"acc_{tag}_{corpus}.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    for entry in entries:
                        entry["pos_tag"] = tag
                        entry["corpus"] = corpus
                        data.append(entry)
            else:
                print(f"⚠️ Missing file: {file_path}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Merged CSV saved to: {output_csv}")


import json
import pandas as pd

def build_instance_table(tagged_path, corpus_type, hedge_words=None):
    with open(tagged_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for convo in data:
        convo_id = convo["conversation_id"]
        seen_tokens = set()
        utterances = convo["utterances"]

        pair_id = 0
        for i in range(len(utterances) - 1):
            u1 = utterances[i]
            u2 = utterances[i + 1]

            # Only include utterance pairs with different roles
            if u1["role"] == u2["role"]:
                continue

            row = {
                "corpus_type": corpus_type,
                "conversation_id": convo_id,
                "utterance_pair_id": pair_id
            }

            # --- Relative length difference ---
            len1 = len(u1["content"].split())
            len2 = len(u2["content"].split())
            avg_len = (len1 + len2) / 2 if (len1 + len2) else 1
            row["rel_length_diff"] = (len2 - len1) / avg_len

            # --- Relative POS frequency differences ---
            pos_a = u1.get("pos_counts", {})
            pos_b = u2.get("pos_counts", {})
            total_a = sum(pos_a.values()) or 1
            total_b = sum(pos_b.values()) or 1

            all_tags = set(pos_a.keys()).union(pos_b.keys())
            for tag in sorted(all_tags):
                ratio_b = pos_b.get(tag, 0) / total_b
                ratio_a = pos_a.get(tag, 0) / total_a
                row[f"rel_diff_{tag}"] = round(ratio_b - ratio_a, 4)

            # --- Hedgeword difference ---
            if hedge_words:
                h1 = sum(1 for hw in hedge_words if hw in u1["content"].lower())
                h2 = sum(1 for hw in hedge_words if hw in u2["content"].lower())
                row["hedge_diff"] = h2 - h1
            else:
                row["hedge_diff"] = None

            # --- Shared proper nouns (PROPN) ---
            propn_a = set(t[0].lower() for t in u1.get("pos_tags", []) if t[1] == "PROPN")
            propn_b = set(t[0].lower() for t in u2.get("pos_tags", []) if t[1] == "PROPN")
            row["propn_shared"] = len(propn_a.intersection(propn_b))

            # --- Novelty in B ---
            tokens_b = set(u2["content"].lower().split())
            row["novelty_b"] = len(tokens_b - seen_tokens)
            seen_tokens.update(tokens_b)

            rows.append(row)
            pair_id += 1

    return pd.DataFrame(rows)



def calculate_pairwise_asymmetric_accommodation(df):
    pos_tags = sorted(set(col.replace("pos_", "").replace("_a", "") 
                          for col in df.columns if col.startswith("pos_") and col.endswith("_a")))

    for tag in pos_tags:
        col_a = f"pos_{tag}_a"
        col_b = f"pos_{tag}_b"

        if col_a not in df.columns or col_b not in df.columns:
            continue

        # Total pairs where A used tag
        a_has_tag = df[df[col_a] == 1]
        triggered = len(a_has_tag[a_has_tag[col_b] == 1])
        triggered_base = len(a_has_tag)

        # Total B-utterances with tag
        base_triggered = len(df[df[col_b] == 1])
        base_total = len(df)

        # Compute probabilities
        p_triggered = triggered / triggered_base if triggered_base else 0
        p_base = base_triggered / base_total if base_total else 0
        acc_score = round(p_triggered - p_base, 4)

        # Store result in new column
        df[f"acc_{tag}_asym"] = acc_score

        print(f"Calculated C_m for {tag}: {acc_score} (P(B|A)={p_triggered:.3f}, P(B)={p_base:.3f})")

    return df


def calculate_true_asymmetric_accommodation_by_corpus(df):
    import pandas as pd

    # Extract all POS tags from rel_diff_<tag> columns
    pos_tags = sorted(set(col.replace("rel_diff_", "")
                          for col in df.columns if col.startswith("rel_diff_")))

    corpus_types = sorted(df["corpus_type"].unique())
    results = []

    for corpus in corpus_types:
        sub_df = df[df["corpus_type"] == corpus]

        for tag in pos_tags:
            # Use rel_diff_<TAG> to infer POS presence
            col = f"rel_diff_{tag}"
            if col not in sub_df.columns:
                continue

            # POS used by A (assume A used m if they had any positive contribution)
            sub_df = sub_df.copy()
            sub_df["a_used"] = sub_df[col].apply(lambda x: 1 if x < 0 else 0)
            sub_df["b_used"] = sub_df[col].apply(lambda x: 1 if x > 0 else 0)

            triggered_base = sub_df[sub_df["a_used"] == 1]
            triggered = triggered_base["b_used"].sum()
            base_total = sub_df[sub_df["a_used"] == 0]
            base_triggered = base_total["b_used"].sum()

            p_triggered = triggered / len(triggered_base) if len(triggered_base) else 0
            p_base = base_triggered / len(base_total) if len(base_total) else 0
            acc_score = round(p_triggered - p_base, 4)

            results.append({
                "corpus_type": corpus,
                "pos_tag": tag,
                "acc_asym_score": acc_score
            })

    return pd.DataFrame(results)
