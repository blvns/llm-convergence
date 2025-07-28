import json
from scipy.stats import ttest_rel
import argparse
from pathlib import Path

def main(args):

	f_counts = {}

	test_data = []
	for p in args.test_paths:
		with open(p, 'r', encoding="utf-8") as f:
			d = json.load(f)
			test_data.append(d)

	with open(args.ref_path, 'r', encoding="utf-8") as f:
		ref_data = json.load(f)

	features = list(ref_data[list(ref_data.keys())[0]].keys())
	sig_data = {t: {} for t in args.test_names}
	c = 0
	for feat in features:
		for t, t_data in zip(args.test_names, test_data):
			c += 1
			example_keys = [k for k in t_data.keys() if k != 'ALL']
			ref_arr = [ref_data[k][feat] for k in example_keys]
			t_arr = [t_data[k][feat] for k in example_keys]
			result = ttest_rel(t_arr, ref_arr)
			sig = result.statistic
			pval = result.pvalue
			sig_data[t][feat] = (sig, pval)

			if feat == 'novel_tok':
				if feat not in f_counts: f_counts[feat] = 0
				if sig < 0 and pval < 0.05:
					f_counts[feat] += 1
			elif 'liwc' in feat or feat == 'utter_diff' or feat == 'propn':
				if feat not in f_counts: f_counts[feat] = 0
				if sig > 0 and pval < 0.05:
					f_counts[feat] += 1

	print(f_counts)
	print(c)


	#write out sig test results
	#write out analysis results
	out_path = './stylometrics/significance/{}/stats.json'.format(args.corpus)
	Path('./stylometrics/significance/{}/'.format(args.corpus)).mkdir(parents=True, exist_ok=True)

	with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sig_data, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Significance test of stylometric features.")
	parser.add_argument("--ref_path", 
						type=str, 
						required=True,
						help="Path to control setting .json.")
	parser.add_argument("--test_paths", 
						type=str, 
						nargs='+',
						required=True,
						help="Path(s) to test settings .json.")
	parser.add_argument("--test_names", 
						type=str, 
						nargs='+',
						required=True,
						help="Names for each test setting.")
	parser.add_argument("--corpus", 
						type=str, 
						required=True,
						choices=['movie', 'dailydialog', 'npr'])

	args = parser.parse_args()
	main(args)
