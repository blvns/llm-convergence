import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_rel
import json

sns.set_style('whitegrid')
sns.set_context('talk') 

FEATURES = ["utter_diff", "liwc", "propn", "novel_tok"]
FEAT_NAMES = ["Utterance Len", 'LIWC Agreement', 'PROPN Overlap', 'Novel Tokens']

HUMAN_BASELINE = {
	'dailydialog': [0.69102,0.53759,0.01460,0.75742],
	'movie': [0.58664, 0.53215,0.02719,0.75639],
	'npr': [0.45905, 0.53007,0.25837,0.58561],
}

RAND_BASELINE = {
	'dailydialog': [0.66348, 0.50481, 0.0005, 0.78874],
	'movie': [0.58447, 0.52271,0.00000,0.78985],
	'npr': [0.46025, 0.48035,0.00959,0.58921],
}

GEMMA_PT = {'dailydialog': [[0.69385, 0.57911,0.02190,0.65732],
							[0.67167, 0.57470,0.01825,0.66582],
							[0.67753, 0.57359,0.01356,0.66968],
							[0.67461, 0.56564,0.02033,0.68695]],
			'movie': [[0.63923, 0.58475,0.03421,0.64280],
					  [0.62373, 0.57981,0.03158,0.65521],
					  [0.61104, 0.58613,0.02807,0.66392],
					  [0.62077, 0.59398,0.03114,0.65530]],
			'npr': [[0.63923, 0.51606, 0.12287, 0.44413],
					[0.62373, 0.52087, 0.11754, 0.50963],
					[0.61104, 0.52570, 0.12500, 0.51644],
					[0.62077, 0.52608, 0.12180, 0.51689]],
			}

GEMMA_IT = {'dailydialog': [[0.73334, 0.56969, 0.05266,0.69551],
							[0.66628, 0.51836,0.05735,0.75824],
							[0.64808, 0.51796,0.05214,0.75992],
							[0.61049, 0.49306,0.06726,0.76123]],
			'movie': [[0.62703, 0.55655,0.06711, 0.70714],
					  [0.58937, 0.51274, 0.07061, 0.72819],
					  [0.59290, 0.52782,0.08597, 0.72619],
					  [0.53554, 0.46521,0.10219,0.71779]],
			'npr': [[0.62703, 0.52380, 0.16261, 0.64512],
					[0.58937, 0.52234, 0.15164, 0.67585],
					[0.59290, 0.52603, 0.17266, 0.65817],
					[0.53554, 0.51925, 0.16078, 0.68015]],
			}

LLAMA_PT = {'dailydialog': [[0.68975, 0.57859, 0.019291, 0.66624],
							[0.68722, 0.57327, 0.02190, 0.68130],
							[0.69062, 0.57630,0.02033,0.67796],
							[0.65438, 0.57083,0.01668,0.67603]],
			'movie': [[0.63248, 0.58302,0.03772,0.63531],
					  [0.61824, 0.58764,0.03114,0.65910],
					  [0.62951, 0.58892,0.03947,0.65774],
					  [0.59195, 0.58594,0.03509,0.67501]],
			'npr': [[0.63248, 0.52466, 0.10627, 0.45246],
					[0.61824, 0.52563, 0.11510, 0.52262],
					[0.62951, 0.53303, 0.11480, 0.50684],
					[0.59195, 0.53057, 0.14495, 0.49636]],
			}

LLAMA_IT = {'dailydialog': [[0.68893, 0.53960,0.05057,0.70558],
							[0.70523, 0.54633,0.04484,0.72480],
							[0.71776, 0.55260,0.03702,0.72800],
							[0.65438, 0.52757,0.03858,0.72950]],
			'movie': [[0.58722, 0.50072,0.08553, 0.67083],
					  [0.61592, 0.52352,0.07061,0.69595],
					  [0.61986, 0.54813,0.05570, 0.67503],
					  [0.59195, 0.53864,0.06535,0.69817]],
			'npr': [[0.58722, 0.52972, 0.26675, 0.60536],
					[0.61592, 0.52786, 0.23097, 0.60754],
					[0.61986, 0.53835, 0.21894, 0.56910],
					[0.59195, 0.53203, 0.26249, 0.59674]],
			}


GEMMA_X = [1, 4, 12, 27]
LLAMA_X = [1, 3, 8, 40]
LLAMA_PT_COLOR = "#ED6E9C"
LLAMA_IT_COLOR = "#D81B60"
GEMMA_PT_COLOR = "#6DB2EE"
GEMMA_IT_COLOR = "#1A81DB"
PT_MARKER = "o"
IT_MARKER = "v"


def calc_liwc_deltas(dataset):
	#load datafiles
	human_path = f"./stylometrics/analysis/{dataset}/human/stats.json"
	with open(human_path, 'r', encoding="utf-8") as f:
		human_data = json.load(f)

	random_path = f"./stylometrics/analysis/{dataset}/random/stats.json"
	with open(random_path, 'r', encoding="utf-8") as f:
		random_data = json.load(f)

	model_paths = f"./stylometrics/analysis/{dataset}/Llama-3.2-1B/stats.json ./stylometrics/analysis/{dataset}/Llama-3.2-1B-Instruct/stats.json ./stylometrics/analysis/{dataset}/Llama-3.2-3B/stats.json ./stylometrics/analysis/{dataset}/Llama-3.2-3B-Instruct/stats.json ./stylometrics/analysis/{dataset}/Llama-3.1-8B/stats.json ./stylometrics/analysis/{dataset}/Llama-3.1-8B-Instruct/stats.json ./stylometrics/analysis/{dataset}/Llama-3.1-70B/stats.json ./stylometrics/analysis/{dataset}/Llama-3.1-70B-Instruct/stats.json ./stylometrics/analysis/{dataset}/gemma-3-1b-pt/stats.json ./stylometrics/analysis/{dataset}/gemma-3-1b-it/stats.json ./stylometrics/analysis/{dataset}/gemma-3-4b-pt/stats.json ./stylometrics/analysis/{dataset}/gemma-3-4b-it/stats.json ./stylometrics/analysis/{dataset}/gemma-3-12b-pt/stats.json ./stylometrics/analysis/{dataset}/gemma-3-12b-it/stats.json ./stylometrics/analysis/{dataset}/gemma-3-27b-pt/stats.json ./stylometrics/analysis/{dataset}/gemma-3-27b-it/stats.json"
	model_paths = model_paths.split(" ")

	llama_pt_paths = [p for p in model_paths if ("/Llama" in p and "B/stats" in p)]
	llama_pt_data = []
	print(llama_pt_paths)
	for p in llama_pt_paths:
		with open(p, 'r', encoding="utf-8") as f:
			d = json.load(f)
			llama_pt_data.append(d)

	llama_it_paths = [p for p in model_paths if ("/Llama" in p and "Instruct/stats" in p)]
	llama_it_data = []
	print(llama_it_paths)
	for p in llama_it_paths:
		with open(p, 'r', encoding="utf-8") as f:
			d = json.load(f)
			llama_it_data.append(d)

	gemma_pt_paths = [p for p in model_paths if ("/gemma" in p and "pt/stats" in p)]
	gemma_pt_data = []
	print(gemma_pt_paths)
	for p in gemma_pt_paths:
		with open(p, 'r', encoding="utf-8") as f:
			d = json.load(f)
			gemma_pt_data.append(d)

	gemma_it_paths = [p for p in model_paths if ("/gemma" in p and "it/stats" in p)]
	gemma_it_data = []
	print(gemma_it_paths)

	for p in gemma_it_paths:
		with open(p, 'r', encoding="utf-8") as f:
			d = json.load(f)
			gemma_it_data.append(d)

	liwc_columns = ["ppron", "ipron", "article", "conj", "prep", "auxvb", "adverb", "negate", "quant"]
	#create random delta maps
	random_baseline = np.array([random_data["ALL"][f"liwc_{k}"] for k in liwc_columns])
	random_baseline = random_baseline[None]
	rand_delta_llama_pt = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in llama_pt_data]) - random_baseline
	rand_delta_llama_it = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in llama_it_data]) - random_baseline
	rand_delta_gemma_pt = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in gemma_pt_data]) - random_baseline
	rand_delta_gemma_it = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in gemma_it_data]) - random_baseline
	
	#create random sig maps
	example_keys = [k for k in llama_pt_data[0].keys() if k != 'ALL']
	random_scores = {}
	for liwc_k in liwc_columns:
		random_scores[liwc_k] = [random_data[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]

	rand_sig_llama_pt = []
	for d in llama_pt_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, random_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		rand_sig_llama_pt.append(row_sig)

	rand_sig_llama_it = []
	for d in llama_it_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, random_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		rand_sig_llama_it.append(row_sig)

	rand_sig_gemma_pt = []
	for d in gemma_pt_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, random_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		rand_sig_gemma_pt.append(row_sig)

	rand_sig_gemma_it = []
	for d in gemma_it_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, random_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		rand_sig_gemma_it.append(row_sig)


	#create human delta maps
	human_baseline = np.array([human_data["ALL"][f"liwc_{k}"] for k in liwc_columns])
	human_baseline = human_baseline[None]
	human_delta_llama_pt = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in llama_pt_data]) - human_baseline
	human_delta_llama_it = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in llama_it_data]) - human_baseline
	human_delta_gemma_pt = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in gemma_pt_data]) - human_baseline
	human_delta_gemma_it = np.array([[d["ALL"][f"liwc_{k}"] for k in liwc_columns] for d in gemma_it_data]) - human_baseline

	#create human sig maps
	example_keys = [k for k in llama_pt_data[0].keys() if k != 'ALL']
	human_scores = {}
	for liwc_k in liwc_columns:
		human_scores[liwc_k] = [human_data[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]

	human_sig_llama_pt = []
	for d in llama_pt_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, human_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		human_sig_llama_pt.append(row_sig)

	human_sig_llama_it = []
	for d in llama_it_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, human_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		human_sig_llama_it.append(row_sig)

	human_sig_gemma_pt = []
	for d in gemma_pt_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, human_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		human_sig_gemma_pt.append(row_sig)

	human_sig_gemma_it = []
	for d in gemma_it_data:
		row_sig = []
		for liwc_k in liwc_columns:
			d_scores = [d[ex_k][f"liwc_{liwc_k}"] for ex_k in example_keys]
			result = ttest_rel(d_scores, human_scores[liwc_k])
			if result.pvalue < 0.05: row_sig.append(1)
			else: row_sig.append(0)
		human_sig_gemma_it.append(row_sig)

	#get vmin, vmax
	maps = [rand_delta_llama_pt, rand_delta_llama_it, rand_delta_gemma_pt, rand_delta_gemma_it, human_delta_llama_pt, human_delta_llama_it, human_delta_gemma_pt, human_delta_gemma_it]
	mins = [x.min() for x in maps]
	maxes = [x.max() for x in maps]
	vmin = min(mins)
	vmax = max(maxes)
	if abs(vmin) > vmax: vmax=abs(vmin)
	else: vmin = -1.*vmax

	return rand_delta_llama_pt, rand_delta_llama_it, rand_delta_gemma_pt, rand_delta_gemma_it, \
	rand_sig_llama_pt, rand_sig_llama_it, rand_sig_gemma_pt, rand_sig_gemma_it, \
	human_delta_llama_pt, human_delta_llama_it, human_delta_gemma_pt, human_delta_gemma_it, \
	human_sig_llama_pt, human_sig_llama_it, human_sig_gemma_pt, human_sig_gemma_it, vmin, vmax
'''
#Fine-grained LIWC Heatmap (DD)
h_pos = 338
h_neg = 208
cm = sns.diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, center='light', as_cmap=True)

rand_delta_llama_pt, rand_delta_llama_it, rand_delta_gemma_pt, rand_delta_gemma_it, \
rand_sig_llama_pt, rand_sig_llama_it, rand_sig_gemma_pt, rand_sig_gemma_it, \
human_delta_llama_pt, human_delta_llama_it, human_delta_gemma_pt, human_delta_gemma_it, \
human_sig_llama_pt, human_sig_llama_it, human_sig_gemma_pt, human_sig_gemma_it, vmin, vmax = calc_liwc_deltas(dataset='npr')

sns.heatmap(np.multiply(rand_delta_llama_pt, rand_sig_llama_pt), vmin=vmin, vmax=vmax, cmap=cm, cbar=True)
plt.tight_layout()
plt.savefig('liwc_heatmap_colorbar.png', dpi=800)
plt.clf()

f,((ax00, ax01), (ax10, ax11), (ax20, ax21), (ax30, ax31)) = plt.subplots(4,2)
f.set_size_inches(8, 8)
ax00.get_shared_y_axes().join(ax01)
ax10.get_shared_y_axes().join(ax11)
ax20.get_shared_y_axes().join(ax21)
ax30.get_shared_y_axes().join(ax31)
ax00.get_shared_x_axes().join(ax10, ax20, ax30)
ax01.get_shared_x_axes().join(ax11, ax21, ax31)

g00 = sns.heatmap(np.multiply(rand_delta_llama_pt, rand_sig_llama_pt), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax00, \
		yticklabels=['1B', '3B', '8B', '70B'], xticklabels=[])
g01 = sns.heatmap(np.multiply(human_delta_llama_pt, human_sig_llama_pt), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax01, \
		yticklabels=[], xticklabels=[])
g10 = sns.heatmap(np.multiply(rand_delta_llama_it, rand_sig_llama_it), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax10, \
		yticklabels=['1B', '3B', '8B', '70B'], xticklabels=[])
g11 = sns.heatmap(np.multiply(human_delta_llama_it, human_sig_llama_it), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax11, \
		yticklabels=[], xticklabels=[])
g20 = sns.heatmap(np.multiply(rand_delta_gemma_pt, rand_sig_gemma_pt), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax20, \
		yticklabels=['1B', '4B', '12B', '27B'], xticklabels=[])
g21 = sns.heatmap(np.multiply(human_delta_gemma_pt, human_sig_gemma_pt), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax21, \
		yticklabels=[], xticklabels=[])
g30 = sns.heatmap(np.multiply(rand_delta_gemma_it, rand_sig_gemma_it), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax30, \
		xticklabels=["ppron", "ipron", "article", "conj", "prep", "auxvb", "adverb", "negate", "quant"], yticklabels=['1B', '4B', '12B', '27B'])
ax30.set_xticklabels(["ppron", "ipron", "article", "conj", "prep", "auxvb", "adverb", "negate", "quant"], rotation=45, ha="right")
g31 = sns.heatmap(np.multiply(human_delta_gemma_it, human_sig_gemma_it), vmin=vmin, vmax=vmax, cmap=cm, cbar=False, ax=ax31, \
		xticklabels=["ppron", "ipron", "article", "conj", "prep", "auxvb", "adverb", "negate", "quant"], yticklabels=[])
ax31.set_xticklabels(["ppron", "ipron", "article", "conj", "prep", "auxvb", "adverb", "negate", "quant"], rotation=45, ha="right")

plt.tight_layout()
plt.savefig('liwc_npr_heatmap.png', dpi=800)
plt.clf()
'''



for d in ['npr']:#['dailydialog', 'movie', 'npr']:
	human = HUMAN_BASELINE[d]
	rand = RAND_BASELINE[d]
	gemma_pt = np.array(GEMMA_PT[d])
	gemma_it = np.array(GEMMA_IT[d])
	llama_pt = np.array(LLAMA_PT[d])
	llama_it = np.array(LLAMA_IT[d])

	print(d)
	#scatter plot for each feature
	for i, (f, c) in enumerate(zip(FEAT_NAMES, FEATURES)):
		if c != "liwc": continue

		human_f = human[i]
		rand_f = rand[i]
		gemma_pt_f = gemma_pt[:, i]
		gemma_it_f = gemma_it[:, i]
		llama_pt_f = llama_pt[:, i]
		llama_it_f = llama_it[:, i]
		plt.axhline(human_f, linestyle="-", color="#525252", alpha=0.85, zorder=1, label='Human')
		plt.scatter(GEMMA_X, gemma_pt_f, color=GEMMA_PT_COLOR, marker=PT_MARKER, label='Gemma (PT)', zorder=2)
		plt.scatter(GEMMA_X, gemma_it_f, color=GEMMA_IT_COLOR, marker=IT_MARKER, label='Gemma (IT)', zorder=2)
		plt.scatter(LLAMA_X, llama_pt_f, color=LLAMA_PT_COLOR, marker=PT_MARKER, label='LLaMA (PT)', zorder=2)
		plt.scatter(LLAMA_X, llama_it_f, color=LLAMA_IT_COLOR, marker=IT_MARKER, label='LLaMA (IT)', zorder=2)
		
		plt.axhline(rand_f, linestyle=":", color="#525252", zorder=-100, label='Rand.')
		plt.xticks([0, 10, 20, 30, 40], [0, 10, 20, 30, 70])
		plt.xlabel("Parameter Count")
		#legend = plt.legend()
		#legend.get_frame().set_edgecolor('#525252')

		plt.tight_layout()
		plt.savefig('accomm_{}_{}2.png'.format(d, c), dpi=800)
		plt.clf()

		#corr study 
		x = [1, 4, 12, 27]+[1, 3, 8, 70]
		y = [abs(g) for g in gemma_pt_f]+[abs(g-human_f) for g in llama_pt_f]
		r = pearsonr(x,y)
		print(c, "pt", r)

		x = [1, 4, 12, 27]+[1, 3, 8, 70]
		y = [abs(g) for g in gemma_it_f]+[abs(g-human_f) for g in llama_it_f]
		r = pearsonr(x,y)
		print(c, "it", r)
	print(' ')
quit()


'''
Lookback plots
'''
'''
for c in ['dailydialog', 'movie', 'npr']:
	scores = {}
	for m in ["human", "Llama-3.2-1B", "Llama-3.2-1B-Instruct", "Llama-3.2-3B", "Llama-3.2-3B-Instruct","Llama-3.1-8B", "Llama-3.1-8B-Instruct","Llama-3.1-70B","Llama-3.1-70B-Instruct", "gemma-3-1b-pt", "gemma-3-1b-it","gemma-3-4b-pt", "gemma-3-4b-it","gemma-3-12b-pt", "gemma-3-12b-it","gemma-3-27b-pt", "gemma-3-27b-it"]:
		scores[m] = {"utter_diff":[], "liwc_mean":[], "propn":[], "novel_tok":[]}
		#load data from file and org
		fpath = "./stylometrics/analysis/{}/{}/stepwise_stats.json".format(c, m)
		with open(fpath, "r", encoding="utf-8") as f:
			data = json.load(f)
		for k in scores[m].keys():
			for i in range(0,5): scores[m][k].append(data[str(i)][k])

	y_labels = {'utter_diff':'Average Agreemeent',
				'liwc_mean':'Average Agreemeent',
				'propn':'Average Overlap',
				'novel_tok':'% Novel Tokens'}
    #generate line plot dataset/feature pair
	for k in scores['human'].keys():
		x = [i for i in range(1,6)]

		fig, ax = plt.subplots()

		#human line
		y = scores['human'][k]
		ax.plot(x, y, label="Human", color="#525252")

		#gemma_pt line
		y_data = []
		for _, m in enumerate(["gemma-3-1b-pt", "gemma-3-4b-pt", "gemma-3-12b-pt", "gemma-3-27b-pt"]):
			y_data.append(scores[m][k])
		y_data = list(map(list, zip(*y_data)))
		y_avg = [sum(row)/len(row) for row in y_data]
		y_min = [y_avg[i]-min(row) for i, row in enumerate(y_data)]
		y_max = [max(row)-y_avg[i] for i, row in enumerate(y_data)]
		#plt.errorbar(x, y_avg, label="Gemma (PT)", yerr=[y_min, y_max], color="#6DB2EE", marker="o")
		ax.plot(x, y_avg, label="Gemma (PT)", color="#6DB2EE", marker="o")

		#gemma_it line
		y_data = []
		for m_idx, m in enumerate(["gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]):
			y_data.append(scores[m][k])
		y_data = list(map(list, zip(*y_data)))
		y_avg = [sum(row)/len(row) for row in y_data]
		#y_min = [y_avg[i]-min(row) for i, row in enumerate(y_data)]
		#y_max = [max(row)-y_avg[i] for i, row in enumerate(y_data)]
		#plt.errorbar(x, y_avg, label="Gemma (IT)", yerr=[y_min, y_max], color="#1A81DB", marker="v")
		ax.plot(x, y_avg, label="Gemma (IT)", color="#1A81DB", marker="v")

		#llama_pt line
		y_data = []
		for m_idx, m in enumerate(["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.1-70B"]):
			y_data.append(scores[m][k])
		y_data = list(map(list, zip(*y_data)))
		y_avg = [sum(row)/len(row) for row in y_data]
		#y_min = [y_avg[i]-min(row) for i, row in enumerate(y_data)]
		#y_max = [max(row)-y_avg[i] for i, row in enumerate(y_data)]
		#plt.errorbar(x, y_avg, label="Llama (PT)", yerr=[y_min, y_max], color="#ED6E9C", marker="o")
		ax.plot(x, y_avg, label="Llama (PT)", color="#ED6E9C", marker="o")

		#llama_it line
		y_data = []
		for m_idx, m in enumerate(["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"]):
			y_data.append(scores[m][k])
		y_data = list(map(list, zip(*y_data)))
		y_avg = [sum(row)/len(row) for row in y_data]
		#y_min = [y_avg[i]-min(row) for i, row in enumerate(y_data)]
		#y_max = [max(row)-y_avg[i] for i, row in enumerate(y_data)]
		#plt.errorbar(x, y_avg, label="Llama (IT)", yerr=[y_min, y_max], color="#D81B60", marker="v")
		ax.plot(x, y_avg, label="Llama (IT)", color="#D81B60", marker="v")

		ax.axvspan(1.5, 2.5, facecolor='gray', alpha=0.3)
		ax.axvspan(3.5, 4.5, facecolor='gray', alpha=0.3)
		ax.grid(axis='x')


		#ax.set_ylabel(y_labels[k])
		if k == 'propn': plt.legend(framealpha=1, fontsize=14)#, loc=(2.5, 0.035)
		plt.tight_layout()
		plt.savefig("./stepwise_{}_{}.png".format(c, k), dpi=800)
		plt.clf()

'''



names = ['Random', 'Human', 'Gemma (PT)', 'Gemma (IT)', 'LLaMA (PT)', 'LLAMA (IT)']
datasets = ['DailyDialog', 'Movie', 'NPR']

'''
utterance bar chart
'''
gemma_pt_avg = []
gemma_pt_max = []
gemma_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_PT[d][i][0] for i in range(0, len(GEMMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_pt_avg.append(d_avg)
	gemma_pt_max.append(max(d_arr)-d_avg)
	gemma_pt_min.append(d_avg-min(d_arr))

gemma_it_avg = []
gemma_it_max = []
gemma_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_IT[d][i][0] for i in range(0, len(GEMMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_it_avg.append(d_avg)
	gemma_it_max.append(max(d_arr)-d_avg)
	gemma_it_min.append(d_avg-min(d_arr))

llama_pt_avg = []
llama_pt_max = []
llama_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_PT[d][i][0] for i in range(0, len(LLAMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_pt_avg.append(d_avg)
	llama_pt_max.append(max(d_arr)-d_avg)
	llama_pt_min.append(d_avg-min(d_arr))

llama_it_avg = []
llama_it_max = []
llama_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_IT[d][i][0] for i in range(0, len(LLAMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_it_avg.append(d_avg)
	llama_it_max.append(max(d_arr)-d_avg)
	llama_it_min.append(d_avg-min(d_arr))


human_bl = [HUMAN_BASELINE['dailydialog'][0],HUMAN_BASELINE['movie'][0],HUMAN_BASELINE['npr'][0]]
random_bl = [RAND_BASELINE['dailydialog'][0],RAND_BASELINE['movie'][0],RAND_BASELINE['npr'][0]]
all_utter = [random_bl, human_bl, gemma_pt_avg, gemma_it_avg,llama_pt_avg,llama_it_avg]
bl_range = [0,0,0]
all_max = [bl_range, bl_range, gemma_pt_max, gemma_it_max,llama_pt_max,llama_it_max]
all_min = [bl_range, bl_range, gemma_pt_min, gemma_it_min,llama_pt_min,llama_it_min]

colors = ["#8F8F8F", "#525252", "#6DB2EE", "#1A81DB", "#ED6E9C", "#D81B60"]

all_utter = [random_bl, human_bl]
bl_range = [0,0,0]
all_max = [bl_range, bl_range]
all_min = [bl_range, bl_range]
colors = ["#8F8F8F", "#525252"]



x = np.arange(len(datasets))  # the label locations
#width = 0.12 # the width of the bars
#multiplier = 0
width = 0.4 # the width of the bars
multiplier = -0.5

#plt.figure(figsize=(10,5))
plt.figure(figsize=(5,5))
for n, data, upper, lower, c in zip(names, all_utter, all_max, all_min, colors):
	offset = width * multiplier
	b1 = plt.bar(x + offset, data, width, yerr=[lower, upper], label=n, color=c)
	multiplier += 1

plt.ylabel('Average Agreemeent')
#plt.xticks(x + width*3, datasets)
plt.xticks(x, datasets)
plt.ylim(0.4, 0.8)
plt.tight_layout()
plt.savefig('utter_baseline_bar.png', dpi=800)
plt.clf()


'''
LIWC bar chart
'''
gemma_pt_avg = []
gemma_pt_max = []
gemma_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_PT[d][i][1] for i in range(0, len(GEMMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_pt_avg.append(d_avg)
	gemma_pt_max.append(max(d_arr)-d_avg)
	gemma_pt_min.append(d_avg-min(d_arr))

gemma_it_avg = []
gemma_it_max = []
gemma_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_IT[d][i][1] for i in range(0, len(GEMMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_it_avg.append(d_avg)
	gemma_it_max.append(max(d_arr)-d_avg)
	gemma_it_min.append(d_avg-min(d_arr))

llama_pt_avg = []
llama_pt_max = []
llama_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_PT[d][i][1] for i in range(0, len(LLAMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_pt_avg.append(d_avg)
	llama_pt_max.append(max(d_arr)-d_avg)
	llama_pt_min.append(d_avg-min(d_arr))

llama_it_avg = []
llama_it_max = []
llama_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_IT[d][i][1] for i in range(0, len(LLAMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_it_avg.append(d_avg)
	llama_it_max.append(max(d_arr)-d_avg)
	llama_it_min.append(d_avg-min(d_arr))


human_bl = [HUMAN_BASELINE['dailydialog'][1],HUMAN_BASELINE['movie'][1],HUMAN_BASELINE['npr'][1]]
random_bl = [RAND_BASELINE['dailydialog'][1],RAND_BASELINE['movie'][1],RAND_BASELINE['npr'][1]]
all_liwc = [random_bl, human_bl, gemma_pt_avg, gemma_it_avg,llama_pt_avg,llama_it_avg]
bl_range = [0,0,0]
all_max = [bl_range, bl_range, gemma_pt_max, gemma_it_max,llama_pt_max,llama_it_max]
all_min = [bl_range, bl_range, gemma_pt_min, gemma_it_min,llama_pt_min,llama_it_min]

colors = ["#8F8F8F", "#525252", "#6DB2EE", "#1A81DB", "#ED6E9C", "#D81B60"]

all_utter = [random_bl, human_bl]
bl_range = [0,0,0]
all_max = [bl_range, bl_range]
all_min = [bl_range, bl_range]
colors = ["#8F8F8F", "#525252"]

x = np.arange(len(datasets))  # the label locations
#width = 0.12 # the width of the bars
#multiplier = 0
width = 0.4 # the width of the bars
multiplier = -0.5

#plt.figure(figsize=(10,5))
plt.figure(figsize=(5,5))
for n, data, upper, lower, c in zip(names, all_liwc, all_max, all_min, colors):
	offset = width * multiplier
	plt.bar(x + offset, data, width, yerr=[lower, upper], label=n, color=c)
	multiplier += 1

plt.ylabel('Average Agreemeent')
#plt.xticks(x + width*3, datasets)
plt.xticks(x, datasets)
plt.ylim(0.4, 0.6)
plt.tight_layout()
plt.savefig('liwc_baseline_bar.png', dpi=800)
plt.clf()


'''
propn bar chart
'''
gemma_pt_avg = []
gemma_pt_max = []
gemma_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_PT[d][i][2] for i in range(0, len(GEMMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_pt_avg.append(d_avg)
	gemma_pt_max.append(max(d_arr)-d_avg)
	gemma_pt_min.append(d_avg-min(d_arr))

gemma_it_avg = []
gemma_it_max = []
gemma_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_IT[d][i][2] for i in range(0, len(GEMMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_it_avg.append(d_avg)
	gemma_it_max.append(max(d_arr)-d_avg)
	gemma_it_min.append(d_avg-min(d_arr))

llama_pt_avg = []
llama_pt_max = []
llama_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_PT[d][i][2] for i in range(0, len(LLAMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_pt_avg.append(d_avg)
	llama_pt_max.append(max(d_arr)-d_avg)
	llama_pt_min.append(d_avg-min(d_arr))

llama_it_avg = []
llama_it_max = []
llama_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_IT[d][i][2] for i in range(0, len(LLAMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_it_avg.append(d_avg)
	llama_it_max.append(max(d_arr)-d_avg)
	llama_it_min.append(d_avg-min(d_arr))


human_bl = [HUMAN_BASELINE['dailydialog'][2],HUMAN_BASELINE['movie'][2],HUMAN_BASELINE['npr'][2]]
random_bl = [RAND_BASELINE['dailydialog'][2],RAND_BASELINE['movie'][2],RAND_BASELINE['npr'][2]]
all_propn = [random_bl, human_bl, gemma_pt_avg, gemma_it_avg,llama_pt_avg,llama_it_avg]
bl_range = [0,0,0]
all_max = [bl_range, bl_range, gemma_pt_max, gemma_it_max,llama_pt_max,llama_it_max]
all_min = [bl_range, bl_range, gemma_pt_min, gemma_it_min,llama_pt_min,llama_it_min]

colors = ["#8F8F8F", "#525252", "#6DB2EE", "#1A81DB", "#ED6E9C", "#D81B60"]

all_utter = [random_bl, human_bl]
bl_range = [0,0,0]
all_max = [bl_range, bl_range]
all_min = [bl_range, bl_range]
colors = ["#8F8F8F", "#525252"]

x = np.arange(len(datasets))  # the label locations
#width = 0.12 # the width of the bars
#multiplier = 0
width = 0.4 # the width of the bars
multiplier = -0.5

#plt.figure(figsize=(10,5))
plt.figure(figsize=(5,5))
for n, data, upper, lower, c in zip(names, all_propn, all_max, all_min, colors):
	offset = width * multiplier
	plt.bar(x + offset, data, width, yerr=[lower, upper], label=n, color=c)
	multiplier += 1

plt.ylabel('Average Overlap')
#plt.xticks(x + width*3, datasets)
plt.xticks(x, datasets)
plt.legend(framealpha=1)
plt.ylim(0.0, 0.3)
plt.tight_layout()
plt.savefig('propn_baseline_bar.png', dpi=800)
plt.clf()

'''
Novel token bar chart
'''
gemma_pt_avg = []
gemma_pt_max = []
gemma_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_PT[d][i][-1] for i in range(0, len(GEMMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_pt_avg.append(d_avg)
	gemma_pt_max.append(max(d_arr)-d_avg)
	gemma_pt_min.append(d_avg-min(d_arr))

gemma_it_avg = []
gemma_it_max = []
gemma_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [GEMMA_IT[d][i][-1] for i in range(0, len(GEMMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	gemma_it_avg.append(d_avg)
	gemma_it_max.append(max(d_arr)-d_avg)
	gemma_it_min.append(d_avg-min(d_arr))

llama_pt_avg = []
llama_pt_max = []
llama_pt_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_PT[d][i][-1] for i in range(0, len(LLAMA_PT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_pt_avg.append(d_avg)
	llama_pt_max.append(max(d_arr)-d_avg)
	llama_pt_min.append(d_avg-min(d_arr))

llama_it_avg = []
llama_it_max = []
llama_it_min = []
for d in ['dailydialog', 'movie', 'npr']:
	d_arr = [LLAMA_IT[d][i][-1] for i in range(0, len(LLAMA_IT[d]))]
	d_avg = sum(d_arr)/len(d_arr)
	llama_it_avg.append(d_avg)
	llama_it_max.append(max(d_arr)-d_avg)
	llama_it_min.append(d_avg-min(d_arr))


human_bl = [HUMAN_BASELINE['dailydialog'][-1],HUMAN_BASELINE['movie'][-1],HUMAN_BASELINE['npr'][-1]]
random_bl = [RAND_BASELINE['dailydialog'][-1],RAND_BASELINE['movie'][-1],RAND_BASELINE['npr'][-1]]
all_novel = [random_bl, human_bl, gemma_pt_avg, gemma_it_avg,llama_pt_avg,llama_it_avg]
bl_range = [0,0,0]
all_max = [bl_range, bl_range, gemma_pt_max, gemma_it_max,llama_pt_max,llama_it_max]
all_min = [bl_range, bl_range, gemma_pt_min, gemma_it_min,llama_pt_min,llama_it_min]

colors = ["#8F8F8F", "#525252", "#6DB2EE", "#1A81DB", "#ED6E9C", "#D81B60"]

all_utter = [random_bl, human_bl]
bl_range = [0,0,0]
all_max = [bl_range, bl_range]
all_min = [bl_range, bl_range]
colors = ["#8F8F8F", "#525252"]

x = np.arange(len(datasets))  # the label locations
width = 0.4 # the width of the bars
multiplier = -0.5

#plt.figure(figsize=(10,5))
plt.figure(figsize=(5,5))
for n, data, upper, lower, c in zip(names, all_novel, all_max, all_min, colors):
	offset = width * multiplier
	plt.bar(x + offset, data, width, yerr=[lower, upper], label=n, color=c)
	multiplier += 1

plt.ylabel('% Novel Tokens')
plt.xticks(x, datasets)
plt.ylim(0.4, 0.8)
plt.tight_layout()
plt.savefig('novel_baseline_bar.png', dpi=800)
plt.clf()

#EOF