import os, json


with open(os.environ['LABELS_PATH']) as f:
	idx_labels = [(int(i[0]), i[1]) for i in json.loads(f.read()).items()]
	f.close()

label_dict = dict(sorted(idx_labels, key=lambda i: i[0]))
