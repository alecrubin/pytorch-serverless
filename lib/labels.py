import os, json, collections


with open(os.environ['LABELS_PATH']) as f:
	d = dict(map(lambda k: (int(k[0]), k[1]), json.loads(f.read()).items()))
	f.close()

label_dict = collections.OrderedDict(sorted(d.items(), key=lambda i: i[0]))
