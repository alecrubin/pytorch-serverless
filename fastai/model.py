from .torch_imports import children


def cut_model(m, cut):
	return list(m.children())[:cut] if cut else [m]


def num_features(m):
	c = children(m)
	if len(c) == 0: return None
	for l in reversed(c):
		if hasattr(l, 'num_features'): return l.num_features
		res = num_features(l)
		if res is not None: return res
