from fastai.conv_builder import *
from lib.labels import label_dict


def classification_model(arch=None, **kwargs):
	opts = dict(is_multi=False, is_reg=False, pretrained=False)
	if arch is None: arch = resnext50
	conv = ConvnetBuilder(arch, len(label_dict.values()), **opts, **kwargs)
	conv.model.eval()
	return conv.model