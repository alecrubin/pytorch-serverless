from fastai.conv_builder import *
from lib.utils import get_labels

LABELS_PATH = os.environ['LABELS_PATH']

def classification_model(arch=None, **kwargs):
	opts = dict(is_multi=False, is_reg=False, pretrained=False)
	n_labels = len(get_labels(LABELS_PATH))
	if arch is None: arch = resnext50
	conv = ConvnetBuilder(arch, n_labels, **opts, **kwargs)
	conv.model.eval()
	return conv.model