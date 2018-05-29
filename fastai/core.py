from .imports import *
from .torch_imports import *

from .layers import *
from .model import *
from .initializers import *

def sum_geom(a,r,n): return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))

def is_listy(x): return isinstance(x, (list,tuple))
def is_iter(x): return isinstance(x, collections.Iterable)
def map_over(x, f): return [f(o) for o in x] if is_listy(x) else f(x)
def map_none(x, f): return None if x is None else f(x)


conv_dict = {np.dtype('int8'): torch.LongTensor, np.dtype('int16'): torch.LongTensor,
             np.dtype('int32'): torch.LongTensor, np.dtype('int64'): torch.LongTensor,
             np.dtype('float32'): torch.FloatTensor, np.dtype('float64'): torch.FloatTensor}


def A(*a):
	"""convert iterable object into numpy array"""
	return np.array(a[0]) if len(a) == 1 else [np.array(o) for o in a]


def T(a, half=False, cuda=True):
	"""
	Convert numpy array into a pytorch tensor.
	if Cuda is available and USE_GPU=ture, store resulting tensor in GPU.
	"""
	if not torch.is_tensor(a):
		a = np.array(np.ascontiguousarray(a))
		if a.dtype in (np.int8, np.int16, np.int32, np.int64):
			a = torch.LongTensor(a.astype(np.int64))
		elif a.dtype in (np.float32, np.float64):
			a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
		else:
			raise NotImplementedError(a.dtype)
	if cuda: a = to_gpu(a, async=True)
	return a


def create_variable(x, volatile, requires_grad=False):
	if type(x) != Variable:
		if IS_TORCH_04:
			x = Variable(T(x), requires_grad=requires_grad)
		else:
			x = Variable(T(x), requires_grad=requires_grad, volatile=volatile)
	return x


def V_(x, requires_grad=False, volatile=False):
	"""equivalent to create_variable, which creates a pytorch tensor. """
	return create_variable(x, volatile=volatile, requires_grad=requires_grad)


def V(x, requires_grad=False, volatile=False):
	"""creates a single or a list of pytorch tensors, depending on input x. """
	return map_over(x, lambda o: V_(o, requires_grad, volatile))


def VV_(x):
	"""creates a volatile tensor, which does not require gradients. """
	return create_variable(x, True)


def VV(x):
	"""creates a single or a list of pytorch tensors, depending on input x. """
	return map_over(x, VV_)


def to_np(v):
	"""returns an np.array object given an input of np.array, list, tuple, torch variable or tensor."""
	if isinstance(v, (np.ndarray, np.generic)): return v
	if isinstance(v, (list, tuple)): return [to_np(o) for o in v]
	if isinstance(v, Variable): v = v.data
	if isinstance(v, torch.cuda.HalfTensor): v = v.float()
	return v.cpu().numpy()


IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
USE_GPU = torch.cuda.is_available()


def to_gpu(x, *args, **kwargs):
	"""puts pytorch variable to gpu, if cuda is avaialble and USE_GPU is set to true. """
	return x.cuda(*args, **kwargs) if USE_GPU else x


def noop(*args, **kwargs): return


def split_by_idxs(seq, idxs):
	"""A generator that returns sequence pieces, seperated by indexes specified in idxs."""
	last = 0
	for idx in idxs:
		if not (-len(seq) <= idx < len(seq)):
			raise KeyError(f'Idx {idx} is out-of-bounds')
		yield seq[last:idx]
		last = idx
	yield seq[last:]


def one_hot(a, c): return np.eye(c)[a]


def partition(a, sz):
	"""splits iterables a in equal parts of size sz"""
	return [a[i:i+sz] for i in range(0, len(a), sz)]


def partition_by_cores(a):
	return partition(a, len(a)//num_cpus()+1)


def num_cpus():
	try:
		return len(os.sched_getaffinity(0))
	except AttributeError:
		return os.cpu_count()
