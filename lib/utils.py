import os
import boto3
import cv2
import numpy as np
import urllib.request


s3_client = boto3.client('s3')


def download_file(bucket_name, object_key_name, file_path):
	"""  Downloads a file from an S3 bucket.
	:param bucket_name: S3 bucket name
	:param object_key_name: S3 object key name
	:param file_path: path to save downloaded file
	"""
	s3_client.download_file(bucket_name, object_key_name, file_path)


def get_labels(path):
	"""  Get labels from a text file.
	:param path: path to text file
	:return: list of labels
	"""
	with open(path, encoding='utf-8', errors='ignore') as f:
		labels = [line.strip() for line in f.readlines()]
		f.close()
	return labels


def open_image_url(url):
	"""  Opens an image using OpenCV from a URL.
	:param url: url path of the image
	:return: the image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
	"""
	flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
	url = str(url)
	resp = urllib.request.urlopen(url)
	try:
		im = np.asarray(bytearray(resp.read()))
		im = cv2.imdecode(im, flags).astype(np.float32)/255
		if im is None: raise OSError(f'File from url not recognized by opencv: {url}')
		return im
	except Exception as e:
		raise OSError(f'Error handling image from url at: {url}') from e


def open_image(path):
	""" Opens an image using OpenCV given the file path.
	:param path: the file path of the image
	:return: the image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
	"""
	flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
	path = str(path)
	if not os.path.exists(path):
		raise OSError(f'No such file or directory: {path}')
	elif os.path.isdir(path):
		raise OSError(f'Is a directory: {path}')
	else:
		try:
			im = cv2.imread(str(path), flags).astype(np.float32)/255
			if im is None: raise OSError(f'File not recognized by opencv: {path}')
			return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		except Exception as e:
			raise OSError(f'Error handling image at: {path}') from e
