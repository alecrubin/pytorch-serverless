import os
import boto3
import cv2
import numpy as np
import urllib.request


s3_client = boto3.client('s3')


def download_file(bucket_name, object_key_name, file_path):
	s3_client.download_file(bucket_name, object_key_name, file_path)


def get_s3_object(bucket_name, object_key_name, metadata=False):
	resp = s3_client.get_object(Bucket=bucket_name, Key=object_key_name)
	assert resp['ResponseMetadata']['HTTPStatusCode'] == 200, resp
	if metadata:
		return resp['Body'].read(), resp['Metadata']
	else:
		return resp['Body'].read()


def open_image_url(url):
	flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
	resp = urllib.request.urlopen(str(url))
	try:
		im = np.asarray(bytearray(resp.read()))
		im = cv2.imdecode(im, flags).astype(np.float32)/255
		if im is None: raise OSError(f'File from url not recognized by opencv: {url}')
		return im
	except Exception as e:
		raise OSError('Error handling image from url at: {}'.format(url)) from e


def open_image(fn):
	""" Opens an image using OpenCV given the file path.

	Arguments:
			fn: the file path of the image

	Returns:
			The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
	"""
	flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
	if not os.path.exists(fn):
		raise OSError('No such file or directory: {}'.format(fn))
	elif os.path.isdir(fn):
		raise OSError('Is a directory: {}'.format(fn))
	else:
		try:
			im = cv2.imread(str(fn), flags).astype(np.float32)/255
			if im is None: raise OSError(f'File not recognized by opencv: {fn}')
			return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		except Exception as e:
			raise OSError('Error handling image at: {}'.format(fn)) from e
