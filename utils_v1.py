#!/usr/bin/env python

# environment = readlif with nd2 installed
# 
# https://pypi.org/project/nd2/

import nd2
import os
import numpy as np
import cv2
from skimage.exposure import rescale_intensity

def convert_and_scale(image, convert_16_to_8bit, scale_image, auto_scale, scale_min, scale_max, multichan):
	"converts 16bit image numpy array (x, y, channel) to 8bit, if asked, and applies scaling factors"
	if convert_16_to_8bit:
		image = (image/256).astype('uint8')
	if multichan:
		if scale_image and auto_scale:
			for i in range(image.shape[2]): # scale each channel
				image[:,:,i] = rescale_intensity(image[:,:,i], in_range='image', out_range='dtype')
		elif scale_image:
			for i in range(image.shape[2]): # scale each channel
				image[:,:,i] = rescale_intensity(image[:,:,i], in_range=(scale_min[i], scale_max[i]), out_range='dtype')
	else:
		if scale_image and auto_scale:
			image = rescale_intensity(image, in_range='image', out_range='dtype')
		elif scale_image:
			image = rescale_intensity(image, in_range=(scale_min[0], scale_max[0]), out_range='dtype')
	return image
	
def laplacian_variance(img):
	return np.var(cv2.Laplacian(img, cv2.CV_64F, ksize=21))

def find_best_z_plane_id(img_list):
	"takes a list of z planes and return most in focus z by laplacian variance"
	if len(img_list) == 1: # if single z, just return the z plane
		max_var = 0
	else:
		lap_vars = []
		for img in img_list:
			lap_vars.append(laplacian_variance(img))
		max_var = lap_vars.index(max(lap_vars)) # z plane with max laplacian variance = most in focus (sharpest edges)
	return max_var

#def auto_z_slices(img_list):
#
#	 use laplacian variance to determine which z slices are good
#	 
#	 return same output as make_max_projection

def make_max_projection(img_list, start_slice, end_slice):
	if end_slice == 'end':
		img_list = img_list[start_slice:]
	else:
		img_list = img_list[start_slice:end_slice]
	return np.max(img_list, axis=0)

def multiple_channels(image_array, z_stack):
	multiple = False
	if z_stack:
		if image_array.ndim == 4: # this means that there is more than 1 channel
			multiple = True
	elif image_array.ndim == 3: # this means that there is more than 1 channel
		multiple = True
	return multiple

def get_single_z(image, convert_16_to_8bit, scale_image, auto_scale, scale_min, scale_max, z_stack, find_best_z, max_projection, auto_slice, start_slice, end_slice, multichan):
	"takes image numpy array in the shape of (z, channels, x, y) and outputs numpy array of best z plane for each channel in the shape of (x, y, channel); also converts and scales; this is the key function"
	# assumes image is a z stack
		
	if multichan:
		if image.dtype=='uint16':
			composite = np.full((image.shape[-2], image.shape[-1], image.shape[-3]), 0, dtype=np.uint16)
		elif image.dtype=='uint8':
			composite = np.full((image.shape[-2], image.shape[-1], image.shape[-3]), 0, dtype=np.uint8)
		for i in range(image.shape[-3]): # for each channel of z-stack; only works if image is a z stack
			z_list = [image[k, i] for k in range(image.shape[0])]
			if find_best_z:				   
				best_z = find_best_z_plane_id(z_list)
				composite[:,:,i] = z_list[best_z]
			elif max_projection:
				#if auto_slice:
					#best_z = auto_z_slices(z_list)
				best_z = make_max_projection(z_list, start_slice, end_slice)
				composite[:,:,i] = best_z
	else:
		if image.dtype=='uint16':
			composite = np.full((image.shape[-2], image.shape[-1]), 0, dtype=np.uint16)
		elif image.dtype=='uint8':
			composite = np.full((image.shape[-2], image.shape[-1]), 0, dtype=np.uint8)
		z_list = [image[k] for k in range(image.shape[0])]
		if find_best_z:				   
			best_z = find_best_z_plane_id(z_list)
			composite = z_list[best_z]
		elif max_projection:
			#if auto_slice:
				#best_z = auto_z_slices(z_list)
			best_z = make_max_projection(z_list, start_slice, end_slice)
			composite = best_z

	composite = convert_and_scale(composite, convert_16_to_8bit, scale_image, auto_scale, scale_min, scale_max, multichan)
	
	return composite

def extract_from_nd2(filename, inpath, dir_name, convert_16_to_8bit, scale_image, auto_scale, scale_min, scale_max, z_stack, find_best_z, max_projection, auto_slice, start_slice, end_slice, return_composite, channel_order, return_channels, extension):
	# Append output file name with settings
	name_addon = ''
	if convert_16_to_8bit:
		name_addon += '_8bit'
	else:
		name_addon += '_16bit'
	if scale_image:
		if auto_scale:
			name_addon += '_autoScale'
		else:
			min_string = ''
			max_string = ''
			for no in scale_min:
				min_string += '-'+str(no)
			for no in scale_max:
				max_string += '-'+str(no)
			name_addon += '_manScale_'+min_string+'_'+max_string
	if z_stack:
		if find_best_z:
			name_addon += '_bestZ'
		elif max_projection:
			name_addon += '_maxProj_'+str(start_slice)+'-'+str(end_slice)
		
	img = nd2.imread(os.path.join(inpath, filename))
	
	multichan = multiple_channels(img, z_stack)
	
	if z_stack:
		comp = get_single_z(img, convert_16_to_8bit, scale_image, auto_scale, scale_min, scale_max, z_stack, find_best_z, max_projection, auto_slice, start_slice, end_slice, multichan)
	else:
		comp = convert_and_scale(img, convert_16_to_8bit, scale_image, auto_scale, scale_min, scale_max, multichan)
	
	if multichan == False:
		cv2.imwrite(os.path.join(inpath, dir_name, filename[:-4]+name_addon+extension), comp)
	else:
		if return_channels:
			for i in range(comp.shape[2]):
				cv2.imwrite(os.path.join(inpath, dir_name, filename[:-4]+'_channel'+str(i)+name_addon+extension), comp[:,:,i])
		if return_composite:
			channel_dict = {'red':2, 'green':1, 'blue':0} # default channel order is BGR
			channel_order_translated = [channel_dict[x] for x in channel_order]
			if comp.dtype=='uint16':
				composite = np.full((comp.shape[0], comp.shape[1], 3), 0, dtype=np.uint16)
			elif comp.dtype=='uint8':
				composite = np.full((comp.shape[0], comp.shape[1], 3), 0, dtype=np.uint8)
			for i in range(min(comp.shape[2], 3)):
				composite[:,:,channel_order_translated[i]] = comp[:,:,i]
			cv2.imwrite(os.path.join(inpath, dir_name, filename[:-4]+'_composite'+name_addon+extension), composite)
	# if name is too long, won't output file
	print('.', end='', flush=True)