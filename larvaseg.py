#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: Larva segmentation

:Authors: (c) Artem Lutov <lua@lutan.ch>
:Date: 2020-08-120
"""
import os
import cv2 as cv
import numpy as np
# from scipy.io import loadmat
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import iglob
# import json


#class NumpyDataEncoder(json.JSONEncoder):
#	"""JSON serializer of numpy types"""
#	def default(self, obj):
#		if isinstance(obj, np.integer):
#			return int(obj)
#		elif isinstance(obj, np.floating):
#			return float(obj)
#		elif isinstance(obj, np.ndarray):
#			return obj.tolist()
#		return super(NumpyDataEncoder, self).default(obj)


if __name__ == '__main__':
	parser = ArgumentParser(description='Larva segmentation.',
		formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
	parser.add_argument("-d", "--outp-dir", default=None,
		help='Output directory. The input directory is used by default.')
	parser.add_argument('input', metavar='INPUT', nargs='+',
		help='Wildcards of input video files to be processed')
	args = parser.parse_args()

	# # Load annotations
	# 0			area
	# 1			perimeter
	# 2			spine length
	# 3			fluorescence
	# 4			frame number
	# 5			larval number (identity)
	# 6			x coordinate
	# 7			y coordinate
	# B7Larva1 .. B7Larva4: array[array]]
	# anno = loadmat('v44_v45/B7L.mat')

	# Prepare output directory
	if args.outp_dir and not os.path.exists(args.outp_dir):
		os.makedirs(args.outp_dir)

	nfiles = 0
	for wlc in args.input:
		for ifname in iglob(wlc):
			nfiles += 1
			print('Processing {} ...'.format(ifname))
			#with open(ifname, 'rb') as finp:
			cap = cv.VideoCapture(ifname)
			fps = cap.get(cv.CAP_PROP_FPS)
			width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
			#ofname = oa.path.split(ifname)[1]
			ofname = os.path.join(args.outp_dir, 'v4_v5_f.05_clahe32.avi')
			print('	fps: {}, size: {}x{}\n\toutput: {}'.format(fps, width, height, ofname))
			# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
			# Define the fps to be equal to 10. Also frame size is passed.
			# FMP4, MJPG, XVID
			#fourcc = cv.VideoWriter_fourcc(*'MJPG')
			out = cv.VideoWriter(ofname, cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))
			if not cap:
				exit(1)
			title = os.path.split(ifname)[1]
			#cv.namedWindow(title, cv.WINDOW_NORMAL)
			#wWnd = 1600
			#rfont = width / wWnd
			#cv.resizeWindow(title, wWnd, int(height / rfont))
			#rfont = max(1, rfont)
			nframe = 0
			while cap.isOpened():
				# Capture frame-by-frame
				ret, frame = cap.read()
				if not ret:
					assert frame is None
					break
				nframe += 1
				#_, _, w, _ = cv.getWindowImageRect(title)
				#rfont = max(1, width / w)
				#cv.putText(frame,'Recording ...',
				#	(20, 20 + int(24 * rfont)),  # bottomLeftCornerOfText
				#	cv.FONT_HERSHEY_SIMPLEX, 
				#	rfont,  # Font size
				#	(7, 7, 255),  # Color
				#	1 + round(rfont)) # Line thickness, px
				cv.imshow(title, frame)
				# Convert to grayscale
				gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
				# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
				clahe = cv.createCLAHE(clipLimit=32.0, tileGridSize=(8,8))  # clipLimit=40 / 255  Threshold for contrast limiting.
				gray = clahe.apply(gray)
				# Adaprive optimal thresholds: THRESH_OTSU, THRESH_TRIANGLE
				ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # THRESH_TRIANGLE
				cv.imshow(title + '_thresh', thresh)
				# cv.imshow(title + '_gray', frame)

				# Noise removal
				kernel = np.ones((3, 3), np.uint8)  # 3, 3 
				opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2) # orig: 2
				# opening = cv.dilate(thresh, kernel, iterations=2) # 3
				cv.imshow(title + '_opening', opening)
				# opening = thresh
				# # Background identification
				sure_bg = cv.dilate(opening, kernel, iterations=3) # orig: 3
				# #sure_bg = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel, iterations=2) # 2
				# cv.imshow(title + '_bg', sure_bg)
				# # sure_bg = opening
				# Foreground identification
				dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)  # 5 or 3
				ret, sure_fg = cv.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)  # 0.7 [0.5 - deliv]]
				# Remaining areas
				sure_fg = np.uint8(sure_fg)
				unknown = cv.subtract(sure_bg, sure_fg)
				# cv.imshow(title + '_unknown', unknown)
				
				# Marker labelling
				ret, markers = cv.connectedComponents(sure_fg)
				# Add one to all labels so that sure background is not 0, but 1
				markers = markers + 1
				# Mark the region of unknown with zero
				markers[unknown == 255] = 0
				# Apply watershed
				markers = cv.watershed(frame, markers)
				frame[markers == -1] = [255,0,0]
				cv.imshow(title + '_gray', frame)
				
				# Save output video
				out.write(frame)
				
				#cv.imwrite("frame%d.jpg" % count, frame)
				# Wait 10 ms for a 'q' to exit or capture the next frame
				key = cv.waitKey(10) & 0xFF
				# Quit: escape, e or q
				if key in (27, ord('e'), ord('q')):
					break
				# Pause: Spacebar or p
				elif key in (32, ord('p')):
					cv.waitKey(-1)
					
			cap.release()
			out.release()
			cv.destroyAllWindows() # destroy all opened windows
