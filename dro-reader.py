#!/usr/bin/env python
#SEC, 2015
#derived from http://opencvpython.blogspot.com/2012/04/simple-digit-recognition-ocr-in-opencv.html
import numpy as np
import cv2
import argparse
import sys
import time

def magnitudes(v):
	return np.sqrt(np.sum(v**2,axis=-1))
def close(a,b):
  return absolute(a-b)<1e-5


def find_perspective_markers(args):
	cap = cv2.VideoCapture(args.video_filename+'.mov')
	ret,frame = cap.read()
	def print_coords(event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN:
			print "[%d,%d]"%(x,y)
			#cv2.circle(img,(x,y),100,(255,0,0),-1)
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',print_coords)
	cv2.imshow('image',frame)
	cv2.waitKey(0)


def get_contours(args,frame):
	rows,cols,ch = frame.shape
	H = args.height; W = args.width;
	gray = frame
	gray[...,2] = 0 #set red to zero to make green and yellow both green
	gray = 1-gray[...,1] #pick out green only
	#preimage = np.float32([[137,83],[1016,55],[193,516],[989,474]])
	preimage = np.float32([[154,144],[647,130],[191,445],[656,421]])
	target = np.float32([[0,0],[W,0],[0,H],[W,H]])
	M = cv2.getPerspectiveTransform(preimage,target)
	frame = cv2.warpPerspective(frame,M,(W,H))
	gray = cv2.warpPerspective(gray,M,(W,H))
	
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,W+1,40)
	thresh = cv2.dilate(thresh,np.ones((2,2),np.uint8),iterations=2)

	if args.mode=='tune':
		cv2.imshow('tune',thresh)
		cv2.waitKey(0)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	#print 'found %d contours'%len(contours)
	return frame,thresh,contours

def train(args):
	cap = cv2.VideoCapture(args.video_filename+'.mov')
	samples =  np.empty((0, args.roi_side*args.roi_side))
	responses = []

	for fn in range(args.n_training_frames):
		for skip in range(args.n_skip_frames): cap.grab()
		ret,frame = cap.read() #remember, B,G,R order
		frame,thresh,contours = get_contours(args,frame)
		
		keys = [i for i in range(48, 58)] # keyboard mappings for 0-9; user may type in this range when prompted
		key = 0
		H = args.height; W = args.width;
		for contour in contours:
			if 4000 > cv2.contourArea(contour) > 300:
				[x, y, w, h] = cv2.boundingRect(contour)
				#if (H/4 > h > H/6) and (W/8 > w > W/30) and (x > w/3):
				if (H/4 > h > H/6) and (x > W/3):
					cv2.rectangle(frame, (x+w/2-args.standard_w/2, y), (x+w/2+args.standard_w/2, y+h), (0, 0, 255), 2)
					cv2.drawContours(frame, contour, -1, (255,0,0), 3)
					roi = thresh[y:y+h, x:x+w]
					roi_small = cv2.resize(roi, (args.roi_side, args.roi_side))
					cv2.imshow('norm', frame)
					key = cv2.waitKey(0)
				if key == 27:
					sys.exit()
				elif key in keys:
					sample = roi_small.reshape((1,args.roi_side*args.roi_side))
					samples = np.append(samples,sample,0)
					# save input in 'responses'
					responses.append(int(chr(key)))
	print "training complete"
	np.savetxt('general-samples.data', samples)
	responses = np.array(responses, np.float32)
	responses = responses.reshape((responses.size,1))
	np.savetxt('general-responses.data', responses)

def classify(args):
	H = args.height; W = args.width;

	samples = np.loadtxt('general-samples.data', np.float32)
	responses = np.loadtxt('general-responses.data', np.float32)
	responses = responses.reshape((responses.size,1))
	start = time.time()
	model = cv2.KNearest()
	model.train(samples, responses)
	print 'training took %.3f s'%(time.time()-start)
	
	cap = cv2.VideoCapture(args.video_filename+'.mov')
	fps = 0;
	while True:
		start = time.time()
		ret, frame = cap.read()
		frame,thresh,contours = get_contours(args,frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		for contour in contours:
			if 4000 > cv2.contourArea(contour) > 400:
				[x, y, w, h] = cv2.boundingRect(contour)
				if (H/4 > h > H/6) and (x > W/3):
					cv2.rectangle(frame, (x+w/2-args.standard_w/2, y), (x+w/2+args.standard_w/2, y+h), (0, 0, 255), 2)
					roi = thresh[y:y+h, x:x+w]
					roi_small = cv2.resize(roi, (args.roi_side, args.roi_side))
					roi_small = roi_small.reshape((1,args.roi_side*args.roi_side))
					roi_small = np.float32(roi_small)
					retval, results, neigh_resp, dists = model.find_nearest(roi_small, k = 3)
					string = str(int((results[0][0])))
					cv2.putText(frame, string, (x-w/4, y+h), 2, 3, (255, 0, 0))	
		fps = (1-args.alpha)*fps + args.alpha/(time.time()-start)
		print "frame rate: %.1f fps"%fps
		cv2.imshow('Video', frame)
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-M','--mode',choices=('find_markers','tune','train','classify'))
	parser.add_argument("-f","--video_filename", default='edm-readout', help="Base filename for video input (no extension, assumed .mov)")
	parser.add_argument("-nf","--n_training_frames",type=int,default=2,help="Number of frames to train on")
	parser.add_argument("-sf","--n_skip_frames",type=int,default=100,help="Number of frames to skip between training frames")
	parser.add_argument("-W","--width",type=int,default=600,help="width of corrected panel")
	parser.add_argument("-H","--height",type=int,default=400,help="height of corrected panel")
	parser.add_argument("-a","--alpha",type=float,default=.1,help="smoothing factor for fps measurement")
	parser.add_argument("-roi","--roi_side",type=int,default=20,help="nxn pixels for roi for KNN")
	parser.add_argument("-sw","--standard_w",type=int,default=38,help="width of a number")
	args = parser.parse_args()
	if args.mode=='train':
		train(args)
	elif args.mode=='tune':
		cap = cv2.VideoCapture(args.video_filename+'.mov')
		ret,frame = cap.read() #remember, B,G,R order
		get_contours(args,frame)
	elif args.mode=='find_markers':
		find_perspective_markers(args)
	elif args.mode=='classify':
		classify(args)