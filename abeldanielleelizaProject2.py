# Authors: Imoleayo Abel, Eliza Bailey, and Danielle Sullivan.
# ENGR 027: Computer Vision (Spring 2013)
# Project 2
#
# March 25th 2013


import cv2, numpy, sys, math, cvk2, Tkinter, struct
from random import choice

####################################################################################################

# Try to get an integer argument:
try:
    device = int(sys.argv[1])
    del sys.argv[1]
except (IndexError, ValueError):
    device = 0

# If we have no further arguments, open the device. Otherwise, get the filename.
if len(sys.argv) == 1:
    capture = cv2.VideoCapture(device)
    if capture:
        print 'Opened device number', device
else:
    capture = cv2.VideoCapture(sys.argv[1])
    if capture:
        print 'Opened file', sys.argv[1]
# Bail if error.
if not capture:
    print 'Error opening video capture!'
    sys.exit(1)

# Fetch the first frame and bail if none.
ok, frame = capture.read()
if not ok or frame is None:
    print 'No frames in video'
    sys.exit(1)

####################################################################################################

fps = 10
fourcc, ext = (struct.unpack('i', 'DIVX')[0], 'avi')
videofilename = 'abeldanielleelizaProject2.' + ext
videofilename2 = 'abeldanielleelizaProject2Warped.' + ext
writer = cv2.VideoWriter(videofilename, fourcc, fps, (frame.shape[1], frame.shape[0]))
writerWarped = cv2.VideoWriter(videofilename2, fourcc, fps, (frame.shape[1], frame.shape[0]))
if (not writer) or (not writerWarped):
    print 'Error opening writer'
else:
    print 'Opened', videofilename, 'and', videofilename2, 'for output.'

####################################################################################################
#							 #																	
# 		TIC-TAC-TOE A.I. 	 #
#							 #
##############################

# winning combos for tic-tac-toe
win_combo = numpy.array([
                         [1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9],
                         [1, 4 ,7],
                         [2, 5, 8],
                         [3, 6, 9],
                         [1, 5, 9],
                         [3, 5, 7],
                         ])

####################################################################################################

def xor(a,b):
    # xor helper function
    if a == b:
        return False
    else:
        return True

####################################################################################################

def check_fit_combo(w_combo, player, opponent):

	position = 99

	# check if player has 2 squares in any of the winning combos. If so, return 
	# 3rd box if free
	for success_triple in w_combo:
		poss = []     # array of items player has in current winning set
		for item in player:
			# check if item is in current winning set
			if item in success_triple:
				poss.append(item)
				# once you have 2 items, return missing third item if it is free
				if len(poss) == 2:
					# check if 3rd box required to win is free 
					for item in success_triple:
						if item not in (poss + player + opponent):
							return item
	return position 


####################################################################################################

def pickMove(human_squares, comp_squares):
    
    # SCENARIO A: if the board is empty
    if len(human_squares) == 0 and len(comp_squares) == 0:
        return choice([1,3,7,9])			# return position 1, 3, 7 or 9 randomly

    # SCENARIO B: if it is the second turn, and the human player went first
    if (len(human_squares) == 1) and (len(comp_squares) == 0):
        if (human_squares[0] in [1,3,7,9]): return 5
        return choice([1,3,7,9])		# return position 1 3 7 or 9 randomly

    # SCENARIO C - offense - check if comp could possibly win
    offense_position = check_fit_combo(win_combo, comp_squares, human_squares)
    if offense_position != 99: return offense_position

    # SCENARIO D1 - defense - check if the human player chould possibly win
    defense_position = check_fit_combo(win_combo, human_squares, comp_squares)                      
    if defense_position != 99: return defense_position

    # SCENARIO D2 - defensive strategic move
    if (len(comp_squares) == 1) and (comp_squares[0] == 5) and (len(human_squares) == 2):
        if (2 in human_squares) and ((7 in human_squares) or (9 in human_squares)):
            return choice[1,3]
        if (4 in human_squares) and ((3 in human_squares) or (9 in human_squares)):
            return choice[1,7]
        if (6 in human_squares) and ((1 in human_squares) or (7 in human_squares)):
            return choice[3,9]
        if (8 in human_squares) and ((1 in human_squares) or (3 in human_squares)):
            return choice[7,9]

	# SCENARIO E1 - offensive strategic move
    if (len(comp_squares) == 1) and (len(human_squares) == 1) and (human_squares[0] == 5):
        content = comp_squares[0]
        possibility1 = [1,3,7,9][[9,7,3,1].index(content)]
        possibility2 = choice([[6,8],[4,8],[2,6],[2,4]][[1,3,7,9].index(content)])
        return choice([possibility1,possibility2])

    # important strategic info: are boxes 1, 3, 7, 9 free?
    one_free = False
    three_free = False
    seven_free = False
    nine_free = False           
    if (1 not in human_squares) and (1 not in comp_squares): one_free = True
    if (3 not in human_squares) and (3 not in comp_squares): three_free = True
    if (7 not in human_squares) and (7 not in comp_squares): seven_free = True
    if (9 not in human_squares) and (9 not in comp_squares): nine_free = True

    # SCENARIO E2 - offensive stategic move II
    if xor(9 in comp_squares, 1 in comp_squares) and (three_free or seven_free):
        if three_free and seven_free: return choice([3,7])
        if (not seven_free) and three_free: return 3
        if (not three_free) and seven_free: return 7

    # SCENARIO E3 - offensive strategic move III
    if xor(3 in comp_squares, 7 in comp_squares) and (nine_free or one_free):
        if nine_free and one_free: return choice([1,9])        
        if (not nine_free) and one_free: return 1                
        if (not one_free) and nine_free: return 9

    # F - random selection 
    L = [1,2,3,4,5,6,7,8,9]
    while 1:
        pos1 = choice(L)
        if ((pos1 not in human_squares)  and (pos1 not in comp_squares)):
            return pos1
        else:
        	L.remove(pos1)

#####################################
#  		End of TIC-TAC-TOE A.I. 	#           
####################################################################################################
####################################################################################################

# get dimensions of display
screen = Tkinter.Tk()
screenWidth = screen.winfo_screenwidth()
screenHeight = screen.winfo_screenheight()
screen.destroy()

# make dimension of game board half the size of shorter screen dimension
if screenHeight < screenWidth:
	shorter = screenHeight
else:
	shorter = screenWidth
outBox = int(shorter/2)						

# coordinates of points to be projected for computing homography
p7 = (screenWidth/2 - outBox/4, screenHeight - outBox/3) 
p1 = (p7[0], p7[1] - outBox)
p2 = (p1[0] + outBox/2, p1[1])
p3 = (p1[0] + outBox, p1[1])
p4 = (p1[0], p1[1] + outBox/2)
p5 = (p2[0], p4[1])
p6 = (p3[0], p4[1])
p8 = (p2[0], p7[1])
p9 = (p3[0], p7[1])
projectedPoints = [p1,p2,p3,p4,p5,p6,p7,p8,p9]

# some color definitions
white = (255,255,255)
black = (0,0,0)
red = (0,0,255)
blue = (255,0,0)

####################################################################################################

def updateStatus(image,text):

	# updates status message at the top of game board

	global p1, white, black, outBox, screenWidth

	# end positions of bounding rectangle for game status
	feedPosTopLeft = (p1[0] - outBox/6, p1[1] - outBox/3)
	feedPosBottomRight = (screenWidth, p1[1] - outBox/6)  

	# first overlay white rectangle on possibly old feed
	feedBox = cv2.rectangle(image, feedPosTopLeft, feedPosBottomRight, white, -1)

	# bottom left position of text
	text_pos = (feedPosTopLeft[0] + 16, feedPosBottomRight[1] - 16)

	# then add updated text
	cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
		   		0.6, black, 1, cv2.CV_AA)
   
	return image 

####################################################################################################

def updateGameFeed(image,feed_list):

	# feedlist holds [event1String, event1Color, event2String, event2Color, ....]

	global p1, white, black, outBox, globalfeedPosTopLeft
	global globalfeedPosBottomRight, num_lines, feed_height
	
	feedBackGroundColor = (192,192,192)

	# end positions of bounding box for game feed
	globalfeedPosTopLeft = (60, p1[1])
	globalfeedPosBottomRight = (p7[0] - 150, p7[1] + 50)

	# draw background retangle
	cv2.rectangle(image, globalfeedPosTopLeft, globalfeedPosBottomRight, feedBackGroundColor, -1)
	# draw line separating feed title and actual feed
	cv2.line(image,globalfeedPosTopLeft,(globalfeedPosBottomRight[0],globalfeedPosTopLeft[1]),
		     black,5,cv2.CV_AA)
	# bottom left position of title of feed
	title_pos = (globalfeedPosTopLeft[0] + 16, globalfeedPosTopLeft[1] - 16)
	title = "Game Feed:"
	# include feed title
	cv2.putText(image, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, black, 1, cv2.CV_AA)

	num_lines = 5  						  # desired number of feeds (event queue) to show
	feed_height = outBox/9  			  # height of bounding rectangle of one event

	# feed_list array holds all events so far, so we have to decide what index of the array we're
	# going to start reading feed from. This simulates the scrolling effect
	if len(feed_list) > 2*num_lines:
		start_index = (len(feed_list))/2 - num_lines
	else:
		start_index = 0

	n_feed = min(len(feed_list)/2,num_lines) # determine how many lines of feed to display

	# display feeds
	for i in range(start_index, start_index + n_feed):
		feedPosTopLeft = (globalfeedPosTopLeft[0], 
			              globalfeedPosTopLeft[1] + ((i-start_index)*feed_height))
		feedPosBottomRight = (globalfeedPosBottomRight[0], 
			                  globalfeedPosTopLeft[1] + ((i-start_index+1)*feed_height))
		text_pos = (feedPosTopLeft[0] + 16, feedPosBottomRight[1] - 16)
		feedBox = cv2.rectangle(image, feedPosTopLeft, feedPosBottomRight, feedBackGroundColor, -1)
		cv2.putText(image, feed_list[2*i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
			        0.6, feed_list[(2*i) + 1], 1, cv2.CV_AA)

	return image 

####################################################################################################

def makeXTemplates(size):

	# size is length of side of the largest square template
	out = []

	# make five template sizes
	for i in range(5):
		modified_size = int((0.3 + (0.2*i))*size)	# update size of template

		# for each size, make a thin, normal, and thick template
		for j in range(3):
			thickness = (2*j) + 1
			# 4 endpoints of 'X'
			topleft = (int(0.1*modified_size),int(0.1*modified_size))	  
			bottomright = (int(0.9*modified_size),int(0.9*modified_size))  
			topright = (bottomright[0], topleft[1])
			bottomleft = (topleft[0], bottomright[1])
			# make template
			template = numpy.zeros((modified_size,modified_size),dtype='uint8')
			cv2.line(template,topleft,bottomright,white,thickness,cv2.CV_AA)
			cv2.line(template,bottomleft,topright,white,thickness,cv2.CV_AA)
			# add some blur (helps with template matching)
			blurred_temp = cv2.GaussianBlur(template, (0, 0), 4)
			out.append(blurred_temp)
			#cv2.imwrite("ADETemplateX/"+str((3*i)+j)+".png",blurred_temp)

	return out

####################################################################################################

def makeOTemplates(size):
	
	# size is length of side of the largest square template
	out = []
	curr = 0

	# make four different template sizes
	for i in range(5):
		modified_size = int((0.3 + (0.2*i))*size)

		# for each size, make a thin, normal and thick template
		for j in range(3):
			thickness = (2*j) + 1
			radius = int(0.4*modified_size)
			# make circular templates
			template = numpy.zeros((modified_size,modified_size),dtype='uint8')
			cv2.circle(template, (modified_size/2,modified_size/2), radius, 
					   (255,255,255), thickness, cv2.CV_AA)
			blurred_temp = cv2.GaussianBlur(template, (0, 0), 4)
			out.append(blurred_temp)
			#cv2.imwrite("./ADETemplateO/"+str(curr)+".png",blurred_temp)
			curr += 1
			# make elliptical templates
			angles = [0,45,90,135]
			eccentricity = 0.8
			for angle in angles:
				template = numpy.zeros((modified_size,modified_size),dtype='uint8')
				cv2.ellipse(template, (modified_size/2,modified_size/2), 
					        (int(eccentricity*radius),radius), angle, 0, 360, white, thickness)
				blurred_temp = cv2.GaussianBlur(template, (0, 0), 4)
				out.append(blurred_temp)
				#cv2.imwrite("./ADETemplateO/"+str(curr)+".png",blurred_temp)
				curr += 1	

	return out

####################################################################################################

def checkWin(b_dict):

	# uses board dictionary to check if a character has won, returns a list of winning
	# character and positions on board that constitute the win. Returns 0 if there's no winner

	# check that boxes 1,2,3 are filled with the same character
	if (b_dict[1] and b_dict[2] and b_dict[3] and (b_dict[1] == b_dict[2]) \
		and (b_dict[1] == b_dict[3])):
		return [b_dict[1],1,2,3]
	# check that boxes 4,5,6 are filled with the same character
	elif (b_dict[4] and b_dict[5] and b_dict[6] and (b_dict[4] == b_dict[5]) \
		and (b_dict[4] == b_dict[6])):
		return [b_dict[4],4,5,6]
	# check that boxes 7,8,9 are filled with the same character
	elif (b_dict[7] and b_dict[8] and b_dict[9] and (b_dict[7] == b_dict[8]) \
		and (b_dict[7] == b_dict[9])):
		return [b_dict[7],7,8,9]
	# check that boxes 1,4,7 are filled with the same character
	elif (b_dict[1] and b_dict[4] and b_dict[7] and (b_dict[1] == b_dict[4]) \
		and (b_dict[1] == b_dict[7])):
		return [b_dict[1],1,4,7]
	# check that boxes 2,5,8 are filled with the same character
	elif (b_dict[2] and b_dict[5] and b_dict[8] and (b_dict[2] == b_dict[5]) \
		and (b_dict[2] == b_dict[8])):
		return [b_dict[2],2,5,8]
	# check that boxes 3,6,9 are filled with the same character
	elif (b_dict[3] and b_dict[6] and b_dict[9] and (b_dict[3] == b_dict[6]) \
		and (b_dict[3] == b_dict[9])):
		return [b_dict[3],3,6,9]
	# check that boxes 1,5,9 are filled with the same character
	elif (b_dict[1] and b_dict[5] and b_dict[9] and (b_dict[1] == b_dict[5]) \
		and (b_dict[1] == b_dict[9])):
		return [b_dict[1],1,5,9]
	# check that boxes 3,5,7 are filled with the same character
	elif (b_dict[3] and b_dict[5] and b_dict[7] and (b_dict[3] == b_dict[5]) \
		and (b_dict[3] == b_dict[7])):
		return [b_dict[3],3,5,7]
	# no winner
	else:
		return 0

####################################################################################################

def updateWinBoard(image,color,w_combo):

	# draw line through winning triple

	if w_combo in [[1,2,3],[4,5,6],[7,8,9]]:
		start = (p1[0], p1[1] + outBox/6 + (w_combo[0]/3)*(outBox/3))
		end = (start[0] + outBox, start[1])
	elif w_combo in [[1,4,7],[2,5,8],[3,6,9]]:
		start = (p1[0] + outBox/6 + ((w_combo[0]-1)%3)*(outBox/3), p1[1])
		end = (start[0], start[1] + outBox)
	elif w_combo in [[1,5,9],[3,5,7]]:
		start = projectedPoints[w_combo[0] - 1]
		end = projectedPoints[8 - (w_combo[0] - 1)]

	cv2.line(image,start,end,color,5,cv2.CV_AA)

	return image

####################################################################################################

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))

# make fullscreen window to project to
cv2.namedWindow('Tic-Tac-Toe', cv2.cv.CV_WINDOW_NORMAL)
cv2.cv.MoveWindow('Tic-Tac-Toe',0,0)
cv2.cv.ResizeWindow('Tic-Tac-Toe',screenWidth,screenHeight)

# initialize variables defined within if-blocks so that they have a global scope
reference_background = 0
frame_grey = 0
mask = 0
instance_pos = []
n_instance = 8			# number of frames in which the position of a blob does not change before
						# we consider it an interesting object
cameraPoints = []		# array of position of projected points
n_frame = 1

backGround = numpy.empty((screenHeight,screenWidth),dtype='uint8') # image to be projected

####################################################################################################

# generate points for homography
while 1:
	cv2.rectangle(backGround,(0,0),(screenWidth,screenHeight),white,-1)	# white background
	if n_frame <= 21:
		# do nothing with captured frame to make sure camera settles 
		backGround = updateStatus(backGround,"system initializing"+((n_frame%3)+1)*".")
		cv2.imshow('Tic-Tac-Toe',backGround)
		cv2.waitKey(500)
		ok, frame = capture.read()
		if writer:
			writer.write(frame)
		if n_frame == 21:
			# set reference background
			ref_grey = numpy.empty((frame.shape[0],frame.shape[1]), 'uint8')                       
			cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY, ref_grey)
			reference_background = ref_grey
	else:
		raw_index = n_frame - 22
		index = raw_index/n_instance
		instance = raw_index%n_instance
		# break after 9th projected point has been registered
		if index > 8:
			backGround = updateStatus(backGround, "finishing initialization")
			cv2.imshow('Tic-Tac-Toe',backGround)
			break

		# make filled black circle
		cv2.circle(backGround, (projectedPoints[index]), 20, black, -1, cv2.CV_AA)
		# update system status
		backGround = updateStatus(backGround,"establishing homography"+(((n_frame-12)%3)+1)*".")
		cv2.imshow('Tic-Tac-Toe',backGround)		# project circle
		cv2.waitKey(1000/(n_instance))				# wait a few seconds
		ok, frame = capture.read()					# capture view with camera
		if writer:									# write to output video
			writer.write(frame)
		# convert frame to grayscale
		frame_grey = numpy.empty((frame.shape[0],frame.shape[1]), 'uint8')   
		cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY, frame_grey)
		# get eroded thresholded difference image
		diff = cv2.absdiff(frame_grey,reference_background)
		mask = numpy.zeros(diff.shape, 'uint8') 
		cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY, mask)
		mask_erode = cv2.erode(mask,kernel=element,iterations=1)
		temp = mask_erode.copy()			# temp copy for getting contour info
		
		contours = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		areas = []			# area of blobs in current instance
		pos = []			# centroid of bloba in current instance
		for i in range(len(contours[0])):
			try:
				info = cvk2.getcontourinfo(contours[0][i])
				areas.append(info['area'])
				pos.append(cvk2.a2ti(info['mean']))
			except:
				pass
		# if more than one blob is found, save the centroid of the maximum area blob
		if len(areas) > 0:
			instance_pos.append(pos[areas.index(max(areas))])
		# after enough instances of the same view, check if the centroid of the maximum
		# area blob changes beyond a given threshold
		if instance == (n_instance-1):
			dot_errorThreshold = 20
			dists = []   # array of distances between centroids in adjacent instances
			for j in range(len(instance_pos)):
				dists.append(math.sqrt(sum([(p-q)**2 for p,q in zip(instance_pos[j],
					                                  instance_pos[(j+1)%len(instance_pos)])])))
			# if there is no consistency in centroid of max area blobs in n_instance frames
			if (len(dists) == 0) or (max(dists) > dot_errorThreshold):
				instance_pos = []					# empty array of centroid of max area blobs
				n_frame -= instance 				# rewind frame count index
				k = cv2.waitKey(5)
				# Check for ESC hit:
				if k % 0x100 == 27:
					break
				continue	# ignore rest of loop and restart assertion of current projected 
							# circle's centroid
			# if there's consistency in n_instance frames, compute average centroid location
			avg_pos = tuple([sum(p)/len(p) for p in zip(*instance_pos)])
			cameraPoints.append(avg_pos)        # append to camera points
			instance_pos = []

	k = cv2.waitKey(5)
	# Check for ESC hit:
	if k % 0x100 == 27:
		break
	n_frame += 1

# get homography matrix 
H = cv2.findHomography(numpy.array(cameraPoints,dtype=numpy.float32), \
	                   numpy.array(projectedPoints,dtype=numpy.float32),0)[0]

####################################################################################################

# get boundary of ONLY game board in warped camera image
campoint0 = numpy.array(cameraPoints[0],dtype=numpy.float32)
campoint1 = numpy.array(cameraPoints[8],dtype=numpy.float32)
[[[xleft,yup]]] = cv2.perspectiveTransform(numpy.reshape(campoint0,(1,1,2)), H)
[[[xright,ydown]]] = cv2.perspectiveTransform(numpy.reshape(campoint1,(1,1,2)), H)

####################################################################################################

def getLoc(pos):

	# return corresponding game board box index of pos. Frame of reference of pos is the game 
	# board portion of the warped camera image with the top left corner of the board the origin

	global xleft, xright, yup, ydown

	# coordinate of pos
	x = pos[0]
	y = pos[1]

	# equation of four vertical lines and four horizontal lines defining board
	x1 = 0
	x2 = (xright - xleft)/3
	x3 = 2*(xright - xleft)/3
	x4 = xright - xleft

	y1 = 0
	y2 = (ydown - yup)/3
	y3 = 2*(ydown - yup)/3
	y4 = ydown - yup

	# get board box index
	if ((y1 <= y) and (y <= y2)):
		if ((x1 <= x) and (x <= x2)):
			return 1
		elif ((x2 <= x) and (x <= x3)):
			return 2
		elif ((x3 <= x) and (x <= x4)):
			return 3
	elif ((y2 <= y) and (y <= y3)):
		if ((x1 <= x) and (x <= x2)):
			return 4
		elif ((x2 <= x) and (x <= x3)):
			return 5
		elif ((x3 <= x) and (x <= x4)):
			return 6
	elif ((y3 <= y) and (y <= y4)):
		if ((x1 <= x) and (x <= x2)):
			return 7
		elif ((x2 <= x) and (x <= x3)):
			return 8
		elif ((x3 <= x) and (x <= x4)):
			return 9
	return 0

####################################################################################################

def getChar(image,XTemps,OTemps):

	# do template matching and return character of board: 'X', 'O' or invalid (0)

	XVals = []  # array of maximum values from matching with X templates
	OVals = []	# array of maximum values from matching with O templates

	# add maximum match values using all templates to appropraite array
	for template in XTemps:
		XVals.append(numpy.max(cv2.matchTemplate(image,template,cv2.cv.CV_TM_CCORR_NORMED)))
	for template in OTemps:
		OVals.append(numpy.max(cv2.matchTemplate(image,template,cv2.cv.CV_TM_CCORR_NORMED)))

	maxX = max(XVals)	# maximum X matxh value
	maxO = max(OVals)	# maximum O match value

	tauBigX = 0.62
	tauBigO = 0.3
	bandWidthX = 0.1
	bandWidthO = 0.05

	if (maxX > tauBigX) and ((maxX-maxO) > bandWidthX):
		return 'X'
	elif (maxO > tauBigO) and ((maxO-maxX) > bandWidthO):
		return 'O'
	else:
		return 0

####################################################################################################

# endpoints of lines on board
pb1 = (p1[0] + outBox/3, p1[1])
pb2 = (p1[0] + (2*outBox)/3, p1[1])
pb3 = (p1[0], p1[1] + outBox/3)
pb4 = (p3[0], p1[1] + outBox/3)
pb5 = (p1[0], p1[1] + (2*outBox)/3)
pb6 = (p3[0], p1[1] + (2*outBox)/3)
pb7 = (p1[0] + outBox/3, p7[1])
pb8 = (p1[0] + (2*outBox)/3, p7[1])

# dictionary to keep track of gameplay
# Format: key = Board box ID (1->9)
#		  Values = Character in board box corresponding to key, "" if none
board_dict = {}
for i in range(1,10):
	board_dict[i] = ""

# initialize varibles set inside of if loops and needed on the outside. This is done to avoid 
# problems that arise the first time these variables are needed outside the if-loops where their 
# are set.
prior_board = 0   				# warped board from previous camera view
curr_status = ""				# system stauts on top of board
camboard_grey_pre = 0
mask = 0
diff_gray = 0
empty_board = 0 				# warped empty board
color = 0
win_triple = 0
frames_since_game_over = 0
prior_count = 0
red_box_count = 0
game_over = False	
first_game = True
new_game = True	
new_game_tobe = True
have_empty_board = False
box1_color = black
box2_color = black
single_player = False
erase_check_mark = False
display_box = True	
display_box_tobe = True
prior_valid = True				# flag to indicate if prior_board is valid. This is set to False if
								# a new valid character is found on board and the projection of a 
								# highlight on the new entry cause the reference image for the next
								# image to be invalid (as a result of the highlight)
gameFeed = []

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
small_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
instance_pos = []

# make templates
X_templates = makeXTemplates((xright-xleft)/3)
O_templates = makeOTemplates((xright-xleft)/3)

# annotated_background is image with only highlights of entries so far. This is added to the 
# background after all the statuses and feeds are updated by using a corresponding mask to set
# pixel values on the background 
annotated_backGround = numpy.zeros((screenHeight,screenWidth,3),dtype='uint8')
annotated_backGround_mask = numpy.zeros((screenHeight,screenWidth),dtype='uint8')

# rectBoard is the camera's view of the board in the last frame. This serves the purpose of a 
# real time video of the camera's view
rectBoard = numpy.zeros((screenHeight,screenWidth,3),dtype='uint8')
rectBoard_mask = numpy.zeros((screenHeight,screenWidth),dtype='uint8')

# key: player index (1=>Player 1), (-1=>Player 2)
# values: array of players character, and list of taken box positions
player_dict = {1:["",[]],-1:["",[]]}

# dictionary of players real names: will be updated appropriately if playing against computer
player_real_names = {1:"Player 1", -1:"Player 2"} 
curr_player = 1

####################################################################################################

def computerMoveUpdates(move, char, color, image, image_mask):

	# updates annotation images for computer move

	global xleft, xright, yup

	# variables to help decide what portion of image to annotate
	side = xright - xleft 				# width of board
	# boundary values for box on board corresponding to 'move'
	xmin = ((move - 1)%3)*(side/3)
	xmax = xmin + (side/3)
	ymin = ((move - 1)/3)*(side/3)
	ymax = ymin + (side/3)
	
	back = numpy.zeros((side/3,side/3,3),dtype='uint8')			# color annotation
	back_gray = numpy.zeros((side/3,side/3),dtype='uint8')		# grayscale anotation
	back_mask = numpy.zeros((side/3,side/3),dtype='uint8')		# thresholded binary annotation

	# draw annotation
	if char == 'X':
		ptopleft = (int(0.2*side/3),int(0.2*side/3))
		pbottomright = (int(0.8*side/3),int(0.8*side/3))
		ptopright = (pbottomright[0],ptopleft[1])
		pbottomleft = (ptopleft[0],pbottomright[1])
		cv2.line(back,ptopleft,pbottomright,color,5,cv2.CV_AA)
		cv2.line(back,pbottomleft,ptopright,color,5,cv2.CV_AA)
	elif char == 'O':
		radius = int(0.7*(side/6))
		cv2.circle(back, (int(side/6),int(side/6)), radius, color, 5, cv2.CV_AA)

	# get grayscale and thresholded binary images of annotation
	cv2.cvtColor(back, cv2.cv.CV_RGB2GRAY, back_gray)
	cv2.threshold(back_gray, 10, 255, cv2.THRESH_BINARY, back_mask)

	binarymask = back_mask.view(numpy.bool)			# boolean view of binary image

	image_portion = image[(yup+ymin):(yup+ymax), (xleft+xmin):(xleft+xmax)]	
	image_portion_mask = image_mask[(yup+ymin):(yup+ymax), (xleft+xmin):(xleft+xmax)]					
	image_portion[binarymask] = color 		# set color for annotation
	image_portion_mask[binarymask] = 1 		# set boolean value for mask

	return [image, image_mask]

####################################################################################################

n_frame = 1
backGround = numpy.empty((screenHeight,screenWidth,3),dtype='uint8')
rectBoard = numpy.zeros((screenHeight,screenWidth,3),dtype='uint8')
rectBoard_mask = numpy.zeros((screenHeight,screenWidth),dtype='uint8')
while 1:
	
	################################################################################################
								# interactively get user's gaming mode #
	while new_game:			
		cv2.rectangle(backGround,(0,0),(screenWidth,screenHeight),white,-1)   # white background
		backGround = updateStatus(backGround,curr_status)
		checkBoxSize = 100 	# size of square checkboxes

		# coordinates of points defining checkboxes
		box1topright = (pb1[0],pb3[1])
		box1bottomleft = (box1topright[0]-checkBoxSize, box1topright[1]+checkBoxSize)
		box2topright = (pb1[0],pb5[1])
		box2bottomleft = (box2topright[0]-checkBoxSize, box2topright[1]+checkBoxSize)

		# location of prompt and texts in fron of checkboxes
		text1Pos = (box1topright[0]+100, box1bottomleft[1]-30)
		text2Pos = (box2topright[0]+100, box2bottomleft[1]-30)
		toptextpos = (box1bottomleft[0], box1topright[1]-50)
		toptext = ""
		text1 = ""
		text2 = ""

		# display box is False if we want user to erase their prior check-mark and True when we
		# need checkboxes to be displayed
		if display_box:			
			# if user already chose to play against the computer A.I, set texts for next prompt
			if single_player:
				toptext = "Who goes first?"
				text1 = "Computer"
				text2 = "Player"
			# otherwise, set prompt to ask for desired game-mode
			else:
				toptext = "Please check desired box."
				text1 = "Single-Player"
				text2 = "Multi-Player"
			# add appropriate text
			if have_empty_board:
				cv2.rectangle(backGround, box1bottomleft, box1topright, box1_color, 5)
				cv2.rectangle(backGround, box2bottomleft, box2topright, box2_color, 5)
				cv2.putText(backGround, text1, text1Pos, cv2.FONT_HERSHEY_SIMPLEX, 
					   		0.7, black, 1, cv2.CV_AA)
				cv2.putText(backGround, text2, text2Pos, cv2.FONT_HERSHEY_SIMPLEX, 
					   		0.7, black, 1, cv2.CV_AA)

			# before we set display_box to False, make boundary of user choice red for 5 frames
			if not display_box_tobe:
				if red_box_count < 5:
					red_box_count += 1
					continue
				if box1_color == red:
					box1_color = black
				if box2_color == red:
					box2_color = black
				red_box_count = 0
				display_box = False
				display_box_tobe = True

		# if display box is False, we want to prompt user to erase their previous check-mark
		else:
			toptext = "Please erase check mark"

		# add appropriate prompt
		cv2.putText(backGround, toptext, toptextpos, cv2.FONT_HERSHEY_SIMPLEX,
					0.7, black, 1, cv2.CV_AA)
		cv2.imshow("Tic-Tac-Toe",backGround)						# project backGround
		cv2.waitKey(100)											# hold for a bit

		ok, frame = capture.read()									# capture a frame
		cam_board = cv2.warpPerspective(frame, H, (1000,1000))		# warp camera view 
		camboard_pre = cam_board[yup:ydown, xleft:xright] 			# portion of screen with prompts
		camboard_grey = numpy.zeros((camboard_pre.shape[0],camboard_pre.shape[1]),dtype='uint8')
		cv2.cvtColor(camboard_pre, cv2.cv.CV_RGB2GRAY, camboard_grey)

		if writer and writerWarped:
			writer.write(frame)
			writerWarped.write(cam_board)

		# prior board is invalid after user erases their check-point and we have to wait a few 
		# frames to make the prior board the board with the checkboxes. If this isnt done, readding
		# the box would be confused for a user entry.
		if not prior_valid:
			prior_board = camboard_grey
			prior_count += 1
			if prior_count == 5:
				prior_valid = True
				prior_count = 0
			k = cv2.waitKey(5)
			# Check for ESC key hit:
			if k % 0x100 == 27:
				sys.exit(1)
			continue

		# we want to ignore first 6 frames completely
		if n_frame > 6:	
			diff = cv2.absdiff(prior_board,camboard_grey)
			mask = numpy.zeros((diff.shape[0],diff.shape[1]),dtype='uint8')
			cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY, mask)
			mask_erode = cv2.erode(mask,kernel=small_element,iterations=3)

			# make sure 5 pairs of adjacent frames do not differ before setting background
			if n_frame <= 11:
				if cv2.minMaxLoc(mask_erode)[1] > 0:
					# There is an obstruction
					curr_status = "finishing initialization"+((n_frame%3)+1)*"."
					k = cv2.waitKey(5)
					# Check for ESC hit:
					if k % 0x100 == 27:
						sys.exit(1)		# quit program
					n_frame = 6
					continue
				prior_board = camboard_grey

			# now we start checking for obstructions/entries on the board
			elif n_frame > 11:
				if not display_box:			# we need user to erase their last entry
					# compare view to initial empty view
					empty_diff = cv2.absdiff(camboard_grey,empty_board)
					mask = numpy.zeros((empty_diff.shape[0],empty_diff.shape[1]),dtype='uint8')
					cv2.threshold(empty_diff, 50, 255, cv2.THRESH_BINARY, mask)
					empty_diff_erode = cv2.erode(mask,kernel=small_element,iterations=2)
					if cv2.minMaxLoc(empty_diff_erode)[1] == 0:
						# board has been erased
						# if all information is got from user already
						if new_game_tobe == False:
							new_game = False
							box1_color = black
							box2_color = black
							n_frame = 1
							break 		# break out of while loop
						# if we need more info from user (a.k.a user chose single player mode)
						else:
							display_box = True
							prior_valid = False        
							k = cv2.waitKey(5)
							# Check for ESC key hit:
							if k % 0x100 == 27:
								sys.exit(1)
							continue
					k = cv2.waitKey(5)
					# Check for ESC key hit:
					if k % 0x100 == 27:
						sys.exit(1)
					continue	

				# first check for obstruction
				bigger_erosion = cv2.erode(mask,kernel=element,iterations=2)
				if cv2.minMaxLoc(bigger_erosion)[1] > 0:
					# There is an obstruction
					curr_status = "obstruction!!!"
					k = cv2.waitKey(5)
					# Check for ESC key hit:
					if k % 0x100 == 27:
						sys.exit(1)		# quit program
					continue

				# no obstruction
				temp = mask_erode.copy()
				contours = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
				areas = []
				pos = []
				# save blob area and position of all blobs
				for i in range(len(contours[0])):
					try:
						info = cvk2.getcontourinfo(contours[0][i])
						areas.append(info['area'])
						pos.append(cvk2.a2ti(info['mean']))
					except:
						pass
				if len(areas) > 0:
					curr_status = "processing entry"+((n_frame%3)+1)*"."
					position = pos[areas.index(max(areas))]  # position of largest area blob
					loc = getLoc(position)					 # get position on board of blob
					# a blob in box 7 is the bottom check-box
					if loc == 7:
						curr_status = ""
						box2_color = red
						display_box_tobe = False
						new_game_tobe = False
						# player goes before computer
						if single_player:    	
							player_real_names[-1] = "Computer"
							player_real_names[1] = "Player"
					# a blob in box 4 at the top check-box
					elif loc == 4:
						curr_status = ""
						box1_color = red 
						display_box_tobe = False
						if single_player:		# computer going first
							player_real_names[1] = "Computer"
							player_real_names[-1] = "Player"
							new_game_tobe = False
						# single player box selected
						else:					
							single_player = True
							new_game_tobe = True
					else:
						curr_status = "invalid choice; please erase and check appropriate box."
				else:
					curr_status = "Ready"

		# we are doing nothing except setting the prior image in the first 5 frames
		else:		
			curr_status = "finishing initialization"+((n_frame%3)+1)*"."
			prior_board = camboard_grey
			if n_frame == 3:
				empty_board = camboard_grey
				have_empty_board = True

		k = cv2.waitKey(5)
		if k % 0x100 == 27:
			# Check for ESC hit
			break
		n_frame += 1

	################################################################################################
	
	cv2.rectangle(backGround,(0,0),(screenWidth,screenHeight),white,-1)   # white background

	# add board lines to background
	cv2.line(backGround,pb1,pb7,black,5,cv2.CV_AA)
	cv2.line(backGround,pb3,pb4,black,5,cv2.CV_AA)
	cv2.line(backGround,pb5,pb6,black,5,cv2.CV_AA)
	cv2.line(backGround,pb2,pb8,black,5,cv2.CV_AA)

	if n_frame <= 11:
		# system is not ready until 11 successive steady frames
		if first_game:
			curr_status = "finishing initialization"+((n_frame%3) + 1)*"."
		else:
			curr_status = "initializing new game"+((n_frame%3) + 1)*"."
		if n_frame == 11:
			empty_board = camboard_grey

	backGround = updateStatus(backGround,curr_status)			# update system status
	backGround = updateGameFeed(backGround,gameFeed)			# update game feed
	bmask = annotated_backGround_mask.view(numpy.bool) 			# boolean mask		
	backGround[bmask] = annotated_backGround[bmask] 	# higlight background with entries so far
	rectBoard_bmask = rectBoard_mask.view(numpy.bool)
	backGround[rectBoard_bmask] = rectBoard[rectBoard_bmask]
	# if there's a winner, put a line through the winning characters
	if win_triple != 0:
		win_color = [red,blue][['X','O'].index(player_dict[curr_player][0])]
		backGround = updateWinBoard(backGround,win_color,win_triple)

	cv2.imshow('Tic-Tac-Toe',backGround)						# project background
	cv2.waitKey(1000/n_instance)								# wait for system to settle

	ok, frame = capture.read()									# capture frame
	cam_board = cv2.warpPerspective(frame, H, (1000,1000))		# warp camera view

	# save views in output video files
	if writer and writerWarped:
		writer.write(frame)
		writerWarped.write(cam_board)

	# extract grayscale image of only the board portion of warped camera view
	camboard_grey_pre = numpy.empty((cam_board.shape[0],cam_board.shape[1]),dtype='uint8')
	cv2.cvtColor(cam_board, cv2.cv.CV_RGB2GRAY, camboard_grey_pre)
	camboard_grey = camboard_grey_pre[yup:ydown, xleft:xright]

	# update mask and resized rectified board view
	(topX, topY) = (globalfeedPosTopLeft[0], globalfeedPosTopLeft[1] + num_lines*feed_height)
	view_width = globalfeedPosBottomRight[0] - globalfeedPosTopLeft[0]
	view_height = globalfeedPosBottomRight[1] - topY
	destination = rectBoard[topY:topY+view_height, topX:topX+view_width]
	rectified_Board = cam_board[yup:ydown, xleft:xright]
	cv2.resize(rectified_Board, (destination.shape[1], destination.shape[0]),
														 destination, 0, 0, cv2.INTER_LANCZOS4)
	cv2.cvtColor(rectBoard, cv2.cv.CV_RGB2GRAY, rectBoard_mask)

	if game_over:
		if frames_since_game_over == 20:
			# clear annotation image and its mask
			win_triple = 0
			annotated_backGround = numpy.zeros((screenHeight,screenWidth,3),dtype='uint8')
			annotated_backGround_mask = numpy.zeros((screenHeight,screenWidth),dtype='uint8')
		if frames_since_game_over > 20:
			diff = cv2.absdiff(empty_board,camboard_grey)
			cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY, mask)
			mask_erode = cv2.erode(mask,kernel=element,iterations=4)
			# if board has been erased, update all initiation values to prepare new game
			if cv2.minMaxLoc(mask)[1] == 0:		
				game_over = False
				first_game = False
				new_game = True
				new_game_tobe = True
				prior_valid = True	
				single_player = False
				erase_check_mark = False
				have_empty_board = False
				display_box = True
				display_box_tobe = True
				red_box_count = 0			
				n_frame = 1
				gameFeed = []
				curr_player = 1
				player_dict = {1:["",[]],-1:["",[]]}
				board_dict = {}
				for i in range(1,10):
					board_dict[i] = ""
				frames_since_game_over = -1
		frames_since_game_over += 1

		k = cv2.waitKey(5)
		# Check for ESC hit:
		if k % 0x100 == 27:
			break
		continue

	# check if prior_board is valid (invalid if we had to highlight new entry) and update if needed
	if not prior_valid:
		for m in range(11):
			# choose 11th frame and make it the new prior_board
			ok, frame = capture.read()
			cam_board = cv2.warpPerspective(frame, H, (1000,1000))
			cv2.cvtColor(cam_board, cv2.cv.CV_RGB2GRAY, camboard_grey_pre)
			camboard_grey = camboard_grey_pre[yup:ydown, xleft:xright]
		prior_board = camboard_grey
		prior_valid = True			# update flag denoting validity of prior_board
		k = cv2.waitKey(5)
		# Check for ESC hit:
		if k % 0x100 == 27:
			break		
		continue

	# start taking image difference after 5th frame. We want adjacent image differences to be 
	# zero for the next 6 frames (until 11th frame) before we set reference board
	if n_frame > 5:
		diff = cv2.absdiff(camboard_grey,prior_board)
		mask = numpy.zeros(diff.shape, 'uint8')
		cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY, mask)  

	    # erode with small structural element so that an X for example, in first 11 frames will be 
	    # seen as an obstruction while a small spec due to noise wouldnt. We'll erode more for 
	    # frames after the 11th to check for obstructions
		mask_erode = cv2.erode(mask,kernel=small_element,iterations=1)

		if n_frame <= 11:
			# if an obstruction is found						
			if cv2.minMaxLoc(mask_erode)[1] > 0:
				curr_status = "remove obstruction to finish initialization"
				k = cv2.waitKey(5)
				# Check for ESC hit:
				if k % 0x100 == 27:
					break				
				continue		# repeat, don't update n_frame
			# no obstruction 
			prior_board = camboard_grey
			if n_frame == 11:
				# set empty board after 11th frame, this will be used to check when the board has
				# been erased after the game
				empty_board = camboard_grey

		elif n_frame > 11:
			raw_index = n_frame - 12
			index = raw_index/n_instance
			instance = raw_index%n_instance

			# check for board obstruction
			bigger_erosion = cv2.erode(mask,kernel=element,iterations=2)
			if cv2.minMaxLoc(bigger_erosion)[1] > 0:
				curr_status = "obstruction!!!"
				k = cv2.waitKey(5)
				# Check for ESC hit:
				if k % 0x100 == 27:
					break
				continue 		# repeat, dont update frame

			# no big obstruction, now checking for X or O
			# nothing on board yet
			if cv2.minMaxLoc(mask_erode)[1] == 0:
				instance_pos = []
				n_frame -= instance
				curr_status = "Ready"
				player_name = player_real_names[curr_player]

				# if computer is current player
				if player_name == "Computer":
					curr_status = "Computer deciding"
					backGround = updateStatus(backGround, curr_status)
					cv2.imshow('Tic-Tac-Toe',backGround)		# project board with current status
					cv2.waitKey(2000)	# wait for 2 seconds
					# get computer move
					comp_move = pickMove(player_dict[-1*curr_player][1],player_dict[curr_player][1])
					comp_char = player_dict[curr_player][0]
					# if computer is going first, set its character as 'O'
					if not comp_char:		
						player_dict[curr_player][0] = 'O'	  
						player_dict[-1*curr_player][0] = 'X'	# set opponent's character
						board_dict[comp_move] = 'O'
						color = blue
						gameFeed.append("Computer is O")
						gameFeed.append(black)
						gameFeed.append("Player is X")
						gameFeed.append(black)
					# if this is not first move of game
					else:					
						if comp_char == 'O':
							color = blue
							board_dict[comp_move] = 'O'			  # update game board with comp_move
						else: 
							color = red
							board_dict[comp_move] = 'X'			  # update game board with comp_move
					comp_char = player_dict[curr_player][0]
					player_dict[curr_player][1].append(comp_move) # update computer's moves so far
					# update game feed
					gameFeed.append(player_name + " entered "+comp_char+" in box "+str(comp_move))
					gameFeed.append(color)
					# update annotated background and its mask
					[annotated_backGround, annotated_backGround_mask] = \
										computerMoveUpdates(comp_move, comp_char, color, 
											     	annotated_backGround, annotated_backGround_mask)
					
					check_win = checkWin(board_dict)	# check if computer wins
					if check_win or ("" not in board_dict.values()):
						game_over = True
						if check_win:
							curr_status = "Game Over: " + player_name + " is the winner! Erase " + \
										  "board for new game."
							gameFeed.append(player_name + " is winner!")
							gameFeed.append(black)
							win_triple = check_win[1:]
						else:
							curr_status = "Game Over: Tie!; Erase " + \
										  "board for new game."
							gameFeed.append("Game ended as a tie.")
							gameFeed.append(black)
					# computer didn't win, update current player
					else:
						curr_player *= -1

					# current image is different from what camera captured because of computer's
					# annotation.
					prior_valid = False
					k = cv2.waitKey(5)
					# Check for ESC hit:
					if k % 0x100 == 27:
						break
					continue

				# computer is not current player
				k = cv2.waitKey(5)
				# Check for ESC hit:
				if k % 0x100 == 27:
					break
				continue 	# repeat, dont update frame

			# something on board
			curr_status = "processing entry"+((instance%3)+1)*"."
			temp = mask_erode.copy()
			# get contour info
			contours = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
			areas = []
			pos = []
			# save blob area and position of all blobs
			for i in range(len(contours[0])):
				try:
					info = cvk2.getcontourinfo(contours[0][i])
					areas.append(info['area'])
					pos.append(cvk2.a2ti(info['mean']))
				except:
					pass
			# save position of maximum area blob in current frame
			if len(areas) > 0:
				instance_pos.append(pos[areas.index(max(areas))])

			# if we have n_instance (8) frames with blobs, check to make sure their location is 
			# consistent
			if (instance == (n_instance-1)) and (len(instance_pos) != 0):
				dot_errorThreshold = 10
				# check if centroid of maximum area blobs stays consistent in the n_instance frames
				dists = []
				for j in range(len(instance_pos)):
					dists.append(math.sqrt(sum([(p-q)**2 for p,q in zip(instance_pos[j],
														  instance_pos[(j+1)%len(instance_pos)])])))
				# centroids inconsistent: refresh (and check another n_instance frames)
				if max(dists) > dot_errorThreshold:
					instance_pos = []
					n_frame -= instance
					k = cv2.waitKey(5)
					# Check for ESC hit:
					if k % 0x100 == 27:
						break
					continue
				# centroids consistent: compute average position
				avg_pos = tuple([sum(p)/len(p) for p in zip(*instance_pos)])
				instance_pos = []

				# use average position to compute location on game board
				board_loc = getLoc(avg_pos)
				char = ""
				if board_loc and (not board_dict[board_loc]):	# if board_loc is free
					# get box on board game (1/9)th of board for template matching
					side = xright - xleft
					xmin = ((board_loc - 1)%3)*(side/3)
					xmax = xmin + (side/3)
					ymin = ((board_loc - 1)/3)*(side/3)
					ymax = ymin + (side/3)
					image_portion = mask_erode[ymin:ymax, xmin:xmax]   # get the (1/9)th region

					# dilate image portion twice: first for possible annotation (highlighting the 
					# the new user input), then dilate a little more for template matching
					first_dilate = cv2.dilate(image_portion,kernel=small_element,iterations=4)
					portion_dilate = cv2.dilate(first_dilate,kernel=small_element,iterations=1)
					char = getChar(portion_dilate,X_templates,O_templates)	# do template matching

					if char: # if entry is an 'X' or 'O'
						player_name = player_real_names[curr_player]
						if char == 'X':			# choose color for highlighting and feed update
							color = red
						elif char == 'O':
							color = blue

						# if first player is moving for the first time, set characters for both 
						# players and update gameFeed appropriately
						if not player_dict[1][0]:
							player_dict[1][0] = char 				# set player 1's character
							player_dict[1][1].append(board_loc)		# update player 1's boxes
							# set player 2's character
							player_dict[-1][0] = ['X','O'][(['X','O'].index(char)+1)%2]
							# update game feed
							gameFeed.append(player_name + " is " + char)
							gameFeed.append(black)
							gameFeed.append(player_real_names[-1*curr_player] + " is " + \
								                                             player_dict[-1][0])
							gameFeed.append(black)
						else:
							if player_dict[curr_player][0] != char:
								curr_status = player_name + ", you are " + \
								              player_dict[curr_player][0] + "; erase and retry."
								k = cv2.waitKey(5)
								# Check for ESC hit:
								if k % 0x100 == 27:
									break
								continue
							player_dict[curr_player][1].append(board_loc)

						board_dict[board_loc] = char 	# update board_dictionary with new entry

						# update image with highlighted entry to be added at top of loop
						binarymask = first_dilate.view(numpy.bool)	# use first dilated image
						# view only portions of annotation images, and annotation mask
						backGround_portion = annotated_backGround[(yup+ymin):(yup+ymax),
																  (xleft+xmin):(xleft+xmax)]	
						backGround_portion_mask = annotated_backGround_mask[(yup+ymin):(yup+ymax),
																		  (xleft+xmin):(xleft+xmax)]					
						backGround_portion[binarymask] = color 	# set color for annotation
						backGround_portion_mask[binarymask] = 255 # set boolean value for mask

						prior_valid = False		# invalidate prior_image due to annotation
						gameFeed.append(player_name + " entered " + char + " in box " + \
									    str(board_loc))
						gameFeed.append(color)

						check_win = checkWin(board_dict)
						# if there's a win or a tie
						if check_win or ("" not in board_dict.values()):
							game_over = True
							# if someone won		
							if check_win:
								curr_status = "Game Over: " + player_name + " is the winner! " + \
											  "Erase board for new game."
								gameFeed.append(player_name + " is winner!")
								gameFeed.append(black)
								win_triple = check_win[1:]
							# if game ended as a tie
							else:
								curr_status = "Game Over: Tie!; Erase board for new game."
								gameFeed.append("Game ended as a tie.")
								gameFeed.append(black)
						else:
							curr_player *= -1 		# update current player
					# if getChar returned "" instead of 'X' or 'O' a.k.a (invalid entry)
					else:
						curr_status = "invalid entry in box " + str(board_loc) + \
									             ", please erase and try again."
						k = cv2.waitKey(5)
						# Check for ESC hit:
						if k % 0x100 == 27:
							break
						continue
				else:
					# entry not inside board or in already taken box
					if not board_loc:
						# most likely will not happen since we are only considering blobs on
						# game board, but this could take care of some edge/boundary issues
						curr_status = "entry not on game-board; erase and retry"
					# entry in already taken box
					elif board_dict[board_loc]:
						curr_status = "box " + str(board_loc) + " already taken; erase and retry"

					k = cv2.waitKey(5)
					# Check for ESC hit:
					if k % 0x100 == 27:
						break
					continue

	else:
		prior_board = camboard_grey
	k = cv2.waitKey(5)
	# Check for ESC hit:
	if k % 0x100 == 27:
		break

	n_frame += 1

####################################################################################################