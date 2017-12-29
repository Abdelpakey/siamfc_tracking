''' 
Referred to example 4-1 in the book "Learning OpenCV: Computer Vision with the OpenCV Library"
Example 4-1. Toy program for using a mouse to draw boxes on the screen
Converted to Python by Abid.K	--mail me at abidrahman2@gmail.com
'''
########################################################################################

import cv2
import numpy as np

box=[0,0,0,0]
drawing_box = False
drawing_finished = False

def get_bbox_from_image(image):
	
	#	creating mouse callback function
	def my_mouse_callback(event,x,y,flags,param):
		global drawing_box, drawing_finished
		if event==cv2.EVENT_LBUTTONDOWN:
			drawing_box=True
			[box[0],box[1],box[2],box[3]]=[x,y,0,0]
			# print box[0]

		if event==cv2.EVENT_LBUTTONUP:
			drawing_box=False
			if box[2]<0:
				box[0]+=box[2]
				box[2]*=-1
			if box[3]<0:
				box[1]+=box[3]
				box[3]*=-1
			drawing_finished = True
				
		if event==cv2.EVENT_MOUSEMOVE:
			if drawing_box:
				box[2]=x-box[0]
				box[3]=y-box[1]	
			
	# # 	function to draw the rectangle, added flag -1 to fill rectangle. If you don't want to fill, just delete it.		
	def draw_box(img,box):
		cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,0),-1)

	#	main program	
	# #	make a clone of image
	# temp=image.copy()

	window_name = "Label Bbox"
	cv2.namedWindow(window_name)
	cv2.setMouseCallback(window_name,my_mouse_callback,image)
	print("\nPlease pick your bounding box in the image by using the mouse.\n")
	while(1):
		if drawing_finished:
			print("Final box %s"%(box))
			break
		temp = image.copy()
		if drawing_box:
			draw_box(temp,box)
			# print(box)
		cv2.imshow(window_name,temp)
		if cv2.waitKey(20)%0x100==27:
			break

	cv2.destroyWindow(window_name)

	return box

if __name__ == '__main__':
	image=cv2.imread("./data/conveyor/265.jpg")
	bbox = get_bbox_from_image(image)
	print("GOT BOX %s"%(bbox))

# {
# 	{
# 		"name": "dorabot.mp4",
# 		"data": [
# 			{
# 				"object": "box1",
# 				"bbox": [[x,y,w,h],[x,y,w,h],...]
# 			},
# 			{
# 				"object": "box2",
# 				"bbox": [[x,y,w,h],[x,y,w,h],...]
# 			}
# 		]
# 	},
# 	{

# 	}
	
# }