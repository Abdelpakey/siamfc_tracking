import xml.etree.ElementTree as ET
import os
import os.path as osp
import glob
import sys
import numpy as np
import cv2

sys.path.append('../')

from utils import natural_sort

RED = (0,0,255)
CLASS_IDS = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808', \
				 'n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045', \
				 'n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']

if __name__ == '__main__':
	root_dir = "/home/vincent/hd/datasets/ILSVRC2015"
	root_annot = os.path.join(root_dir, "Annotations/VID/train")
	root_img = os.path.join(root_dir, "Data/VID/train")

	img_ext = "JPEG"

	skip_frame = 1

	for folder1 in os.listdir(root_img):
		root_img_folder1 = osp.join(root_img, folder1)
		root_annot_folder1 = osp.join(root_annot, folder1)
		for folder2 in [f for f in os.listdir(root_img_folder1) if not f.endswith('.txt')]:
			img_folder = osp.join(root_img_folder1, folder2)
			print("Reading %s..."%(img_folder))
			annot_folder = osp.join(root_annot_folder1, folder2) 

			img_files = [f for f in os.listdir(img_folder) if f.endswith(img_ext)] # glob.glob(osp.join(img_folder, "*.JPEG")) 
			img_files = natural_sort(img_files)
			# annot_files = [f for f in os.listdir(sample_annot_folder) if f.endswith("xml")]
			# annot_files = natural_sort(annot_files)

			tracker_ids = set()
			for i,f in enumerate(img_files):
				if skip_frame > 0 and i % (skip_frame + 1) != 0:
					continue
				base_name = f.replace("."+img_ext,"")
				annot_f = osp.join(annot_folder, base_name + ".xml")
				img_f = osp.join(img_folder, f)
				img = cv2.imread(img_f)
				if img is None:
					print("Could not find %s! Skipping it"%(img_f))
					continue

				et = ET.parse(annot_f)
				et_root = et.getroot()
				et_objs = et_root.findall('object')
				for ix, obj in enumerate(et_objs):
					obj_cls = obj.find('name').text
					obj_id = int(obj.find("trackid").text)
					tracker_ids.add(obj_id)
					obj_bbox = obj.find('bndbox')
					bbox = [obj_bbox.find('xmin').text,obj_bbox.find('ymin').text,obj_bbox.find('xmax').text,obj_bbox.find('ymax').text]
					bbox = np.array(bbox, np.int32)

					pt1 = bbox[:2]
					pt2 = bbox[2:4]
					cv2.rectangle(img, tuple(pt1), tuple(pt2), RED)
					mid_pt = np.mean([pt1,pt2],axis=0).astype(np.int)
					cv2.putText(img, "%d"%(obj_id), tuple(mid_pt), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

				cv2.imshow("bboxes", img)
				cv2.waitKey(20)
			print("tracker_ids: ", tracker_ids)
