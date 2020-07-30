import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

class DetectorTF2:

	def __init__(self, path_to_checkpoint, path_to_labelmap, class_id=None, threshold=0.5):
		# class_id is list of ids for desired classes, or None for all classes in the labelmap
		self.class_id = class_id
		self.Threshold = threshold
		# Loading label map
		label_map = label_map_util.load_labelmap(path_to_labelmap)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)

		tf.keras.backend.clear_session()
		self.detect_fn = tf.saved_model.load(path_to_checkpoint)


	def DetectFromImage(self, img):
		im_height, im_width, _ = img.shape
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		input_tensor = np.expand_dims(img, 0)
		detections = self.detect_fn(input_tensor)

		bboxes = detections['detection_boxes'][0].numpy()
		bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
		bscores = detections['detection_scores'][0].numpy()
		det_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores, im_width, im_height)

		return det_boxes


	def ExtractBBoxes(self, bboxes, bclasses, bscores, im_width, im_height):
		bbox = []
		for idx in range(len(bboxes)):
			if self.class_id is None or bclasses[idx] in self.class_id:
				if bscores[idx] >= self.Threshold:
					y_min = int(bboxes[idx][0] * im_height)
					x_min = int(bboxes[idx][1] * im_width)
					y_max = int(bboxes[idx][2] * im_height)
					x_max = int(bboxes[idx][3] * im_width)
					class_label = self.category_index[int(bclasses[idx])]['name']
					bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])
		return bbox


	def DisplayDetections(self, image, boxes_list, det_time=None):
		if not boxes_list: return image  # input list is empty
		img = image.copy()
		for idx in range(len(boxes_list)):
			x_min = boxes_list[idx][0]
			y_min = boxes_list[idx][1]
			x_max = boxes_list[idx][2]
			y_max = boxes_list[idx][3]
			cls =  str(boxes_list[idx][4])
			score = str(np.round(boxes_list[idx][-1], 2))

			text = cls + ": " + score
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
			cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
			cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

		if det_time != None:
			fps = round(1000. / det_time, 1)
			fps_txt = str(fps) + " FPS"
			cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		return img

