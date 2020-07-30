import os
import cv2
import time
import argparse

from detector import DetectorTF2


def DetectFromVideo(detector, Video_path, save_output=False, output_dir='output/'):

	cap = cv2.VideoCapture(Video_path)
	if save_output:
		output_path = os.path.join(output_dir, 'detection_'+ Video_path.split("/")[-1])
		frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

	while (cap.isOpened()):
		ret, img = cap.read()
		if not ret: break

		timestamp1 = time.time()
		det_boxes = detector.DetectFromImage(img)
		elapsed_time = round((time.time() - timestamp1) * 1000) #ms
		img = detector.DisplayDetections(img, det_boxes, det_time=elapsed_time)

		cv2.imshow('TF2 Detection', img)
		if cv2.waitKey(1) == 27: break

		if save_output:
			out.write(img)

	cap.release()
	if save_output:
		out.release()


def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):

	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			print(image_path)
			img = cv2.imread(image_path)
			det_boxes = detector.DetectFromImage(img)
			img = detector.DisplayDetections(img, det_boxes)

			cv2.imshow('TF2 Detection', img)
			cv2.waitKey(0)

			if save_output:
				img_out = os.path.join(output_dir, file.name)
				cv2.imwrite(img_out, img)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
	parser.add_argument('--model_path', help='Path to frozen detection model',
						default='models/efficientdet_d0_coco17_tpu-32/saved_model')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
	                    default='models/mscoco_label_map.pbtxt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
	                    type=str, default=None) # example input "1,3" to detect person and car
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)
	parser.add_argument('--images_dir', help='Directory to input images)', default='data/samples/images/')
	parser.add_argument('--video_path', help='Path to input video)', default='data/samples/pedestrian_test.mp4')
	parser.add_argument('--output_directory', help='Path to output images and video', default='data/samples/output')
	parser.add_argument('--video_input', help='Flag for video input, default: False', action='store_true')  # default is false
	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized, default: False',
	                    action='store_true')  # default is false
	args = parser.parse_args()

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not os.path.exists(args.output_directory):
			os.makedirs(args.output_directory)

	# instance of the class DetectorTF2
	detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

	if args.video_input:
		DetectFromVideo(detector, args.video_path, save_output=args.save_output, output_dir=args.output_directory)
	else:
		DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output, output_dir=args.output_directory)

	print("Done ...")
	cv2.destroyAllWindows()
