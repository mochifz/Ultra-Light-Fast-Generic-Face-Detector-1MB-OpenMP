import torch
import torch.nn as nn
import numpy as np
import time
#import pnnx
#pnnx.export(ssd, f"mb_tiny.pt", torch.rand(1, 3, 240, 320, dtype=torch.float))
class ConvReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
		super(ConvReLU, self).__init__()
		self.conv = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			groups=groups
		)
		self.relu = nn.ReLU()
	def forward(self, x):
		return self.relu(self.conv(x))

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.backbone = nn.Sequential(
			ConvReLU(3, 16, kernel_size=3, stride=2, padding=1, groups=1),
			ConvReLU(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
			ConvReLU(16, 32, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(32, 32, kernel_size=3, stride=2, padding=1, groups=32),
			ConvReLU(32, 32, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
			ConvReLU(32, 32, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(32, 32, kernel_size=3, stride=2, padding=1, groups=32),
			ConvReLU(32, 64, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
			ConvReLU(64, 64, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
			ConvReLU(64, 64, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
			ConvReLU(64, 64, kernel_size=1, stride=1, padding=0, groups=1),
		)
		self.cls_head_1 = nn.Sequential(
			ConvReLU(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
			nn.Conv2d(64, 6, kernel_size=1, stride=1)
		)
		self.reg_head_1 = nn.Sequential(
			ConvReLU(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
			nn.Conv2d(64, 12, kernel_size=1, stride=1)
		)
		self.transition_1 = nn.Sequential(
			ConvReLU(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
			ConvReLU(64, 128, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(128, 128, kernel_size=3,
					   stride=1, padding=1, groups=128),
			ConvReLU(128, 128, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(128, 128, kernel_size=3,
					   stride=1, padding=1, groups=128),
			ConvReLU(128, 128, kernel_size=1, stride=1, padding=0, groups=1),
		)
		self.cls_head_2 = nn.Sequential(
			ConvReLU(128, 128, kernel_size=3,
					   stride=1, padding=1, groups=128),
			nn.Conv2d(128, 4, kernel_size=1, stride=1)
		)
		self.reg_head_2 = nn.Sequential(
			ConvReLU(128, 128, kernel_size=3,
					   stride=1, padding=1, groups=128),
			nn.Conv2d(128, 8, kernel_size=1, stride=1)
		)
		self.transition_2 = nn.Sequential(
			ConvReLU(128, 128, kernel_size=3,
					   stride=2, padding=1, groups=128),
			ConvReLU(128, 256, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(256, 256, kernel_size=3,
					   stride=1, padding=1, groups=256),
			ConvReLU(256, 256, kernel_size=1, stride=1, padding=0, groups=1),
		)
		self.cls_head_3 = nn.Sequential(
			ConvReLU(256, 256, kernel_size=3,
					   stride=1, padding=1, groups=256),
			nn.Conv2d(256, 4, kernel_size=1, stride=1)
		)
		self.reg_head_3 = nn.Sequential(
			ConvReLU(256, 256, kernel_size=3,
					   stride=1, padding=1, groups=256),
			nn.Conv2d(256, 8, kernel_size=1, stride=1)
		)
		self.extra_layers = nn.Sequential(
			ConvReLU(256, 64, kernel_size=1, stride=1, padding=0, groups=1),
			ConvReLU(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
			ConvReLU(64, 256, kernel_size=1, stride=1, padding=0, groups=1),
		)
		self.cls_head_4 = nn.Conv2d(
			256, 6, kernel_size=3, stride=1, padding=1)
		self.reg_head_4 = nn.Conv2d(
			256, 12, kernel_size=3, stride=1, padding=1)
		self.softmax = nn.Softmax(dim=2)


	def load_pretrained_weights(self, state_dict_path):
		state_dict = torch.load(state_dict_path, map_location='cpu')
		model_keys = list(self.state_dict().keys())
		loaded_keys = list(state_dict.keys())

		new_state_dict = {
			model_key: state_dict[loaded_key] 
			for model_key, loaded_key in zip(model_keys, loaded_keys)
		}
		self.load_state_dict(new_state_dict)
	def heads(self, x):
		cls1 = self.cls_head_1(x).permute(0, 2, 3, 1).reshape(1, 3600, 2)
		reg1 = self.reg_head_1(x).permute(0, 2, 3, 1).reshape(1, 3600, 4)

		feat2 = self.transition_1(x)

		cls2 = self.cls_head_2(feat2).permute(0, 2, 3, 1).reshape(1, 600, 2)
		reg2 = self.reg_head_2(feat2).permute(0, 2, 3, 1).reshape(1, 600, 4)

		feat3 = self.transition_2(feat2)

		cls3 = self.cls_head_3(feat3).permute(0, 2, 3, 1).reshape(1, 160, 2)
		reg3 = self.reg_head_3(feat3).permute(0, 2, 3, 1).reshape(1, 160, 4)

		feat4 = self.extra_layers(feat3)

		cls4 = self.cls_head_4(feat4).permute(0, 2, 3, 1).reshape(1, 60, 2)
		reg4 = self.reg_head_4(feat4).permute(0, 2, 3, 1).reshape(1, 60, 4)

		all_cls = torch.cat((cls1, cls2, cls3, cls4), dim=1)
		locations = torch.cat((reg1, reg2, reg3, reg4), dim=1)
		return all_cls, locations
	def forward(self, x):
		all_cls, locations = self.heads(self.backbone(x))
		scores = self.softmax(all_cls)
		return scores, locations

def read_ppm(file_path):
	with open(file_path, 'rb') as f:
		magic = f.readline().decode('ascii').strip()
		if magic != 'P6':
			raise ValueError(f"Unsupported PPM format: {magic}")

		line = f.readline().decode('ascii').strip()
		while line.startswith('#'):
			line = f.readline().decode('ascii').strip()

		width, height = map(int, line.split())

		max_val = int(f.readline().decode('ascii').strip())

		data = f.read()

		if max_val == 255:
			dtype = np.uint8
		else:
			dtype = np.uint16

		img = np.frombuffer(data, dtype=dtype)
		img = img.reshape((height, width, 3))

	return img

def write_ppm(file_path, image):
	height, width = image.shape[:2]

	with open(file_path, 'wb') as f:
		f.write(b'P6\n')
		f.write(f'{width} {height}\n'.encode('ascii'))
		f.write(b'255\n')

		if image.dtype != np.uint8:
			image = np.clip(image, 0, 255).astype(np.uint8)

		f.write(image.tobytes())

def resize_image(image, target_size):
	src_h, src_w, _ = image.shape
	dst_w, dst_h = target_size

	output = np.zeros((dst_h, dst_w, 3), dtype=np.float32)

	scale_x = src_w / dst_w
	scale_y = src_h / dst_h

	for dy in range(dst_h):
		for dx in range(dst_w):
			src_x = (dx + 0.5) * scale_x - 0.5
			src_y = (dy + 0.5) * scale_y - 0.5

			x0 = int(np.floor(src_x))
			y0 = int(np.floor(src_y))
			x1 = min(x0 + 1, src_w - 1)
			y1 = min(y0 + 1, src_h - 1)

			x0 = max(0, x0)
			y0 = max(0, y0)

			dx_weight = src_x - x0
			dy_weight = src_y - y0

			for c in range(3):
				val = (image[y0, x0, c] * (1 - dx_weight) * (1 - dy_weight) +
					   image[y0, x1, c] * dx_weight * (1 - dy_weight) +
					   image[y1, x0, c] * (1 - dx_weight) * dy_weight +
					   image[y1, x1, c] * dx_weight * dy_weight)
				output[dy, dx, c] = val

	return output

def preprocess_image(image, input_size=(320, 240), mean=127.0, std=128.0):
	return torch.from_numpy(((resize_image(image, input_size) - mean) / std).transpose(2, 0, 1))

def decode_boxes(locations, priors, center_variance=0.1, size_variance=0.2):
	cx = locations[:, 0] * center_variance * priors[:, 2] + priors[:, 0]
	cy = locations[:, 1] * center_variance * priors[:, 3] + priors[:, 1]
	w = np.exp(locations[:, 2] * size_variance) * priors[:, 2]
	h = np.exp(locations[:, 3] * size_variance) * priors[:, 3]
	x1 = np.clip(cx - w / 2.0, 0, 1)
	y1 = np.clip(cy - h / 2.0, 0, 1)
	x2 = np.clip(cx + w / 2.0, 0, 1)
	y2 = np.clip(cy + h / 2.0, 0, 1)

	return np.stack([x1, y1, x2, y2], axis=1)

def draw_boxes_on_image(image, boxes, scores, labels=None, color=(0, 255, 0), thickness=2):
	img_copy = image.copy().astype(np.uint8)

	height, width = img_copy.shape[:2]

	for i, box in enumerate(boxes):
		x1 = int(np.clip(box[0] * width, 0, width - 1))
		y1 = int(np.clip(box[1] * height, 0, height - 1))
		x2 = int(np.clip(box[2] * width, 0, width - 1))
		y2 = int(np.clip(box[3] * height, 0, height - 1))

		for t in range(thickness):
			if y1 - t >= 0:
				img_copy[max(0, y1 - t), x1:x2+1] = color
			if y2 + t < height:
				img_copy[min(height-1, y2 + t), x1:x2+1] = color
			if x1 - t >= 0:
				img_copy[y1:y2+1, max(0, x1 - t)] = color
			if x2 + t < width:
				img_copy[y1:y2+1, min(width-1, x2 + t)] = color

	return img_copy

def generate_priors(image_size=(320, 240), min_boxes=[[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]],
					strides=[8.0, 16.0, 32.0, 64.0]):
	in_w, in_h = image_size
	priors = []

	for index, min_box in enumerate(min_boxes):
		scale_w = in_w / strides[index]
		scale_h = in_h / strides[index]

		fm_w = int(np.ceil(in_w / strides[index]))
		fm_h = int(np.ceil(in_h / strides[index]))

		for j in range(fm_h):
			for i in range(fm_w):
				x_center = (i + 0.5) / scale_w
				y_center = (j + 0.5) / scale_h

				for k in min_box:
					w = k / in_w
					h = k / in_h

					priors.append([x_center, y_center, w, h])

	return np.clip(np.array(priors, dtype=np.float32), 0, 1)

def standard_nms(boxes, scores, iou_threshold=0.3, top_k=-1):
	if len(boxes) == 0:
		return []

	sorted_indices = np.argsort(scores)[::-1]

	keep_indices = []
	while len(sorted_indices) > 0:
		current_idx = sorted_indices[0]
		keep_indices.append(current_idx)

		if top_k > 0 and len(keep_indices) >= top_k:
			break

		if len(sorted_indices) == 1:
			break

		current_box = boxes[current_idx]
		remaining_boxes = boxes[sorted_indices[1:]]

		xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
		yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
		xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
		yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])

		w = np.maximum(0, xx2 - xx1)
		h = np.maximum(0, yy2 - yy1)
		intersection = w * h

		area_current = (current_box[2] - current_box[0]) * \
			(current_box[3] - current_box[1])
		area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
			remaining_boxes[:, 3] - remaining_boxes[:, 1])
		union = area_current + area_remaining - intersection

		ious = intersection / (union + 1e-6)

		keep_mask = ious <= iou_threshold
		sorted_indices = sorted_indices[1:][keep_mask]

	return keep_indices

def detect_faces(model, image, score_threshold=0.6, iou_threshold=0.3, top_k=-1):
	priors = generate_priors(image_size=(320, 240))

	input_tensor = preprocess_image(image, input_size=(320, 240))

	model.eval()
	with torch.no_grad():
		start_time = time.time()
		scores, locations = model(input_tensor.unsqueeze(0))
		print(f"Detection took {time.time() - start_time:.2f} seconds")

	scores_np = scores.numpy()[0]
	locations_np = locations.numpy()[0]

	face_scores = scores_np[:, 1]

	boxes = decode_boxes(locations_np, priors)

	mask = face_scores > score_threshold
	filtered_boxes = boxes[mask]
	filtered_scores = face_scores[mask]

	if len(filtered_scores) == 0:
		return np.array([]).reshape(0, 4), np.array([])

	picked_indices = standard_nms(
		filtered_boxes, filtered_scores, iou_threshold, top_k)

	final_boxes = filtered_boxes[picked_indices]
	final_scores = filtered_scores[picked_indices]

	return final_boxes, final_scores

def detect_ppm(ppm_file_path, output_file=None):
	image = read_ppm(ppm_file_path)
	model = Model()
	model.load_pretrained_weights("mb_tiny.pth")
	#torch.save(model.backbone.state_dict(), "backbone.pth")
	model.float()
	model.eval()
	boxes, scores = detect_faces(model, image)
	if len(scores) > 0:
		result_image = draw_boxes_on_image(image, boxes, scores)

		if output_file is None:
			output_file = ppm_file_path.replace('.ppm', '_result.ppm')

		write_ppm(output_file, result_image)

		print(f"Detected {len(scores)} face(s)")
		for i, (box, score) in enumerate(zip(boxes, scores)):
			print(
				f"  Face {i+1}: score={score:.4f}, box=[{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]")

	else:
		print("No faces detected")

	return boxes, scores

if __name__ == "__main__":
	detect_ppm("test.ppm")
