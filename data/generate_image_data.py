from PIL import Image, ImageDraw
import os
import zCurve as z
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_image_data(classes, frame_width=10, frame_height=6, num_frames=30, samples=5):
	square_size = 1

	for cls in classes:
		output_dir = "data_img/"+cls
		os.makedirs(output_dir, exist_ok=True)

		for s in range(samples):
			# Generate frames
			for i in range(num_frames):
				img = Image.new("RGB", (frame_width, frame_height), "black")
				draw = ImageDraw.Draw(img)

				# Compute square position
				if cls == "lefttoright":
					x = int((frame_width - square_size) * i / (num_frames - 1))
				elif cls == "righttoleft":
					x = int(frame_width - frame_width * i / (num_frames - 1) - square_size)
				elif cls == "stopinmiddle":
					x = int((frame_width - square_size) * i / (num_frames - 1)) if i < num_frames // 2 else x
				elif cls == "waittocross":
					x = frame_width // 3 + np.sin(np.pi * 2 * (i / num_frames))
				y = (frame_height - square_size) // 2

				sizechange = 0.2*np.sin(np.pi * 2 * (i / num_frames) + s)  # vary square width as leg movement sim

				draw.rectangle([x, y, x + square_size + sizechange, y + square_size*2], fill="yellow")
				os.makedirs(output_dir+"/"+cls+f"_{s:03d}", exist_ok=True)
				img.save(os.path.join(output_dir+"/"+cls+f"_{s:03d}", cls+f"_{s:03d}_{i:04d}.png"))


def encode_images(classes):
	for cls in classes:
		parent_folder = "data_img/"+cls+"/"
		folders = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]

		# FOR EACH SAMPLE
		for folder_path in folders:
			# FOR EACH FRAME
			data = []
			for f, filename in enumerate(sorted(os.listdir(parent_folder + folder_path))):
				if filename.lower().endswith(('.png')):
					img_path = os.path.join(parent_folder+folder_path, filename)
					with Image.open(img_path) as img:
						img = img.convert('L')  # Ensure consistent color mode
						pixels = np.array(img).flatten()  # Flatten to 1D
						pixels[pixels > 0] = 1
						data.append(pixels.tolist())

			# CREATE DATAFRAME
			column_names = [f'pix{i}' for i in range(len(data[0]))]
			df = pd.DataFrame(data, columns=column_names)
			df.to_csv(parent_folder+folder_path+'.csv', index=False)
			df = df.astype(int)

			# ENCODE
			seq = [z.interlace(*(int(x>0) for x in row), bits_per_dim=2) for row in df.values]
			df = pd.DataFrame(seq, columns=["sfc_encoded"])
			df.to_csv(os.path.join("data_csv", f"{folder_path}.csv"), index=False, float_format='%.4f')

def plot_sfc_images(classes):
	for cls in classes:
		data_dir = "data_csv/"
		csv_files = sorted([f for f in os.listdir(data_dir) if cls in f and f.endswith('.csv')])
		csv_files.sort()

		for i, csv_file in enumerate(csv_files[:10]):
			plt.figure()
			df = pd.read_csv(os.path.join(data_dir, csv_file))
			plt.scatter(df['sfc_encoded'], np.arange(len(df)))

			plt.title("SFC-encoded frames: " + csv_file)
			plt.xlabel("Morton index")
			plt.ylabel("Frames")
			plt.tight_layout()
			plt.grid(True)
			plt.savefig("data_img/"+cls+f"/{csv_file[:-4]}_csp.png")
			plt.close()

def plot_sfc_images2(classes):
	for cls in classes:
		data_dir = "data_csv/"
		csv_files = sorted([f for f in os.listdir(data_dir) if cls in f and f.endswith('.csv')])
		csv_files.sort()

		for i, csv_file in enumerate(csv_files[:1]):
			plt.figure()
			df = pd.read_csv(os.path.join(data_dir, csv_file))

			# Draw vertical bars in the background at each sfc_encoded position
			for x in df['sfc_encoded']:
				plt.axvspan(x - 0.5, x + 0.5, color='lightgray', alpha=0.5)

			plt.scatter(df['sfc_encoded'], np.arange(len(df)))

			plt.title(cls + " sample")
			plt.xlabel("Morton index")
			plt.ylabel("Frames")
			plt.tight_layout()
			# plt.grid(True)
			os.makedirs("data_img/"+cls, exist_ok=True)
			plt.savefig("data_img/"+cls+f"/{csv_file[:-4]}_csp.png")
			plt.close()


if __name__ == "__main__":
	classes = ["lefttoright", "righttoleft", "stopinmiddle", "waittocross", 'lefturn', 'rightturn', 'ra', 'lanechange']
	img_width = 10
	img_height = 6
	generate_image_data(classes, img_width, img_height, samples=5)
	encode_images(classes, img_width, img_height)
	# plot_sfc_images(classes)
	plot_sfc_images2(classes)