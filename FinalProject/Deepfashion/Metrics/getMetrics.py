from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.metrics import adapted_rand_error

def extract_features(image_array):
	model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
	processed_image = preprocess_input(image_array)
	features = model.predict(processed_image)
	return features

def calculate_fid(generated_array, real_array):
	generated_features = extract_features(generated_array)
	real_features = extract_features(real_array)
	
	mu1, sigma1 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
	mu2, sigma2 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
	
	(fid_score,) = calculate_fid_score(mu1, mu2, sigma1, sigma2)
	return fid_score

def calculate_fid_score(mu1, mu2, sigma1, sigma2, eps=1e-6):
	diff = mu1 - mu2
	
	covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; '
		       'adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
	
	tr_covmean = np.trace(covmean)
	return ((diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean) / real_array.shape[0])

def calculate_dice(generated_array, real_array):
	generated_segmentation = slic(generated_array)
	real_segmentation = slic(real_array)
	
	dice_score = calculate_dice_score(generated_segmentation, real_segmentation)
	return dice_score

def calculate_dice_score(segmentation1, segmentation2):
	return 1 - adapted_rand_error(segmentation1, segmentation2)

# Load images
generated_img = Image.open('generated/generated_1.jpg')
real_img = Image.open('original/original_1.jpg')

# Convert images to numpy arrays
generated_array = np.array(generated_img)
real_array = np.array(real_img)

# Add batch dimension
generated_array = np.expand_dims(generated_array, axis=0)
real_array = np.expand_dims(real_array, axis=0)


# Calculate FID score
fid_score = calculate_fid(generated_array, real_array)
print("FID Score:", fid_score)

# Calculate Dice score
dice_score = calculate_dice(generated_array, real_array)
print("Dice Score:", dice_score)