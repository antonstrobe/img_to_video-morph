import cv2
import numpy as np
import os
from PIL import Image
import random
import tempfile

def resize_and_blur_background(image, target_size=(1920, 1080)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    blurred_background = cv2.GaussianBlur(resized, (0, 0), 100)
    delta_w = target_size[0] - new_size[0]
    delta_h = target_size[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    bordered = cv2.copyMakeBorder(blurred_background, top, bottom, left, right, cv2.BORDER_REFLECT)
    bordered[top:top+new_size[1], left:left+new_size[0]] = resized
    return bordered

def create_alpha_channel(image):
    # Создать альфа-канал, где 1 - область изображения, 0 - черный фон
    alpha_channel = np.all(image[:, :, :3] != [0, 0, 0], axis=2).astype(np.uint8) * 255
    return alpha_channel

def add_alpha_channel(image, alpha_channel):
    # Добавить альфа-канал к изображению
    image_with_alpha = cv2.merge((image, alpha_channel))
    return image_with_alpha

def random_points(image, num_points=10):
    h, w = image.shape[:2]
    return [(random.randint(0, w), random.randint(0, h)) for _ in range(num_points)]

def morph_images(img1, img2, alpha):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    morphed_img = (1 - alpha) * img1 + alpha * img2
    return np.uint8(morphed_img)

def create_morph_video(images, output_video, fps=60, morph_time=3, hold_time=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (1920, 1080))

    for i in range(len(images) - 1):
        print(f"Processing morphing between image {i} and {i+1}")
        start_img = cv2.imread(images[i])
        end_img = cv2.imread(images[i + 1])

        for _ in range(hold_time * fps):
            out.write(start_img)

        for frame in range(morph_time * fps):
            alpha = frame / (morph_time * fps)
            morphed_img = morph_images(start_img, end_img, alpha)
            out.write(morphed_img)

    final_img = cv2.imread(images[-1])
    for _ in range(hold_time * fps):
        out.write(final_img)

    out.release()
    print("Morphing video created successfully.")

if __name__ == "__main__":
    images_folder = '.'
    temp_folder = tempfile.mkdtemp()

    images = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    processed_images = []

    for img_name in images:
        img_path = os.path.join(images_folder, img_name)
        image = cv2.imread(img_path)
        print(f"Processing image: {img_name}")
        processed_image = resize_and_blur_background(image)
        
        alpha_channel = create_alpha_channel(processed_image)
        processed_image_with_alpha = add_alpha_channel(processed_image, alpha_channel)
        
        temp_image_path = os.path.join(temp_folder, img_name)
        cv2.imwrite(temp_image_path, processed_image_with_alpha)
        processed_images.append(temp_image_path)

    create_morph_video(processed_images, 'output.mp4')
