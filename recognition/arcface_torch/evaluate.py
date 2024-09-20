import os
import argparse
import shutil
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from retinaface import RetinaFace

from backbones import get_model

def embed_image(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        if isinstance(img, str) and os.path.isfile(img):
            img = cv2.imread(img)
            img = cv2.resize(img, (112, 112))
        else:
            raise ValueError("The provided img path is not valid.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    with torch.no_grad():
        feat = net(img).numpy()
    return feat

def crop_image(image, facial_area):
    x1, y1, x2, y2 = facial_area
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def detect_and_crop_largest_face(img_path):
    # Read the image
    img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"Error reading image {img_path}")
    
    # Detect faces in the image
    faces = RetinaFace.detect_faces(img)
    
    if not faces:
        raise ValueError(f"No faces detected in image {img_path}")
    
    # Find the largest face
    largest_face = None
    max_area = 0
    for key in faces.keys():
        face = faces[key]
        facial_area = face['facial_area']
        area = (facial_area[2] - facial_area[0]) * (facial_area[3] - facial_area[1])
        if area > max_area:
            max_area = area
            largest_face = facial_area
    
    if largest_face is None:
        raise ValueError(f"No faces detected in image {img_path}")
    
    # Crop the largest face region from the image
    cropped_face = crop_image(img, largest_face)
    return cropped_face

def detect_face(img_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        try:
            cropped_face = detect_and_crop_largest_face(img_path)
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, cropped_face)
        except ValueError as e:
            print(e)

def embed_identities(weight, name, img_folder):
    face_embeddings = []
    labels = []
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        face_embedding = embed_image(weight, name, img_path)
        face_embeddings.append(face_embedding)
        labels.append(img_name)
    
    # Convert the list of embeddings to a NumPy array (matrix)
    face_embeddings_matrix = np.vstack(face_embeddings)
    
    return face_embeddings_matrix, labels

def find_most_similar_embedding(weight, name, img_path, face_embeddings_matrix, face_labels):
    # Detect and crop the largest face in the given image
    cropped_face = detect_and_crop_largest_face(img_path)
    
    # Save the cropped face to a new image file in a temporary folder
    temp_crop_folder = 'temp_crop_faces'
    if not os.path.exists(temp_crop_folder):
        os.makedirs(temp_crop_folder)
    cropped_face_filename = os.path.join(temp_crop_folder, os.path.basename(img_path))
    cv2.imwrite(cropped_face_filename, cropped_face)
    
    # Embed the cropped face
    img_embedding = embed_image(weight, name, cropped_face_filename)
    
    # Ensure the embeddings are 2D arrays
    img_embedding = img_embedding.flatten().reshape(1, -1)
    face_embeddings_matrix = face_embeddings_matrix.reshape(face_embeddings_matrix.shape[0], -1)
    
    # Compute cosine similarity between the given image's embedding and each embedding in the matrix
    similarities = cosine_similarity(img_embedding, face_embeddings_matrix)
    
    # Find the index of the embedding with the highest similarity
    most_similar_index = np.argmax(similarities)
    
    # Return the index and the corresponding label
    return most_similar_index, face_labels[most_similar_index]

def load_image_paths_and_labels(label_file):
    image_paths = []
    labels = []
    with open(label_file, 'r') as file:
        for line in file:
            path, label = line.strip().split(',')
            image_paths.append(path)
            labels.append(label)
    return image_paths, labels

def main(weight, name, identity_folder, label_file):
    # Load image paths and labels from the label file
    image_paths, labels = load_image_paths_and_labels(label_file)
    
    # Step 1: Detect and crop the largest face in each image in the identity folder and save them to the temp folder
    temp_identity_folder = 'temp_identity_faces'
    detect_face(identity_folder, temp_identity_folder)
    
    # Step 2: Embed every image in the temp identity folder into a matrix with labels
    face_embeddings_matrix, face_labels = embed_identities(weight, name, temp_identity_folder)
    
    # Initialize counters for accuracy calculation
    correct_predictions = 0
    total_predictions = 0
    
    # Step 3: Iterate over all loaded image paths and find the most similar embedding for each one
    for img_path, label in zip(image_paths, labels):
        try:
            most_similar_index, most_similar_label = find_most_similar_embedding(weight, name, img_path, face_embeddings_matrix, face_labels)
            total_predictions += 1
            most_similar_label = most_similar_label.split('.')[0]

            if most_similar_label == label:
                correct_predictions += 1
                print(f"Image: {img_path} - Most similar image: {most_similar_label} (Index: {most_similar_index}) - Correct")
            else:
                print(f"Image: {img_path} - Most similar image: {most_similar_label} (Index: {most_similar_index}) - Incorrect")
        except ValueError as e:
            print(e)
    
    # Calculate and print accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Step 4: Delete the temporary folders
    shutil.rmtree(temp_identity_folder)
    shutil.rmtree('temp_crop_faces')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate face recognition model using ArcFace embeddings.')
    parser.add_argument('--weight', type=str, required=True, help='Path to the model weight file.')
    parser.add_argument('--network', type=str, default='r50', help='Model name (r50, r100, etc.).')
    parser.add_argument('--identity_folder', type=str, required=True, help='Path to the folder containing identity images.')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file containing image paths and labels.')
    
    args = parser.parse_args()
    
    main(args.weight, args.network, args.identity_folder, args.label_file)