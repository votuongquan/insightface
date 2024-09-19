import argparse
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from retinaface import RetinaFace

from backbones import get_model

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

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

def detect_face(img_path):
    img = cv2.imread(img_path)
    faces = RetinaFace.detect_faces(img)
    if faces:
        # Assuming the first detected face is the one we want
        face = faces[list(faces.keys())[0]]
        facial_area = face['facial_area']
        x1, y1, x2, y2 = facial_area
        cropped_face = img[y1:y2, x1:x2]
        return cropped_face
    else:
        raise ValueError("No face detected in the image")

def verify_images(weight, name, img1, img2, threshold=0.5):
    # Detect and crop faces in both images
    face1 = detect_face(img1)
    face2 = detect_face(img2)
    
    # Save the cropped faces temporarily
    cv2.imwrite('temp_face1.jpg', face1)
    cv2.imwrite('temp_face2.jpg', face2)
    
    # Extract embeddings for both cropped faces
    embedding1 = inference(weight, name, 'temp_face1.jpg')
    embedding2 = inference(weight, name, 'temp_face2.jpg')
    
    # Compute cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    # Determine if the images are similar based on the threshold
    is_similar = similarity > threshold
    return is_similar, similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img1', type=str, required=True, help='Path to the first image')
    parser.add_argument('--img2', type=str, required=True, help='Path to the second image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    args = parser.parse_args()
    
    is_similar, similarity = verify_images(args.weight, args.network, args.img1, args.img2, args.threshold)
    print(f"Similarity: {similarity}")
    print(f"Are the images similar? {'Yes' if is_similar else 'No'}")