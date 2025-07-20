import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import pickle
import argparse
from collections import Counter


# --- Helper Functions ---

def get_image_paths(coco_path):
    """Walks through the COCO directory to get all image paths."""
    image_paths = []
    for root, _, files in os.walk(coco_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def extract_features(image_paths, processor, model, device, max_images=None):
    """
    Extracts object features and labels from a list of images using DETR.
    """
    all_features = []
    all_labels = []

    if max_images:
        image_paths = image_paths[:max_images]

    for image_path in tqdm(image_paths, desc="Extracting object features"):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                # Get encoder's last hidden state
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]  # Shape: (1, num_queries, hidden_size)

                # Get object detection results to find out which queries correspond to objects
                logits = outputs.logits
                probas = logits.softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.9  # Confidence threshold

                # Collect features and labels for detected objects
                object_features = last_hidden_state[0, keep, :].cpu().numpy()
                object_labels = probas[keep].argmax(-1).cpu().numpy()

                if object_features.shape[0] > 0:
                    all_features.append(object_features)
                    all_labels.extend([model.config.id2label[label_id] for label_id in object_labels])

        except Exception as e:
            print(f"Skipping image {image_path} due to error: {e}")
            continue

    return np.vstack(all_features), all_labels


def build_library(features, labels, num_clusters):
    """
    Builds the prototype library using K-Means clustering.
    """
    print(f"Starting K-Means clustering with k={num_clusters}...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)

    # Associate each cluster with its most frequent label
    cluster_labels = {}
    for i in range(num_clusters):
        # Find all labels for data points in the current cluster
        labels_in_cluster = [labels[j] for j, cluster_id in enumerate(clusters) if cluster_id == i]
        if labels_in_cluster:
            # Find the most common label
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            cluster_labels[i] = most_common_label
        else:
            cluster_labels[i] = "unknown"

    prototype_library = {
        "prototypes": kmeans.cluster_centers_,  # The cluster centroids
        "cluster_to_label": cluster_labels  # Mapping from cluster index to string label
    }

    return prototype_library


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a visual prototype library from the COCO dataset.")
    parser.add_argument("--coco_path", type=str, required=True,
                        help="Path to the COCO dataset (e.g., train2017 directory).")
    parser.add_argument("--output_path", type=str, default="./checkpoints/prototype_library.pkl",
                        help="Path to save the final prototype library.")
    parser.add_argument("--num_clusters", type=int, default=512, help="Number of prototypes (K in K-Means).")
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Maximum number of images to process from COCO (for faster processing).")
    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load DETR model
    print("Loading DETR model...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    model.eval()

    # 1. Extract features from all images in the COCO dataset
    image_paths = get_image_paths(args.coco_path)
    features, labels = extract_features(image_paths, processor, model, device, args.max_images)

    # 2. Build library using K-Means
    prototype_library = build_library(features, labels, args.num_clusters)

    # 3. Save the library to a file
    print(f"Saving prototype library to {args.output_path}...")
    with open(args.output_path, "wb") as f:
        pickle.dump(prototype_library, f)

    print("Prototype library built successfully.")
    print(f"Total prototypes: {len(prototype_library['prototypes'])}")