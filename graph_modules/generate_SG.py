# THis file generates scene graph for images in the dataset. 
# This file takes a json file containg bounding box for every image as its input.
# This input json file is genertaed using groundedSAM.

import json
import numpy as np
import torch
import math
import sys
sys.path.append('./')
from model.relationnet import RelationNet

spatial_relations = ["on", "next to", "behind", "in front of",  "above", "across", "below", "inside", "under", "left" , "right", "in" , "None"]


def instanciate_objects(object_list):
    element_counts = {}

    converted_list = []
    for element in object_list:
        element = element.split(' ')[0]
        if element not in element_counts:
            element_counts[element] = 0
        converted_list.append(f"{element}_{element_counts[element]}")
        element_counts[element] += 1
    
    return converted_list

def predict_relation(object_features, relation_model):
    with torch.no_grad():
        outputs = relation_model(object_features)
        # print(outputs)
        _, predicted = torch.max(outputs, 1)
        predicted_relation = predicted.item()
    return predicted_relation


# Function to generate relation labels for object pairs
def generate_relation_labels(image_data):

    W,H = 640, 480
    phrases = image_data["phrases"]
    
    phrases = instanciate_objects(phrases)

    boxes = image_data["boxes"]
    n_boxes = len(boxes)

    n_pairs = n_boxes+5

    relations = {}
    # Generate random pairs of object indices
    object_pairs = np.random.choice(range(n_boxes), n_pairs * 2, replace = True)

    for i in range(0, len(object_pairs), 2):
        pair1 = object_pairs[i]
        pair2 = object_pairs[i + 1]

        obj1 = phrases[pair1]
        bbox1 = boxes[pair1]

        obj2 = phrases[pair2]
        bbox2 = boxes[pair2]

        x1, y1, x2, y2 = bbox1
        x1,y1,w1,h1 = x1, y1, x2-x1, y2-y1
        
        x1_prime, y1_prime, x2_prime, y2_prime = bbox2
        # print(x1_prime, y1_prime, x2_prime, y2_prime)
        x1_prime,y1_prime,w1_prime,h1_prime = x1_prime, y1_prime, x2_prime-x1_prime, y2_prime-y1_prime
        # print(x1_prime, y1_prime, w1_prime, h1_prime)

        # Extract features for object 1
        feat1 =  torch.tensor([x1/W, y1/H, (x1+w1)/W, (y1+h1)/H, w1/W, h1/H, (w1*h1)/(W*H)])
        feat2 =  torch.tensor([x1_prime/W, y1_prime/H, (x1_prime+w1_prime)/W, (y1_prime+h1_prime)/H, w1_prime/W, h1_prime/H, (w1_prime*h1_prime)/(W*H)])
        scale_inv = torch.tensor([(x1-x1_prime)/w1_prime, (y1-y1_prime)/h1_prime, math.log(w1/w1_prime), math.log(h1/h1_prime)])

        feat = torch.cat([feat1, feat2, scale_inv], dim=0).unsqueeze(0)
        
        # Use your model to predict relation (replace predict_relation with your function)
        predicted_relation = predict_relation(feat, relation_model)
        # print(spatial_relations[predicted_relation])
        
        # Store the relation between obj1 and obj2
        relation_key = f"{obj1}-{obj2}"
        relations[relation_key] = spatial_relations[predicted_relation]

    return relations, phrases

configs = {
    "input_dim": 18,
    "hidden_dim": 13,
    "num_classes": 13
}


relation_model = RelationNet(configs)  # Assuming this is the correct initialization method for your RelationNet model
relation_model.load_state_dict(torch.load("trained_relationnet.pth"))  # Assuming you have a pre-trained RelationNet model
relation_model.eval()


# Load your JSON data
with open('./datasets/detection_results_50salads.json', 'r') as f:
    data = json.load(f)

# Iterate through folders and images to generate relations
result = {}
for folder_key, folder_data in data.items():
    result[folder_key] = {}
    for image_key, image_data in folder_data.items():
        result[folder_key][image_key]  = {}
        result[folder_key][image_key]["relations"], result[folder_key][image_key]["objects"] = generate_relation_labels(image_data)  # Adjust n_pairs as needed

# Save the result to a new JSON file
with open('./datasets/objects_relations_50salads.json', 'w') as f:
    json.dump(result, f, indent=4)
