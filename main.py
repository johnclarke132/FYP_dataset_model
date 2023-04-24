import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import json
import cv2
import os
import numpy as np
import glob
from google.cloud import storage

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "fypdataset-b5446ff94dc1.json"
#
# # df = pd.read_json('bicepcurl.jsonl', lines=True)
#
# # load the jsonl file
# with open('bicepcurl.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f]
#
# # create X and y
#
# # Create a client to interact with the storage service
# client = storage.Client()
#
# X = []
# y = []
# for row in data:
#     video_uri = row['videoGcsUri']
#     bucket_name, blob_name = video_uri.replace('gs://', '').split('/', 1)
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     video_bytes = blob.download_as_bytes()
#     print(video_bytes)
#
#     # Convert the video bytes to a numpy array of frames
#     cap = cv2.VideoCapture(cv2.CAP_ANY)
#     cap.open(video_bytes)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # preprocess the frame as needed
#         frame = cv2.resize(frame, (224, 224))
#         frame = np.array(frame, dtype=np.float32)
#         frames.append(frame)
#     cap.release()
#     frames = np.array(frames)
#
#     # reshape the frames array and add to X
#     if len(frames.shape) == 3:
#         frames = np.expand_dims(frames, axis=0)
#     frames = np.squeeze(frames)
#     frames = frames.reshape(-1, frames.shape[2], frames.shape[3], frames.shape[4])
#     X.append(frames)
#
#     # add the annotation to y
#     y.append(row['timeSegmentAnnotations'][0]['displayName'])
#
# # print(X,y)

#
# step 1, set up the captures:
#
caps = []
videoList = glob.glob(r'\dataset_location_test.txt')
for path in videoList:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("error opening ", path)
    else:
        caps.append(cap)
#
# step 2, iterate through videoframes, collect images, and stitch them
#
while True:
    frames = []  # frames for one timestep
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            # return # game over
            pass
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB color space
        frames.append(frame)
        print("frame appended")
        # cap.release()
    break

    # stitch a new frame from frames
    # append to output video
print("frames[] complete")


# convert the frames to a numpy array and reshape to (num_frames, height, width, channels)
frames = np.array(frames)
frames = frames.reshape((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
# return the feature vector (e.g., average or max pool over frames)
feature_vector = np.mean(frames, axis=0)
print("frames converted and reshaped")

X_train_features = [feature_vector for video_path in X_train]
X_test_features = [feature_vector for video_path in X_test]
print("test, train features created")

# train a classification model
from sklearn.svm import SVC

model = SVC()
model.fit(X_train_features, y_train)

# evaluate the model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
