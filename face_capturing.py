import cv2
import dlib
import numpy as np
import time
import os
import glob
from PIL import Image, ImageDraw
from collections import Counter
from landmark_detection import measurements_calculation
from prediction import predict_face_shape
import warnings
warnings.filterwarnings('ignore')

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this model from dlib's website

# Define the 3D model points of facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals
resolution_width = 1080
resolution_height = 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)

focal_length = resolution_height
center = (resolution_width / 2, resolution_height / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]], dtype="double")

# Assuming no lens distortion
dist_coeffs = np.zeros((4, 1))

def clear_folder(directory):
    files = glob.glob(os.path.join(directory, '*.jpg'))
    for f in files:
        os.remove(f)

# Directory to save the captured images
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)
clear_folder(output_dir)

# Number of images to capture
num_images_to_capture = 11
images_captured = 0

# Capture 10 images
while images_captured < num_images_to_capture:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        image_points = np.array([
            (landmarks[30][0], landmarks[30][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),    # Chin
            (landmarks[36][0], landmarks[36][1]),  # Left eye left corner
            (landmarks[45][0], landmarks[45][1]),  # Right eye right corner
            (landmarks[48][0], landmarks[48][1]),  # Left Mouth corner
            (landmarks[54][0], landmarks[54][1])   # Right mouth corner
        ], dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
        transformation_matrix = np.hstack((rotation_matrix, translation_vector))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(transformation_matrix)
        pitch, yaw, roll = [(euler_angles[i][0]) for i in range(3)]

        # Save the frame
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"headpose_{images_captured + 1}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f'Image {images_captured + 1} captured: {filepath}')
        images_captured += 1

        # Wait for 2 seconds before capturing the next image
        time.sleep(2)

    frame_display = cv2.resize(frame.copy(), (resolution_width, resolution_height))
    cv2.imshow('Head Pose Estimation', frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Run webcam without capturing images
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        # Draw the landmarks on the frame
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        image_points = np.array([
            (landmarks[30][0], landmarks[30][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),    # Chin
            (landmarks[36][0], landmarks[36][1]),  # Left eye left corner
            (landmarks[45][0], landmarks[45][1]),  # Right eye right corner
            (landmarks[48][0], landmarks[48][1]),  # Left Mouth corner
            (landmarks[54][0], landmarks[54][1])   # Right mouth corner
        ], dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
        transformation_matrix = np.hstack((rotation_matrix, translation_vector))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(transformation_matrix)
        pitch, yaw, roll = [(euler_angles[i][0]) for i in range(3)]

        # Display the roll, pitch, and yaw on the frame
        cv2.putText(frame, f"Roll: {roll:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame_display = cv2.resize(frame.copy(), (resolution_width, resolution_height))
    cv2.imshow('Live Webcam Feed with Landmarks and Head Pose', frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

measurements_calculation()
prediction_list = predict_face_shape()
print(prediction_list)


# Create a Counter object
counter = Counter(prediction_list)

# Find the most common element
mce, mcc = counter.most_common(1)[0]
confidence = round((mcc/len(prediction_list)*100),0)

face_shapes_dict = {0: {"Face Shape": "Heart shape", "Suitable Spectacles": ["Cateye frames"]},
                    1: {"Face Shape": "Oblong shape", "Suitable Spectacles": ["Oversized frames", "Rectangular frames"]},
                    2: {"Face Shape": "Oval shape", "Suitable Spectacles": ["Round frames", "Rectangular frames"]},
                    3: {"Face Shape": "Round shape", "Suitable Spectacles": ["Cateye frames", "Aviators", "Wayfarers"]},
                    4: {"Face Shape": "Square shape", "Suitable Spectacles": ["Rectangle frames", "Square frames"]}}

spectacles_images_dict = {0: "Spectacle_Images/Cateye.png",
                          1: "Spectacle_Images/Oversized&Rectangle.png",
                          2: "Spectacle_Images/Round&Rectangle.png",
                          3: "Spectacle_Images/Cat&Avia&Wayf.png",
                          4: "Spectacle_Images/Square&Rect.png"}


print(f"The most frequent value and it's count is: {mce, mcc}")
print(face_shapes_dict[mce])
print(confidence)

#-------------------------------------------------
img = cv2.imread('captured_images/headpose_11.jpg')
spect_img = cv2.imread(spectacles_images_dict[mce])
font = cv2.FONT_HERSHEY_SIMPLEX
position_1 = (10, 25)
position_2 = (10, 55)# Specify the (x, y) coordinates where you want to place the text
font_scale = 1
font_color = (0, 255, 0)  # White color (BGR format)
thickness = 2
cv2.putText(img, f'Confidence level: {confidence}%', position_1, font, font_scale, font_color, thickness)
cv2.putText(img, f'Face Shape: {face_shapes_dict[mce]['Face Shape']}', position_2, font, font_scale, font_color, thickness)

user_image_path = input("Enter the name of the image to save: ")
cv2.imwrite(f'saved_images/{user_image_path}.jpg', img)
cv2.imshow('User Image', img)

# Create a named window
cv2.namedWindow('Resized_Window', cv2.WINDOW_NORMAL)

# Resize the window
cv2.resizeWindow('Resized_Window', 800, 600)  # Width: 800, Height: 600

# Display the image in the resized window
cv2.imshow('Resized_Window', spect_img)

cv2.waitKey(0)
cv2.destroyAllWindows()







