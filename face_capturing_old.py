import cv2
import dlib
import numpy as np
import math
import time
from landmark_detection import measurements_calculation
from prediction import predict_face_shape

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this model from dlib's website

# Define the 3D model points of facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
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


def draw_axis(frame, landmarks, rotation_vector, translation_vector):
    # Define the length of the axis
    axis_length = 100.0

    # Define the axis in 3D space
    axis = np.float32([
        [axis_length, 0, 0],  # X-axis (Red)
        [0, axis_length, 0],  # Y-axis (Green)
        [0, 0, axis_length]  # Z-axis (Blue)
    ]).reshape(-1, 3)

    # Project the 3D axis points to the 2D image plane
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Draw the axes
    corner = tuple(landmarks[30].ravel())  # Nose tip
    imgpts = imgpts.astype(int)
    frame = cv2.line(frame, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # X-axis in red
    frame = cv2.line(frame, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # Y-axis in green
    frame = cv2.line(frame, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # Z-axis in blue
    return frame


image_captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get the landmarks/parts for the face
        shape = predictor(gray, face)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        # 2D image points. If you change the index, you change the landmark.
        image_points = np.array([
            (landmarks[30][0], landmarks[30][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),  # Chin
            (landmarks[36][0], landmarks[36][1]),  # Left eye left corner
            (landmarks[45][0], landmarks[45][1]),  # Right eye right corner
            (landmarks[48][0], landmarks[48][1]),  # Left Mouth corner
            (landmarks[54][0], landmarks[54][1])  # Right mouth corner
        ], dtype="double")

        # Solve PnP problem
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs)

        # Get rotational matrix
        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

        # Combine the rotation matrix and translation vector to form the transformation matrix
        transformation_matrix = np.hstack((rotation_matrix, translation_vector))

        print(transformation_matrix)

        # Decompose the projection matrix to get the Euler angles
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(transformation_matrix)
        pitch, yaw, roll = [(euler_angles[i][0]) for i in range(3)]

        print(euler_angles)

        # Display the angles on the frame
        if not image_captured:
            cv2.putText(frame, f"Roll: {roll:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw the 3D axis
            frame = draw_axis(frame, landmarks, rotation_vector, translation_vector)

        # Capture image when roll value is within [-10, 10]
        if not image_captured and -10 < roll < 10 and (170 <= pitch <= 180 or -170 <= pitch <= -180):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_captured = True

    # Resize the frame to 1080 pixels for display
    frame_display = cv2.resize(frame.copy(), (resolution_width, resolution_height))

    # Display the resulting frame
    cv2.imshow('Head Pose Estimation', frame_display)

    # Break the loop and save the image when roll is within [-10, 10]
    if image_captured:
        # Save the original frame without overlays
        cv2.imwrite(f'headpose_replaced.png', frame_display)
        print(f'Image captured at headpose_replaced')
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()