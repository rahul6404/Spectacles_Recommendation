from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
import dlib
from imutils import face_utils
import os

app = Flask(__name__)

# Load the detector and predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

camera = cv2.VideoCapture(0)

# Reference points for head pose estimation
object_pts = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float64)

# Camera internals
size = (640, 480)
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]], dtype="double")

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Initialize previous frame for motion detection
previous_frame = None

# Buffer to store recent MAR values for stability check
recent_mar_values = []

# Track captured image status
captured_image_path = 'captured_image.jpg'
image_captured = False

# Global variable to control capturing state
capturing = True


@app.route('/')
def index():
    return render_template('index.html', captured_image=image_captured)


def align_face(frame, landmarks):
    # Extract the coordinates of the left and right eye
    left_eye_pts = landmarks[36:42]
    right_eye_pts = landmarks[42:48]

    left_eye_center = left_eye_pts.mean(axis=0).astype("int")
    right_eye_center = right_eye_pts.mean(axis=0).astype("int")

    # Compute the angle between the eye centroids
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # Determine the scale of the new image by computing the ratio of the distance between the eyes in the current frame to a known distance (e.g., 100 pixels)
    desired_right_eye_x = 1.0 - 0.35
    dist_between_eyes = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - 0.35)
    desired_dist *= 100
    scale = desired_dist / dist_between_eyes

    # Get the center of both eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

    # Grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Update the translation component of the matrix
    tX = size[0] * 0.5
    tY = size[1] * 0.4
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    # Apply the affine transformation
    (w, h) = (size[0], size[1])
    output = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

    return output


def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


def detect_face_and_pose(frame):
    global previous_frame
    global recent_mar_values

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) != 1:
        return frame, None, False, False, True  # Return invalid if no face or multiple faces are detected

    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)

    # Head pose estimation
    image_pts = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],  # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left Mouth corner
        landmarks[54]  # Right mouth corner
    ], dtype=np.float64)

    _, rotation_vector, translation_vector = cv2.solvePnP(object_pts, image_pts, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    pitch, yaw, roll = euler_angles.flatten()

    # Motion detection
    motion_detected = False
    if previous_frame is not None:
        diff_frame = cv2.absdiff(previous_frame, gray)
        _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        motion_detected = np.sum(thresh_frame) > 10000  # Threshold for motion

    previous_frame = gray.copy()

    # Check if face is frontal
    is_frontal = -10 < yaw < 10 and (170 <= pitch <= 180 or -170 <= pitch <= -180) and -10 < roll < 10

    # Check mouth aspect ratio
    mouth = landmarks[48:68]  # Mouth landmarks
    mar = mouth_aspect_ratio(mouth)

    # Update buffer with recent MAR values
    recent_mar_values.append(mar)
    if len(recent_mar_values) > 10:  # Keep last 10 frames
        recent_mar_values.pop(0)

    # Calculate average MAR for stability check
    average_mar = np.mean(recent_mar_values)

    # Define MAR thresholds
    MAR_THRESHOLD = 0.5
    MAR_STABLE_THRESHOLD = 0.45  # Stricter threshold for stability

    # Check if mouth is open
    mouth_open = average_mar > MAR_THRESHOLD
    mouth_stable = average_mar < MAR_STABLE_THRESHOLD

    # Obstruction detection: Check if any landmarks around the mouth are missing
    obstruction_detected = any((point == (0, 0)).all() for point in mouth)

    # Image is invalid if the mouth is open or if there's an obstruction
    invalid_image = mouth_open or obstruction_detected or not mouth_stable

    return frame, landmarks, is_frontal, motion_detected, invalid_image


def gen_frames():
    global capturing
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, landmarks, is_frontal, motion_detected, invalid_image = detect_face_and_pose(frame)
            if landmarks is not None:
                if is_frontal and not motion_detected and not invalid_image and capturing:
                    cv2.imwrite(captured_image_path, processed_frame)
                    global image_captured
                    image_captured = True
                    capturing = False  # Stop capturing

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_image', methods=['POST'])
def capture_image():
    global camera
    global image_captured
    global capturing

    if capturing:
        success, frame = camera.read()
        if success:
            processed_frame, _, is_frontal, motion_detected, invalid_image = detect_face_and_pose(frame)
            if is_frontal and not motion_detected and not invalid_image:
                filename = os.path.join(IMAGE_DIR, 'captured_image.jpg')
                cv2.imwrite(filename, processed_frame)
                image_captured = True
                capturing = False  # Stop capturing
                return jsonify({'status': 'success', 'message': 'Image captured successfully'})
            else:
                return jsonify(
                    {'status': 'error', 'message': 'Failed to capture image. Ensure face is frontal and still.'})
        return jsonify({'status': 'error', 'message': 'Error capturing image'})
    return jsonify({'status': 'error', 'message': 'Currently not capturing'})


@app.route('/image_status', methods=['GET'])
def image_status():
    global image_captured
    return jsonify({'image_captured': image_captured})


@app.route('/accept')
def accept():
    global image_captured
    global capturing
    image_captured = False
    capturing = True  # Restart capturing
    return redirect(url_for('index', message='Image accepted.'))


@app.route('/recapture')
def recapture():
    global image_captured
    global capturing
    if os.path.exists(captured_image_path):
        os.remove(captured_image_path)
    image_captured = False
    capturing = True  # Restart capturing
    return redirect(url_for('index', message='Image discarded. Please recapture.'))


@app.route('/image')
def image():
    if os.path.exists(captured_image_path):
        return Response(
            open(captured_image_path, 'rb').read(),
            mimetype='image/jpeg'
        )
    return redirect(url_for('index', message='No image to display'))


IMAGE_DIR = 'static/captured_images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

captured_image_path = os.path.join(IMAGE_DIR, 'captured_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)