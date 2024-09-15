import dlib
import cv2
import numpy as np
import os
import csv
import glob

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')


# angle calculation function
def calculate_angle(vector_1, vector_2):
    # Normalize the vectors
    vector1_normalized = vector_1 / np.linalg.norm(vector_1)
    vector2_normalized = vector_2 / np.linalg.norm(vector_2)

    # Calculate the dot product
    dot_product = np.dot(vector1_normalized, vector2_normalized)

    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return round(angle_degrees, 2)

# Euclidean Distance function
def calculate_distance(point_1, point_2):
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

# CSV file setup
csv_filename = 'merged_file_5.csv'
# Define the headers for the CSV file
headers = ['r1', 'r2', 'r3', 'r4', 'chin_angle_1', 'face_shape']

# Main Function
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use detector to find face landmarks
    faces = detector(gray_image)

    # Initialising an empty numpy array
    coordinates_list = []

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        # Draw a rectangle around the face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Look for the landmarks
        landmarks = predictor(gray_image, face)

        # Loop through all the points
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Print the landmark number and its coordinates
            #print(f'Landmark #{n}: ({x}, {y})')
            coordinates_list.append((x, y))

            # Draw a circle on each landmark point
            cv2.circle(image, (x, y), 2, (255, 255, 0), -1)

            # Put a number (n) near each point
            cv2.putText(image, str(n), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Display the output
    cv2.imshow('81 Landmarks with Numbering', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # changing the list into numpy array
    coordinates_list = np.array(coordinates_list)
    # print(coordinates_list)

    # Facial Distances
    forehead_midpoint = ((coordinates_list[69][0] + coordinates_list[72][0])/2,
                         (coordinates_list[69][1] + coordinates_list[72][1])/2)
    forehead_width = calculate_distance(coordinates_list[75], coordinates_list[79])
    facial_length  = calculate_distance(forehead_midpoint, coordinates_list[8])
    cheekbone_width = calculate_distance(coordinates_list[27], coordinates_list[45]) * 2
    jawline_length = calculate_distance(coordinates_list[8], coordinates_list[12])
    D8 = calculate_distance(coordinates_list[3], coordinates_list[13])
    # Ratios

    ratio_1 = round(forehead_width / jawline_length, 2)
    ratio_2 = round(facial_length / cheekbone_width, 2)
    ratio_3 = round(facial_length / jawline_length, 2)
    ratio_4 = round(forehead_width / cheekbone_width, 2)
    ratio_5 = round(D8 / facial_length, 2)
    ratio_6 = round(forehead_width / facial_length, 2)

    # Angle Measurements
    chin_tip = np.array(coordinates_list[8])
    lip_tip  = np.array(coordinates_list[57])
    jaw_edge_tip_1 = np.array(coordinates_list[10])

    chin_to_nose_vector = lip_tip - chin_tip
    jawline_vector_1 = jaw_edge_tip_1 - chin_tip

    chin_angle_1 = calculate_angle(chin_to_nose_vector, jawline_vector_1)

    print(forehead_width, facial_length, cheekbone_width, jawline_length)
    print(ratio_1, ratio_2, ratio_3,ratio_4, ratio_5, ratio_6)

process_image("C:/Users/rahul/Downloads/sandeep_face.jpg")



