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

csv_filename = 'merged_file_5.csv'
headers = ['normalized_coordinates', 'face_shape']

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
    # cv2.imshow('81 Landmarks with Numbering', image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    # changing the list into numpy array
    coordinates_list = np.array(coordinates_list)
    coordinates_list = coordinates_list - coordinates_list[30]
    # print(coordinates_list)

    # Normalized coordinates

    # Assuming you have a numpy array called 'coordinates' with shape (81, 2)
    # Each row contains (x, y) coordinates as tuples

    # Extract x and y coordinates
    x_coords = coordinates_list[:, 0]
    y_coords = coordinates_list[:, 1]

    # Normalize x and y coordinates
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Avoid division by zero (in case x_max - x_min is zero)
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    # Normalize x and y coordinates
    normalized_x = (x_coords) / x_range
    normalized_y = (y_coords) / y_range

    normalized_coordinates = []
    for i in range(81):
        normalized_coordinates.append((normalized_x[i], normalized_y[i]))

    # normalized_coordinates = np.array(normalized_coordinates)

    return [normalized_coordinates]




# Open the CSV file in write mode
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(headers)
    # Loop through all the files in the folder
    for subfolder in os.listdir('face_shapes'):
        subfolder_path = os.path.join('face_shapes', subfolder)
        if os.path.isdir(subfolder_path):
            # Loop through each image file in the subfolder
            for image_file in glob.glob(os.path.join(subfolder_path, '*.jpg')):
                # Call the function to process each image and get the measurements
                measurements = process_image(image_file)
                # Write the measurements along with the filename and target to the CSV
                writer.writerow(measurements + [subfolder])

