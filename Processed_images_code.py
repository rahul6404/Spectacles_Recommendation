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
    vector1_normalized = vector_1 / np.linalg.norm(vector_1)
    vector2_normalized = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(vector1_normalized, vector2_normalized)
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    return round(angle_degrees, 2)

# Euclidean Distance function
def calculate_distance(point_1, point_2):
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

# Normalization function
def normalized_distance(distance, values):
    total = np.sum(values)
    return (distance / total)

# CSV file setup
csv_filename = 'merged_file_new.csv'
headers = ['chin_angle_1', 'chin_angle_2', 'cheek_bone_angle', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4',
           'ratio_5','ratio_6','n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8','n9','n10','n11', 'face_shape']

# Main Function
def process_image(image_path, subfolder_name):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    coordinates_list = []

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray_image, face)
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            coordinates_list.append((x, y))
            cv2.circle(image, (x, y), 2, (255, 255, 0), -1)
            #cv2.putText(image, str(n), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.destroyAllWindows()
    coordinates_list = np.array(coordinates_list)

    try:
        forehead_midpoint = ((coordinates_list[69][0] + coordinates_list[72][0])/2, (coordinates_list[69][1] + coordinates_list[72][1])/2)
        forehead_width = calculate_distance(coordinates_list[75], coordinates_list[79])
        cheekbone_width = calculate_distance(coordinates_list[36], coordinates_list[45])
        jawline_length = calculate_distance(coordinates_list[8], coordinates_list[12])
        facial_length = calculate_distance(forehead_midpoint, coordinates_list[8])
        D5 = calculate_distance(coordinates_list[2], coordinates_list[14])
        D6 = calculate_distance(coordinates_list[4], coordinates_list[12])
        D7 = calculate_distance(coordinates_list[6], coordinates_list[10])
        D8 = calculate_distance(coordinates_list[3], coordinates_list[13])
        eye_width = calculate_distance(coordinates_list[42], coordinates_list[45])
        eye_to_eyebrow = calculate_distance(coordinates_list[46], coordinates_list[24])
        nose_width = calculate_distance(coordinates_list[31], coordinates_list[35])
        forehead_height = calculate_distance(coordinates_list[72], coordinates_list[24])

        values = [eye_width, eye_to_eyebrow, nose_width, forehead_height, forehead_width,
                  cheekbone_width, jawline_length, facial_length, D5, D6, D7]

        n1 = normalized_distance(forehead_width, values)
        n2 = normalized_distance(cheekbone_width, values)
        n3 = normalized_distance(jawline_length, values)
        n4 = normalized_distance(facial_length, values)
        n5 = normalized_distance(D5, values)
        n6 = normalized_distance(D6, values)
        n7 = normalized_distance(D7, values)
        n8 = normalized_distance(eye_width, values)
        n9 = normalized_distance(eye_to_eyebrow, values)
        n10 = normalized_distance(nose_width, values)
        n11 = normalized_distance(forehead_height, values)

        chin_tip = np.array(coordinates_list[8])
        lip_tip = np.array(coordinates_list[57])
        jaw_edge_tip_1r = np.array(coordinates_list[10])
        jaw_edge_tip_2r = np.array(coordinates_list[12])
        jaw_edge_tip_1l = np.array(coordinates_list[6])
        jaw_edge_tip_2l = np.array(coordinates_list[4])
        nose_tip = np.array(coordinates_list[30])
        cheek_tip = np.array(coordinates_list[14])

        chin_to_nose_vector = lip_tip - chin_tip
        jawline_vector_1r = jaw_edge_tip_1r - chin_tip
        jawline_vector_2r = jaw_edge_tip_2r - chin_tip
        jawline_vector_1l = jaw_edge_tip_1l - chin_tip
        jawline_vector_2l = jaw_edge_tip_2l - chin_tip
        cheek_to_nose_vector = nose_tip - cheek_tip
        jawline_vector_3 = jaw_edge_tip_2r - cheek_tip

        chin_angle_1 = (calculate_angle(chin_to_nose_vector, jawline_vector_1r) + calculate_angle(chin_to_nose_vector, jawline_vector_1l)) / 2
        chin_angle_2 = (calculate_angle(chin_to_nose_vector, jawline_vector_2r) + calculate_angle(chin_to_nose_vector, jawline_vector_2l)) / 2
        cheek_bone_angle = calculate_angle(cheek_to_nose_vector, jawline_vector_3)

        ratio_1 = round(forehead_width / jawline_length, 2)
        ratio_2 = round(facial_length / cheekbone_width, 2)
        ratio_3 = round(facial_length / jawline_length, 2)
        ratio_4 = round(forehead_width / cheekbone_width, 2)
        ratio_5 = round(D8 / facial_length, 2)
        ratio_6 = round(forehead_width / facial_length, 2)

        cv2.putText(image, f'Folder: {subfolder_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f'Chin Angle 1: {chin_angle_1}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        save_path = os.path.join('processed_images', subfolder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, os.path.basename(image_path)), image)

        return [chin_angle_1, chin_angle_2, cheek_bone_angle, ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6,
                n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11]
    except IndexError as e:
        print(f"Index out of Range: {e}")

with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for subfolder in os.listdir('face_shapes'):
        subfolder_path = os.path.join('face_shapes', subfolder)
        if os.path.isdir(subfolder_path):
            for image_file in glob.glob(os.path.join(subfolder_path, '*.jpg')):
                measurements = process_image(image_file, subfolder)
                if measurements is not None:
                    writer.writerow(measurements + [subfolder])
                else:
                    continue
