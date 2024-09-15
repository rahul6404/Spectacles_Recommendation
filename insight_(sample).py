import cv2
from insightface.app import FaceAnalysis

# Initialize the FaceAnalysis app with the buffalo-l model
app = FaceAnalysis(name='buffalo_l')

# Prepare the model (set ctx_id to -1 to use CPU, or to your GPU ID to use GPU)
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load your image (replace 'your_image.jpg' with the path to your sample image)
img = cv2.imread('face_shapes/Round_face_shape/round (180).jpg')

# Add 50px black padding around the image
padded_img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Perform face analysis on the padded image
faces = app.get(padded_img)

# Draw all landmarks on the padded image
for face in faces:
    n = 0
    for landmark in face.landmark_3d_68:  # This assumes the model provides 68 3D landmarks
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(padded_img, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(padded_img, str(n), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        n += 1

# Save the output image with padding
cv2.imwrite('New_img_2.jpg', padded_img)
