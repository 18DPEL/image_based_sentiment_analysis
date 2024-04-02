from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

# Path to the image you want to analyze
img_path = 'download.jpeg'

# Load the image
img = cv2.imread(img_path)

# Analyze the sentiment of the image
results = DeepFace.analyze(img_path, actions=['emotion'])

# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Find the face with the highest emotion accuracy
max_emotion_face = max(results, key=lambda x: max(x['emotion'].values()))

# Get the highest accuracy emotion
emotion = max_emotion_face['emotion']
max_emotion = max(emotion, key=emotion.get)

# Get the bounding box coordinates of the face
x, y, w, h = max_emotion_face['region']['x'], max_emotion_face['region']['y'], max_emotion_face['region']['w'], max_emotion_face['region']['h']

# Draw a bounding box around the face
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Annotate the image with the highest accuracy emotion
plt.text(x, y-10, f"{max_emotion}: {emotion[max_emotion]:.2f}%", color='red', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

# Show the image with the bounding box and annotations
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
