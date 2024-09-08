import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('garbage.pt')

# Path to the image
image_path = "images/trash.jpg"

# Read the image
frame = cv2.imread(image_path)

# Run YOLOv8 inference on the image
resized_frame = cv2.resize(frame, (600, 500))
results = model(resized_frame, classes = [0, 2, 3, 4, 5, 6])

# Visualize the results on the frame
annotated_frame = results[0].plot()

# Display the annotated frame
cv2.imshow("YOLOv8 Inference", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
