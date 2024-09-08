import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('garbage.pt')

# Open the video file
video_path = "home.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        resized_frame = cv2.resize(frame, (600, 500))
        results = model(resized_frame, classes=[0, 63, 66, 67, 73])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # print('Results:', results[0].names)
        # print('Type:', type(results[0]))

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
