import cv2
import dlib
import os
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import face_recognition


detector = dlib.get_frontal_face_detector()



def extract_faces(video_path, output_dir, name_prefix):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize an empty list to store face coordinates
    faces = []

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize a tqdm progress bar
    progress_bar = tqdm(total=total_frames, desc='Extracting Faces')

    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Break if we have reached the end of the video
        if not ret:
            break

        # Update the progress bar
        progress_bar.update(1)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        rects = detector(gray, 0)

        # Loop through the detected faces
        for i, rect in enumerate(rects):
            # Extract the face from the frame
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            face = frame[y1:y2, x1:x2]

            # Save the face as an image file
            filename = f"{name_prefix}_{i}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, face)

            # Add the face coordinates to the list
            faces.append({
                "filename": filename,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    # Release the video file
    cap.release()

    # Close the tqdm progress bar
    progress_bar.close()

    # Return the list of face coordinates
    return faces


def save_faces_csv(faces, output_path):
    # Convert the list of face dictionaries to a DataFrame
    df = pd.DataFrame(faces)

    # Save the DataFrame as a CSV file
    df.to_csv(output_path, index=False)


# Set the input video path
video_path = "video.mp4"

output_dir = "output_dir"

name_prefix = "face"

faces = extract_faces(video_path, output_dir, name_prefix)

csv_path = "encodings.csv"

save_faces_csv(faces, csv_path)

save_faces_csv(csv_path)

def get_face_vectors(output_dir, output_csv_file):
    # Initialize an empty list to store face feature vectors
    vectors = []

    # Initialize a tqdm progress bar
    progress_bar = tqdm(os.listdir(output_dir), desc='Getting Face Vectors')

    # Loop through the face image files in the output directory
    for filename in progress_bar:
        # Load the face image
        filepath = os.path.join(output_dir, filename)
        face = cv2.imread(filepath)

        # Convert the face to a feature vector using a pre-trained face recognition model
        vector = face_recognition.face_encodings(face)

        # Add the feature vector to the list
        if len(vector) > 0:
            vectors.append(vector[0])
            # Append the feature vector to the CSV file
            with open(output_csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(vector[0])

    # Return the list of face feature vectors
    return vectors



def cluster_faces(vectors, n_clusters):
    # Perform k-means clustering on the face feature vectors
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(vectors)

    # Return the list of cluster labels
    return labels

# Set the number of clusters to create
n_clusters = 5

# Call the get_face_vectors function to get a list of face feature vectors
vectors = get_face_vectors(output_dir)

# Call the cluster_faces function to cluster the face feature vectors and get a list of cluster labels
labels = cluster_faces(vectors, n_clusters)


# Create a directory for each cluster
for i in range(n_clusters):
    cluster_dir = os.path.join(output_dir, f"cluster_{i}")
    os.makedirs(cluster_dir, exist_ok=True)

# Loop through the face image files in the output directory
for i, filename in enumerate(os.listdir(output_dir)):
    # Get the cluster label for the face
    label = labels[i]

    # Move the face image to the corresponding cluster directory
    src_path = os.path.join(output_dir, filename)
    dst_dir = os.path.join(output_dir, f"cluster_{label}")
    dst_path = os.path.join(dst_dir, filename)
    os.rename(src_path, dst_path)
    
    
def main():
    # Define the paths to the input video file, output directory, and CSV file
    input_video_path = 'path/to/input/video/file.mp4'
    output_dir = 'path/to/output/directory'
    csv_path = 'path/to/output/csv/file.csv'

    # Extract faces from the input video file and save them to the output directory
    extract_faces(input_video_path, output_dir)

    # Get the face feature vectors from the output directory and save them to the CSV file
    get_face_vectors(output_dir, csv_path)
    
if __name__ == '__main__':
    video_path = 'path/to/video.mp4'
    output_dir = 'path/to/output/directory'
    csv_path = 'path/to/output/csv/file.csv'

    # Extract faces from the video and save them to the output directory
    extract_faces(video_path, output_dir)

    # Create a CSV file of face feature vectors from the extracted faces
    create_csv(output_dir, csv_path)
