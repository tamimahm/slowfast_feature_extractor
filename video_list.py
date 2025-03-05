import os
import csv

def create_vid_list_csv(dataset_dir, output_csv="vid_list.csv"):
    """
    Create a vid_list.csv file for the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        output_csv (str): Name of the output CSV file.
    """
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"The dataset directory '{dataset_dir}' does not exist.")

    # Create the dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)

    # List to store video paths
    video_paths = []

    # Traverse the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".mp4"):  # Only process .mp4 files
                # Get the relative path to the video file
                rel_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                video_paths.append([rel_path, ""])  # Add video path and empty label

    # Write to vid_list.csv
    output_path = os.path.join(dataset_dir, output_csv)
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_path", "label"])  # Write header
        writer.writerows(video_paths)  # Write video paths

    print(f"Created {output_csv} in {dataset_dir} with {len(video_paths)} videos.")

# Example usage
dataset_directory = "D:/try_videos_slowfast"  # Replace with your dataset directory
create_vid_list_csv(dataset_directory)