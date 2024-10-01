import cv2

def resize_and_adjust_video(input_path, output_path, target_width=100, target_height=50, target_frames=75):
    print(f"Opening video file: {input_path}")
    
    # Open the video file
    video_capture = cv2.VideoCapture(input_path)
    
    # Check if the video was successfully opened
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    print(f"Video file {input_path} opened successfully.")
    
    # Get original video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original FPS: {fps}, Frame Count: {frame_count}")

    # Define codec and create VideoWriter object to save the resized video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    frames = []
    print("Reading and resizing frames...")

    # Read the video and resize each frame
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No more frames to read.")
            break
        # Resize the frame to match LipNet's expected input
        resized_frame = cv2.resize(frame, (target_width, target_height))
        frames.append(resized_frame)

    print(f"Total frames read: {len(frames)}")

    if len(frames) == 0:
        print(f"Error: No frames read from video {input_path}")
        return

    # Adjust the number of frames to match the target
    if len(frames) > target_frames:
        frames = frames[:target_frames]
        print(f"Trimming video to {target_frames} frames.")
    elif len(frames) < target_frames:
        while len(frames) < target_frames:
            frames.append(frames[-1])  # Repeat last frame
        print(f"Extending video to {target_frames} frames by repeating the last frame.")

    # Write the resized frames to a new video file
    for frame in frames:
        out.write(frame)
    
    print(f"Video resizing and frame adjustment complete. Saved to {output_path}")

    # Release video resources
    video_capture.release()
    out.release()

# Example usage:
resize_and_adjust_video("path_to_input_video.mp4", "path_to_output_video.mpg")
