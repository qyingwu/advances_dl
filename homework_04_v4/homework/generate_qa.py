import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def get_kart_and_track_names(data_dir: str) -> tuple[dict, set]:
    """
    Read all info.json files to build kart and track name mappings.
    
    Args:
        data_dir: Directory containing the info.json files
        
    Returns:
        Tuple of (kart_id_to_name mapping, set of track names)
    """
    kart_names = {}  # Will map track_id to kart_name
    track_names = set()
    
    # Process all info files
    for info_file in Path(data_dir).glob("*_info.json"):
        try:
            with open(info_file) as f:
                info = json.load(f)
                
            # Add track name
            if "track" in info:
                track_names.add(info["track"].lower())
                
            # Add kart names and their IDs from detections
            if "karts" in info and "detections" in info:
                for detection_frame in info["detections"]:
                    for detection in detection_frame:
                        class_id, track_id = map(int, detection[:2])
                        if class_id == 1:  # Kart class
                            # If we have this kart's name in the karts list
                            if track_id < len(info["karts"]):
                                kart_name = info["karts"][track_id].lower()
                                kart_names[str(track_id)] = kart_name
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
            
    return kart_names, track_names

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # First pass: find the center kart
    image_center = (img_width / 2, img_height / 2)
    min_dist_to_center = float('inf')
    center_kart_id = None

    for detection in frame_detections:
        class_id, track_id = map(int, detection[:2])
        if class_id != 1:  # Only process karts
            continue

        x1, y1, x2, y2 = map(float, detection[2:])
        # Scale coordinates
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if box is too small or outside image
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Calculate center point of the box
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        # Calculate distance to image center
        dist_to_center = ((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2) ** 0.5

        # Update center kart if this one is closer
        if dist_to_center < min_dist_to_center:
            min_dist_to_center = dist_to_center
            center_kart_id = track_id

    # Second pass: draw the boxes with correct colors
    for detection in frame_detections:
        class_id, track_id = map(int, detection[:2])
        if class_id != 1:  # Only process karts
            continue

        x1, y1, x2, y2 = map(float, detection[2:])
        # Scale coordinates
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if box is too small or outside image
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Set color based on whether this is the center kart
        if track_id == center_kart_id:
            color = (255, 0, 0)  # Red for ego/center kart
        else:
            color = (0, 255, 0)  # Green for other karts

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    # Get track name from the info file
    track_name = info.get("track", "unknown")
    return track_name.lower()


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Get the correct detection frame
    if view_index >= len(info["detections"]):
        return []

    frame_detections = info["detections"][view_index]
    
    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    # Calculate image center
    image_center = (img_width / 2, img_height / 2)
    
    # Process all karts first to find their centers
    karts = []
    min_dist_to_center = float('inf')
    center_kart_idx = None
    
    for detection in frame_detections:
        class_id, track_id = map(int, detection[:2])
        
        # Only process karts
        if class_id != 1:
            continue
            
        x1, y1, x2, y2 = map(float, detection[2:])
        
        # Scale coordinates
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        # Skip if box is too small or outside image
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        if x2_scaled <= 0 or x1_scaled >= img_width or y2_scaled <= 0 or y1_scaled >= img_height:
            continue
            
        # Calculate center point
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # Get kart name
        kart_name = info["karts"][track_id].lower() if track_id < len(info["karts"]) else f"kart_{track_id}"
        
        # Calculate distance to image center
        dist_to_center = ((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2) ** 0.5
        
        kart_info = {
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_center_kart": False  # Will be updated after finding center kart
        }
        
        # Update center kart if this one is closer to image center
        if dist_to_center < min_dist_to_center:
            min_dist_to_center = dist_to_center
            center_kart_idx = len(karts)
            
        karts.append(kart_info)
    
    # Mark the center kart if we found any karts
    if center_kart_idx is not None:
        karts[center_kart_idx]["is_center_kart"] = True
    
    return karts


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []
    
    # Get visible kart objects
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    
    # If no karts are visible, return empty list
    if not kart_objects:
        return []
        
    # Read the full info file for additional data
    with open(info_path) as f:
        info = json.load(f)
        
    # Get the center kart (ego vehicle) and other visible karts
    center_kart = None
    other_karts = []
    for kart in kart_objects:
        if kart["is_center_kart"]:
            center_kart = kart
        else:
            other_karts.append(kart)
            
    # If no center kart was found, return empty list
    if not center_kart:
        return []
        
    # Get image file name with train/ prefix
    image_file = f"train/{Path(info_path).stem.replace('_info', '')}_{view_index:02d}_im.jpg"
        
    # 1. Ego car question
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": center_kart["kart_name"],
        "image_file": image_file
    })
    
    # 2. Track information question
    track_name = extract_track_info(info_path)
    qa_pairs.append({
        "question": "What track is being raced on?",
        "answer": track_name,
        "image_file": image_file
    })
    
    # 3. Questions about other visible karts
    if other_karts:
        # Get positions relative to ego kart
        for kart in other_karts:
            # Simply use y-axis for front/back determination
            is_in_front = kart["center"][1] < center_kart["center"][1]
            front_back = "front" if is_in_front else "back"
            
            # Use x-axis for left/right
            is_on_left = kart["center"][0] < center_kart["center"][0]
            left_right = "left" if is_on_left else "right"
            
            qa_pairs.extend([
                {
                    "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                    "answer": front_back,
                    "image_file": image_file
                },
                {
                    "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                    "answer": left_right,
                    "image_file": image_file
                },
                {
                    "question": f"Where is {kart['kart_name']} relative to the ego car?",
                    "answer": f"{front_back} and {left_right}",
                    "image_file": image_file
                }
            ])
            
    # 4. Race position questions
    if "distance_down_track" in info:
        # Only consider visible karts for position questions
        visible_kart_ids = {kart["instance_id"] for kart in kart_objects}
        positions = [(i, dist) for i, dist in enumerate(info["distance_down_track"]) if i in visible_kart_ids]
        positions.sort(key=lambda x: x[1], reverse=True)  # Sort by distance, highest first
        
        # Find ego kart's position among visible karts
        ego_position = None
        for pos, (kart_idx, _) in enumerate(positions, 1):
            if kart_idx == center_kart["instance_id"]:
                ego_position = pos
                break
                
        if ego_position is not None:
            qa_pairs.append({
                "question": "What position is the ego car in?",
                "answer": str(ego_position),
                "image_file": image_file
            })
            
            # Add question about who is leading if ego isn't first
            if ego_position > 1:
                leader_idx = positions[0][0]
                leader_name = info["karts"][leader_idx].lower()
                qa_pairs.append({
                    "question": "Who is leading the race?",
                    "answer": leader_name,
                    "image_file": image_file
                })
                
    # 5. Velocity questions
    if "velocity" in info:
        frame_velocities = info["velocity"]
        if len(frame_velocities) > center_kart["instance_id"]:
            ego_velocity = frame_velocities[center_kart["instance_id"]]
            speed = (ego_velocity[0]**2 + ego_velocity[1]**2 + ego_velocity[2]**2)**0.5
            
            # Simple velocity question
            qa_pairs.append({
                "question": "Is the ego car moving fast or slow?",
                "answer": "fast" if speed > 20 else "slow",  # Threshold can be adjusted
                "image_file": image_file
            })
            
            # Compare velocities with other visible karts
            for kart in other_karts:
                if kart["instance_id"] < len(frame_velocities):
                    other_velocity = frame_velocities[kart["instance_id"]]
                    other_speed = (other_velocity[0]**2 + other_velocity[1]**2 + other_velocity[2]**2)**0.5
                    if abs(other_speed - speed) > 5:  # Significant difference threshold
                        qa_pairs.append({
                            "question": f"Is {kart['kart_name']} going faster or slower than the ego car?",
                            "answer": "faster" if other_speed > speed else "slower",
                            "image_file": image_file
                        })
                
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def main(
    data_dir: str = "data",
    output_dir: str = "data/train_grader",
    img_width: int = 150,
    img_height: int = 100,
    min_box_size: int = 5,
    views_per_scene: int = 10,
):
    """
    Main function to generate question-answer pairs for the dataset.
    
    Args:
        data_dir: Root directory containing the data
        output_dir: Directory to save output files (default: data/train_grader)
        img_width: Width to scale images to
        img_height: Height to scale images to
        min_box_size: Minimum size of bounding boxes to consider
        views_per_scene: Number of views per scene to process
    """
    # Convert to Path objects
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    train_dir = data_path / "train"
    
    # Verify directories exist
    if not train_dir.exists():
        raise ValueError(f"Training data directory {train_dir} does not exist")
        
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get kart and track names from the training data
    print(f"Scanning {train_dir} for kart and track names...")
    kart_names, track_names = get_kart_and_track_names(str(train_dir))
    print(f"Found {len(kart_names)} karts and {len(track_names)} tracks in the data")
    
    if not kart_names and not track_names:
        print("Warning: No karts or tracks found. Please check the data directory structure.")
        print(f"Expected structure: {train_dir}/XXXXX_info.json")
        return
    
    # Process all info files in train directory
    qa_pairs = []
    info_files = list(train_dir.glob("*_info.json"))
    
    if not info_files:
        print(f"No info files found in {train_dir}")
        return
        
    print(f"\nProcessing {len(info_files)} info files from training data...")
    
    # Process each info file
    for info_file in sorted(info_files):
        scene_id = info_file.stem.replace("_info", "")
        print(f"Processing scene {scene_id}...")
        
        # Generate QA pairs for each view in the scene
        scene_qa_pairs = []
        for view_idx in range(views_per_scene):
            scene_qa_pairs.extend(
                generate_qa_pairs(
                    str(info_file),
                    view_idx,
                    img_width=img_width,
                    img_height=img_height
                )
            )
        
        print(f"Generated {len(scene_qa_pairs)} QA pairs for scene {scene_id}")
        qa_pairs.extend(scene_qa_pairs)
            
    # Save all QA pairs to a single file
    if qa_pairs:
        output_file = output_path / "balanced_qa_pairs.json"
        with open(output_file, "w") as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"\nSaved {len(qa_pairs)} QA pairs to {output_file}")
    else:
        print("No QA pairs were generated")


def generate(
    data_dir: str = "data",
    output_dir: str = "data/train_grader",
    img_width: int = 150,
    img_height: int = 100,
    min_box_size: int = 5,
    views_per_scene: int = 10,
):
    """Generate QA pairs for all splits in the dataset."""
    return main(
        data_dir=data_dir,
        output_dir=output_dir,
        img_width=img_width,
        img_height=img_height,
        min_box_size=min_box_size,
        views_per_scene=views_per_scene,
    )

def check(info_file: str, view_index: int):
    """Check QA pairs for a specific info file and view index."""
    return check_qa_pairs(info_file, view_index)

if __name__ == "__main__":
    fire.Fire({
        "generate": generate,
        "check": check
    })
