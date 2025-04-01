import cv2
import numpy as np
import argparse
import os
import glob

# Argument parser for configurable parameters
parser = argparse.ArgumentParser(description='Tomato Ripeness Annotation Tool')
parser.add_argument('--image_dir', type=str, default='data/test/images', help='Directory containing images to annotate')
parser.add_argument('--annotation_dir', type=str, default='data/test/annotations', help='Directory to save annotations')
parser.add_argument('--max_side', type=int, default=600, help='Maximum size of the longest side when resizing images')
args = parser.parse_args()

# Create the annotations directory if it doesn't exist
os.makedirs(args.annotation_dir, exist_ok=True)

# Get list of images to annotate
image_extensions = ['*.jpg', '*.png']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
image_paths.sort()  # Sort the list of image paths

# Filter out images that already have annotations
annotated_images = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(args.annotation_dir, '*.txt')))
image_paths = [p for p in image_paths if os.path.splitext(os.path.basename(p))[0] not in annotated_images]

if not image_paths:
    print("All images have been annotated.")
    exit()

current_image_index = 0  # Index of the current image

# Initialize global variables
current_size = 50  # Default size for new squares
waiting_for_occlusion_input = False  # Flag to indicate if we are waiting for occlusion input
occlusion_status = None  # Variable to store occlusion status
yes_rect = [0, 0, 0, 0]  # Rectangle for "Yes" option
no_rect = [0, 0, 0, 0]   # Rectangle for "No" option

def compute_ripeness_level(h):
    if (h >= 160 and h <= 180) or (h >= 0 and h < 2):
        return 6
    elif 2 <= h < 6:
        return 5
    elif 6 <= h < 10:
        return 4
    elif 10 <= h < 20:
        return 3
    elif 20 <= h < 35:
        return 2
    elif 35 <= h <= 100:
        return 1
    else:
        return 0  # If h doesn't fall into any of these categories

def compute_weight_from_hue(h):
    if (h >= 160 and h <= 180) or (h >= 0 and h < 2):
        return 1.0
    elif 2 <= h < 6:
        # Map h from 2-6 to weight from 1.0 to 0.8
        return 1.0 - ((h - 2) / (6 - 2)) * (1.0 - 0.8)
    elif 6 <= h < 10:
        # Map h from 6-10 to weight from 0.8 to 0.6
        return 0.8 - ((h - 6) / (10 - 6)) * (0.8 - 0.6)
    elif 10 <= h < 20:
        # Map h from 10-20 to weight from 0.6 to 0.4
        return 0.6 - ((h - 10) / (20 - 10)) * (0.6 - 0.4)
    elif 20 <= h < 35:
        # Map h from 20-35 to weight from 0.4 to 0.2
        return 0.4 - ((h - 20) / (35 - 20)) * (0.4 - 0.2)
    elif 35 <= h <= 100:
        return 0.0
    else:
        return 0.0  # Default weight

def compute_weight_from_manual_level(level):
    level_weight_mapping = {
        6: 1.0,
        5: 0.8,
        4: 0.6,
        3: 0.4,
        2: 0.2,
        1: 0.0
    }
    return level_weight_mapping.get(level, 0.0)

def compute_hue_and_level(square, hsv_image, fixed_size):
    center = square['center']
    size = square['size']
    top_left = (center[0] - size // 2, center[1] - size // 2)
    bottom_right = (center[0] + size // 2, center[1] + size // 2)

    # Ensure the coordinates are within the image boundaries
    top_left = (max(top_left[0], 0), max(top_left[1], 0))
    bottom_right = (
        min(bottom_right[0], fixed_size[0] - 1),
        min(bottom_right[1], fixed_size[1] - 1),
    )

    # Extract the region of interest (ROI)
    roi = hsv_image[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]

    if roi.size > 0:
        # Extract the H channel
        h_channel = roi[:, :, 0].astype(np.float32)

        # Deal with wraparound (values near 0 and values near 180)
        h_channel_wrapped = np.where(h_channel > 150, h_channel - 180, h_channel)

        # Compute the mean H value
        mean_h_wrapped = np.mean(h_channel_wrapped)

        # Wrap the result back to [0, 180] range
        mean_h = (mean_h_wrapped + 180) % 180

        # Compute the ripeness level automatically
        auto_level = compute_ripeness_level(mean_h)

        # Compute the weight from hue
        weight = compute_weight_from_hue(mean_h)

        # Update the square with hue, level, and weight
        square['h_value'] = mean_h
        square['auto_level'] = auto_level
        square['weight'] = weight

def mouse_callback(event, x, y, flags, param):
    global squares, current_size, hsv_image, fixed_size
    global waiting_for_occlusion_input, yes_rect, no_rect, occlusion_status

    if waiting_for_occlusion_input:
        # Handle clicks on Yes/No options
        if event == cv2.EVENT_LBUTTONDOWN:
            if yes_rect[0] <= x <= yes_rect[2] and yes_rect[1] <= y <= yes_rect[3]:
                # Clicked on Yes
                occlusion_status = 1  # Occluded
                # Save data and proceed
                save_annotation()
                waiting_for_occlusion_input = False
            elif no_rect[0] <= x <= no_rect[2] and no_rect[1] <= y <= no_rect[3]:
                # Clicked on No
                occlusion_status = 0  # Not occluded
                # Save data and proceed
                save_annotation()
                waiting_for_occlusion_input = False
    else:
        # Existing code for adding squares and adjusting size
        # If left mouse button is clicked, add a new square with the clicked point as the center
        if event == cv2.EVENT_LBUTTONDOWN:
            if squares:  # If there is an existing square, compute the hue and level for the previous one
                prev_square = squares[-1]
                if prev_square.get('h_value') is None:
                    compute_hue_and_level(prev_square, hsv_image, fixed_size)
            # Add the new square with no H value or level yet
            squares.append({
                'center': (x, y),
                'size': current_size,
                'h_value': None,
                'auto_level': None,
                'manual_level': None,
                'weight': None
            })

        # If the scroll wheel is moved, adjust the size of the last added square
        elif event == cv2.EVENT_MOUSEWHEEL and squares:
            last_square = squares[-1]  # Get the last added square
            size = last_square['size']
            if flags > 0:
                size += 10  # Increase size
            else:
                size = max(10, size - 10)  # Decrease size, but not less than 10
            last_square['size'] = size  # Update the size of the last square

def save_annotation():
    global ripeness_percentage, occlusion_status, image_path
    # Save the ripeness percentage and occlusion status to a txt file
    annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    annotation_path = os.path.join(args.annotation_dir, annotation_filename)
    with open(annotation_path, 'w') as f:
        f.write(f'{ripeness_percentage} {occlusion_status}\n')
    print(f'Annotation saved to {annotation_path}')

# Main loop to process images
while current_image_index < len(image_paths):
    image_path = image_paths[current_image_index]
    print(f'Annotating image: {image_path}')
    squares = []  # Reset squares for the new image
    ripeness_percentage = None  # Reset ripeness percentage
    occlusion_status = None  # Reset occlusion status
    waiting_for_occlusion_input = False  # Reset state

    # Load the image
    image = cv2.imread(image_path)

    # Resize the image so that the longest side is args.max_side pixels, and the other side scales proportionally
    max_side = args.max_side  # Desired length for the longest side
    height, width = image.shape[:2]

    if width > height:
        scale_ratio = max_side / width
    else:
        scale_ratio = max_side / height

    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)
    fixed_size = (new_width, new_height)

    # Resize the image to the new size
    image = cv2.resize(image, fixed_size)

    # Convert the image to HSV color space for H value extraction
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a window and set the mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        img_copy = image.copy()  # Create a copy of the image to draw on

        # Draw all the squares stored in the 'squares' list
        for square in squares:
            center = square['center']
            size = square['size']
            h_value = square.get('h_value')
            auto_level = square.get('auto_level')
            manual_level = square.get('manual_level')
            weight = square.get('weight')

            top_left = (center[0] - size // 2, center[1] - size // 2)
            bottom_right = (center[0] + size // 2, center[1] + size // 2)
            cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 2)

            # If the H value is computed, display hue and level near the square
            if h_value is not None:
                if manual_level is not None:
                    text = f'Hue: {h_value:.2f}, Level: {manual_level} (Manual)'
                else:
                    text = f'Hue: {h_value:.2f}, Level: {auto_level}'
                # Get image dimensions
                img_height, img_width = img_copy.shape[:2]

                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Initial text position
                text_x = top_left[0]
                text_y = top_left[1] - 10  # 10 pixels above the rectangle

                # Adjust text_x if it goes beyond the left or right edge
                if text_x < 0:
                    text_x = 0
                elif text_x + text_width > img_width:
                    text_x = img_width - text_width

                # Adjust text_y if it goes above the top edge
                if text_y - text_height < 0:
                    text_y = bottom_right[1] + text_height + 10  # Place text below the rectangle

                text_position = (text_x, text_y)
                cv2.putText(img_copy, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # If ripeness percentage is calculated, display it in the top-left corner
        if ripeness_percentage is not None:
            percentage_text = f'Ripeness: {ripeness_percentage * 100:.2f}%'
            cv2.putText(img_copy, percentage_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        # Display occlusion status if available
        if occlusion_status is not None:
            occlusion_text = f'Occlusion: {"Yes" if occlusion_status == 1 else "No"}'
            # Display occlusion status at the bottom-left corner
            img_height, img_width = img_copy.shape[:2]
            text_size, _ = cv2.getTextSize(occlusion_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 4)
            cv2.putText(img_copy, occlusion_text, (10, img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        # If waiting for occlusion input, display Yes and No options
        if waiting_for_occlusion_input:
            # Define positions
            img_height, img_width = img_copy.shape[:2]
            yes_text = 'Yes(1)'
            no_text = 'No(0)'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            # Get text size
            (yes_text_width, yes_text_height), _ = cv2.getTextSize(yes_text, font, font_scale, thickness)
            (no_text_width, no_text_height), _ = cv2.getTextSize(no_text, font, font_scale, thickness)

            # Positions for Yes and No
            margin = 10
            y_position = img_height - margin - max(yes_text_height, no_text_height) - 50  # Adjusted position

            # Yes on the left
            yes_x1 = margin
            yes_y1 = y_position
            yes_x2 = yes_x1 + yes_text_width + margin
            yes_y2 = yes_y1 + yes_text_height + margin
            yes_rect = [yes_x1, yes_y1, yes_x2, yes_y2]

            # No next to Yes
            no_x1 = yes_x2 + margin
            no_y1 = yes_y1
            no_x2 = no_x1 + no_text_width + margin
            no_y2 = no_y1 + no_text_height + margin
            no_rect = [no_x1, no_y1, no_x2, no_y2]

            # Draw rectangles
            cv2.rectangle(img_copy, (yes_x1, yes_y1), (yes_x2, yes_y2), (0, 0, 255), -1)  # Red for "Yes"
            cv2.rectangle(img_copy, (no_x1, no_y1), (no_x2, no_y2), (0, 255, 0), -1)      # Green for "No"

            # Put text
            cv2.putText(img_copy, yes_text, (yes_x1 + margin // 2, yes_y2 - margin // 2), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(img_copy, no_text, (no_x1 + margin // 2, no_y2 - margin // 2), font, font_scale, (255, 255, 255), thickness)

        # Show the image with the drawn squares and H values
        cv2.imshow("Image", img_copy)

        # Listen for keypresses
        key = cv2.waitKey(1) & 0xFF

        # Press 'Esc' or 'q' to quit the program
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            exit()
        # Press 'r' to undo the last added square
        elif key == ord('r') and squares and not waiting_for_occlusion_input:
            squares.pop()  # Remove the last square
        # Press '0' to compute and display the H value and level of the current square automatically
        elif key == ord('0') and squares and not waiting_for_occlusion_input:
            current_square = squares[-1]
            if current_square.get('h_value') is None:
                compute_hue_and_level(current_square, hsv_image, fixed_size)
        # Press '1'-'6' to manually assign ripeness level, and automatically compute hue
        elif key in [ord(str(i)) for i in range(1, 7)] and squares and not waiting_for_occlusion_input:
            current_square = squares[-1]
            # Manually assign level
            manual_level = int(chr(key))
            current_square['manual_level'] = manual_level
            # Compute hue if not already computed
            if current_square.get('h_value') is None:
                compute_hue_and_level(current_square, hsv_image, fixed_size)
            # Compute weight from manual level
            current_square['weight'] = compute_weight_from_manual_level(manual_level)
        # Press 's' to compute ripeness percentage and prepare for occlusion input
        elif key == ord('s') and not waiting_for_occlusion_input:
            # Ensure all squares have been processed
            for square in squares:
                if square.get('h_value') is None:
                    compute_hue_and_level(square, hsv_image, fixed_size)
                # Compute weight if not already computed
                if square.get('weight') is None:
                    if square.get('manual_level') is not None:
                        square['weight'] = compute_weight_from_manual_level(square['manual_level'])
                    elif square.get('h_value') is not None:
                        # Compute weight based on automatic level
                        square['weight'] = compute_weight_from_hue(square['h_value'])

            total_weight = 0
            count = 0
            for square in squares:
                if square.get('weight') is not None:
                    total_weight += square['weight']
                    count += 1
            if count > 0:
                ripeness_percentage = total_weight / count
            else:
                ripeness_percentage = 0

            # Prepare for occlusion input
            waiting_for_occlusion_input = True
            print("Please click on 'Yes' or 'No' to indicate occlusion status.")
        # Press 'd' to move to the next image
        elif key == ord('d'):
            if waiting_for_occlusion_input:
                print("Please select occlusion status before proceeding.")
            elif ripeness_percentage is None:
                print("Please compute ripeness percentage before proceeding.")
            else:
                # Move to the next image
                cv2.destroyAllWindows()
                current_image_index += 1
                break  # Exit the inner loop to load the next image

    cv2.destroyAllWindows()

print("Annotation completed for all images.")
