# Tomato Ripeness Annotation Tool

This project provides a tool for annotating tomato ripeness using a simple graphical user interface (GUI) built with OpenCV. The tool allows the user to select regions of interest (ROIs) in images, compute hue-based ripeness levels, and record occlusion status.

## Features
- Annotate images for tomato ripeness estimation by selecting regions of interest.
- Automatically compute ripeness levels based on hue values.
- Manual annotation of ripeness levels by pressing keys '1' to '6'.
- Compute overall ripeness percentage by averaging weights of selected regions.
- Record occlusion status of images with Yes/No options.
- Save annotations to text files for further analysis.

## Installation
### Requirements
- Python 3.x
- OpenCV
- Numpy

Install dependencies via pip:
```bash
pip install opencv-python-headless opencv-python opencv-contrib-python numpy
```

## Usage
### Command Line Arguments
- `--image_dir`: Directory containing images to annotate (default: `data/test/images`)
- `--annotation_dir`: Directory to save annotations (default: `data/test/annotations`)
- `--max_side`: Maximum size of the longest side when resizing images (default: `600`)

Example:
```bash
python annotation_tool.py --image_dir=data/test/images --annotation_dir=data/test/annotations --max_side=600
```

### Key Commands
- `Left-click`: Add a square region for annotation.
- `Mouse wheel`: Adjust the size of the square.
- `r`: Remove the last added square.
- `0`: Compute hue and ripeness level automatically.
- `1`-`6`: Manually assign ripeness levels.
- `s`: Compute ripeness percentage and proceed to occlusion input.
- `d`: Move to the next image.
- `Esc` or `q`: Quit the program.

### Annotation Output
Annotations are saved as text files with the format:
```
<ripeness_percentage> <occlusion_status>
```
Where:
- `<ripeness_percentage>` is a floating-point value representing the average ripeness weight.
- `<occlusion_status>` is `1` for occluded and `0` for not occluded.

## License
MIT License

## Author
Jiacheng Rong