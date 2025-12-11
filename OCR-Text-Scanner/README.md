# OCR Text Scanner

A comprehensive GUI-based printed text scanner built with PyQt5 and PyTesseract. This application provides advanced OCR capabilities with features like ROI selection, live camera input, and text overlay preview.

## Features

### Core Functionality

- **GUI Interface**: Clean, modern PyQt5-based interface
- **Image Loading**: Support for multiple image formats (PNG, JPG, JPEG, BMP, TIFF, GIF)
- **OCR Processing**: High-quality text extraction using PyTesseract
- **Text Display**: Clear presentation of extracted text in a dedicated panel

### Advanced Features

- **ROI Selection**: Click and drag to select specific regions for OCR processing
- **Live Camera Input**: Real-time camera feed with OCR capabilities
- **Text Overlay**: Visual preview showing detected text boundaries on the image
- **Image Preprocessing**: Multiple enhancement options for better OCR accuracy
  - Contrast enhancement
  - Noise reduction
  - Adaptive thresholding

### User Interface

- **Tabbed Interface**: Organized layout with separate tabs for results and settings
- **Resizable Panels**: Adjustable layout for optimal viewing
- **Progress Indicators**: Visual feedback during OCR processing
- **Customizable Settings**: Toggle preprocessing options and display preferences

## Installation

### Prerequisites

- Python 3.7 or higher
- Tesseract OCR engine

### Install Tesseract OCR

#### macOS (using Homebrew)

```bash
brew install tesseract
```

#### Ubuntu/Debian

```bash
sudo apt-get install tesseract-ocr
```

#### Windows

1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH

### Install Python Dependencies

```bash
cd OCR-Text-Scanner
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
cd src
python main.py
```

### Basic Operations

#### 1. Load an Image

- Click "Load Image" button
- Select an image file from your computer
- The image will appear in the left panel

#### 2. Select Region of Interest (ROI)

- Click and drag on the image to select a specific area
- A red rectangle will show your selection
- Use "Clear ROI" to remove the selection

#### 3. Start Camera Feed

- Click "Start Camera" to begin live video feed
- Click "Stop Camera" to end the feed
- OCR can be performed on live camera frames

#### 4. Run OCR

- Click "Run OCR" to extract text from the image
- Results will appear in the "OCR Results" tab
- If "Show Text Overlay" is enabled, detected text boundaries will be highlighted

### Settings Configuration

#### Preprocessing Options

- **Enhance Contrast**: Improves text visibility in low-contrast images
- **Denoise**: Reduces image noise for cleaner text detection
- **Apply Threshold**: Converts image to binary for better OCR accuracy

#### OCR Options

- **Process ROI Only**: Limits OCR to the selected region
- **Show Text Overlay**: Displays detected text boundaries on the image

## Project Structure

```
OCR-Text-Scanner/
├── src/
│   └── main.py              # Main application file
├── assets/                  # Sample images (optional)
├── tests/                   # Test files (optional)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Technical Details

### Dependencies

- **PyQt5**: GUI framework for the user interface
- **pytesseract**: Python wrapper for Tesseract OCR engine
- **opencv-python**: Computer vision library for image processing
- **Pillow**: Python Imaging Library for image handling
- **numpy**: Numerical computing library

### Key Components

#### ImageLabel Class

- Custom QLabel widget for image display
- Handles ROI selection with mouse events
- Manages image scaling and coordinate conversion

#### CameraThread Class

- Separate thread for camera operations
- Prevents GUI freezing during video capture
- Emits frames for real-time processing

#### OCRProcessor Class

- Static methods for image preprocessing
- Text extraction with confidence scoring
- Bounding box detection for overlay display

#### MainWindow Class

- Main application window and UI management
- Coordinates all components and user interactions
- Handles settings and configuration

### Image Preprocessing Pipeline

1. **Grayscale Conversion**: Reduces computational complexity
2. **Histogram Equalization**: Enhances contrast uniformly
3. **Median Filtering**: Removes noise while preserving edges
4. **Otsu Thresholding**: Automatic binary threshold selection

## Troubleshooting

### Common Issues

#### "Tesseract not found" Error

- Ensure Tesseract is installed and in your system PATH
- On macOS: `brew install tesseract`
- On Windows: Add Tesseract installation directory to PATH

#### Camera Not Working

- Check camera permissions in system settings
- Ensure no other applications are using the camera
- Try restarting the application

#### Poor OCR Results

- Enable preprocessing options in Settings tab
- Select a smaller ROI around the text
- Ensure good lighting and image quality
- Try different image formats or resolutions

#### GUI Not Responding

- Check that all dependencies are installed correctly
- Ensure you're running Python 3.7 or higher
- Try running from command line to see error messages

## Performance Tips

1. **Use ROI Selection**: Process only the text area for faster results
2. **Enable Preprocessing**: Improves accuracy for challenging images
3. **Good Lighting**: Ensure adequate lighting for camera input
4. **High Contrast**: Use images with clear text-background distinction

## Contributing

Feel free to contribute to this project by:

- Reporting bugs and issues
- Suggesting new features
- Submitting pull requests
- Improving documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **Tesseract OCR**: Google's open-source OCR engine
- **PyQt5**: Cross-platform GUI toolkit
- **OpenCV**: Computer vision and image processing library
