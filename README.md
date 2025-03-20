# Chess Move Advisor | Screenshot Detection with Stockfish

<p>
    <img alt="python" src="https://img.shields.io/badge/Python-1E90FF?logo=python&logoColor=white">
    <img alt="opencv" src="https://img.shields.io/badge/OpenCV-1F3AF7?logo=opencv&logoColor=01F701">
    <img alt="numpy" src="https://img.shields.io/badge/numpy-4AA6C9?logo=numpy&logoColor=white">
    <img alt="stockfish" src="https://img.shields.io/badge/Stockfish-232323?logo=lichess&logoColor=white">
    <img alt="pyqt5" src="https://img.shields.io/badge/PyQt5-41CD52?logo=qt&logoColor=white">
</p>

This application allows you to capture a screenshot of a chess game (such as on Chess.com), automatically detect the board position, and get the best move recommendation from Stockfish chess engine.

## Features

- **Screen Capture**: Automatically takes a screenshot of your display
- **Chess Board Detection**: Identifies a chess board on your screen using precise color detection
- **Piece Recognition**: Uses OpenCV template matching to detect chess pieces and their positions
- **FEN Generation**: Converts the detected board position to FEN notation
- **Stockfish Integration**: Analyzes the position and suggests the best move
- **Visual Overlay**: Displays the recommended move directly on your screen with an arrow
- **Team Selection**: Option to play as white or black pieces
- **Piece Style Customization**: Change the visual style of pieces used for detection
- **Debug Mode**: View the detection results for troubleshooting

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PyQt5
- Stockfish engine
- Chess (Python library)
- PIL (Pillow)
- Requests
- Screeninfo

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install opencv-python numpy pyqt5 pillow requests chess screeninfo
```

3. Download Stockfish engine from [the official site](https://stockfishchess.org/download/) and update the path in the code:

```python
stockfish_path = "path/to/your/stockfish/executable"
```

## Usage

Run the application with:

```bash
python main.py
```

### How to use:

1. Launch the application
2. Select which side you're playing (White or Black)
3. Click "Capture Screen" to analyze the current chess position on your screen
4. An arrow will be displayed on the chess board showing the recommended move
5. Click "Remove Arrows" to clear the overlay

### Configuration Options:

- **Debug Mode**: Toggle to view detailed information about the piece detection process
- **Team Selection**: Choose whether you're playing as White or Black
- **Piece Style**: Change the piece style to match the visual theme on Chess.com

## Customization

You can adjust several parameters in the code:

- `chessPieceThreshold`: Detection sensitivity for each piece type
- `CHESS_BOARD_OUTPUT_DIR`: Where to save output images
- `CHESS_PIECE_DIR`: Where to store chess piece templates

## How It Works

1. The application captures a screenshot of your display
2. It identifies a chess board using color detection
3. Each square is analyzed to detect chess pieces using template matching
4. The detected position is converted to FEN notation
5. Stockfish analyzes the position and suggests the best move
6. A visual overlay displays the suggested move on your screen

## License

This project is licensed under the MIT License.
