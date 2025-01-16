import cv2
import numpy as np
import os
import requests
from io import BytesIO
import time
from PIL import ImageGrab, Image
import chess.engine
from screeninfo import get_monitors
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QProgressBar, QLineEdit, QCompleter
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor
import sys
import tkinter as tk
from PIL import ImageTk

CHESS_BOARD_OUTPUT_DIR = os.path.join('dist')
CHESS_PIECE_DIR = os.path.join('chess_piece')
SHOW_IMAGE = True
EXPORT_IMAGE = True

# Create chess_piece directory if it doesn't exist
os.makedirs(CHESS_PIECE_DIR, exist_ok=True)

# The threshold values for each chess piece can be adjusted according to the image quality
chessPieceThreshold = {
    'K': 0.40, #king - increased threshold
    'k': 0.35, #king_black
    'Q': 0.40, #queen - increased threshold
    'q': 0.35, #queen_black
    'B': 0.20, #bishop
    'b': 0.15, #bishop_black
    'N': 0.20, #knight
    'n': 0.15, #knight_black
    'R': 0.20, #rook
    'r': 0.15, #rook_black
    'P': 0.15, #pawn - decreased threshold
    'p': 0.10, #pawn_black
}

def download_piece_image(piece):
    """Download chess piece image from chess.com and save to local folder"""
    color = 'b' if piece.islower() else 'w'
    filename = f'{color}{piece.lower()}.png'
    filepath = os.path.join(CHESS_PIECE_DIR, filename)
    
    # If file already exists, load it
    if os.path.exists(filepath):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (48, 48))  # Increased piece size
        return img
    
    # Otherwise download it
    url = f'https://www.chess.com/chess-themes/pieces/neo/300/{color}{piece.lower()}.png'
    response = requests.get(url)
    if response.status_code == 200:
        # Convert to PIL Image
        img = Image.open(BytesIO(response.content))
        # Convert to RGBA if not already
        img = img.convert('RGBA')
        # Save the original image
        img.save(filepath)
        # Convert to numpy array
        img_array = np.array(img)
        # Convert from RGBA to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
        # Resize piece images to be larger (scaled proportionally with board)
        img_bgr = cv2.resize(img_bgr, (300, 300))  # Increased from 48 to 300 to match board scale
        return img_bgr
    return None

# Download and prepare chess piece images
chessPieceImages = dict()
for piece in chessPieceThreshold.keys():
    piece_image = download_piece_image(piece)
    if piece_image is not None:
        chessPieceImages[piece] = (piece_image, chessPieceThreshold[piece])
    else:
        print(f"Failed to download image for piece: {piece}")

def remove_board_colors(image):
    """Remove chess board colors to improve piece detection"""
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the colors to remove (in BGR)
    dark_green = np.array([82, 149, 115])  # #739552
    light_beige = np.array([208, 236, 235])  # #EBECD0 (BGR order)
    
    # Convert colors to HSV
    dark_green_hsv = cv2.cvtColor(np.uint8([[dark_green]]), cv2.COLOR_BGR2HSV)[0][0]
    light_beige_hsv = cv2.cvtColor(np.uint8([[light_beige]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Create masks for both colors with tolerance
    tolerance = 80  # Increased tolerance significantly
    dark_mask = cv2.inRange(hsv, 
                           dark_green_hsv - tolerance, 
                           dark_green_hsv + tolerance)
    light_mask = cv2.inRange(hsv, 
                            light_beige_hsv - tolerance, 
                            light_beige_hsv + tolerance)
    
    # Combine masks
    board_mask = cv2.bitwise_or(dark_mask, light_mask)
    
    # Create bright red background (BGR: 0, 0, 255)
    result = image.copy()
    result[board_mask > 0] = [0, 0, 255]  # Bright red
    
    return result

def get_piece_color(square_img):
    """Determine if a piece is present based on specific piece colors"""
    # Define the piece colors in BGR with wider range
    white_colors = [
        np.array([249, 249, 249]),  # #F9F9F9
        np.array([240, 240, 240]),  # Slightly darker white
    ]
    black_colors = [
        np.array([87, 89, 93]),     # #5D5957
        np.array([70, 70, 70]),     # Slightly darker black
    ]
    
    # Increase tolerance for better detection
    tolerance = 15
    
    # Check for white pieces
    white_pixels = 0
    for white_color in white_colors:
        white_mask = cv2.inRange(square_img, 
                                white_color - tolerance,
                                white_color + tolerance)
        white_pixels += np.sum(white_mask > 0)
    
    # Check for black pieces
    black_pixels = 0
    for black_color in black_colors:
        black_mask = cv2.inRange(square_img, 
                                black_color - tolerance,
                                black_color + tolerance)
        black_pixels += np.sum(black_mask > 0)
    
    # Increase minimum pixel threshold
    min_pixels = 20
    
    if white_pixels < min_pixels and black_pixels < min_pixels:
        return None
        
    return 'white_piece' if white_pixels > black_pixels else 'black_piece'

def draw_move_arrow(image, move_str, square_size=300):
    """Draw an arrow showing the chess move"""
    # Chess square to coordinate conversion
    file_to_x = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    rank_to_y = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}
    
    # Extract from and to squares (e.g., "e2e4" or "g1f3")
    from_file = move_str[0]
    from_rank = move_str[1]
    to_file = move_str[2]
    to_rank = move_str[3]
    
    # Convert to pixel coordinates
    from_x = file_to_x[from_file] * square_size + square_size // 2
    from_y = rank_to_y[from_rank] * square_size + square_size // 2
    to_x = file_to_x[to_file] * square_size + square_size // 2
    to_y = rank_to_y[to_rank] * square_size + square_size // 2
    
    # Draw the arrow
    cv2.arrowedLine(image, 
                    (from_x, from_y), 
                    (to_x, to_y),
                    (0, 255, 255),  # Yellow color
                    thickness=10,
                    tipLength=0.3)

def detectPieceOfChess(boardImage):
    # Remove board colors first
    boardImage = remove_board_colors(boardImage)
    
    # Create a copy of the image to draw on
    displayImage = boardImage.copy()
    
    # Calculate square size
    square_size = 300  # Each square is 300x300
    
    # Store all detections
    all_detections = []
    
    # Process pieces in specific order - prioritize kings and queens
    piece_order = ['K', 'k', 'Q', 'q']  # Start with kings and queens
    piece_order.extend([p for p in sorted(chessPieceImages.keys()) 
                       if p.isupper() and p not in ['K', 'Q']])  # Other white pieces
    piece_order.extend([p for p in sorted(chessPieceImages.keys()) 
                       if p.islower() and p not in ['k', 'q']])  # Other black pieces
    
    # Iterate through each square on the board
    for row in range(8):
        for col in range(8):
            # Calculate square coordinates
            x = col * square_size
            y = row * square_size
            
            # Draw debug square
            cv2.rectangle(displayImage, 
                         (x, y), 
                         (x + square_size, y + square_size), 
                         (255, 0, 0),  # Blue color
                         3)  # Thickness
            
            # Extract the current square
            square = boardImage[y:y+square_size, x:x+square_size]
            
            # Determine piece color before detection
            piece_color = get_piece_color(square)
            
            if piece_color:
                # Filter piece order based on color
                if piece_color == 'white_piece':
                    current_pieces = [p for p in piece_order if p.isupper()]
                else:  # black_piece
                    current_pieces = [p for p in piece_order if p.islower()]
                
                # Process each piece type
                max_matches = []
                
                for piece in current_pieces:
                    pieceImage = chessPieceImages[piece][0]
                    
                    try:
                        # Resize piece image to match square size
                        pieceImage = cv2.resize(pieceImage, (square_size, square_size))
                        
                        # Convert both images to grayscale for template matching
                        squareGray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                        pieceImageGray = cv2.cvtColor(pieceImage, cv2.COLOR_BGR2GRAY)
                        mask = cv2.resize(pieceImage[:,:,3], (square_size, square_size))
                        
                        # Template matching on just this square
                        result = cv2.matchTemplate(squareGray, pieceImageGray, cv2.TM_CCOEFF_NORMED, mask=mask)
                        max_val = np.max(result)
                        max_matches.append((piece, max_val))
                        
                    except Exception as e:
                        print(f"Error processing piece {piece}: {str(e)}")
                        continue
                
                # Sort matches by value and get best match
                max_matches.sort(key=lambda x: x[1], reverse=True)
                if max_matches:
                    best_piece, best_val = max_matches[0]
                    
                    # Draw rectangle for the entire square
                    rectangleColor = (0, 255, 0)  # Bright green
                    cv2.rectangle(displayImage, 
                                (x, y), 
                                (x + square_size, y + square_size), 
                                rectangleColor, 
                                6)
                    
                    # Draw piece name with larger font
                    textColor = (0, 0, 255) if best_piece.islower() else (255, 0, 0)
                    textPosition = (x + square_size//4, y + square_size//2)
                    cv2.putText(displayImage, best_piece, textPosition, 
                              cv2.FONT_HERSHEY_SIMPLEX, 3.0, textColor, 6)
                    
                    # Add debug text for match value
                    cv2.putText(displayImage, 
                              f"{best_val:.2f}", 
                              (x + 10, y + square_size - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              1.0, 
                              (0, 255, 255), 
                              2)
                    
                    all_detections.append((x, y, best_piece))
    
    # After all detections, before showing image
    if all_detections:
        fen = convert_detections_to_fen(all_detections, 2400)
        print(f"FEN: {fen}")
        
        try:
            best_move = get_best_move(fen)
            print(f"Best move: {best_move}")
            
            # Draw arrow for the move
            draw_move_arrow(displayImage, best_move)
            
            # Also draw the text
            cv2.putText(displayImage, 
                      f"Best move: {best_move}", 
                      (10, 2350),
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      2.0, 
                      (0, 255, 255), 
                      3)
                      
        except Exception as e:
            print(f"Error getting best move: {str(e)}")
    
    if EXPORT_IMAGE:
        timestamp = str(int(time.time()))
        cv2.imwrite(os.path.join(CHESS_BOARD_OUTPUT_DIR, f'board_{timestamp}.jpg'), displayImage)
    
    return displayImage, all_detections  # Return both image and detections

def draw_overlay_arrow(screenshot, board_coords, move_str, root):
    """Draw arrow overlay on screen"""
    x, y, w, h = board_coords  # Get board position from detection
    print(f"Drawing overlay at: {x}, {y}, {w}x{h}")  # Debug print
    
    # Create overlay window
    overlay = tk.Toplevel(root)
    overlay.title("")
    overlay.attributes('-alpha', 0.7)  # Set transparency
    overlay.attributes('-topmost', True)  # Keep on top
    overlay.overrideredirect(True)  # Remove window decorations
    overlay.attributes('-transparentcolor', 'black')  # Make black background transparent
    
    # Position overlay
    overlay.geometry(f"{w}x{h}+{x}+{y}")
    
    # Create canvas for drawing
    canvas = tk.Canvas(overlay, width=w, height=h, 
                      highlightthickness=0, bg='black')
    canvas.pack()
    
    # Convert chess coordinates to screen coordinates
    file_to_x = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    rank_to_y = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}
    
    square_size = w // 8
    
    # Calculate arrow coordinates
    from_file = move_str[0]
    from_rank = move_str[1]
    to_file = move_str[2]
    to_rank = move_str[3]
    
    from_x = file_to_x[from_file] * square_size + square_size // 2
    from_y = rank_to_y[from_rank] * square_size + square_size // 2
    to_x = file_to_x[to_file] * square_size + square_size // 2
    to_y = rank_to_y[to_rank] * square_size + square_size // 2
    
    print(f"Drawing arrow from ({from_x}, {from_y}) to ({to_x}, {to_y})")  # Debug print
    
    # Draw arrow on canvas
    canvas.create_line(from_x, from_y, to_x, to_y,
                      fill='yellow', width=5,
                      arrow=tk.LAST, arrowshape=(16, 20, 6))
    
    # Add small close button in corner
    close_btn = tk.Button(overlay, text="×", 
                         command=overlay.destroy,
                         bg='#2c2c2c',
                         fg='white',
                         font=('Arial', 12),
                         relief='flat',
                         width=2,
                         height=1)
    close_btn.place(x=0, y=0)
    
    return overlay

def detect_chessboard(image):
    """Detect and crop chessboard using exact chess.com colors"""
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the colors to detect (in BGR)
    dark_green = np.array([82, 149, 115])  # #739552
    light_beige = np.array([208, 236, 235])  # #EBECD0
    
    # Convert colors to HSV
    dark_green_hsv = cv2.cvtColor(np.uint8([[dark_green]]), cv2.COLOR_BGR2HSV)[0][0]
    light_beige_hsv = cv2.cvtColor(np.uint8([[light_beige]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Create masks for both colors with small tolerance
    tolerance = 30
    dark_mask = cv2.inRange(hsv, 
                           dark_green_hsv - tolerance, 
                           dark_green_hsv + tolerance)
    light_mask = cv2.inRange(hsv, 
                            light_beige_hsv - tolerance, 
                            light_beige_hsv + tolerance)
    
    # Combine masks to get all board squares
    board_mask = cv2.bitwise_or(dark_mask, light_mask)
    
    # Find contours of the board
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find the largest contour that could be the chessboard
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate the square size (assuming 8x8 board)
    square_size = min(w, h) // 8
    
    # Calculate the total board size
    board_size = square_size * 8
    
    # Crop the chessboard region
    cropped = image[y:y+board_size, x:x+board_size]
    
    # Resize to 2400x2400 for better piece detection
    cropped = cv2.resize(cropped, (2400, 2400))
    
    return cropped, (x, y, board_size, board_size)  # Return both cropped image and original coordinates

class DetectionWindow(QWidget):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Chess Detection Results")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        # Convert OpenCV image to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Convert to QImage
        qimage = QImage(image.data, w, h, w*3, QImage.Format_RGB888)
        
        # Create label to display image
        self.image_label = QLabel()
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        
        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
        # Resize window to fit image
        self.resize(800, 800)

def show_detection_window(image, root):
    """Show detection results in a new window"""
    # Convert OpenCV image to PIL format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # Resize to a reasonable size while maintaining aspect ratio
    image.thumbnail((800, 800))
    
    # Create Tkinter window
    debug_window = tk.Toplevel(root)
    debug_window.title("Chess Detection Results")
    debug_window.attributes('-topmost', True)
    
    # Convert PIL image to PhotoImage
    photo = ImageTk.PhotoImage(image)
    
    # Create label to display image
    label = tk.Label(debug_window, image=photo)
    label.image = photo  # Keep a reference!
    label.pack(padx=10, pady=10)
    
    def safe_destroy():
        try:
            debug_window.destroy()
            if debug_window in window.debug_windows:
                window.debug_windows.remove(debug_window)
        except:
            pass
    
    # Add proper close button
    close_btn = tk.Button(debug_window, text="Close", 
                         command=safe_destroy)
    close_btn.pack(pady=5)
    
    # Handle window close button (X)
    debug_window.protocol("WM_DELETE_WINDOW", safe_destroy)
    
    # Keep track of window
    window.debug_windows.append(debug_window)
    return debug_window

def get_best_move(fen):
    """Get best move from Stockfish"""
    print("Starting Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\theob\Documents\stockfish\stockfish-windows-x86-64-avx2.exe")
    
    try:
        print(f"Creating board from FEN: {fen}")
        board = chess.Board(fen)
        
        print("Getting best move...")
        result = engine.play(board, chess.engine.Limit(time=2.0))
        best_move = result.move
        
        # Get the move in UCI format (e.g., "e2e4")
        move_str = best_move.uci()
        print(f"Found best move: {move_str}")
        
        return move_str
        
    finally:
        print("Closing Stockfish...")
        engine.quit()

def convert_detections_to_fen(all_detections, board_size):
    """Convert piece detections to FEN string"""
    # Initialize 8x8 empty board
    board = [['' for _ in range(8)] for _ in range(8)]
    square_size = board_size // 8
    
    # Fill in detected pieces
    for x, y, piece in all_detections:
        row = y // square_size
        col = x // square_size
        board[row][col] = piece
    
    # Convert to FEN
    fen_rows = []
    for row in board:
        empty_count = 0
        fen_row = ''
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    # Join rows with '/'
    fen = '/'.join(fen_rows)
    
    # Add additional FEN fields (assuming white to move, all castling available)
    fen += ' w KQkq - 0 1'
    
    return fen

class ChessDetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # Store references to windows
        self.debug_window = None
        self.current_overlay = None
        self.debug_enabled = False
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Create title bar
        self.title_bar = QWidget()
        self.title_bar.setStyleSheet("background-color: #2c2c2c;")
        self.title_bar.setFixedHeight(30)
        
        # Use horizontal layout for title bar
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)
        
        # Add title text
        self.title_label = QLabel("Chess Piece Detector")
        self.title_label.setStyleSheet("color: white;")
        self.title_label.setFont(QFont('Arial', 10))
        title_layout.addWidget(self.title_label)
        
        # Add close button
        self.close_button = QPushButton("×")
        self.close_button.setStyleSheet("""
            QPushButton {
                color: white;
                border: none;
                background: transparent;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #c93537;
            }
        """)
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)
        title_layout.addWidget(self.close_button)
        
        # Add title bar to main layout
        self.layout.addWidget(self.title_bar)
        
        # Create content area
        content = QWidget()
        content.setStyleSheet("background-color: #363636;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)  # Increased spacing between elements
        
        # Create capture button
        self.capture_btn = QPushButton("Capture Screen")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        self.capture_btn.setCursor(Qt.PointingHandCursor)
        self.capture_btn.clicked.connect(self.capture_screen)
        content_layout.addWidget(self.capture_btn)
        
        # Create remove arrows button
        self.remove_arrows_btn = QPushButton("Remove Arrows")
        self.remove_arrows_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        self.remove_arrows_btn.setCursor(Qt.PointingHandCursor)
        self.remove_arrows_btn.clicked.connect(self.remove_arrows)
        content_layout.addWidget(self.remove_arrows_btn)
        
        # Add debug toggle switch
        self.debug_checkbox = QPushButton("Debug Mode: Off")
        self.debug_checkbox.setCheckable(True)
        self.debug_checkbox.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 5px;
                min-height: 20px;
                text-align: left;
                padding-left: 15px;
            }
            QPushButton:checked {
                background-color: #45a049;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:checked:hover {
                background-color: #3d8b41;
            }
        """)
        self.debug_checkbox.clicked.connect(self.toggle_debug)
        content_layout.addWidget(self.debug_checkbox)
        
        # Create piece style input with auto-complete
        style_container = QHBoxLayout()
        style_label = QLabel("Style:")
        style_label.setStyleSheet("color: white;")
        style_container.addWidget(style_label)
        
        self.style_input = QLineEdit()
        self.style_input.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 3px;
                margin-left: 5px;
            }
            QLineEdit:focus {
                background-color: #5a5a5a;
            }
        """)
        self.style_input.setText("neo")
        self.style_input.setPlaceholderText("Enter piece style")
        
        # Add auto-complete
        completer = QCompleter(["neo", "cases", "chess24", "classic", "modern"])
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.style_input.setCompleter(completer)
        
        # Connect returnPressed instead of textChanged
        self.style_input.returnPressed.connect(self.on_style_enter)
        
        style_container.addWidget(self.style_input)
        content_layout.addLayout(style_container)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 2px;
                background-color: #4a4a4a;
                height: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #45a049;
                border-radius: 2px;
            }
        """)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)
        content_layout.addWidget(self.progress_bar)
        
        # Create instructions label
        instructions = QLabel("Press button and switch\nto chess board window")
        instructions.setStyleSheet("color: white;")
        instructions.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(instructions)
        
        # Add content to main layout
        self.layout.addWidget(content)
        
        # Window setup
        self.setMinimumSize(200, 300)  # Slightly increased height
        self.resize(200, 300)
        
        # For dragging
        self.dragging = False
        self.offset = QPoint()

    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_enabled = self.debug_checkbox.isChecked()
        self.debug_checkbox.setText("Debug Mode: On" if self.debug_enabled else "Debug Mode: Off")
        if not self.debug_enabled and self.debug_window:
            self.debug_window.close()
            self.debug_window = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.title_bar.geometry().contains(event.pos()):
            self.dragging = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(self.mapToGlobal(event.pos() - self.offset))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def capture_screen(self):
        """Capture and process the screen"""
        # Close any existing windows
        if self.current_overlay:
            self.current_overlay.close()
            self.current_overlay = None
        
        if self.debug_window:
            self.debug_window.close()
            self.debug_window = None
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        QApplication.processEvents()  # Update UI
        
        # Get screen size including all monitors
        monitors = []
        for m in get_monitors():
            monitors.append({
                'left': m.x,
                'top': m.y,
                'width': m.width,
                'height': m.height
            })
        
        # Calculate total bounding box
        left = min(m['left'] for m in monitors)
        top = min(m['top'] for m in monitors)
        right = max(m['left'] + m['width'] for m in monitors)
        bottom = max(m['top'] + m['height'] for m in monitors)
        
        # Capture the entire screen area immediately
        screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        self.progress_bar.setValue(25)
        QApplication.processEvents()
        
        cropped_board, board_coords = detect_chessboard(screenshot_cv)
        if cropped_board is None:
            print("Could not detect chessboard")
            self.progress_bar.setValue(0)
            return
        
        self.progress_bar.setValue(50)
        QApplication.processEvents()
        
        displayImage, detections = detectPieceOfChess(cropped_board)
        
        self.progress_bar.setValue(75)
        QApplication.processEvents()
        
        # Show debug window only if enabled
        if self.debug_enabled:
            self.show_detection_window(displayImage)
        
        # Get best move and draw overlay
        if detections:
            fen = convert_detections_to_fen(detections, 2400)
            try:
                best_move = get_best_move(fen)
                self.show_move_overlay(board_coords, best_move)
                self.progress_bar.setValue(100)
            except Exception as e:
                print(f"Error processing move: {str(e)}")
                self.progress_bar.setValue(0)
        else:
            self.progress_bar.setValue(0)

    def show_detection_window(self, image):
        """Show detection results in a new PyQt window"""
        # Convert OpenCV image to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create new window
        if self.debug_window is None:
            self.debug_window = QWidget()
            self.debug_window.setWindowTitle("Chess Detection Results")
            self.debug_window.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            
            # Create layout
            layout = QVBoxLayout()
            self.debug_window.setLayout(layout)
            
            # Create close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.debug_window.close)
            
            # Add widgets to layout
            self.image_label = QLabel()
            layout.addWidget(self.image_label)
            layout.addWidget(close_btn)
        
        # Convert to QImage and update label
        qimage = QImage(image.data, w, h, w*3, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(
            800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Show window
        self.debug_window.show()

    def show_move_overlay(self, board_coords, move_str):
        """Show move overlay using PyQt"""
        x, y, w, h = board_coords
        
        # Create overlay window
        overlay = QWidget()
        overlay.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setGeometry(x, y, w, h)
        
        # Create a canvas for drawing
        class ArrowCanvas(QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setAttribute(Qt.WA_TranslucentBackground)
                
            def paintEvent(self, event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                
                # Convert chess coordinates to screen coordinates
                file_to_x = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
                rank_to_y = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}
                
                square_size = w // 8
                
                # Calculate arrow coordinates
                from_file = move_str[0]
                from_rank = move_str[1]
                to_file = move_str[2]
                to_rank = move_str[3]
                
                from_x = file_to_x[from_file] * square_size + square_size // 2
                from_y = rank_to_y[from_rank] * square_size + square_size // 2
                to_x = file_to_x[to_file] * square_size + square_size // 2
                to_y = rank_to_y[to_rank] * square_size + square_size // 2
                
                # Draw arrow
                pen = QPen(QColor(255, 255, 0))  # Yellow color
                pen.setWidth(5)
                painter.setPen(pen)
                
                # Draw line
                painter.drawLine(from_x, from_y, to_x, to_y)
                
                # Draw arrowhead
                angle = np.arctan2(to_y - from_y, to_x - from_x)
                arrow_size = 20
                arrow_angle = np.pi / 6  # 30 degrees
                
                # Calculate arrowhead points and convert to integers
                p1 = QPoint(
                    int(to_x - arrow_size * np.cos(angle - arrow_angle)),
                    int(to_y - arrow_size * np.sin(angle - arrow_angle))
                )
                p2 = QPoint(
                    int(to_x - arrow_size * np.cos(angle + arrow_angle)),
                    int(to_y - arrow_size * np.sin(angle + arrow_angle))
                )
                
                # Draw arrowhead
                painter.drawLine(to_x, to_y, p1.x(), p1.y())
                painter.drawLine(to_x, to_y, p2.x(), p2.y())
        
        # Create and add the canvas
        canvas = ArrowCanvas(overlay)
        canvas.resize(w, h)
        
        # Add small close button
        close_btn = QPushButton("×", overlay)
        close_btn.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #2c2c2c;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #c93537;
            }
        """)
        close_btn.setFixedSize(20, 20)
        close_btn.clicked.connect(overlay.close)
        close_btn.move(0, 0)
        
        # Store reference to prevent garbage collection
        self.current_overlay = overlay
        overlay.show()

    def remove_arrows(self):
        """Remove all arrow overlays"""
        if self.current_overlay:
            self.current_overlay.close()
            self.current_overlay = None

    def on_style_enter(self):
        """Handle style change when Enter is pressed"""
        new_style = self.style_input.text().strip().lower()  # Convert to lowercase
        if not new_style:
            return
        
        try:
            # Clear the chess pieces directory
            chess_piece_dir = CHESS_PIECE_DIR  # Use the same directory as startup
            if os.path.exists(chess_piece_dir):
                for file in os.listdir(chess_piece_dir):
                    file_path = os.path.join(chess_piece_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        
            # Reset progress bar
            self.progress_bar.setValue(0)
            QApplication.processEvents()
            
            # Download new pieces using the same method as startup
            for piece in chessPieceThreshold.keys():
                # Update progress
                progress = int((list(chessPieceThreshold.keys()).index(piece) + 1) / len(chessPieceThreshold) * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()
                
                color = 'b' if piece.islower() else 'w'
                filename = f'{color}{piece.lower()}.png'
                filepath = os.path.join(CHESS_PIECE_DIR, filename)
                
                # Use the exact same URL as startup
                url = f'https://www.chess.com/chess-themes/pieces/{new_style}/300/{color}{piece.lower()}.png'
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Convert to PIL Image
                    img = Image.open(BytesIO(response.content))
                    # Convert to RGBA if not already
                    img = img.convert('RGBA')
                    # Save the original image
                    img.save(filepath)
                else:
                    print(f"Failed to download {piece}: {response.status_code}")
                    return  # Stop if any piece fails
            
            # Reset progress bar after download
            self.progress_bar.setValue(0)
            
            # Clear focus from input
            self.style_input.clearFocus()
            
        except Exception as e:
            print(f"Error updating chess pieces: {e}")
            self.progress_bar.setValue(0)

# Create and run application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChessDetectorWindow()
    window.show()
    sys.exit(app.exec_())

