import cv2
import numpy as np
import os
import requests
from io import BytesIO
import time
import tkinter as tk
from PIL import ImageGrab, Image

CHESS_BOARD_OUTPUT_DIR = os.path.join('dist')
CHESS_PIECE_DIR = os.path.join('chess_piece')
SHOW_IMAGE = True
EXPORT_IMAGE = True

# Create chess_piece directory if it doesn't exist
os.makedirs(CHESS_PIECE_DIR, exist_ok=True)

# The threshold values for each chess piece can be adjusted according to the image quality
chessPieceThreshold = {
    'B': 0.20, #bishop
    'b': 0.02, #bishop_black
    'K': 0.20, #king
    'k': 0.02, #king_black
    'N': 0.20, #knight
    'n': 0.02, #knight_black
    'P': 0.20, #pawn
    'p': 0.01, #pawn_black
    'Q': 1.95, #queen
    'q': 0.02, #queen_black
    'R': 0.20, #rook
    'r': 0.02, #rook_black
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
    # Define the piece colors in BGR
    white_piece = np.array([249, 249, 249])  # #F9F9F9
    black_piece = np.array([87, 89, 93])     # #5D5957
    
    # Create masks for piece colors with small tolerance
    tolerance = 10
    white_mask = cv2.inRange(square_img, 
                            white_piece - tolerance,
                            white_piece + tolerance)
    black_mask = cv2.inRange(square_img, 
                            black_piece - tolerance,
                            black_piece + tolerance)
    
    # Check if either color is present
    has_white = np.any(white_mask > 0)
    has_black = np.any(black_mask > 0)
    
    if not (has_white or has_black):
        return None
        
    # If we have piece colors, proceed with full detection
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    
    # Count dark and light pixels
    dark_pixels = np.sum(gray < 100)
    light_pixels = np.sum(gray > 200)
    total_pixels = gray.size
    
    dark_percentage = dark_pixels / total_pixels
    light_percentage = light_pixels / total_pixels
    
    # If significant dark pixels, it's a black piece
    if dark_percentage > 0.15:
        return 'black_piece'
    # If significant light pixels, it's a white piece
    elif light_percentage > 0.15:
        return 'white_piece'
    
    return None

def detectPieceOfChess(boardImage):
    # Remove board colors first
    boardImage = remove_board_colors(boardImage)
    
    # Create a copy of the image to draw on
    displayImage = boardImage.copy()
    
    # Calculate square size
    square_size = 300  # Each square is 300x300
    
    # Store all detections
    all_detections = []
    
    # Process pieces in specific order
    piece_order = ['Q']  # Start with white queen
    piece_order.extend([p for p in sorted(chessPieceImages.keys()) if p.isupper() and p != 'Q'])  # Other white pieces
    piece_order.extend([p for p in sorted(chessPieceImages.keys()) if p.islower()])  # Black pieces
    
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
    
    if EXPORT_IMAGE:
        timestamp = str(int(time.time()))
        cv2.imwrite(os.path.join(CHESS_BOARD_OUTPUT_DIR, f'board_{timestamp}.jpg'), displayImage)
    
    if SHOW_IMAGE:
        cv2.namedWindow('Chess Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Chess Detection', displayImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        return None
    
    # Find the largest contour that could be the chessboard
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate the square size (assuming 8x8 board)
    square_size = min(w, h) // 8
    
    # Calculate the total board size
    board_size = square_size * 8
    
    # Calculate center of the detected region
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate new coordinates for exact board size
    x = center_x - board_size // 2
    y = center_y - board_size // 2
    
    # Crop the chessboard region
    cropped = image[y:y+board_size, x:x+board_size]
    
    # Resize to 2400x2400 for better piece detection
    cropped = cv2.resize(cropped, (2400, 2400))
    
    return cropped

def capture_screen():
    # Add a small delay to allow user to switch to the chess board window
    root.iconify()  # Minimize the window
    time.sleep(2)  # Wait 2 seconds
    
    # Capture the screen
    screenshot = ImageGrab.grab()
    # Convert PIL image to OpenCV format
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Add chessboard detection
    cropped_board = detect_chessboard(screenshot)
    if cropped_board is None:
        print("Could not detect chessboard")
        root.deiconify()
        return
        
    root.deiconify()
    detectPieceOfChess(cropped_board)

# Create GUI window
root = tk.Tk()
root.title("Chess Piece Detector")
root.geometry("200x100")

# Create capture button
capture_btn = tk.Button(root, text="Capture Screen", command=capture_screen)
capture_btn.pack(expand=True)

# Create instructions label
instructions = tk.Label(root, text="Press button and switch\nto chess board window", justify=tk.CENTER)
instructions.pack(expand=True)

root.mainloop()