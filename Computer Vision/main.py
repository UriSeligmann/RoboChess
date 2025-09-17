from FENCalculator import getFen


def main():
    # Starting position
    current_fen = "RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w KQkq - 0 1"
    print(f"\nInitial position: {current_fen}")
    
    image_count = 0
    while True:
        try:
            print("\nPress Enter to capture next position (or 'q' to quit)...")
            key = input()
            if key.lower() == 'q':
                break
            
            # Process the image
            new_fen = getFen("Capture.png", current_fen)
            print(f"Move detected! New position:")
            print(f"FEN: {new_fen}")
            
            # Update current position
            current_fen = new_fen
            image_count += 1

        except ValueError as e:
            print(f"Couldn't determine move: {e}")
            input("Press Enter to retry...")
        except Exception as e:
            print(f"Error: {e}")
            input("Press Enter to retry...")

if __name__ == "__main__":
    main()