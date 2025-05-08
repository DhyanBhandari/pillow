# pip install numpy pygame scikit-learn joblib
# install these libraries if not already installed
#  to run this code python sleep_monitor.py
import os
import time
import joblib
import numpy as np
import pygame
from datetime import datetime

class SleepMonitor:
    def __init__(self, model_path='Pillow_model.pkl', music_folder='relaxation_music'):
        """
        Initialize the sleep monitor with a trained model and music directory.
        
        Args:
            model_path: Path to the trained model
            music_folder: Folder containing relaxation music files
        """
        # Load the model
        self.model = joblib.load(model_path)
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # Set up music directory
        self.music_folder = music_folder
        if not os.path.exists(music_folder):
            os.makedirs(music_folder)
            print(f"Created music folder at {music_folder}")
            print("Please add relaxation music files (.mp3, .wav) to this folder")
        
        # Get list of music files
        self.music_files = self._get_music_files()
        
        # Tracking variables
        self.current_state = None
        self.music_playing = False
        self.consecutive_danger_count = 0
        self.consecutive_normal_count = 0
    
    def _get_music_files(self):
        """Get list of music files from the music folder."""
        if not os.path.exists(self.music_folder):
            return []
        
        music_files = []
        for file in os.listdir(self.music_folder):
            if file.endswith(('.mp3', '.wav')):
                music_files.append(os.path.join(self.music_folder, file))
        
        return music_files
    
    def play_music(self):
        """Start playing relaxation music."""
        if not self.music_files:
            print("No music files found in the music folder.")
            return False
        
        if self.music_playing:
            return True
        
        # Choose a random music file
        music_file = np.random.choice(self.music_files)
        
        try:
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play(-1)  # -1 means loop indefinitely
            self.music_playing = True
            print(f"Playing relaxation music: {os.path.basename(music_file)}")
            return True
        except Exception as e:
            print(f"Error playing music: {e}")
            return False
    
    def stop_music(self):
        """Stop playing relaxation music."""
        if self.music_playing:
            pygame.mixer.music.stop()
            self.music_playing = False
            print("Stopped relaxation music")
    
    def read_input_file(self, input_file):
        """Read sensor data from input file."""
        try:
            with open(input_file, 'r') as f:
                data = f.read().strip()
            
            # Parse data
            values = data.split('#')
            if len(values) != 4:
                raise ValueError("Input file should contain 4 values separated by #")
            
            # Convert to float and reshape for model
            features = np.array([float(val) for val in values]).reshape(1, -1)
            return features
        except Exception as e:
            print(f"Error reading input file: {e}")
            return None
    
    def predict_state(self, features):
        """Use the model to predict sleep state."""
        try:
            prediction = self.model.predict(features)
            return prediction[0]
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def is_danger_state(self, state):
        """Check if the state is a danger state."""
        return state and state.startswith("Danger")
    
    def process_input(self, input_file):
        """Process input file and take appropriate action."""
        features = self.read_input_file(input_file)
        if features is None:
            return
        
        state = self.predict_state(features)
        self.current_state = state
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - Detected: {state}")
        
        # Write to output file
        with open("output.txt", "w") as f:
            f.write(f"['{state}']")
        
        # Write to result file
        with open("PILLOW_RESULT.txt", "w") as f:
            f.write(f"PILLOW_TEST: Date- {timestamp} detected ['{state}']")
        
        # Decide whether to play or stop music
        if self.is_danger_state(state):
            self.consecutive_danger_count += 1
            self.consecutive_normal_count = 0
            
            # Start playing music after 3 consecutive danger readings
            if self.consecutive_danger_count >= 3 and not self.music_playing:
                self.play_music()
        else:
            self.consecutive_danger_count = 0
            self.consecutive_normal_count += 1
            
            # Stop music after 5 consecutive non-danger readings
            if self.consecutive_normal_count >= 5 and self.music_playing:
                self.stop_music()
    
    def monitor(self, input_file="input.txt", interval=10):
        """
        Continuously monitor sleep state and manage music.
        
        Args:
            input_file: Path to input file
            interval: Seconds between checks
        """
        print(f"Starting sleep monitor. Checking {input_file} every {interval} seconds.")
        print(f"Relaxation music will play when danger state is detected for 3 consecutive readings.")
        
        try:
            while True:
                if os.path.exists(input_file):
                    self.process_input(input_file)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nSleep monitor stopped.")
            self.stop_music()
            pygame.mixer.quit()


if __name__ == "__main__":
    # Create the monitor
    monitor = SleepMonitor()
    
    # Check if music files exist
    if not monitor.music_files:
        print("\nWARNING: No music files found. Please add relaxation music to the 'relaxation_music' folder.")
        print("Supported formats: .mp3, .wav\n")
    
    # Start monitoring
    monitor.monitor()