import pygame
import numpy as np
import scipy.signal
import pyaudio
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Parameters
N = 150  # Grid size
R = 10   # Kernel radius
SCREEN_SIZE = (450, 450)  # Pygame window size
center = N//2
radius = 5

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

def list_microphones():
    p = pyaudio.PyAudio()
    info = []
    print("\nAvailable Microphones:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            print(f"Index {i}: {dev_info['name']}")
            info.append(dev_info)
    p.terminate()
    return info

def select_microphone():
    mics = list_microphones()
    if not mics:
        print("No microphones found!")
        return 0
    selected = input("Select microphone index (press Enter for default): ").strip()
    return int(selected) if selected.isdigit() else 0

# Initialize screen
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Interactive Lenia')
clock = pygame.time.Clock()

# Initialize Lenia grid with colorful noise pattern
A = np.random.random((N, N)) * 0.2
A[center-radius:center+radius, center-radius:center+radius] = np.random.random((2*radius, 2*radius)) * 0.5

# Create kernel (Bert Chan's version)
def gaussian_2d(x, y, sigma=1):
    r = np.sqrt(x**2 + y**2) / R
    k = np.zeros_like(r)
    mask = (r < 1)
    k[mask] = np.exp(4 - 1/(1 - r[mask]**2))
    return k

x, y = np.meshgrid(np.linspace(-R, R, 2*R+1), np.linspace(-R, R, 2*R+1))
K = gaussian_2d(x, y, sigma=5)
K /= np.sum(K)

# Growth function with adjusted parameters
def growth(x, beta=1/3, sigma=0.15):
    return 2 * np.exp(-(x - beta)**2 / (2 * sigma**2)) - 1

# Convert Lenia grid to surface with enhanced colors
def grid_to_surface(grid, audio_factor):
    rgb_array = (grid * 255).astype(np.uint8)
    colored = np.zeros((N, N, 3), dtype=np.uint8)
    
    # Base colors (blue-green mix)
    colored[..., 0] = rgb_array * 0.3  # Blue
    colored[..., 1] = rgb_array * 0.8  # Green
    colored[..., 2] = rgb_array * 0.2  # Red
    
    # Add yellow highlights for high activity areas
    highlights = (grid > 0.7)
    colored[highlights, 1] = 255  # Green
    colored[highlights, 2] = 255  # Red
    
    # Add audio-reactive coloring
    if audio_factor > 0.05:
        colored[..., 0] = np.minimum(255, colored[..., 0] + (audio_factor * 50))  # More blue
        colored[..., 2] = np.minimum(255, colored[..., 2] + (audio_factor * 50))  # More red
    
    surface = pygame.surfarray.make_surface(colored)
    return pygame.transform.scale(surface, SCREEN_SIZE)

# Audio setup
selected_mic = select_microphone()
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=selected_mic,
                frames_per_buffer=CHUNK)

def get_audio_level():
    try:
        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        level = np.sqrt(np.mean(audio_data**2))
        return level * 8.0  # Increased sensitivity
    except Exception as e:
        print(f"Error reading audio: {e}")
        return 0

# Main loop
running = True
paused = False
font = pygame.font.Font(None, 36)

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                paused = not paused
            elif event.key == K_r:
                # Reset with active center and noise
                A = np.random.random((N, N)) * 0.2
                reset_radius = 20
                A[center-reset_radius:center+reset_radius, 
                  center-reset_radius:center+reset_radius] = np.random.random((2*reset_radius, 2*reset_radius)) * 0.5
    
    if not paused:
        # Get audio level
        audio_level = get_audio_level()
        audio_factor = np.clip(audio_level / 0.1, 0, 10)
        
        # Add noise based on audio
        # noise_level = 0.05 * (1 + audio_factor)
        noise_level = 0
        noise = np.random.random(A.shape) * noise_level
        
        # Update Lenia grid
        U = scipy.signal.convolve2d(A, K, mode='same', boundary='wrap')
        G = growth(U)
        A = np.clip(A + (0.4 * G * (1 + audio_factor)) + noise, 0, 1)
        
        # Add energy injection when sound is detected
        if audio_factor > 0.1:
            reaction_radius = min(int(20 * (1 + audio_factor)), N//4)
            start_x = max(center-reaction_radius, 0)
            end_x = min(center+reaction_radius, N)
            start_y = max(center-reaction_radius, 0)
            end_y = min(center+reaction_radius, N)
            
            # Create gradient injection
            xx, yy = np.meshgrid(
                np.linspace(-1, 1, end_x-start_x),
                np.linspace(-1, 1, end_y-start_y)
            )
            gradient = np.exp(-(xx**2 + yy**2) / (2 * 0.5**2))
            A[start_x:end_x, start_y:end_y] += gradient * audio_factor * 0.2
        
        # Prevent pattern death
        if np.mean(A) < 0.05:
            reset_radius = 20
            A[center-reset_radius:center+reset_radius, 
              center-reset_radius:center+reset_radius] = np.random.random((2*reset_radius, 2*reset_radius)) * 0.5
            A += np.random.random(A.shape) * 0.1
        
        # Draw the grid with audio-reactive colors
        surface = grid_to_surface(A, audio_factor)
        screen.blit(surface, (0, 0))
        
        # Display audio level
        level_text = f"Audio Level: {audio_factor:.2f}"
        level_surface = font.render(level_text, True, (255, 255, 255))
        screen.blit(level_surface, (10, 10))
        
        pygame.display.flip()
    
    clock.tick(30)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()