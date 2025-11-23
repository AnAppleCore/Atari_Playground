"""Visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path


class VideoRecorder:
    """Record gameplay as video/GIF."""

    def __init__(self, output_path: str, fps: int = 15):
        """
        Initialize video recorder.

        Args:
            output_path: Path to save video/GIF
            fps: Frames per second (default: 15, lower = slower playback)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video."""
        # Convert frame to RGB if needed
        if len(frame.shape) == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[0] == 4:  # Frame stack
            frame = frame[0]  # Use first frame
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 1:  # Single channel
            frame = np.stack([frame[:, :, 0]] * 3, axis=-1)

        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        self.frames.append(frame)

    def save(self, format: str = "mp4"):
        """
        Save video.

        Args:
            format: 'gif' or 'mp4' (default: mp4)
        """
        if not self.frames:
            print("Warning: No frames to save")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "mp4":
                # Try to save as MP4 with ffmpeg
                try:
                    imageio.mimsave(
                        str(self.output_path),
                        self.frames,
                        fps=self.fps,
                        codec='libx264',
                        pixelformat='yuv420p'
                    )
                except Exception:
                    # Fallback: save as MP4 without specific codec
                    imageio.mimsave(str(self.output_path), self.frames, fps=self.fps)
            elif format == "gif":
                imageio.mimsave(str(self.output_path), self.frames, fps=self.fps)
            else:
                raise ValueError(f"Unsupported format: {format}")

            file_size = self.output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ Video saved: {self.output_path}")
            print(f"  Frames: {len(self.frames)}, FPS: {self.fps}, Size: {file_size:.1f} MB")
        except Exception as e:
            print(f"✗ Error saving video: {e}")
            print(f"  Make sure ffmpeg is installed: pip install imageio-ffmpeg")

    def reset(self):
        """Reset frames."""
        self.frames = []


class MetricsPlotter:
    """Plot training metrics."""
    
    def __init__(self):
        """Initialize plotter."""
        self.metrics = {}
    
    def add_metric(self, name: str, value: float):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def plot(self, output_path: str):
        """Plot metrics."""
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(15, 4))
        
        if len(self.metrics) == 1:
            axes = [axes]
        
        for idx, (name, values) in enumerate(self.metrics.items()):
            axes[idx].plot(values)
            axes[idx].set_title(name)
            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("Value")
            axes[idx].grid(True)
        
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

