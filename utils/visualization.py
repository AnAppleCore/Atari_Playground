"""Visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import torch


class VideoRecorder:
    """Record gameplay as video/GIF."""

    def __init__(self, output_path: str, fps: int = 30):
        """
        Initialize video recorder.

        Args:
            output_path: Path to save video/GIF
            fps: Frames per second (default: 30 for better quality)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frames = []

    def add_frame(self, frame):
        """Add a frame to the video."""
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        
        frame = np.asarray(frame)
        
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif len(frame.shape) == 3:
            if frame.shape[0] == 4:
                frame = frame[0]
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[0] == 1:
                frame = frame[0]
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[2] == 1:
                frame = np.stack([frame[:, :, 0]] * 3, axis=-1)
            elif frame.shape[2] == 3:
                pass
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            else:
                raise ValueError(f"Unexpected frame shape: {frame.shape}")
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        frame = np.ascontiguousarray(frame)
        
        if frame.shape[0] < 84 or frame.shape[1] < 84:
            try:
                import cv2
                target_size = (max(84, frame.shape[1]), max(84, frame.shape[0]))
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            except ImportError:
                scale_h = max(1, 84 / frame.shape[0])
                scale_w = max(1, 84 / frame.shape[1])
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale_h), int(w * scale_w)
                frame = np.repeat(np.repeat(frame, scale_h, axis=0), scale_w, axis=1)[:new_h, :new_w]
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        self.frames.append(frame)

    def save(self, format: str = "mp4"):
        """Save video."""
        if not self.frames:
            print("Warning: No frames to save")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "mp4":
                try:
                    import imageio_ffmpeg
                    writer = imageio_ffmpeg.write_frames(
                        str(self.output_path),
                        size=(self.frames[0].shape[1], self.frames[0].shape[0]),
                        fps=self.fps,
                        codec='libx264',
                        quality=8,
                        pixelformat='yuv420p',
                        bitrate='2M',
                    )
                    writer.send(None)
                    for frame in self.frames:
                        writer.send(frame)
                    writer.close()
                except (ImportError, Exception):
                    try:
                        imageio.mimsave(
                            str(self.output_path),
                            self.frames,
                            fps=self.fps,
                            codec='libx264',
                            quality=8,
                            pixelformat='yuv420p',
                            macro_block_size=None,
                        )
                    except Exception:
                        imageio.mimsave(str(self.output_path), self.frames, fps=self.fps)
            elif format == "gif":
                try:
                    imageio.mimsave(
                        str(self.output_path),
                        self.frames,
                        fps=self.fps,
                        duration=1.0/self.fps,
                    )
                except Exception:
                    imageio.mimsave(str(self.output_path), self.frames, fps=self.fps)
            else:
                raise ValueError(f"Unsupported format: {format}")

            file_size = self.output_path.stat().st_size / (1024 * 1024)
            duration = len(self.frames) / self.fps
            print(f"✓ Video saved: {self.output_path}")
            print(f"  Frames: {len(self.frames)}, FPS: {self.fps}, Duration: {duration:.2f}s, Size: {file_size:.1f} MB")
        except Exception as e:
            print(f"✗ Error saving video: {e}")
            print(f"  Make sure ffmpeg is installed: pip install imageio-ffmpeg")
            import traceback
            traceback.print_exc()

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
    
    def plot(self, output_path: str, max_cols: int = 2):
        """Plot metrics in a compact grid layout.

        Args:
            output_path: Path to save the figure
            max_cols: Maximum number of subplots per row (default: 2)
        """
        if not self.metrics:
            print("No metrics to plot.")
            return

        num_metrics = len(self.metrics)
        ncols = min(max_cols, num_metrics)
        nrows = int(np.ceil(num_metrics / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            squeeze=False,
        )

        # Plot each metric into its own subplot
        for idx, (name, values) in enumerate(sorted(self.metrics.items())):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]
            
            if name == "episode_reward" and len(values) > 10:
                ax.plot(values, alpha=0.3, color='blue', label='Raw')
                window = min(10, len(values) // 5)
                if window > 1:
                    moving_avg = []
                    for i in range(len(values)):
                        start = max(0, i - window + 1)
                        moving_avg.append(sum(values[start:i+1]) / (i - start + 1))
                    ax.plot(moving_avg, color='red', linewidth=2, label=f'MA({window})')
                    ax.legend()
            else:
                ax.plot(values)
            
            ax.set_title(name.replace('_', ' ').title())
            xlabel = "Episode" if name == "episode_reward" else "Step"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

        for idx in range(num_metrics, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            fig.delaxes(axes[row][col])

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

