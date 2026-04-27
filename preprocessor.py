import torch
import torch.nn.functional as F

class NormalPreprocessor:
    def __init__(self):
        pass

    def __call__(self, frame, out_size=(112, 112), device=None):
        if frame is None:
            return None
        
        # If frame is a list (for DQN sampling), handle it
        if isinstance(frame, list):
            frames = torch.cat(frame, dim=0)
        else:
            frames = frame

        # Ensure we are on the correct device immediately
        if device is not None:
            frames = frames.to(device)

        # Optimization: Perform all operations in a vectorized way on GPU
        # VizDoom typically provides (B, C, H, W) or (C, H, W)
        # Normalize to [0, 1]
        frames = frames.float() / 255.0

        # Resize using GPU-accelerated interpolation
        if frames.shape[-2:] != out_size:
            frames = F.interpolate(frames, size=out_size, mode='bilinear', align_corners=False)

        return frames
    
    def close(self):
        pass

base_preprocessor = NormalPreprocessor()
