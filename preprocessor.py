import torch
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import os

class NormalPreprocessor :

    def __init__(self) :
        self.pool = ThreadPoolExecutor(max_workers=os.cpu_count())

    def _process_individual_frame(self,frame,out_size=(60,45)) :

        if frame is None :
            return None

        frame = frame.detach().cpu()

        frame = frame.squeeze(0).permute(1,2,0)

        frame = torch.clip(frame,0,255).numpy().astype(np.uint8)

        # ic(frame)

        img = Image.fromarray(frame).resize(out_size,Image.Resampling.BILINEAR)

        # img.save("processed.png")

        arr = np.asarray(img,dtype = np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

        return tensor

    def __call__(self,frame,out_size=(60,45),device=None) :

        if isinstance(frame,list) :
            processed = self.pool.map(lambda f : self._process_individual_frame(f,out_size),frame)
            if device is not None :
                processed = [p if p is None else p.to(device) for p in processed]
        else :
            processed = self._process_individual_frame(frame,out_size)
            if processed is None :
                return None
            if device is not None :
                processed = processed.to(device)

        return processed
    
    def close(self) :
        self.pool.shutdown(wait=True)

base_preprocessor = NormalPreprocessor()