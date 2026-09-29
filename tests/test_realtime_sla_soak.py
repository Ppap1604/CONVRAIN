import pytest
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import psutil
import os
import gc
from unittest.mock import MagicMock

from convrain.pipeline import ConvRainPipeline
from convrain.config import load_config
from convrain.ingest import Scan
from convrain.intensity import RainRateLUT

def test_pipeline_memory_soak_and_sla():
    cfg = load_config(None)
    # Use smaller domain to speed up testing
    cfg["domain"] = {"lon_min": 100.0, "lat_min": 10.0, "lon_max": 102.0, "lat_max": 12.0}
    
    pipe = ConvRainPipeline(cfg, None, None, None, RainRateLUT())
    
    process = psutil.Process(os.getpid())
    gc.collect()
    mem_start = process.memory_info().rss
    
    latencies = []
    t = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i in range(15):
        lon_grid, lat_grid = np.meshgrid(np.linspace(100.0, 102.0, 100), np.linspace(10.0, 12.0, 100))
        ir = np.full((100, 100), 290.0, dtype=np.float32)
        
        # Add a moving synthetic cloud
        r, c = int(50 + i*2), int(50 + i*2)
        if r < 100 and c < 100:
            ir[max(0, r-10):min(100, r+10), max(0, c-10):min(100, c+10)] = 210.0
            
        bt = {"B13": ir, "B08": ir - 5.0, "B10": ir - 10.0, "B11": ir, "B14": ir, "B15": ir, "B16": ir}
        scan = Scan(time=t, bt=bt, lat=lat_grid.astype(np.float32), lon=lon_grid.astype(np.float32))
        
        t0 = time.time()
        res = pipe.step(scan)
        t1 = time.time()
        
        latencies.append(t1 - t0)
        t += timedelta(minutes=10)
        
    gc.collect()
    mem_end = process.memory_info().rss
    mem_diff_mb = (mem_end - mem_start) / (1024 * 1024)
    
    # SLA Criteria: should be very fast for 100x100 grid (e.g. < 2 seconds)
    mean_latency = np.mean(latencies)
    assert mean_latency < 5.0, f"SLA Failed: Latency {mean_latency:.2f}s is too high"
    
    # Memory Leak Criteria: < 50MB increase over the run
    assert mem_diff_mb < 50.0, f"Memory Leak Failed: Increased by {mem_diff_mb:.2f} MB"
