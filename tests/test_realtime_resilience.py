import pytest
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add scripts to path so we can import run_realtime
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import run_realtime

def test_cleanup_old_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "json").mkdir()
    
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=4)).timestamp()
    new_time = (now - timedelta(days=1)).timestamp()
    
    # old json
    f1 = out / "json" / "objects_old.json"
    f1.touch()
    os.utime(f1, (old_time, old_time))
    
    # new json
    f2 = out / "out_new" / "objects_new.json"
    f2.parent.mkdir()
    f2.touch()
    os.utime(f2, (new_time, new_time))
    
    # state.pkl (old)
    f3 = out / "state.pkl"
    f3.touch()
    os.utime(f3, (old_time, old_time))
    
    run_realtime.cleanup_old_files(out, keep_days=3)
    
    assert not f1.exists()  # Old file should be deleted
    assert not (out / "json").exists() # Empty dir should be deleted
    assert f2.exists()      # New file should be kept
    assert f3.exists()      # state.pkl should be kept, even if old

def test_run_realtime_broken_state_recovery(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    state = out / "state.pkl"
    state.write_text("invalid pickle data") # Write bad pickle
    
    watch = tmp_path / "watch"
    watch.mkdir()
    
    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", [
        "run_realtime.py",
        "--watch-dir", str(watch),
        "--out", str(out),
        "--once"
    ])
    
    # main should not crash despite invalid state.pkl
    run_realtime.main()
