import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from convrain.ingest import read_ahi_hsd

def test_read_ahi_hsd_mock():
    # Setup mock config
    cfg = {
        "domain": {"lon_min": 97.0, "lat_min": 5.0, "lon_max": 106.0, "lat_max": 21.0},
        "satellite": {"bands": ["B13", "B08"], "sub_lon": 140.7, "orbit_radius_km": 42164.0}
    }
    
    mock_satpy = MagicMock()
    mock_scene_cls = MagicMock()
    mock_satpy.Scene = mock_scene_cls
    
    mock_scene = MagicMock()
    mock_scene_cls.return_value = mock_scene
    mock_cropped = MagicMock()
    mock_scene.crop.return_value = mock_cropped
    
    mock_b13 = MagicMock()
    mock_b08 = MagicMock()
    mock_b13.values = np.full((10, 10), 280.0, dtype=np.float32)
    mock_b08.values = np.full((10, 10), 250.0, dtype=np.float32)
    
    lons, lats = np.meshgrid(np.linspace(97, 106, 10), np.linspace(5, 21, 10))
    mock_area = MagicMock()
    mock_area.get_lonlats.return_value = (lons, lats)
    mock_b13.attrs = {"area": mock_area, "start_time": datetime(2026, 9, 29, 0, 0)}
    
    mock_cropped.__getitem__.side_effect = lambda k: {"B13": mock_b13, "B08": mock_b08}[k]
    
    with patch.dict("sys.modules", {"satpy": mock_satpy}):
        scan = read_ahi_hsd(["dummy_file.dat"], cfg)
        
    assert scan.shape == (10, 10)
    assert "B13" in scan.bt
    assert "B08" in scan.bt
    assert scan.time.year == 2026
    assert scan.time.tzinfo == timezone.utc
    assert scan.quality_ok is True
    assert scan.sat_zenith is not None

def test_read_ahi_hsd_import_error():
    cfg = {
        "domain": {"lon_min": 97.0, "lat_min": 5.0, "lon_max": 106.0, "lat_max": 21.0},
        "satellite": {"bands": ["B13"], "sub_lon": 140.7, "orbit_radius_km": 42164.0}
    }
    # Simulate satpy not installed by patching sys.modules with satpy=None
    with patch.dict("sys.modules", {"satpy": None}):
        with pytest.raises(ImportError, match="ต้องติดตั้ง satpy"):
            read_ahi_hsd(["dummy_file.dat"], cfg)
