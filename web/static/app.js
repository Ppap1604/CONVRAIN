/**
 * CONVRAIN - Satellite Convective Rain Onset Nowcast Web Platform
 * Dual Mode: Modern Dark SpaceX / Tactical + EUMETSAT RDT-CW Light Mode
 * Features:
 *  - Dark Mode & Light Mode seamlessly switchable
 *  - Satellite Cloud Deck (soft, translucent grayscale clouds from satellite imagery)
 *  - RDT Convective Cell Framing (Developing, Active Rain, Decaying, Projected Landfall)
 *  - Sharp Motion Direction Vector Arrows & Trajectory Tracking
 *  - Real-time GPS & Map-Click Point Rain Onset Prediction (ETA Window)
 *  - 10-Minute Satellite Replay Timeline Player
 */

// Global State
let map;
let baseLayers = {};
let currentBase = 'dark';
let isDarkMode = true;

// Layer Groups & Overlays
let satelliteCloudOverlay = null;
let footprintsLayer = null;
let trajectoriesLayer = null;
let impactZonesLayer = null;
let graticuleLayer = null;
let userMarker = null;
let interceptVectorLine = null;

// Layer Visibility Toggles
let showSatelliteClouds = true;
let showTrajectories = true;
let showImpactZones = true;
let showFootprints = true;
let showGraticule = true;

// Data State
let allScans = [];
let currentScanIndex = -1;
let currentScanData = null;
let isPlaying = false;
let playInterval = null;
let activeUserCoord = null;
let hasInitializedBounds = false;

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupEventListeners();
  loadScans();
});

// 1. Initialize Map with Basemaps & Graticule
function initMap() {
  const mapElement = document.getElementById('map');
  if (!mapElement) return;

  map = L.map('map', {
    center: [13.8, 100.8],
    zoom: 8,
    zoomControl: false,
    preferCanvas: true,
  });

  // Custom Zoom Control (Top-Left under Menu)
  L.control.zoom({ position: 'topleft' }).addTo(map);

  // Basemaps (Esri Dark Canvas, RDT Ocean, Satellite Imagery, OSM)
  baseLayers.dark = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    {
      className: 'dark-tiles',
      maxZoom: 16,
      attribution: '&copy; Esri, HERE, Garmin',
    }
  );

  baseLayers.rdt_ocean = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    {
      className: 'rdt-ocean-tiles',
      maxZoom: 16,
      attribution: '&copy; Esri &copy; GEBCO, NOAA, National Geographic',
    }
  );

  baseLayers.satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {
      maxZoom: 19,
      attribution: '&copy; Esri, Maxar',
    }
  );

  baseLayers.osm = L.tileLayer(
    'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap',
    }
  );

  // Default: Dark Mode Basemap
  baseLayers.dark.addTo(map);

  // Dedicated Layer Groups
  footprintsLayer = L.geoJSON(null, {
    style: styleRdtFootprint,
    onEachFeature: onEachFootprintFeature,
  }).addTo(map);

  impactZonesLayer = L.geoJSON(null, {
    style: styleRdtImpactZone,
    onEachFeature: onEachImpactFeature,
  }).addTo(map);

  trajectoriesLayer = L.layerGroup().addTo(map);
  graticuleLayer = L.layerGroup().addTo(map);

  // Build Lat/Lon Coordinate Graticule Grid
  renderLatLonGraticule();

  // Map Click Inspector
  map.on('click', (e) => {
    const lat = parseFloat(e.latlng.lat.toFixed(4));
    const lon = parseFloat(e.latlng.lng.toFixed(4));
    setUserLocation(lat, lon, true);
  });

  setTimeout(() => { if (map) map.invalidateSize(); }, 150);
  setTimeout(() => { if (map) map.invalidateSize(); }, 600);
  window.addEventListener('resize', () => { if (map) map.invalidateSize(); });
}

// 2. Render Lat/Lon Graticule Grid (Adapts to Dark/Light theme)
function renderLatLonGraticule() {
  graticuleLayer.clearLayers();

  const lats = [11, 12, 13, 14, 15, 16, 17];
  const lons = [97, 98, 99, 100, 101, 102, 103, 104, 105];
  const lineColor = isDarkMode ? 'rgba(255, 255, 255, 0.13)' : 'rgba(0, 0, 0, 0.22)';

  lats.forEach((lat) => {
    // Horizontal latitude line
    const line = L.polyline([[lat, 95], [lat, 107]], {
      color: lineColor,
      weight: 0.8,
      opacity: 0.8,
      dashArray: '3, 4',
      interactive: false,
    });
    graticuleLayer.addLayer(line);

    // Degree label along the left side
    const lbl = L.marker([lat, 97.4], {
      icon: L.divIcon({
        className: 'graticule-label',
        html: `${lat}°N`,
        iconSize: [40, 14],
      }),
      interactive: false,
    });
    graticuleLayer.addLayer(lbl);
  });

  lons.forEach((lon) => {
    // Vertical longitude line
    const line = L.polyline([[9, lon], [19, lon]], {
      color: lineColor,
      weight: 0.8,
      opacity: 0.8,
      dashArray: '3, 4',
      interactive: false,
    });
    graticuleLayer.addLayer(line);

    // Degree label along the top
    const lbl = L.marker([16.8, lon], {
      icon: L.divIcon({
        className: 'graticule-label',
        html: `${lon}°E`,
        iconSize: [40, 14],
      }),
      interactive: false,
    });
    graticuleLayer.addLayer(lbl);
  });
}

// 3. Setup UI Event Listeners
function setupEventListeners() {
  // Menu toggle button
  const menuBtn = document.getElementById('btn-rdt-menu');
  const drawer = document.getElementById('rdt-drawer');
  if (menuBtn && drawer) {
    menuBtn.addEventListener('click', () => {
      const isHidden = drawer.style.display === 'none';
      drawer.style.display = isHidden ? 'flex' : 'none';
      menuBtn.innerText = isHidden ? '∧ ย่อเมนู' : '∨ เมนู / Menu';
    });
  }

  // Dataset Switcher (Real Satellite Data <-> Simulation)
  const datasetBtn = document.getElementById('btn-dataset-switch');
  if (datasetBtn) {
    datasetBtn.addEventListener('click', async () => {
      try {
        const dsRes = await fetch('/api/datasets');
        const dsData = await dsRes.json();
        const nextId = dsData.current === 'real' ? 'simulation' : 'real';
        await fetch(`/api/datasets/${nextId}`, { method: 'POST' });
        updateDatasetUI(nextId);
        hasInitializedBounds = false;
        await loadScans();
      } catch (err) {
        console.error('Failed to switch dataset:', err);
      }
    });
  }

  // Theme Switcher (Dark Mode <-> Light RDT)
  const themeBtn = document.getElementById('toggle-theme');
  const themeLbl = document.getElementById('lbl-theme');
  const themeIcon = document.getElementById('icon-theme');
  const legendArrow = document.getElementById('legend-arrow-icon');

  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      isDarkMode = !isDarkMode;
      document.body.classList.toggle('dark-theme', isDarkMode);
      document.body.classList.toggle('light-theme', !isDarkMode);

      if (isDarkMode) {
        if (themeLbl) themeLbl.innerText = 'Dark Mode';
        if (themeIcon) themeIcon.innerText = '🌙';
        if (legendArrow) legendArrow.style.color = '#facc15';

        // Switch to Dark Canvas if currently on RDT Ocean
        if (currentBase === 'rdt_ocean') {
          map.removeLayer(baseLayers.rdt_ocean);
          map.addLayer(baseLayers.dark);
          currentBase = 'dark';
          const baseLbl = document.getElementById('lbl-basemap');
          if (baseLbl) baseLbl.innerText = 'Dark Canvas';
        }
      } else {
        if (themeLbl) themeLbl.innerText = 'RDT Light';
        if (themeIcon) themeIcon.innerText = '☀️';
        if (legendArrow) legendArrow.style.color = '#000000';

        // Switch to RDT Ocean if currently on Dark Canvas
        if (currentBase === 'dark') {
          map.removeLayer(baseLayers.dark);
          map.addLayer(baseLayers.rdt_ocean);
          currentBase = 'rdt_ocean';
          const baseLbl = document.getElementById('lbl-basemap');
          if (baseLbl) baseLbl.innerText = 'RDT Ocean';
        }
      }

      // Refresh Graticule and current scan visualization with new theme colors
      renderLatLonGraticule();
      if (currentScanData) renderScan(currentScanData);
    });
  }

  // Satellite Clouds Toggle
  const btnToggleSatClouds = document.getElementById('toggle-sat-clouds');
  if (btnToggleSatClouds) {
    btnToggleSatClouds.addEventListener('click', () => {
      showSatelliteClouds = !showSatelliteClouds;
      btnToggleSatClouds.classList.toggle('active', showSatelliteClouds);
      if (showSatelliteClouds) {
        if (satelliteCloudOverlay && !map.hasLayer(satelliteCloudOverlay)) {
          satelliteCloudOverlay.addTo(map);
          satelliteCloudOverlay.bringToBack();
        } else if (currentScanData) {
          renderSatelliteCloudLayer(currentScanData);
        }
      } else {
        if (satelliteCloudOverlay && map.hasLayer(satelliteCloudOverlay)) {
          map.removeLayer(satelliteCloudOverlay);
        }
      }
    });
  }

  // Motion Vector Arrows Toggle
  const btnToggleTraj = document.getElementById('toggle-trajectories');
  if (btnToggleTraj) {
    btnToggleTraj.addEventListener('click', () => {
      showTrajectories = !showTrajectories;
      btnToggleTraj.classList.toggle('active', showTrajectories);
      if (showTrajectories) map.addLayer(trajectoriesLayer);
      else map.removeLayer(trajectoriesLayer);
    });
  }

  // Impact Zones Toggle
  const btnToggleImpact = document.getElementById('toggle-impact-zones');
  if (btnToggleImpact) {
    btnToggleImpact.addEventListener('click', () => {
      showImpactZones = !showImpactZones;
      btnToggleImpact.classList.toggle('active', showImpactZones);
      if (showImpactZones) map.addLayer(impactZonesLayer);
      else map.removeLayer(impactZonesLayer);
    });
  }

  // Footprint Frames Toggle
  const btnToggleFootprints = document.getElementById('toggle-footprints');
  if (btnToggleFootprints) {
    btnToggleFootprints.addEventListener('click', () => {
      showFootprints = !showFootprints;
      btnToggleFootprints.classList.toggle('active', showFootprints);
      if (showFootprints) map.addLayer(footprintsLayer);
      else map.removeLayer(footprintsLayer);
    });
  }

  // Graticule Toggle
  const btnToggleGraticule = document.getElementById('toggle-graticule');
  if (btnToggleGraticule) {
    btnToggleGraticule.addEventListener('click', () => {
      showGraticule = !showGraticule;
      btnToggleGraticule.classList.toggle('active', showGraticule);
      if (showGraticule) map.addLayer(graticuleLayer);
      else map.removeLayer(graticuleLayer);
    });
  }

  // Basemap Selector
  const btnToggleBasemap = document.getElementById('toggle-basemap');
  const baseLbl = document.getElementById('lbl-basemap');
  if (btnToggleBasemap) {
    btnToggleBasemap.addEventListener('click', () => {
      if (currentBase === 'dark') {
        map.removeLayer(baseLayers.dark);
        map.addLayer(baseLayers.rdt_ocean);
        currentBase = 'rdt_ocean';
        if (baseLbl) baseLbl.innerText = 'RDT Ocean';
      } else if (currentBase === 'rdt_ocean') {
        map.removeLayer(baseLayers.rdt_ocean);
        map.addLayer(baseLayers.satellite);
        currentBase = 'satellite';
        if (baseLbl) baseLbl.innerText = 'Satellite';
      } else if (currentBase === 'satellite') {
        map.removeLayer(baseLayers.satellite);
        map.addLayer(baseLayers.osm);
        currentBase = 'osm';
        if (baseLbl) baseLbl.innerText = 'OpenStreetMap';
      } else {
        map.removeLayer(baseLayers.osm);
        map.addLayer(baseLayers.dark);
        currentBase = 'dark';
        if (baseLbl) baseLbl.innerText = 'Dark Canvas';
      }
    });
  }

  // GPS Button
  const gpsBtn = document.getElementById('btn-gps');
  if (gpsBtn) gpsBtn.addEventListener('click', requestUserGPS);

  // Check coordinates button
  const checkBtn = document.getElementById('btn-check-coords');
  if (checkBtn) {
    checkBtn.addEventListener('click', () => {
      const lat = parseFloat(document.getElementById('input-lat').value);
      const lon = parseFloat(document.getElementById('input-lon').value);
      if (!isNaN(lat) && !isNaN(lon)) {
        setUserLocation(lat, lon, true);
      } else {
        alert('กรุณากรอกพิกัด Latitude และ Longitude ให้ถูกต้อง');
      }
    });
  }

  // Preset location chips
  document.querySelectorAll('.preset-item').forEach((item) => {
    item.addEventListener('click', () => {
      const lat = parseFloat(item.dataset.lat);
      const lon = parseFloat(item.dataset.lon);
      setUserLocation(lat, lon, true);
    });
  });

  // Timeline Controls
  const playBtn = document.getElementById('btn-play');
  const prevBtn = document.getElementById('btn-prev');
  const nextBtn = document.getElementById('btn-next');
  const slider = document.getElementById('timeline-slider');

  if (playBtn) playBtn.addEventListener('click', togglePlay);
  if (prevBtn) prevBtn.addEventListener('click', () => stepScan(-1));
  if (nextBtn) nextBtn.addEventListener('click', () => stepScan(1));
  if (slider) {
    slider.addEventListener('input', (e) => {
      selectScanIndex(parseInt(e.target.value, 10));
    });
  }
}

// 4. GPS Geolocation
function requestUserGPS() {
  const statusLbl = document.getElementById('gps-status-lbl');
  if (!navigator.geolocation) {
    alert('เบราว์เซอร์ไม่รองรับ HTML5 GPS Geolocation');
    return;
  }

  if (statusLbl) statusLbl.innerText = 'กำลังดึงพิกัด GPS...';

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const lat = parseFloat(pos.coords.latitude.toFixed(4));
      const lon = parseFloat(pos.coords.longitude.toFixed(4));
      if (statusLbl) statusLbl.innerText = 'GPS พร้อมใช้งาน';
      setUserLocation(lat, lon, true);
    },
    (err) => {
      console.warn('GPS Error:', err);
      if (statusLbl) statusLbl.innerText = 'GPS ไม่ตอบสนอง (คลิกบนแผนที่)';
      alert(`ไม่สามารถดึงตำแหน่ง GPS ได้ (${err.message}) กรุณาคลิกเลือกจุดบนแผนที่หรือเลือกจากพิกัดแนะนำ`);
    },
    { enableHighAccuracy: true, timeout: 8000 }
  );
}

// 5. Set & Inspect Target Location
function setUserLocation(lat, lon, panTo = false) {
  activeUserCoord = { lat, lon };

  document.getElementById('input-lat').value = lat;
  document.getElementById('input-lon').value = lon;

  // Open drawer if closed so user sees the forecast
  const drawer = document.getElementById('rdt-drawer');
  if (drawer && drawer.style.display === 'none') {
    drawer.style.display = 'flex';
    document.getElementById('btn-rdt-menu').innerText = '∧ ย่อเมนู';
  }

  // Red Target Crosshair Marker
  if (userMarker) {
    userMarker.setLatLng([lat, lon]);
  } else {
    const crosshairIcon = L.divIcon({
      className: 'user-rdt-marker',
      html: `
        <div class="user-cross-line user-cross-h"></div>
        <div class="user-cross-line user-cross-v"></div>
        <div class="user-center-dot"></div>
      `,
      iconSize: [22, 22],
      iconAnchor: [11, 11],
    });
    userMarker = L.marker([lat, lon], { icon: crosshairIcon, zIndexOffset: 3000 }).addTo(map);
    userMarker.bindTooltip(`พิกัดเป้าหมาย: ${lat}, ${lon}`, { className: 'rdt-tooltip' });
  }

  if (panTo) {
    map.flyTo([lat, lon], Math.max(map.getZoom(), 9), { duration: 0.8 });
  }

  queryPointNowcast(lat, lon);
}

// 6. Query Point Nowcast API
async function queryPointNowcast(lat, lon) {
  const resultCard = document.getElementById('rain-result-card');
  if (!resultCard) return;

  resultCard.style.display = 'block';
  resultCard.innerHTML = `
    <div style="padding: 10px; color: var(--text-muted); font-size: 11px; text-align: center;">
      กำลังวิเคราะห์สภาพฝน ณ พิกัด (${lat}, ${lon}) ...
    </div>
  `;

  const stamp = allScans[currentScanIndex] ? allScans[currentScanIndex].stamp : '';
  const url = `/api/nowcast/point?lat=${lat}&lon=${lon}${stamp ? `&stamp=${stamp}` : ''}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('API response failed');
    const data = await res.json();
    renderPointNowcastResult(data);
  } catch (err) {
    console.error('Point Nowcast Error:', err);
    resultCard.innerHTML = `
      <div style="padding: 10px; color: #f87171; font-size: 11px;">
        เกิดข้อผิดพลาดในการคำนวณข้อมูล
      </div>
    `;
  }
}

// 7. Render Point Onset Alert Box
function renderPointNowcastResult(data) {
  const resultCard = document.getElementById('rain-result-card');
  if (!resultCard) return;

  const m = data.metrics || {};
  const alertLevel = data.alert_level || 'notice';

  if (interceptVectorLine) {
    map.removeLayer(interceptVectorLine);
    interceptVectorLine = null;
  }

  if (data.advection) {
    const adv = data.advection;
    const latlngs = [
      [adv.origin_lat, adv.origin_lon],
      [data.query_location.lat, data.query_location.lon],
    ];

    interceptVectorLine = L.polyline(latlngs, {
      color: '#ef4444',
      weight: 2.2,
      dashArray: '4, 4',
      opacity: 0.95,
    }).addTo(map);

    interceptVectorLine.bindTooltip(`ทิศทางเคลื่อนเข้าหา: อีก ~${adv.arrival_min} นาที (${adv.distance_km} กม.)`, {
      permanent: true,
      className: 'rdt-tooltip',
      direction: 'center',
    });
  }

  const p30 = m.p_within_30min != null ? Math.round(m.p_within_30min * 100) : 0;
  const p60 = m.p_within_60min != null ? Math.round(m.p_within_60min * 100) : 0;

  let etaHtml = '-';
  if (m.eta_window_th) {
    etaHtml = `${m.eta_window_th[0]} – ${m.eta_window_th[1]}`;
  } else if (m.eta_median_th) {
    etaHtml = `~${m.eta_median_th}`;
  }

  let rainRateHtml = m.rain_rate_mm_hr != null ? `${m.rain_rate_mm_hr} มม./ชม.` : 'ไม่มี';

  resultCard.innerHTML = `
    <div class="rdt-alert-box">
      <div class="alert-title-row ${alertLevel}">
        <span>${alertLevel === 'danger' ? '🚨' : alertLevel === 'warning' ? '⚠️' : alertLevel === 'safe' ? '☀️' : 'ℹ️'}</span>
        <span>${data.status_label || ''}</span>
      </div>
      <div class="alert-desc-text">${data.summary_th || ''}</div>

      <div class="rdt-stats-table">
        <div>
          <div class="rdt-stat-lbl">เวลาเริ่มตก (ETA Window)</div>
          <div class="rdt-stat-val" style="color: #facc15;">${etaHtml}</div>
        </div>
        <div>
          <div class="rdt-stat-lbl">อัตราฝนคาดการณ์</div>
          <div class="rdt-stat-val">${rainRateHtml}</div>
        </div>
        <div>
          <div class="rdt-stat-lbl">โอกาสใน 30 นาที</div>
          <div class="rdt-stat-val">${p30}%</div>
        </div>
        <div>
          <div class="rdt-stat-lbl">โอกาสใน 60 นาที</div>
          <div class="rdt-stat-val">${p60}%</div>
        </div>
      </div>

      ${m.distance_km ? `
        <div style="margin-top: 6px; font-size: 10.5px; color: var(--text-muted);">
          กลุ่มเมฆห่างออกไป <strong>${m.distance_km} กม.</strong> (${m.cloud_direction_thai || ''}) ความเร็ว <strong>${m.cloud_speed_kmh || '-'} กม./ชม.</strong>
        </div>
      ` : ''}
    </div>
  `;
}

function updateDatasetUI(datasetId) {
  const isReal = datasetId === 'real';
  const nameLbl = document.getElementById('dataset-name');
  const iconLbl = document.getElementById('dataset-icon');
  const badge = document.getElementById('dataset-badge');
  const sliderSub = document.getElementById('slider-sub-lbl');

  if (nameLbl) nameLbl.innerText = isReal ? 'ข้อมูลจริง Himawari-9' : 'ข้อมูลจำลอง (Simulation)';
  if (iconLbl) iconLbl.innerText = isReal ? '🛰️' : '🧪';
  if (badge) {
    badge.innerText = isReal ? 'REAL SATELLITE' : 'SIMULATION';
    badge.style.background = isReal ? '#0284c7' : '#22c55e';
  }
  if (sliderSub) {
    sliderSub.innerText = isReal
      ? 'HIMAWARI-8/9 AHI SATELLITE + THAILAND RADAR (28 ก.ย. 2026)'
      : 'HIMAWARI-8/9 AHI SATELLITE CONVECTION REPLAY (10-MIN RESOLUTION)';
  }
}

// 8. Load Scans List from API
async function loadScans() {
  try {
    try {
      const dsRes = await fetch('/api/datasets');
      const dsData = await dsRes.json();
      updateDatasetUI(dsData.current);
    } catch (e) {}

    const res = await fetch('/api/scans');
    const data = await res.json();
    allScans = data.scans || [];

    if (allScans.length === 0) return;

    const slider = document.getElementById('timeline-slider');
    slider.min = 0;
    slider.max = allScans.length - 1;
    slider.value = allScans.length - 1;

    document.getElementById('slider-start-lbl').innerText = formatStampTime(allScans[0].stamp);
    document.getElementById('slider-end-lbl').innerText = formatStampTime(allScans[allScans.length - 1].stamp);

    await selectScanIndex(allScans.length - 1);
    // Auto-analyze default point (กทม./สมุทรปราการ) on initial load
    setUserLocation(13.5407, 100.4041, false);
  } catch (err) {
    console.error('Failed to load scans:', err);
  }
}

// 9. Select Specific Scan
async function selectScanIndex(index) {
  if (index < 0 || index >= allScans.length) return;
  currentScanIndex = index;

  const slider = document.getElementById('timeline-slider');
  if (slider) slider.value = index;

  const scan = allScans[index];
  
  // Update Top Tab Title
  const tabTitle = document.getElementById('rdt-product-title');
  if (tabTitle) {
    tabTitle.innerText = `RDT CONVRAIN [${scan.stamp}], Ech000H pour ${formatStampTime(scan.stamp)}`;
  }

  try {
    const res = await fetch(`/api/scans/${scan.stamp}`);
    if (!res.ok) throw new Error('Scan fetch failed');
    currentScanData = await res.json();

    renderScan(currentScanData);

    if (activeUserCoord) {
      queryPointNowcast(activeUserCoord.lat, activeUserCoord.lon);
    }
  } catch (err) {
    console.error('Error loading scan details:', err);
  }
}

// 10. Master Render: Satellite Cloud Deck + Footprints + Motion Vectors
function renderScan(data) {
  const footprints = data.footprints;
  const trajectories = data.trajectories || [];
  const impactZones = data.impact_zones || null;

  // Clear previous vector layers
  footprintsLayer.clearLayers();
  impactZonesLayer.clearLayers();
  trajectoriesLayer.clearLayers();

  // 1. Render Satellite Cloud Shadows Layer (เงาเมฆดาวเทียมจางๆ แบบในรูป)
  renderSatelliteCloudLayer(data);

  // 2. Convective Cell Footprints (RDT Multi-layer Framing)
  if (footprints && footprints.features) {
    footprintsLayer.addData(footprints);
  }

  // 3. Projected Rain Landfall / Impact Zones (Dashed Red Polygons)
  if (impactZones && impactZones.features) {
    impactZonesLayer.addData(impactZones);
  }

  // 4. Sharp Motion Vector Arrows & Trailing Trajectory
  trajectories.forEach((traj) => {
    renderRdtMotionArrow(traj);
  });

  // Auto-center bounds on first load focused on Thailand
  if (!hasInitializedBounds) {
    map.setView([14.8, 101.2], 7);
    hasInitializedBounds = true;
  }

  if (map) map.invalidateSize();
}

// 11. Render Satellite Cloud Deck (เมฆทั้งหมดจากภาพถ่ายดาวเทียมเป็นเงาเทาๆจางๆ)
function renderSatelliteCloudLayer(scanData) {
  if (!showSatelliteClouds) {
    if (satelliteCloudOverlay && map.hasLayer(satelliteCloudOverlay)) {
      map.removeLayer(satelliteCloudOverlay);
    }
    return;
  }

  // Full extent bounding box covering Thailand & surrounding seas
  const minLat = 4.0, maxLat = 22.0;
  const minLon = 93.0, maxLon = 110.0;
  const bounds = [[minLat, minLon], [maxLat, maxLon]];

  // 1024x1024 High-Resolution Offscreen Canvas
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  function toPx(lat, lon) {
    return {
      x: ((lon - minLon) / (maxLon - minLon)) * canvas.width,
      y: ((maxLat - lat) / (maxLat - minLat)) * canvas.height,
    };
  }

  const isDark = isDarkMode;

  // Draw convective cloud canopies (core + spreading anvil canopy blown downwind)
  const allObjects = (scanData && scanData.doc && scanData.doc.objects) || [];
  // จัดลำดับยอดเมฆสูงและเย็นที่สุด เพื่อวาดรัศมีเงาเมฆฟุ้ง 200 เซลล์หลักอย่างรวดเร็ว
  const objects = [...allObjects].sort((a, b) => {
    const ha = (a.cloud_top && a.cloud_top.cloud_top_height_km) || 0;
    const hb = (b.cloud_top && b.cloud_top.cloud_top_height_km) || 0;
    return hb - ha;
  }).slice(0, 200);

  objects.forEach((obj) => {
    const coords = obj.geometry.coordinates; // [lon, lat]
    const center = toPx(coords[1], coords[0]);
    const motion = obj.motion || { speed_ms: 6, direction_deg: 135 };
    const cloudTop = obj.cloud_top || {};
    const topHeight = cloudTop.cloud_top_height_km || 7.0;
    const minBt = cloudTop.min_bt_10p4 || 250;

    // Density and size based on cloud height and coldness
    const coldness = Math.max(0, Math.min(1, (273 - minBt) / 60)); // 0 (warm) to 1 (very cold ~213K)
    const baseRadiusPx = 30 + (topHeight / 14) * 45; // 35 to 80 px
    const anvilRadiusPx = baseRadiusPx * (1.8 + coldness * 1.3); // 70 to 200 px anvil

    // Downwind displacement of the anvil plume
    const radDir = ((motion.direction_deg - 90) * Math.PI) / 180;
    const anvilShift = (motion.speed_ms || 5) * 3.2;
    const anvilCenterX = center.x + Math.cos(radDir) * anvilShift;
    const anvilCenterY = center.y + Math.sin(radDir) * anvilShift;

    // Radial gradient for the soft satellite cloud canopy
    const cloudGrad = ctx.createRadialGradient(
      center.x, center.y, baseRadiusPx * 0.15,
      anvilCenterX, anvilCenterY, anvilRadiusPx
    );

    if (isDark) {
      // Dark Mode: Soft luminous silver-gray satellite clouds
      const cAlpha = 0.38 + coldness * 0.22;
      cloudGrad.addColorStop(0.0, `rgba(240, 246, 255, ${cAlpha})`);
      cloudGrad.addColorStop(0.25, `rgba(220, 232, 245, ${cAlpha * 0.75})`);
      cloudGrad.addColorStop(0.55, `rgba(185, 205, 225, ${cAlpha * 0.40})`);
      cloudGrad.addColorStop(0.82, `rgba(160, 185, 210, ${cAlpha * 0.15})`);
      cloudGrad.addColorStop(1.0, 'rgba(150, 175, 200, 0.0)');
    } else {
      // Light Mode: Soft smoky slate-gray cloud shadows (matching reference photo)
      const cAlpha = 0.36 + coldness * 0.22;
      cloudGrad.addColorStop(0.0, `rgba(95, 110, 130, ${cAlpha})`);
      cloudGrad.addColorStop(0.25, `rgba(115, 130, 150, ${cAlpha * 0.72})`);
      cloudGrad.addColorStop(0.55, `rgba(140, 155, 175, ${cAlpha * 0.38})`);
      cloudGrad.addColorStop(0.82, `rgba(160, 175, 195, ${cAlpha * 0.14})`);
      cloudGrad.addColorStop(1.0, 'rgba(175, 190, 205, 0.0)');
    }

    ctx.save();
    ctx.fillStyle = cloudGrad;
    ctx.beginPath();
    ctx.arc(anvilCenterX, anvilCenterY, anvilRadiusPx, 0, Math.PI * 2);
    ctx.filter = 'blur(18px)';
    ctx.fill();
    ctx.restore();
  });

  // C. Draw footprint contours with soft feathered Gaussian blur
  // This binds the dense cloud core directly to the actual satellite footprint shape
  const allFootprints = (scanData && scanData.footprints && scanData.footprints.features) || [];
  const footprints = allFootprints.slice(0, 200);
  ctx.save();
  ctx.filter = 'blur(12px)';
  footprints.forEach((feat) => {
    if (!feat.geometry || !feat.geometry.coordinates) return;
    const rings = feat.geometry.coordinates;
    rings.forEach((ring) => {
      ctx.beginPath();
      ring.forEach((pt, idx) => {
        const px = toPx(pt[1], pt[0]);
        if (idx === 0) ctx.moveTo(px.x, px.y);
        else ctx.lineTo(px.x, px.y);
      });
      ctx.closePath();
      ctx.fillStyle = isDark ? 'rgba(235, 245, 255, 0.24)' : 'rgba(85, 100, 120, 0.26)';
      ctx.fill();
    });
  });
  ctx.restore();

  // Convert to Data URL and update or create Leaflet ImageOverlay
  const dataUrl = canvas.toDataURL('image/png');
  if (satelliteCloudOverlay) {
    satelliteCloudOverlay.setUrl(dataUrl);
    satelliteCloudOverlay.setBounds(bounds);
    if (!map.hasLayer(satelliteCloudOverlay)) {
      satelliteCloudOverlay.addTo(map);
      satelliteCloudOverlay.bringToBack();
    }
  } else {
    satelliteCloudOverlay = L.imageOverlay(dataUrl, bounds, {
      opacity: 0.88,
      zIndex: 5,
      interactive: false,
    }).addTo(map);
    satelliteCloudOverlay.bringToBack();
  }
}

// 12. Styling Footprints (RDT Convective Multi-Layer Frames)
function styleRdtFootprint(feature) {
  const oid = feature.id || (feature.properties ? feature.properties.object_id : null);
  let status = 'developing';

  if (currentScanData && currentScanData.doc && currentScanData.doc.objects) {
    const found = currentScanData.doc.objects.find((o) => o.object_id === oid);
    if (found) status = found.status;
  }

  if (status === 'raining' || status === 'peak') {
    // Mature / Raining: Bold Red outline with Cyan core fill
    return {
      color: '#ef4444',
      weight: 2.2,
      opacity: 0.95,
      fillColor: '#22d3ee',
      fillOpacity: isDarkMode ? 0.35 : 0.55,
    };
  } else if (status === 'developing') {
    // Developing: Bright Magenta/Fuchsia outline with Yellow core fill
    return {
      color: '#e879f9',
      weight: 2.2,
      opacity: 0.95,
      fillColor: '#fde047',
      fillOpacity: isDarkMode ? 0.35 : 0.50,
    };
  } else {
    // Decaying / Dissipated: Dashed Blue outline
    return {
      color: '#3b82f6',
      weight: 1.8,
      dashArray: '5, 5',
      opacity: 0.85,
      fillColor: '#93c5fd',
      fillOpacity: isDarkMode ? 0.18 : 0.25,
    };
  }
}

function onEachFootprintFeature(feature, layer) {
  const oid = feature.id || (feature.properties ? feature.properties.object_id : null);
  if (!oid) return;

  layer.on('click', () => {
    if (currentScanData && currentScanData.doc) {
      const obj = currentScanData.doc.objects.find((o) => o.object_id === oid);
      if (obj) focusObject(obj);
    }
  });

  layer.bindTooltip(`RDT Cell: ${oid}`, { sticky: true, className: 'rdt-tooltip' });
}

// 13. Styling Projected Landfall (Dashed Red Outline)
function styleRdtImpactZone(feature) {
  return {
    color: '#ef4444',
    weight: 1.8,
    dashArray: '4, 4',
    opacity: 0.9,
    fillColor: '#fca5a5',
    fillOpacity: isDarkMode ? 0.20 : 0.30,
  };
}

function onEachImpactFeature(feature, layer) {
  const p = feature.properties || {};
  const popup = `
    <div style="font-size: 11px; padding: 2px;">
      <div style="font-weight: 700; color: #ef4444; font-size: 12px; margin-bottom: 4px;">
        🎯 จุดคาดการณ์ฝนเริ่มตก (Projected Impact)
      </div>
      <div><strong>Cell ID:</strong> ${p.object_id}</div>
      <div><strong>เวลาคาดการณ์ (ETA):</strong> <span style="font-weight: 700;">${p.eta_th}</span></div>
      ${p.eta_window_th ? `<div><strong>ช่วงเวลา (Window):</strong> ${p.eta_window_th[0]} – ${p.eta_window_th[1]}</div>` : ''}
      <div><strong>ความเร็วเคลื่อนตัว:</strong> ${p.speed_kmh} กม./ชม. (ทิศ ${p.direction_deg}°)</div>
      ${p.rain_rate_mm_hr ? `<div><strong>อัตราฝนคาดการณ์:</strong> ${p.rain_rate_mm_hr} มม./ชม.</div>` : ''}
    </div>
  `;
  layer.bindPopup(popup);
}

// 14. Motion Vector Arrow (Adapts dynamically to Dark/Light Mode)
function renderRdtMotionArrow(traj) {
  const c_lon = traj.origin[0];
  const c_lat = traj.origin[1];
  const dir = traj.direction_deg;
  const speed = traj.speed_ms;

  // Vector length proportional to speed (~30 min displacement)
  const dist_m = speed * 1800;
  const d_lat = (dist_m * Math.cos((dir * Math.PI) / 180)) / 111139.0;
  const d_lon = (dist_m * Math.sin((dir * Math.PI) / 180)) / (111139.0 * Math.cos((c_lat * Math.PI) / 180));

  const end_lat = c_lat + d_lat;
  const end_lon = c_lon + d_lon;

  // Arrow shaft & head colors:
  // In Dark Mode: Luminous amber/yellow (#facc15) with black border for high-contrast tactical readability
  // In Light Mode: Sharp solid black (#000000) with white border
  const arrowShaftColor = isDarkMode ? '#facc15' : '#000000';
  const arrowFill = isDarkMode ? '#fde047' : '#000000';
  const arrowStroke = isDarkMode ? '#000000' : '#ffffff';

  // Arrow shaft
  const arrowShaft = L.polyline([[c_lat, c_lon], [end_lat, end_lon]], {
    color: arrowShaftColor,
    weight: isDarkMode ? 2.4 : 2.2,
    opacity: 1,
  });
  trajectoriesLayer.addLayer(arrowShaft);

  // Sharp arrowhead marker
  const arrowIcon = L.divIcon({
    className: 'black-arrow-icon',
    html: `
      <div style="transform: rotate(${dir}deg); transform-origin: 0px 0px;">
        <svg width="20" height="20" viewBox="0 0 20 20" style="margin-left: -5px; margin-top: -10px; overflow: visible;">
          <polygon points="0,3 17,10 0,17 4,10" fill="${arrowFill}" stroke="${arrowStroke}" stroke-width="1.2" stroke-linejoin="round"/>
        </svg>
      </div>
    `,
    iconSize: [0, 0],
  });
  const arrowHead = L.marker([end_lat, end_lon], { icon: arrowIcon });
  trajectoriesLayer.addLayer(arrowHead);

  // Trailing yellow trajectory track
  if (traj.path && traj.path.length > 2) {
    const yellowTrack = L.polyline(traj.path.map((p) => [p[1], p[0]]), {
      color: '#eab308',
      weight: 1.5,
      opacity: 0.85,
    });
    trajectoriesLayer.addLayer(yellowTrack);
  }
}

// 15. Focus Object
function focusObject(obj) {
  const coords = obj.geometry.coordinates;
  map.flyTo([coords[1], coords[0]], 10, { duration: 0.6 });
}

// 16. Timeline Controls
function togglePlay() {
  const playBtn = document.getElementById('btn-play');
  if (isPlaying) {
    clearInterval(playInterval);
    isPlaying = false;
    playBtn.innerText = '▶';
  } else {
    isPlaying = true;
    playBtn.innerText = '❚❚';
    playInterval = setInterval(() => {
      let next = currentScanIndex + 1;
      if (next >= allScans.length) next = 0;
      selectScanIndex(next);
    }, 1500);
  }
}

function stepScan(direction) {
  let next = currentScanIndex + direction;
  if (next >= 0 && next < allScans.length) {
    selectScanIndex(next);
  }
}

// Helpers
function formatStampTime(stamp) {
  if (!stamp) return '-';
  const m = stamp.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})Z/);
  if (m) {
    const utcHours = parseInt(m[4], 10);
    const mins = m[5];
    const thHours = (utcHours + 7) % 24;
    const thHStr = thHours < 10 ? '0' + thHours : thHours;
    return `${thHStr}:${mins} น. (${m[4]}:${mins}UTC)`;
  }
  return stamp;
}
