/**
 * CONVRAIN - Satellite Convective Rain Onset Nowcast Web Platform
 * Modern 2026 Redesign:
 *  - Unified Header & Telemetry
 *  - Collapsible Glassmorphism Sidebar
 *  - Floating Layer FABs & Collapsible Legend
 *  - Level-of-Detail (LOD) Vector Rendering (clean nationwide overview)
 *  - Seamless Satellite Cloud Shadows
 *  - 2x2 Metrics Grid Point Nowcast
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
let referenceLayer = null;

// Layer Visibility & Mode Toggles
let showSatelliteClouds = true;
let showTrajectories = true;
let showImpactZones = true;
let showFootprints = true;
let showGraticule = true;
let smoothMode = true; // Windy Organic Cloud Smoothing (Chaikin subdivision)
let activeSelectedObjectId = null; // Currently inspected/clicked cell

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

// =====================================================================
// 1. Initialize Map with Basemaps & Graticule
// =====================================================================
function initMap() {
  const mapElement = document.getElementById('map');
  if (!mapElement) return;

  map = L.map('map', {
    center: [14.8, 101.2],
    zoom: 7,
    zoomControl: false,
    preferCanvas: true,
  });

  // Custom Zoom Control (Top-Left)
  L.control.zoom({ position: 'topleft' }).addTo(map);

  // Dedicated Top Labels Pane for Cities & Boundaries (Windy-style top labels)
  map.createPane('labelsPane');
  map.getPane('labelsPane').style.zIndex = 650;
  map.getPane('labelsPane').style.pointerEvents = 'none';

  // Basemaps (Esri Dark Canvas Base, RDT Ocean, Satellite, OSM)
  baseLayers.dark = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    {
      className: 'dark-tiles',
      maxZoom: 16,
      attribution: '&copy; Esri, HERE, Garmin',
    }
  );

  referenceLayer = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
    {
      pane: 'labelsPane',
      maxZoom: 16,
      attribution: '',
    }
  );

  baseLayers.rdt_ocean = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    {
      className: 'rdt-ocean-tiles',
      maxZoom: 16,
      attribution: '&copy; Esri &copy; GEBCO, NOAA',
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

  // Default: Dark Mode Basemap + Top Reference Labels
  baseLayers.dark.addTo(map);
  referenceLayer.addTo(map);

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
    activeSelectedObjectId = null;
    setUserLocation(lat, lon, true);
    if (currentScanData) renderVectorLayersLOD(currentScanData);
  });

  // Level-of-Detail (LOD) update on zoom
  map.on('zoomend', () => {
    if (currentScanData) {
      renderVectorLayersLOD(currentScanData);
    }
  });

  setTimeout(() => { if (map) map.invalidateSize(); }, 200);
  setTimeout(() => { if (map) map.invalidateSize(); }, 600);
  window.addEventListener('resize', () => { if (map) map.invalidateSize(); });
}

// =====================================================================
// 2. Render Lat/Lon Graticule Grid
// =====================================================================
function renderLatLonGraticule() {
  graticuleLayer.clearLayers();

  const lats = [10, 12, 14, 16, 18, 20];
  const lons = [96, 98, 100, 102, 104, 106];
  const lineColor = isDarkMode ? 'rgba(255, 255, 255, 0.10)' : 'rgba(0, 0, 0, 0.16)';

  lats.forEach((lat) => {
    const line = L.polyline([[lat, 94], [lat, 108]], {
      color: lineColor,
      weight: 0.8,
      opacity: 0.8,
      dashArray: '3, 4',
      interactive: false,
    });
    graticuleLayer.addLayer(line);

    const lbl = L.marker([lat, 96.2], {
      icon: L.divIcon({
        className: 'graticule-label',
        html: `${lat}°N`,
        iconSize: [35, 14],
      }),
      interactive: false,
    });
    graticuleLayer.addLayer(lbl);
  });

  lons.forEach((lon) => {
    const line = L.polyline([[8, lon], [21, lon]], {
      color: lineColor,
      weight: 0.8,
      opacity: 0.8,
      dashArray: '3, 4',
      interactive: false,
    });
    graticuleLayer.addLayer(line);

    const lbl = L.marker([18.5, lon], {
      icon: L.divIcon({
        className: 'graticule-label',
        html: `${lon}°E`,
        iconSize: [35, 14],
      }),
      interactive: false,
    });
    graticuleLayer.addLayer(lbl);
  });
}

// =====================================================================
// 3. Setup UI Event Listeners
// =====================================================================
function setupEventListeners() {
  // Sidebar Toggle Buttons
  const sidebar = document.getElementById('app-sidebar');
  const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
  const btnCloseSidebar = document.getElementById('btn-close-sidebar');

  if (btnToggleSidebar && sidebar) {
    btnToggleSidebar.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      setTimeout(() => { if (map) map.invalidateSize(); }, 320);
    });
  }

  if (btnCloseSidebar && sidebar) {
    btnCloseSidebar.addEventListener('click', () => {
      sidebar.classList.add('collapsed');
      setTimeout(() => { if (map) map.invalidateSize(); }, 320);
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

        if (currentBase === 'dark') {
          map.removeLayer(baseLayers.dark);
          map.addLayer(baseLayers.rdt_ocean);
          currentBase = 'rdt_ocean';
          const baseLbl = document.getElementById('lbl-basemap');
          if (baseLbl) baseLbl.innerText = 'RDT Ocean';
        }
      }

      renderLatLonGraticule();
      if (currentScanData) renderScan(currentScanData);
    });
  }

  // Smooth Organic Clouds Toggle (Windy Style vs Raw Contours)
  const btnToggleSmooth = document.getElementById('toggle-smooth-clouds');
  if (btnToggleSmooth) {
    btnToggleSmooth.addEventListener('click', () => {
      smoothMode = !smoothMode;
      btnToggleSmooth.classList.toggle('active', smoothMode);
      const lbl = btnToggleSmooth.querySelector('.layer-label');
      if (lbl) lbl.innerText = smoothMode ? 'เมฆธรรมชาติ (Smooth)' : 'กรอบวิเคราะห์ (Raw)';
      if (currentScanData) {
        renderScan(currentScanData);
      }
    });
  }

  // Satellite Clouds FAB Toggle
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

  // Cloud Opacity Slider (Windy Style)
  const sliderCloudOpacity = document.getElementById('slider-cloud-opacity');
  const lblCloudOpacity = document.getElementById('lbl-cloud-opacity');
  if (sliderCloudOpacity) {
    sliderCloudOpacity.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      if (lblCloudOpacity) lblCloudOpacity.innerText = `${val}%`;
      if (satelliteCloudOverlay) {
        satelliteCloudOverlay.setOpacity(val / 100.0);
      }
    });
  }

  // Motion Vector Arrows FAB Toggle
  const btnToggleTraj = document.getElementById('toggle-trajectories');
  if (btnToggleTraj) {
    btnToggleTraj.addEventListener('click', () => {
      showTrajectories = !showTrajectories;
      btnToggleTraj.classList.toggle('active', showTrajectories);
      if (showTrajectories) map.addLayer(trajectoriesLayer);
      else map.removeLayer(trajectoriesLayer);
    });
  }

  // Impact Zones FAB Toggle
  const btnToggleImpact = document.getElementById('toggle-impact-zones');
  if (btnToggleImpact) {
    btnToggleImpact.addEventListener('click', () => {
      showImpactZones = !showImpactZones;
      btnToggleImpact.classList.toggle('active', showImpactZones);
      if (showImpactZones) map.addLayer(impactZonesLayer);
      else map.removeLayer(impactZonesLayer);
    });
  }

  // Footprint Frames FAB Toggle
  const btnToggleFootprints = document.getElementById('toggle-footprints');
  if (btnToggleFootprints) {
    btnToggleFootprints.addEventListener('click', () => {
      showFootprints = !showFootprints;
      btnToggleFootprints.classList.toggle('active', showFootprints);
      if (showFootprints) map.addLayer(footprintsLayer);
      else map.removeLayer(footprintsLayer);
    });
  }

  // Graticule FAB Toggle
  const btnToggleGraticule = document.getElementById('toggle-graticule');
  if (btnToggleGraticule) {
    btnToggleGraticule.addEventListener('click', () => {
      showGraticule = !showGraticule;
      btnToggleGraticule.classList.toggle('active', showGraticule);
      if (showGraticule) map.addLayer(graticuleLayer);
      else map.removeLayer(graticuleLayer);
    });
  }

  // Basemap Selector Button
  const btnToggleBasemap = document.getElementById('toggle-basemap');
  const baseLbl = document.getElementById('lbl-basemap');
  if (btnToggleBasemap) {
    btnToggleBasemap.addEventListener('click', () => {
      if (currentBase === 'dark') {
        map.removeLayer(baseLayers.dark);
        if (referenceLayer && map.hasLayer(referenceLayer)) map.removeLayer(referenceLayer);
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
        if (referenceLayer && !map.hasLayer(referenceLayer)) map.addLayer(referenceLayer);
        currentBase = 'dark';
        if (baseLbl) baseLbl.innerText = 'Dark Canvas';
      }
    });
  }

  // Collapsible Legend Toggle
  const btnToggleLegend = document.getElementById('btn-toggle-legend');
  const legendPanel = document.getElementById('legend-panel');
  const iconLegendCollapse = document.getElementById('icon-legend-collapse');
  if (btnToggleLegend && legendPanel) {
    btnToggleLegend.addEventListener('click', () => {
      legendPanel.classList.toggle('collapsed');
      if (iconLegendCollapse) {
        iconLegendCollapse.innerText = legendPanel.classList.contains('collapsed') ? '+' : '−';
      }
    });
  }

  // GPS Geolocation Button
  const btnGps = document.getElementById('btn-gps');
  if (btnGps) {
    btnGps.addEventListener('click', requestUserGPS);
  }

  // Coordinates Search Button
  const btnCheckCoords = document.getElementById('btn-check-coords');
  if (btnCheckCoords) {
    btnCheckCoords.addEventListener('click', () => {
      const lat = parseFloat(document.getElementById('input-lat').value);
      const lon = parseFloat(document.getElementById('input-lon').value);
      if (!isNaN(lat) && !isNaN(lon)) {
        setUserLocation(lat, lon, true);
      }
    });
  }

  // Preset Location Chips
  document.querySelectorAll('.preset-chip, .preset-item').forEach((item) => {
    item.addEventListener('click', () => {
      const lat = parseFloat(item.dataset.lat);
      const lon = parseFloat(item.dataset.lon);
      setUserLocation(lat, lon, true);
    });
  });

  // Timeline Playback Controls
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

// =====================================================================
// 4. GPS Geolocation
// =====================================================================
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

// =====================================================================
// 5. Set & Inspect Target Location
// =====================================================================
function setUserLocation(lat, lon, panTo = false) {
  activeUserCoord = { lat, lon };

  const inputLat = document.getElementById('input-lat');
  const inputLon = document.getElementById('input-lon');
  if (inputLat) inputLat.value = lat;
  if (inputLon) inputLon.value = lon;

  // Open sidebar if collapsed so user sees the forecast card
  const sidebar = document.getElementById('app-sidebar');
  if (sidebar && sidebar.classList.contains('collapsed')) {
    sidebar.classList.remove('collapsed');
    setTimeout(() => { if (map) map.invalidateSize(); }, 320);
  }

  // Crosshair Marker
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
    map.flyTo([lat, lon], Math.max(map.getZoom(), 8), { duration: 0.6 });
  }

  queryPointNowcast(lat, lon);
}

// =====================================================================
// 6. Query Point Nowcast API
// =====================================================================
async function queryPointNowcast(lat, lon) {
  const resultCard = document.getElementById('rain-result-card');
  if (!resultCard) return;

  resultCard.style.display = 'block';
  resultCard.innerHTML = `
    <div style="padding: 16px; color: var(--text-muted); font-size: 11px; text-align: center;">
      กำลังวิเคราะห์สภาพฝน ณ พิกัด (${lat}, ${lon}) ...
    </div>
  `;

  const stamp = allScans[currentScanIndex] ? allScans[currentScanIndex].stamp : '';
  const url = `/api/nowcast/point?lat=${lat}&lon=${lon}${stamp ? `&stamp=${stamp}` : ''}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('API response failed');
    const data = await res.json();
    if (data.target_object && data.target_object.object_id) {
      activeSelectedObjectId = data.target_object.object_id;
      if (currentScanData) renderVectorLayersLOD(currentScanData);
    }
    renderPointNowcastResult(data);
  } catch (err) {
    console.error('Point Nowcast Error:', err);
    resultCard.innerHTML = `
      <div style="padding: 12px; color: #f87171; font-size: 11px; background: rgba(239,68,68,0.1); border-radius: 8px;">
        เกิดข้อผิดพลาดในการคำนวณข้อมูลพิกัด
      </div>
    `;
  }
}

// =====================================================================
// 7. Render Point Onset Alert Box (Modern 2x2 Grid)
// =====================================================================
function renderPointNowcastResult(data) {
  const resultCard = document.getElementById('rain-result-card');
  if (!resultCard) return;

  const m = data.metrics || {};
  const alertLevel = data.alert_level || 'notice';

  // Intercept line
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

  let rainRateHtml = m.rain_rate_mm_hr != null ? `${m.rain_rate_mm_hr} มม./ชม.` : '0 มม./ชม.';

  resultCard.innerHTML = `
    <div class="rdt-alert-box">
      <div class="alert-title-row ${alertLevel}">
        <span>${alertLevel === 'danger' ? '🚨' : alertLevel === 'warning' ? '⚠️' : alertLevel === 'safe' ? '☀️' : 'ℹ️'}</span>
        <span>${data.status_label || ''}</span>
      </div>
      <div class="alert-desc-text">${data.summary_th || ''}</div>

      <div class="metrics-grid-2x2">
        <div class="metric-card">
          <div class="metric-card-lbl">เวลาเริ่มตก (ETA Window)</div>
          <div class="metric-card-val highlight-yellow">${etaHtml}</div>
        </div>
        <div class="metric-card">
          <div class="metric-card-lbl">อัตราฝนคาดการณ์</div>
          <div class="metric-card-val highlight-cyan">${rainRateHtml}</div>
        </div>
        <div class="metric-card">
          <div class="metric-card-lbl">โอกาสใน 30 นาที</div>
          <div class="metric-card-val">${p30}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-card-lbl">โอกาสใน 60 นาที</div>
          <div class="metric-card-val">${p60}%</div>
        </div>
      </div>

      ${m.distance_km ? `
        <div class="cell-distance-note">
          กลุ่มเมฆห่างออกไป <strong>${m.distance_km} กม.</strong> (${m.cloud_direction_thai || ''}) ความเร็ว <strong>${m.cloud_speed_kmh || '-'} กม./ชม.</strong>
        </div>
      ` : ''}
    </div>
  `;
}

// =====================================================================
// 8. Update Dataset UI
// =====================================================================
function updateDatasetUI(datasetId) {
  const isReal = datasetId === 'real';
  const nameLbl = document.getElementById('dataset-name');
  const iconLbl = document.getElementById('dataset-icon');
  const badge = document.getElementById('dataset-badge');
  const sliderSub = document.getElementById('slider-sub-lbl');
  const topbarSource = document.getElementById('topbar-source');

  if (nameLbl) nameLbl.innerText = isReal ? 'ข้อมูลจริง Himawari-9' : 'ข้อมูลจำลอง (Simulation)';
  if (iconLbl) iconLbl.innerText = isReal ? '🛰️' : '🧪';
  if (badge) {
    badge.innerText = isReal ? 'LIVE' : 'SIM';
  }
  if (topbarSource) {
    topbarSource.innerText = isReal ? 'Himawari-9 AHI + Radar' : 'Simulation Replay';
  }
  if (sliderSub) {
    sliderSub.innerText = isReal
      ? 'HIMAWARI-8/9 AHI SATELLITE + THAILAND RADAR (28 ก.ย. 2026)'
      : 'HIMAWARI-8/9 AHI SATELLITE CONVECTION REPLAY (10-MIN RESOLUTION)';
  }
}

// =====================================================================
// 9. Load Scans List from API
// =====================================================================
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

    document.getElementById('slider-start-lbl').innerText = `${allScans[0].stamp.slice(9, 11)}:${allScans[0].stamp.slice(11, 13)} UTC (${formatStampTime(allScans[0].stamp)})`;
    document.getElementById('slider-end-lbl').innerText = `${allScans[allScans.length - 1].stamp.slice(9, 11)}:${allScans[allScans.length - 1].stamp.slice(11, 13)} UTC (${formatStampTime(allScans[allScans.length - 1].stamp)})`;

    await selectScanIndex(allScans.length - 1);
    setUserLocation(13.5407, 100.4041, false);
  } catch (err) {
    console.error('Failed to load scans:', err);
  }
}

// =====================================================================
// 10. Select Specific Scan
// =====================================================================
async function selectScanIndex(index) {
  if (index < 0 || index >= allScans.length) return;
  currentScanIndex = index;

  const slider = document.getElementById('timeline-slider');
  if (slider) slider.value = index;

  const scan = allScans[index];
  
  // Update Topbar Time
  const topbarTime = document.getElementById('topbar-time');
  if (topbarTime) {
    topbarTime.innerText = `${formatStampTime(scan.stamp)} (${scan.stamp.slice(9, 11)}:${scan.stamp.slice(11, 13)} UTC)`;
  }

  try {
    const res = await fetch(`/api/scans/${scan.stamp}`);
    if (!res.ok) throw new Error('Scan fetch failed');
    currentScanData = await res.json();

    // Update Topbar Cells Telemetry
    const topbarCells = document.getElementById('topbar-cells');
    if (topbarCells && currentScanData.summary) {
      const s = currentScanData.summary;
      topbarCells.innerText = `${s.total_objects.toLocaleString()} เซลล์ (ฝนตก ${s.raining_count}, ก่อตัว ${s.developing_count})`;
    }

    renderScan(currentScanData);

    if (activeUserCoord) {
      queryPointNowcast(activeUserCoord.lat, activeUserCoord.lon);
    }
  } catch (err) {
    console.error('Error loading scan details:', err);
  }
}

// =====================================================================
// 11. Master Render: Satellite Cloud Deck + LOD Vectors
// =====================================================================
function renderScan(data) {
  // 1. Satellite Cloud Shadows (Offscreen Canvas)
  renderSatelliteCloudLayer(data);

  // 2. Convective Footprints + Impact Zones + Motion Vectors with LOD
  renderVectorLayersLOD(data);

  // Auto-center bounds once on initial load
  if (!hasInitializedBounds) {
    map.setView([14.8, 101.2], 7);
    hasInitializedBounds = true;
  }

  if (map) map.invalidateSize();
}

// =====================================================================
// Chaikin's Corner Cutting Algorithm for Natural Organic Cloud Contours
// =====================================================================
function chaikinSmoothRing(ring, iterations = 2) {
  if (!ring || ring.length < 3) return ring;
  let current = ring;
  const isClosed = (
    Math.abs(current[0][0] - current[current.length - 1][0]) < 1e-6 &&
    Math.abs(current[0][1] - current[current.length - 1][1]) < 1e-6
  );
  let pts = isClosed ? current.slice(0, -1) : current.slice();
  if (pts.length < 3) return ring;

  for (let it = 0; it < iterations; it++) {
    const smoothed = [];
    const len = pts.length;
    for (let i = 0; i < len; i++) {
      const p0 = pts[i];
      const p1 = pts[(i + 1) % len];
      smoothed.push([
        0.75 * p0[0] + 0.25 * p1[0],
        0.75 * p0[1] + 0.25 * p1[1],
      ]);
      smoothed.push([
        0.25 * p0[0] + 0.75 * p1[0],
        0.25 * p0[1] + 0.75 * p1[1],
      ]);
    }
    pts = smoothed;
  }
  if (isClosed) {
    pts.push([pts[0][0], pts[0][1]]);
  }
  return pts;
}

function smoothGeometry(geom, iterations = 2) {
  if (!geom || !geom.coordinates) return geom;
  if (geom.type === 'Polygon') {
    return {
      type: 'Polygon',
      coordinates: geom.coordinates.map((ring) => chaikinSmoothRing(ring, iterations)),
    };
  } else if (geom.type === 'MultiPolygon') {
    return {
      type: 'MultiPolygon',
      coordinates: geom.coordinates.map((poly) =>
        poly.map((ring) => chaikinSmoothRing(ring, iterations))
      ),
    };
  }
  return geom;
}

function smoothFeatureCollection(fc, iterations = 2) {
  if (!fc || !fc.features) return fc;
  return {
    ...fc,
    features: fc.features.map((feat) => {
      if (!feat.geometry) return feat;
      return {
        ...feat,
        geometry: smoothGeometry(feat.geometry, iterations),
      };
    }),
  };
}

// =====================================================================
// 12. Level-of-Detail (LOD) Vector Rendering (Clean, Organic, Non-Cluttered)
// =====================================================================
function renderVectorLayersLOD(data) {
  if (!data) return;

  footprintsLayer.clearLayers();
  impactZonesLayer.clearLayers();
  trajectoriesLayer.clearLayers();

  const z = map ? map.getZoom() : 7;
  let maxCells = 30; // Clean national overview at zoom <= 7
  if (z >= 9) {
    maxCells = 160; // Clean city/local zoom without lagging or drowning the map
  } else if (z >= 8) {
    maxCells = 65; // Regional overview
  }

  const allTrajectories = data.trajectories || [];
  const trajectories = allTrajectories.slice(0, maxCells);
  const activeIds = new Set(trajectories.map((t) => t.object_id));

  // 1. Footprints (Convective Rain Cells)
  if (data.footprints && data.footprints.features) {
    let filteredFootprints = data.footprints.features;
    if (maxCells < 9999) {
      filteredFootprints = data.footprints.features.filter((f) => {
        const oid = f.id || (f.properties && f.properties.object_id);
        return activeIds.has(oid);
      });
      if (filteredFootprints.length < 15) {
        filteredFootprints = data.footprints.features.slice(0, maxCells);
      }
    }

    const processedFootprints = smoothMode
      ? smoothFeatureCollection({ type: 'FeatureCollection', features: filteredFootprints }, 2)
      : { type: 'FeatureCollection', features: filteredFootprints };

    footprintsLayer.addData(processedFootprints);
  }

  // 2. Projected Landfall / Impact Zones (Clean, Focused, Not Tangled)
  if (data.impact_zones && data.impact_zones.features) {
    let allImpacts = data.impact_zones.features;
    let filteredImpacts = [];

    // Always show impact zone for the active selected cell
    if (activeSelectedObjectId) {
      const targetImpact = allImpacts.find(
        (f) => (f.properties && f.properties.object_id) === activeSelectedObjectId
      );
      if (targetImpact) filteredImpacts.push(targetImpact);
    }

    // In smooth mode: only show impact zones for high-threat oncoming cells (max 6 total)
    if (smoothMode) {
      const highThreatImpacts = allImpacts.filter((f) => {
        const p = f.properties || {};
        const p30 = p.p_within_30min || 0;
        const rr = p.rain_rate_mm_hr || 0;
        return (p30 >= 0.70 || rr >= 8.0) && activeIds.has(p.object_id);
      }).slice(0, 6);

      highThreatImpacts.forEach((imp) => {
        if (!filteredImpacts.some((existing) => existing.properties.object_id === imp.properties.object_id)) {
          filteredImpacts.push(imp);
        }
      });
    } else {
      // Raw view: show up to 25
      filteredImpacts = allImpacts.filter((f) => activeIds.has(f.properties && f.properties.object_id)).slice(0, 25);
    }

    const processedImpacts = smoothMode
      ? smoothFeatureCollection({ type: 'FeatureCollection', features: filteredImpacts }, 2)
      : { type: 'FeatureCollection', features: filteredImpacts };

    impactZonesLayer.addData(processedImpacts);
  }

  // 3. Motion Vector Arrows
  trajectories.forEach((traj) => {
    renderRdtMotionArrow(traj);
  });
}

// =====================================================================
// 13. Render Satellite Cloud Deck (Seamless Organic Translucent Shadows)
// =====================================================================
function renderSatelliteCloudLayer(scanData) {
  if (!showSatelliteClouds) {
    if (satelliteCloudOverlay && map.hasLayer(satelliteCloudOverlay)) {
      map.removeLayer(satelliteCloudOverlay);
    }
    return;
  }

  const minLat = 4.0, maxLat = 22.0;
  const minLon = 93.0, maxLon = 110.0;
  const bounds = [[minLat, minLon], [maxLat, maxLon]];

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
  const allObjects = (scanData && scanData.doc && scanData.doc.objects) || [];
  
  // Sort by highest & coldest cloud tops
  const objects = [...allObjects].sort((a, b) => {
    const ha = (a.cloud_top && a.cloud_top.cloud_top_height_km) || 0;
    const hb = (b.cloud_top && b.cloud_top.cloud_top_height_km) || 0;
    return hb - ha;
  }).slice(0, 140);

  // 1. Draw Organic Footprint Envelopes (The actual cloud body from smoothed polygons)
  const allFootprints = (scanData && scanData.footprints && scanData.footprints.features) || [];
  const footprintMap = new Map();
  allFootprints.forEach((f) => {
    const oid = f.id || (f.properties && f.properties.object_id);
    if (oid) footprintMap.set(oid, f);
  });

  ctx.save();
  ctx.filter = 'blur(14px)';
  objects.forEach((obj) => {
    const oid = obj.object_id;
    const feat = footprintMap.get(oid);
    const cloudTop = obj.cloud_top || {};
    const minBt = cloudTop.min_bt_10p4 || 250;
    const coldness = Math.max(0, Math.min(1, (273 - minBt) / 60)); // 0 to 1

    if (feat && feat.geometry && feat.geometry.coordinates) {
      const rings = feat.geometry.coordinates;
      rings.forEach((origRing) => {
        const ring = smoothMode ? chaikinSmoothRing(origRing, 2) : origRing;
        ctx.beginPath();
        ring.forEach((pt, idx) => {
          const px = toPx(pt[1], pt[0]);
          if (idx === 0) ctx.moveTo(px.x, px.y);
          else ctx.lineTo(px.x, px.y);
        });
        ctx.closePath();
        const alpha = isDark ? (0.07 + coldness * 0.11) : (0.06 + coldness * 0.09);
        ctx.fillStyle = isDark
          ? `rgba(230, 242, 255, ${alpha})`
          : `rgba(90, 110, 135, ${alpha})`;
        ctx.fill();
      });
    }
  });
  ctx.restore();

  // 2. Anvil Cirrus Outflow & Diffuse Cloud Halos (Wind-stretched organic plumes)
  ctx.save();
  ctx.filter = 'blur(20px)';
  objects.slice(0, 75).forEach((obj) => {
    const coords = obj.geometry.coordinates;
    const center = toPx(coords[1], coords[0]);
    const motion = obj.motion || { speed_ms: 6, direction_deg: 135 };
    const cloudTop = obj.cloud_top || {};
    const topHeight = cloudTop.cloud_top_height_km || 7.0;
    const minBt = cloudTop.min_bt_10p4 || 250;
    const coldness = Math.max(0, Math.min(1, (273 - minBt) / 60));

    const baseRadiusPx = 16 + (topHeight / 14) * 26;
    const anvilRadiusPx = baseRadiusPx * (1.3 + coldness * 1.0);

    const radDir = ((motion.direction_deg - 90) * Math.PI) / 180;
    const anvilShift = (motion.speed_ms || 5) * 2.0;
    const anvilCenterX = center.x + Math.cos(radDir) * anvilShift;
    const anvilCenterY = center.y + Math.sin(radDir) * anvilShift;

    const cloudGrad = ctx.createRadialGradient(
      center.x, center.y, baseRadiusPx * 0.2,
      anvilCenterX, anvilCenterY, anvilRadiusPx
    );

    const cAlpha = isDark ? (0.09 + coldness * 0.07) : (0.08 + coldness * 0.06);
    if (isDark) {
      cloudGrad.addColorStop(0.0, `rgba(240, 248, 255, ${cAlpha})`);
      cloudGrad.addColorStop(0.40, `rgba(210, 228, 248, ${cAlpha * 0.65})`);
      cloudGrad.addColorStop(0.75, `rgba(170, 195, 225, ${cAlpha * 0.22})`);
      cloudGrad.addColorStop(1.0, 'rgba(150, 180, 210, 0.0)');
    } else {
      cloudGrad.addColorStop(0.0, `rgba(80, 100, 125, ${cAlpha})`);
      cloudGrad.addColorStop(0.40, `rgba(105, 125, 150, ${cAlpha * 0.65})`);
      cloudGrad.addColorStop(0.75, `rgba(140, 160, 185, ${cAlpha * 0.22})`);
      cloudGrad.addColorStop(1.0, 'rgba(165, 185, 205, 0.0)');
    }

    ctx.fillStyle = cloudGrad;
    ctx.beginPath();
    ctx.arc(anvilCenterX, anvilCenterY, anvilRadiusPx, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();

  // 3. Vignette Edge Feathering (Guarantees zero sharp rectangular boundaries at canvas edge)
  ctx.save();
  ctx.globalCompositeOperation = 'destination-in';
  const edgeGradient = ctx.createRadialGradient(
    canvas.width / 2, canvas.height / 2, canvas.width * 0.36,
    canvas.width / 2, canvas.height / 2, canvas.width * 0.50
  );
  edgeGradient.addColorStop(0.0, 'rgba(0, 0, 0, 1.0)');
  edgeGradient.addColorStop(0.75, 'rgba(0, 0, 0, 0.85)');
  edgeGradient.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
  ctx.fillStyle = edgeGradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();

  const dataUrl = canvas.toDataURL('image/png');
  const opacity = 0.30;
  if (satelliteCloudOverlay) {
    satelliteCloudOverlay.setUrl(dataUrl);
    satelliteCloudOverlay.setBounds(bounds);
    satelliteCloudOverlay.setOpacity(opacity);
    if (!map.hasLayer(satelliteCloudOverlay)) {
      satelliteCloudOverlay.addTo(map);
      satelliteCloudOverlay.bringToBack();
    }
  } else {
    satelliteCloudOverlay = L.imageOverlay(dataUrl, bounds, {
      opacity: opacity,
      zIndex: 5,
      interactive: false,
    }).addTo(map);
    satelliteCloudOverlay.bringToBack();
  }
}

// =====================================================================
// 14. Styling Footprints (Windy-Style Smooth Radar Contours)
// =====================================================================
function styleRdtFootprint(feature) {
  const oid = feature.id || (feature.properties ? feature.properties.object_id : null);
  let status = 'developing';
  let rainRate = null;

  if (currentScanData && currentScanData.doc && currentScanData.doc.objects) {
    const found = currentScanData.doc.objects.find((o) => o.object_id === oid);
    if (found) {
      status = found.status;
      if (found.intensity) rainRate = found.intensity.rain_rate_mm_hr;
    }
  }

  const isSelected = activeSelectedObjectId === oid;

  if (status === 'raining' || status === 'peak') {
    let strokeColor = '#ef4444';
    let fillColor = '#06b6d4';
    if (rainRate != null) {
      if (rainRate >= 15.0) { strokeColor = '#ec4899'; fillColor = '#db2777'; }
      else if (rainRate >= 5.0) { strokeColor = '#f97316'; fillColor = '#ea580c'; }
      else if (rainRate >= 2.0) { strokeColor = '#eab308'; fillColor = '#06b6d4'; }
    }

    return {
      color: isSelected ? '#38bdf8' : strokeColor,
      weight: isSelected ? 2.6 : (smoothMode ? 1.3 : 1.6),
      opacity: isSelected ? 1.0 : (smoothMode ? 0.85 : 0.92),
      fillColor: fillColor,
      fillOpacity: isSelected ? 0.28 : (isDarkMode ? 0.12 : 0.15),
      lineJoin: 'round',
      lineCap: 'round',
      className: isSelected ? 'rdt-footprint-selected' : 'rdt-footprint-smooth',
    };
  } else if (status === 'developing') {
    return {
      color: isSelected ? '#38bdf8' : '#f59e0b',
      weight: isSelected ? 2.4 : (smoothMode ? 1.1 : 1.4),
      opacity: isSelected ? 1.0 : (smoothMode ? 0.75 : 0.88),
      fillColor: '#facc15',
      fillOpacity: isSelected ? 0.20 : (isDarkMode ? 0.05 : 0.08),
      lineJoin: 'round',
      lineCap: 'round',
      className: isSelected ? 'rdt-footprint-selected' : 'rdt-footprint-smooth',
    };
  } else {
    return {
      color: '#60a5fa',
      weight: 1.0,
      dashArray: '3, 4',
      opacity: 0.50,
      fillColor: '#93c5fd',
      fillOpacity: 0.02,
      lineJoin: 'round',
      lineCap: 'round',
    };
  }
}

function onEachFootprintFeature(feature, layer) {
  const p = feature.properties || {};
  const oid = feature.id || p.object_id;

  layer.on('click', (e) => {
    L.DomEvent.stopPropagation(e);
    activeSelectedObjectId = oid;
    let obj = null;
    if (currentScanData && currentScanData.doc && currentScanData.doc.objects) {
      obj = currentScanData.doc.objects.find((o) => o.object_id === oid);
    }
    if (obj && obj.geometry && obj.geometry.coordinates) {
      setUserLocation(obj.geometry.coordinates[1], obj.geometry.coordinates[0], false);
    }
    if (currentScanData) renderVectorLayersLOD(currentScanData);
  });

  let obj = null;
  if (currentScanData && currentScanData.doc && currentScanData.doc.objects) {
    obj = currentScanData.doc.objects.find((o) => o.object_id === oid);
  }

  const status = obj ? obj.status : (p.status || 'unknown');
  const statusLabel =
    status === 'raining' || status === 'peak' ? 'ฝนตกแล้ว (Mature Rain)'
    : status === 'developing' ? 'กำลังพัฒนาตัว (Developing)'
    : 'กำลังสลายตัว (Decaying)';

  const rainRate = (obj && obj.intensity && obj.intensity.rain_rate_mm_hr != null)
    ? `${obj.intensity.rain_rate_mm_hr} มม./ชม.` : '-';
  const p30 = (obj && obj.onset && obj.onset.p_within_30min != null)
    ? `${Math.round(obj.onset.p_within_30min * 100)}%` : '-';
  const topH = (obj && obj.cloud_top && obj.cloud_top.cloud_top_height_km)
    ? `${obj.cloud_top.cloud_top_height_km} กม.` : '-';
  const speed = (obj && obj.motion && obj.motion.speed_ms)
    ? `${(obj.motion.speed_ms * 3.6).toFixed(1)} กม./ชม.` : '-';

  layer.bindPopup(`
    <div style="font-size: 11px; line-height: 1.6; min-width: 170px;">
      <div style="font-weight: 800; border-bottom: 1px solid var(--border-prominent); padding-bottom: 3px; margin-bottom: 5px; color: var(--accent-cyan);">
        ${oid}
      </div>
      <div><strong>สถานะ:</strong> ${statusLabel}</div>
      <div><strong>ความสูงยอดเมฆ:</strong> ${topH}</div>
      <div><strong>ความเร็วลมเคลื่อนตัว:</strong> ${speed}</div>
      <div><strong>อัตราฝน:</strong> ${rainRate}</div>
      <div><strong>โอกาสตก 30 นาที:</strong> ${p30}</div>
    </div>
  `);
}

// =====================================================================
// 15. Styling Impact Zones (Projected Rain Landfall)
// =====================================================================
function styleRdtImpactZone(feature) {
  const p = feature.properties || {};
  const isSelected = activeSelectedObjectId === p.object_id;

  return {
    color: isSelected ? '#ff4757' : '#f87171',
    weight: isSelected ? 2.0 : 1.2,
    dashArray: isSelected ? '5, 3' : '4, 4',
    opacity: isSelected ? 0.95 : 0.65,
    fillColor: '#fca5a5',
    fillOpacity: isSelected ? 0.14 : 0.04,
    lineJoin: 'round',
    lineCap: 'round',
  };
}

function onEachImpactFeature(feature, layer) {
  const p = feature.properties || {};
  layer.bindPopup(`
    <div style="font-size: 11px; line-height: 1.5;">
      <div style="font-weight: 800; color: #f87171; border-bottom: 1px solid var(--border-prominent); padding-bottom: 3px; margin-bottom: 4px;">
        🎯 พื้นที่คาดการณ์ฝนตก (~${p.project_min || 30} นาทีข้างหน้า)
      </div>
      <div><strong>เซลล์ต้นทาง:</strong> ${p.object_id}</div>
      <div><strong>เวลาคาดการณ์ (ETA):</strong> ${p.eta_th || '-'}</div>
      <div><strong>ความเร็วเคลื่อนที่:</strong> ${p.speed_kmh || '-'} กม./ชม.</div>
    </div>
  `);
}

// =====================================================================
// 16. Motion Vector Arrow
// =====================================================================
function renderRdtMotionArrow(traj) {
  const c_lon = traj.origin[0];
  const c_lat = traj.origin[1];
  const dir = traj.direction_deg;
  const speed = traj.speed_ms;

  // Displacement over 30 min along motion bearing dir
  const dist_m = speed * 1800;
  const d_lat = (dist_m * Math.cos((dir * Math.PI) / 180)) / 111139.0;
  const d_lon = (dist_m * Math.sin((dir * Math.PI) / 180)) / (111139.0 * Math.cos((c_lat * Math.PI) / 180));

  const end_lat = c_lat + d_lat;
  const end_lon = c_lon + d_lon;

  const arrowShaftColor = isDarkMode ? '#facc15' : '#000000';
  const arrowFill = isDarkMode ? '#fde047' : '#000000';
  const arrowStroke = isDarkMode ? '#000000' : '#ffffff';

  // 1. Arrow Shaft Polyline (from cell center to 30-min forecast position)
  const arrowShaft = L.polyline([[c_lat, c_lon], [end_lat, end_lon]], {
    color: arrowShaftColor,
    weight: isDarkMode ? 2.2 : 2.0,
    opacity: 0.95,
  });
  trajectoriesLayer.addLayer(arrowShaft);

  // 2. Sharp Arrowhead Marker (Base SVG points North (0°), perfectly aligned with bearing)
  const arrowIcon = L.divIcon({
    className: 'rdt-arrow-marker',
    html: `
      <div style="width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; transform: rotate(${dir}deg); transform-origin: 10px 10px;">
        <svg width="20" height="20" viewBox="0 0 20 20" style="overflow: visible;">
          <polygon points="10,2 17,17 10,13 3,17" fill="${arrowFill}" stroke="${arrowStroke}" stroke-width="1.2" stroke-linejoin="round"/>
        </svg>
      </div>
    `,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
  const arrowHead = L.marker([end_lat, end_lon], { icon: arrowIcon });
  trajectoriesLayer.addLayer(arrowHead);
}

// =====================================================================
// 17. Timeline Playback Controls
// =====================================================================
function togglePlay() {
  isPlaying = !isPlaying;
  const playBtn = document.getElementById('btn-play');
  if (playBtn) playBtn.innerText = isPlaying ? '⏸' : '▶';

  if (isPlaying) {
    playInterval = setInterval(() => {
      let nextIdx = currentScanIndex + 1;
      if (nextIdx >= allScans.length) nextIdx = 0;
      selectScanIndex(nextIdx);
    }, 1800);
  } else {
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
  }
}

function stepScan(delta) {
  if (isPlaying) togglePlay();
  let nextIdx = currentScanIndex + delta;
  if (nextIdx < 0) nextIdx = 0;
  if (nextIdx >= allScans.length) nextIdx = allScans.length - 1;
  selectScanIndex(nextIdx);
}

// =====================================================================
// 18. Helper: Format Thai Time
// =====================================================================
function formatStampTime(stamp) {
  try {
    const year = stamp.slice(0, 4);
    const month = stamp.slice(4, 6);
    const day = stamp.slice(6, 8);
    const hour = stamp.slice(9, 11);
    const min = stamp.slice(11, 13);
    const utcDate = new Date(Date.UTC(+year, +month - 1, +day, +hour, +min));
    const thaiDate = new Date(utcDate.getTime() + 7 * 3600 * 1000);
    const thH = String(thaiDate.getUTCHours()).padStart(2, '0');
    const thM = String(thaiDate.getUTCMinutes()).padStart(2, '0');
    return `${thH}:${thM} น.`;
  } catch (e) {
    return stamp;
  }
}
