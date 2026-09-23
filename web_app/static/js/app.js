/**
 * app.js -- BIAN-SNN Interactive Web Dashboard
 * Controls canvas rendering, simulation loop, ray-casting visualization, and REST API sync.
 */

// -- State Variables --
let currentState = null;
let currentMethod = 'astar';
let isPlaying = false;
let playTimer = null;
let speedFactor = 3;

// Canvas setup
const canvas = document.getElementById('gridCanvas');
const ctx = canvas.getContext('2d');

// DOM Elements
const playPauseBtn = document.getElementById('playPauseBtn');
const playIcon = document.getElementById('playIcon');
const playText = document.getElementById('playText');
const stepBtn = document.getElementById('stepBtn');
const resetBtn = document.getElementById('resetBtn');
const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');
const presetSelect = document.getElementById('presetSelect');
const overlayBanner = document.getElementById('overlayBanner');
const overlayBannerText = document.getElementById('overlayBannerText');

// Telemetry DOM Elements
const metricSteps = document.getElementById('metricSteps');
const metricCollisions = document.getElementById('metricCollisions');
const metricPos = document.getElementById('metricPos');
const metricFacing = document.getElementById('metricFacing');

// Sensors DOM Elements
const sensorLeftBar = document.getElementById('sensorLeftBar');
const sensorFrontBar = document.getElementById('sensorFrontBar');
const sensorRightBar = document.getElementById('sensorRightBar');
const sensorLeftVal = document.getElementById('sensorLeftVal');
const sensorFrontVal = document.getElementById('sensorFrontVal');
const sensorRightVal = document.getElementById('sensorRightVal');

// SNN DOM Elements
const neuralPanel = document.getElementById('neuralPanel');
const voltLeft = document.getElementById('voltLeft');
const voltForward = document.getElementById('voltForward');
const voltRight = document.getElementById('voltRight');
const spikeNumLeft = document.getElementById('spikeNumLeft');
const spikeNumForward = document.getElementById('spikeNumForward');
const spikeNumRight = document.getElementById('spikeNumRight');
const dotLeft = document.getElementById('dotLeft');
const dotForward = document.getElementById('dotForward');
const dotRight = document.getElementById('dotRight');

// Decision Log DOM
const decisionActionBadge = document.getElementById('decisionActionBadge');
const decisionDescription = document.getElementById('decisionDescription');

// Method Tabs
const tabs = document.querySelectorAll('.method-tab');
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentMethod = tab.getAttribute('data-method');
        updateMethodBadge();
    });
});

function updateMethodBadge() {
    decisionActionBadge.textContent = currentMethod.toUpperCase();
    if (currentMethod === 'astar') {
        decisionDescription.textContent = 'A* optimal shortest-path planning active.';
    } else if (currentMethod === 'snn') {
        decisionDescription.textContent = 'Spiking Neural Network (LIF) autonomous controller active.';
    } else if (currentMethod === 'rl') {
        decisionDescription.textContent = 'Deep Q-Network (DQN) policy agent active.';
    }
}

// -- API Communication --

async function fetchState() {
    try {
        const res = await fetch('/api/state');
        if (!res.ok) throw new Error('API offline');
        currentState = await res.json();
        render();
        updateTelemetry();
    } catch (err) {
        console.error('Failed to fetch state:', err);
    }
}

async function stepSimulation() {
    if (!currentState) return;
    if (currentState.robot.reached_goal) {
        pauseSimulation();
        showBanner('🎉 Goal Reached Successfully!');
        return;
    }

    try {
        const res = await fetch('/api/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ method: currentMethod })
        });
        const newState = await res.json();
        if (newState.done) {
            pauseSimulation();
            showBanner('Goal Reached!');
            return;
        }
        currentState = newState;
        render();
        updateTelemetry();

        if (currentState.robot.reached_goal) {
            pauseSimulation();
            showBanner('🎉 Goal Reached Successfully!');
        } else if (currentState.last_step && currentState.last_step.collided) {
            showBanner('⚠️ Collision Occurred!', true);
        }
    } catch (err) {
        console.error('Failed to step:', err);
    }
}

async function resetSimulation() {
    pauseSimulation();
    hideBanner();
    const preset = presetSelect.value;
    try {
        const res = await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid_size: 15, preset: preset, seed: Math.floor(Math.random() * 1000) })
        });
        currentState = await res.json();
        render();
        updateTelemetry();
    } catch (err) {
        console.error('Failed to reset:', err);
    }
}

async function toggleObstacle(row, col) {
    try {
        const res = await fetch('/api/toggle_obstacle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row, col })
        });
        const data = await res.json();
        if (data.state) {
            currentState = data.state;
            render();
            updateTelemetry();
        }
    } catch (err) {
        console.error('Failed to toggle obstacle:', err);
    }
}

// -- Play / Pause Loop --

function togglePlay() {
    if (isPlaying) {
        pauseSimulation();
    } else {
        startSimulation();
    }
}

function startSimulation() {
    if (currentState && currentState.robot.reached_goal) {
        resetSimulation().then(() => {
            isPlaying = true;
            updatePlayBtn();
            scheduleNextStep();
        });
        return;
    }
    isPlaying = true;
    updatePlayBtn();
    scheduleNextStep();
}

function pauseSimulation() {
    isPlaying = false;
    clearTimeout(playTimer);
    updatePlayBtn();
}

function updatePlayBtn() {
    if (isPlaying) {
        playIcon.textContent = '⏸';
        playText.textContent = 'Pause';
        playPauseBtn.classList.replace('btn-primary', 'btn-secondary');
    } else {
        playIcon.textContent = '▶';
        playText.textContent = 'Auto Play';
        playPauseBtn.classList.replace('btn-secondary', 'btn-primary');
    }
}

function scheduleNextStep() {
    if (!isPlaying) return;
    const intervalMs = Math.max(80, 700 / speedFactor);
    playTimer = setTimeout(async () => {
        await stepSimulation();
        if (isPlaying) scheduleNextStep();
    }, intervalMs);
}

function showBanner(text, isWarning = false) {
    overlayBannerText.textContent = text;
    overlayBanner.style.background = isWarning ? 'rgba(239, 68, 68, 0.9)' : 'rgba(16, 185, 129, 0.9)';
    overlayBanner.classList.add('visible');
    setTimeout(() => {
        if (!currentState?.robot?.reached_goal) hideBanner();
    }, 2000);
}

function hideBanner() {
    overlayBanner.classList.remove('visible');
}

// -- Canvas Rendering Engine --

function render() {
    if (!currentState) return;

    const width = currentState.width;
    const height = currentState.height;
    const cellSize = canvas.width / width;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Grid Cells & Obstacles
    for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
            const isObs = currentState.grid[r][c] === 1;
            const x = c * cellSize;
            const y = r * cellSize;

            if (isObs) {
                // Wall / Obstacle styling
                ctx.fillStyle = '#1E293B';
                ctx.fillRect(x, y, cellSize, cellSize);

                ctx.strokeStyle = '#334155';
                ctx.lineWidth = 1;
                ctx.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

                // Subtle inner 3D block
                ctx.fillStyle = '#0F172A';
                ctx.fillRect(x + 3, y + 3, cellSize - 6, cellSize - 6);
            } else {
                // Free cell
                ctx.fillStyle = '#0B111E';
                ctx.fillRect(x, y, cellSize, cellSize);

                ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(x, y, cellSize, cellSize);
            }
        }
    }

    // 2. Draw Start Cell
    if (currentState.start) {
        const [sr, sc] = currentState.start;
        drawCellMarker(sc * cellSize, sr * cellSize, cellSize, 'rgba(56, 189, 248, 0.25)', '#38BDF8', 'S');
    }

    // 3. Draw Goal Cell
    if (currentState.goal) {
        const [gr, gc] = currentState.goal;
        drawCellMarker(gc * cellSize, gr * cellSize, cellSize, 'rgba(16, 185, 129, 0.3)', '#10B981', 'G');
    }

    // 4. Draw Planned A* Path (if available)
    if (currentMethod === 'astar' && currentState.astar_path && currentState.astar_path.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 230, 118, 0.65)';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);

        const first = currentState.astar_path[0];
        ctx.moveTo((first[1] + 0.5) * cellSize, (first[0] + 0.5) * cellSize);

        for (let i = 1; i < currentState.astar_path.length; i++) {
            const pt = currentState.astar_path[i];
            ctx.lineTo((pt[1] + 0.5) * cellSize, (pt[0] + 0.5) * cellSize);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // 5. Draw Robot Trajectory Trace
    if (currentState.robot.trajectory && currentState.robot.trajectory.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = currentMethod === 'snn' ? 'rgba(157, 78, 221, 0.5)' : 
                          currentMethod === 'rl'  ? 'rgba(255, 158, 0, 0.5)' : 
                          'rgba(0, 242, 254, 0.35)';
        ctx.lineWidth = 2.5;
        const traj = currentState.robot.trajectory;
        ctx.moveTo((traj[0][1] + 0.5) * cellSize, (traj[0][0] + 0.5) * cellSize);
        for (let i = 1; i < traj.length; i++) {
            ctx.lineTo((traj[i][1] + 0.5) * cellSize, (traj[i][0] + 0.5) * cellSize);
        }
        ctx.stroke();
    }

    // 6. Draw Ray-Casting Sensors
    drawSensorRays(cellSize);

    // 7. Draw Robot
    drawRobot(cellSize);
}

function drawCellMarker(x, y, size, fill, stroke, text) {
    ctx.fillStyle = fill;
    ctx.fillRect(x + 2, y + 2, size - 4, size - 4);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x + 2, y + 2, size - 4, size - 4);

    ctx.fillStyle = stroke;
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + size / 2, y + size / 2);
}

function drawRobot(cellSize) {
    const [r, c] = currentState.robot.pos;
    const cx = (c + 0.5) * cellSize;
    const cy = (r + 0.5) * cellSize;
    const radius = cellSize * 0.38;

    // Glowing aura
    const glow = ctx.createRadialGradient(cx, cy, radius * 0.3, cx, cy, radius * 1.5);
    glow.addColorStop(0, 'rgba(0, 242, 254, 0.4)');
    glow.addColorStop(1, 'transparent');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 1.5, 0, 2 * Math.PI);
    ctx.fill();

    // Body
    ctx.fillStyle = '#0F172A';
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.fill();

    ctx.strokeStyle = currentMethod === 'snn' ? '#9D4EDD' : 
                      currentMethod === 'rl'  ? '#FF9E00' : '#00F2FE';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Heading Arrow
    const orient = currentState.robot.orientation; // 0=N, 1=E, 2=S, 3=W
    const angles = [-Math.PI / 2, 0, Math.PI / 2, Math.PI];
    const angle = angles[orient];

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);

    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.moveTo(radius * 0.75, 0);
    ctx.lineTo(-radius * 0.45, -radius * 0.45);
    ctx.lineTo(-radius * 0.2, 0);
    ctx.lineTo(-radius * 0.45, radius * 0.45);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
}

function drawSensorRays(cellSize) {
    if (!currentState.sensors || !currentState.sensors.raw) return;

    const [r, c] = currentState.robot.pos;
    const cx = (c + 0.5) * cellSize;
    const cy = (r + 0.5) * cellSize;
    const orient = currentState.robot.orientation; // 0=N, 1=E, 2=S, 3=W

    // Orientation angles
    const baseAngle = [-Math.PI / 2, 0, Math.PI / 2, Math.PI][orient];

    const rays = [
        { name: 'left',  angle: baseAngle - Math.PI / 2, dist: currentState.sensors.raw.left,  color: '#38BDF8' },
        { name: 'front', angle: baseAngle,               dist: currentState.sensors.raw.front, color: '#FACC15' },
        { name: 'right', angle: baseAngle + Math.PI / 2, dist: currentState.sensors.raw.right, color: '#F43F5E' },
    ];

    rays.forEach(ray => {
        // Compute hit coordinate
        const totalDist = ray.dist + 1.0;
        const hitX = cx + Math.cos(ray.angle) * totalDist * cellSize;
        const hitY = cy + Math.sin(ray.angle) * totalDist * cellSize;

        // Draw ray line
        ctx.beginPath();
        ctx.strokeStyle = ray.color;
        ctx.lineWidth = 1.75;
        ctx.setLineDash([4, 4]);
        ctx.moveTo(cx, cy);
        ctx.lineTo(hitX, hitY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Spark dot at hit
        ctx.fillStyle = ray.color;
        ctx.shadowColor = ray.color;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(hitX, hitY, 3.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;
    });
}

// -- Telemetry & HUD Update --

function updateTelemetry() {
    if (!currentState) return;

    metricSteps.textContent = currentState.robot.steps;
    metricCollisions.textContent = currentState.robot.collisions;
    metricPos.textContent = `(${currentState.robot.pos[0]}, ${currentState.robot.pos[1]})`;
    metricFacing.textContent = `${currentState.robot.orient_name} (${['NORTH', 'EAST', 'SOUTH', 'WEST'][currentState.robot.orientation]})`;

    // Sensor Meters
    const raw = currentState.sensors.raw;
    const maxR = currentState.sensors.max_range;

    sensorLeftVal.textContent = `${raw.left} / ${maxR} cells`;
    sensorFrontVal.textContent = `${raw.front} / ${maxR} cells`;
    sensorRightVal.textContent = `${raw.right} / ${maxR} cells`;

    sensorLeftBar.style.width = `${(raw.left / maxR) * 100}%`;
    sensorFrontBar.style.width = `${(raw.front / maxR) * 100}%`;
    sensorRightBar.style.width = `${(raw.right / maxR) * 100}%`;

    // Decision info
    if (currentState.last_step && currentState.last_step.action) {
        decisionActionBadge.textContent = currentState.last_step.action;
        let desc = `Executed ${currentState.last_step.action}`;
        if (currentState.last_step.collided) desc += ' · [COLLISION DETECTED]';
        if (currentState.last_step.reached_goal) desc += ' · [TARGET REACHED]';
        decisionDescription.textContent = desc;

        // SNN Specific updates
        if (currentState.last_step.extra && currentState.last_step.extra.motor_spikes) {
            const spk = currentState.last_step.extra.motor_spikes;
            spikeNumLeft.textContent = `${spk[0]} Spikes`;
            spikeNumForward.textContent = `${spk[1]} Spikes`;
            spikeNumRight.textContent = `${spk[2]} Spikes`;

            dotLeft.classList.toggle('firing', spk[0] > 0);
            dotForward.classList.toggle('firing', spk[1] > 0);
            dotRight.classList.toggle('firing', spk[2] > 0);
        }

        if (currentState.last_step.extra && currentState.last_step.extra.motor_voltages) {
            const v = currentState.last_step.extra.motor_voltages;
            voltLeft.textContent = `${v[0].toFixed(1)} mV`;
            voltForward.textContent = `${v[1].toFixed(1)} mV`;
            voltRight.textContent = `${v[2].toFixed(1)} mV`;
        }
    }
}

// -- Canvas Click to Toggle Obstacles --

canvas.addEventListener('click', (e) => {
    if (!currentState) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellSize = rect.width / currentState.width;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);

    toggleObstacle(row, col);
});

// -- Speed Slider --
speedSlider.addEventListener('input', (e) => {
    speedFactor = parseInt(e.target.value);
    speedValue.textContent = `${speedFactor}x`;
});

// -- Preset Change --
presetSelect.addEventListener('change', () => {
    resetSimulation();
});

// -- Button Listeners --
playPauseBtn.addEventListener('click', togglePlay);
stepBtn.addEventListener('click', () => {
    pauseSimulation();
    stepSimulation();
});
resetBtn.addEventListener('click', resetSimulation);

// -- Modal Gallery --
const galleryModal = document.getElementById('galleryModal');
const openGalleryBtn = document.getElementById('openGalleryBtn');
const closeGalleryBtn = document.getElementById('closeGalleryBtn');
const modalFiguresList = document.getElementById('modalFiguresList');

openGalleryBtn.addEventListener('click', async () => {
    galleryModal.classList.add('open');
    try {
        const res = await fetch('/api/figures');
        const data = await res.json();
        modalFiguresList.innerHTML = '';

        if (!data.figures || data.figures.length === 0) {
            modalFiguresList.innerHTML = '<p>No figures found in results/figures directory.</p>';
            return;
        }

        data.figures.forEach(fig => {
            const card = document.createElement('div');
            card.className = 'figure-preview-card';
            card.innerHTML = `
                <h4>${fig}</h4>
                <a href="/api/figures/${fig}" target="_blank">
                    <img src="/api/figures/${fig}" alt="${fig}" loading="lazy">
                </a>
            `;
            modalFiguresList.appendChild(card);
        });
    } catch (err) {
        modalFiguresList.innerHTML = '<p>Failed to load figures.</p>';
    }
});

closeGalleryBtn.addEventListener('click', () => {
    galleryModal.classList.remove('open');
});

// Initialize on page load
fetchState();
