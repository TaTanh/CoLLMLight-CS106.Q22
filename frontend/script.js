/**
 * Draw Road Network & Simulate
 */
id = Math.random().toString(36).substring(2, 15);

BACKGROUND_COLOR = 0x121824; // Premium Dark Navy
LANE_COLOR = 0x1f293d;       // Slate Lane Color
LANE_BORDER_WIDTH = 1;
LANE_BORDER_COLOR = 0x374151; // Grey Border
LANE_INNER_COLOR = 0x4b5563;
LANE_DASH = 10;
LANE_GAP = 12;
TRAFFIC_LIGHT_WIDTH = 4;
MAX_TRAFFIC_LIGHT_NUM = 100000;
ROTATE = 90;

CAR_LENGTH = 5;
CAR_WIDTH = 2;

CAR_COLORS = [
    0x10b981, // Emerald Green
    0x3b82f6, // Bright Blue
    0xf59e0b, // Amber Gold
    0x8b5cf6, // Violet
    0xec4899  // Pink
];
CAR_COLORS_NUM = CAR_COLORS.length;

NUM_CAR_POOL = 10000;
let debugMode = false;

LIGHT_RED = 0xef4444;    // Vivid Red
LIGHT_GREEN = 0x10b981;  // Emerald Green

TURN_SIGNAL_COLOR = 0xFFFFFF;
TURN_SIGNAL_WIDTH   = 1;
TURN_SIGNAL_LENGTH  = 5;

var simulation, roadnet, steps;
var nodes = {};
var edges = {};
var logs;
var gettingLog = false;

let Application = PIXI.Application,
    Sprite = PIXI.Sprite,
    Graphics = PIXI.Graphics,
    Container = PIXI.Container,
    ParticleContainer = PIXI.particles.ParticleContainer,
    Texture = PIXI.Texture,
    Rectangle = PIXI.Rectangle
;

var controls = new function () {
    this.replaySpeedMax = 1;
    this.replaySpeedMin = 0.01;
    this.replaySpeed = 0.5;
    this.paused = false;
};

var trafficLightsG = {};

var app, viewport, renderer, simulatorContainer, carContainer, trafficLightContainer;
var turnSignalContainer;
var carPool;

var cnt = 0;
var frameElapsed = 0;
var totalStep = 0;

var nodeCarNum = document.getElementById("car-num");
var nodeProgressPercentage = document.getElementById("progress-percentage");
var nodeTotalStep = document.getElementById("total-step-num");
var nodeCurrentStep = document.getElementById("current-step-num");
var nodeSelectedEntity = document.getElementById("intersection-selected-badge");

var SPEED = 3, SCALE_SPEED = 1.01;
var LEFT = 37, UP = 38, RIGHT = 39, DOWN = 40;
var MINUS = 189, EQUAL = 187, P = 80;
var LEFT_BRACKET = 219, RIGHT_BRACKET = 221; 
var ONE = 49, TWO = 50;
var SPACE = 32;

var keyDown = new Set();
var turnSignalTextures = [];

let pauseButton = document.getElementById("pause");
let playPauseIcon = document.getElementById("play-pause-icon");
let nodeCanvas = document.getElementById("simulator-canvas");
let replayControlDom = document.getElementById("replay-control");
let replaySpeedDom = document.getElementById("replay-speed");

let loading = false;
let selectedDOM = document.getElementById("intersection-selected-badge");

// Web Dashboard State variables
let activeRunId = "";
let reasoningData = null;
let selectedIntersection = "";
let intersectionIds = [];

/**
 * Handle Web Dashboard API requests
 */

async function loadRunsList() {
    try {
        let resp = await fetch("/api/runs");
        let runs = await resp.json();
        let selector = document.getElementById("run-selector");
        selector.innerHTML = "";
        
        if (runs.length === 0) {
            selector.innerHTML = `<option value="" disabled selected>No completed runs found</option>`;
            return;
        }

        selector.innerHTML = `<option value="" disabled selected>Choose a simulation run...</option>`;
        runs.forEach(run => {
            let option = document.createElement("option");
            option.value = run.id;
            option.text = `${run.name} (${run.has_reasoning ? 'with LLM logs' : 'baseline'})`;
            option.dataset.hasReasoning = run.has_reasoning;
            selector.appendChild(option);
        });
    } catch (e) {
        console.error("Failed to fetch runs list:", e);
    }
}

async function loadSelectedRun(runId, hasReasoning) {
    if (loading) return;
    loading = true;
    
    document.getElementById("spinner").classList.remove("d-none");
    document.getElementById("guide").classList.add("d-none");
    
    try {
        // Fetch roadnetLogFile
        let roadnetResp = await fetch(`/api/run_file?run_id=${runId}&file=roadnetLogFile.json`);
        let roadnetText = await roadnetResp.text();
        
        // Fetch replayLogFile
        let replayResp = await fetch(`/api/run_file?run_id=${runId}&file=replayLogFile.txt`);
        let replayText = await replayResp.text();
        
        // Fetch reasoning log if available
        reasoningData = null;
        if (hasReasoning === "true") {
            try {
                let reasoningResp = await fetch(`/api/run_file?run_id=${runId}&file=reasoning_log.json`);
                if (reasoningResp.ok) {
                    reasoningData = await reasoningResp.json();
                }
            } catch (e) {
                console.error("Failed to load reasoning logs:", e);
            }
        }
        
        // Load final metrics if available
        try {
            let metricsResp = await fetch(`/api/metrics`);
            let metrics = await metricsResp.json();
            let matched = metrics.find(m => runId.includes(m.dataset) && (runId.includes(m.model) || m.model === 'student'));
            if (matched) {
                document.getElementById("metric-att").innerText = `${matched.ATT}s`;
                document.getElementById("metric-awt").innerText = `${matched.AWT}s`;
            } else {
                document.getElementById("metric-att").innerText = "--";
                document.getElementById("metric-awt").innerText = "--";
            }
        } catch (e) {
            console.error("Failed to parse metrics:", e);
        }

        // Initialize simulation variables
        simulation = JSON.parse(roadnetText);
        logs = replayText.split('\n');
        if (logs[logs.length - 1] === "") {
            logs.pop();
        }
        totalStep = logs.length;
        
        controls.paused = false;
        playPauseIcon.className = "fas fa-pause";
        cnt = 0;
        
        setTimeout(function () {
            try {
                drawRoadnet();
            } catch (e) {
                console.error("Drawing roadnet failed:", e);
                alert("Drawing roadnet failed: " + e.message + "\nStack: " + e.stack);
                document.getElementById("spinner").classList.add("d-none");
                loading = false;
                return;
            }
            ready = true;
            loading = false;
            document.getElementById("spinner").classList.add("d-none");
        }, 200);

    } catch (e) {
        alert("Failed to load simulation files from server: " + e.message);
        document.getElementById("spinner").classList.add("d-none");
        loading = false;
    }
}

// GUI Simulation Launcher status polling
let pollInterval = null;
let wasSimulating = false;
function startPollingStatus() {
    if (pollInterval) clearInterval(pollInterval);
    
    pollInterval = setInterval(async () => {
        try {
            let resp = await fetch("/api/simulate_status");
            let data = await resp.json();
            let consoleDOM = document.getElementById("console-output");
            
            if (data.running) {
                wasSimulating = true;
                document.getElementById("console-running-badge").classList.remove("d-none");
                
                // Disable start button
                let startBtn = document.getElementById("btn-run-sim");
                if (startBtn) {
                    startBtn.disabled = true;
                    startBtn.innerHTML = `<i class="fas fa-spinner fa-spin mr-1"></i> Simulating...`;
                }

                consoleDOM.innerHTML = data.output || "Simulation starting...";
                consoleDOM.scrollTop = consoleDOM.scrollHeight;
            } else {
                clearInterval(pollInterval);
                pollInterval = null;
                document.getElementById("console-running-badge").classList.add("d-none");
                
                // Enable start button
                let startBtn = document.getElementById("btn-run-sim");
                if (startBtn) {
                    startBtn.disabled = false;
                    startBtn.innerHTML = `<i class="fas fa-play mr-1"></i> Start Simulation`;
                }

                // Refresh list of runs
                await loadRunsList();

                // Auto-load the run if it has just transitioned from running to stopped
                if (wasSimulating) {
                    wasSimulating = false;
                    let dataset = document.getElementById("sim-dataset").value;
                    let runId = `litepp_eval_${dataset}`;
                    let selector = document.getElementById("run-selector");
                    let matchedOpt = selector.querySelector(`option[value="${runId}"]`);
                    if (matchedOpt) {
                        selector.value = runId;
                        activeRunId = runId;
                        loadSelectedRun(runId, matchedOpt.dataset.hasReasoning);
                    }
                }
            }
        } catch(e) {
            console.error("Polling error:", e);
        }
    }, 1000);
}

// Trigger simulation on click
document.getElementById("btn-run-sim").addEventListener("click", async () => {
    let dataset = document.getElementById("sim-dataset").value;
    let model = document.getElementById("sim-model").value;
    let time = parseInt(document.getElementById("sim-time").value);
    let endpoint = document.getElementById("sim-endpoint").value;
    
    try {
        let resp = await fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset, model, time, endpoint })
        });
        if (resp.ok) {
            document.getElementById("console-output").innerHTML = "Starting simulation on backend...\n";
            startPollingStatus();
        } else {
            let text = await resp.text();
            alert("Error launching simulation: " + text);
        }
    } catch(e) {
        alert("Network error: " + e.message);
    }
});

// Run Selector Change
document.getElementById("run-selector").addEventListener("change", (e) => {
    let selectedOption = e.target.options[e.target.selectedIndex];
    activeRunId = selectedOption.value;
    loadSelectedRun(activeRunId, selectedOption.dataset.hasReasoning);
});

// Intersection Selector Change
document.getElementById("inter-log-selector").addEventListener("change", (e) => {
    selectedIntersection = e.target.value;
    if (selectedIntersection) {
        selectedDOM.innerText = selectedIntersection;
        updateReasoningUI(cnt);
    }
});

function selectIntersection(interId) {
    selectedIntersection = interId;
    document.getElementById("inter-log-selector").value = interId;
    selectedDOM.innerText = interId;
    updateReasoningUI(cnt);
}

function updateReasoningUI(step) {
    let detailsPanel = document.getElementById("reasoning-log-details");
    let placeholder = document.getElementById("reasoning-placeholder");
    
    if (!reasoningData || !selectedIntersection) {
        detailsPanel.classList.add("d-none");
        placeholder.classList.remove("d-none");
        return;
    }
    
    // Find the floor decision timestep (decisions are every 30s)
    let decisionStep = Math.floor(step / 30) * 30;
    let stepKey = decisionStep.toString();
    
    let stepLog = reasoningData[stepKey];
    if (!stepLog || !stepLog[selectedIntersection]) {
        detailsPanel.classList.add("d-none");
        placeholder.classList.remove("d-none");
        placeholder.innerHTML = `<i class="fas fa-exclamation-triangle text-warning h3 mb-2"></i><p class="text-xs">No decision log found for ${selectedIntersection} at step ${step}.</p>`;
        return;
    }
    
    placeholder.classList.add("d-none");
    detailsPanel.classList.remove("d-none");
    
    let interLog = stepLog[selectedIntersection];
    
    // Display Phase 1 (Complexity)
    let complexityLabel = "EASY";
    let complexityReason = "Simple queue configurations.";
    let raJson = {};
    
    try {
        raJson = JSON.parse(interLog.ra_response);
        if (raJson.phase1) {
            complexityLabel = raJson.phase1.answer ? raJson.phase1.answer.toUpperCase() : "EASY";
            complexityReason = raJson.phase1.reason || "Standard traffic volume.";
        }
    } catch (e) {
        console.error("Failed to parse RA response JSON:", e);
    }
    
    let compBadge = document.getElementById("llm-complexity-badge");
    compBadge.innerText = complexityLabel;
    if (complexityLabel === "COMPLEX") {
        compBadge.className = "badge badge-danger text-xxs font-weight-bold";
    } else {
        compBadge.className = "badge badge-success text-xxs font-weight-bold";
    }
    document.getElementById("llm-complexity-reason").innerText = `"${complexityReason}"`;
    
    // Display Phase 2 (Signal Choice)
    let choicePhase = "UNKNOWN";
    let raAnalysis = "--";
    let raFuture = "--";
    let raCompare = "--";
    
    if (raJson.phase2) {
        choicePhase = raJson.phase2.answer || "UNKNOWN";
        raAnalysis = raJson.phase2.traffic_analysis || "--";
        raFuture = raJson.phase2.future_state_summary || "--";
        raCompare = raJson.phase2.signal_comparison || "--";
    }
    
    document.getElementById("llm-choice-badge").innerText = choicePhase;
    document.getElementById("llm-ra-analysis").innerText = raAnalysis;
    document.getElementById("llm-ra-future").innerText = raFuture;
    document.getElementById("llm-ra-compare").innerText = raCompare;
    
    // Display ATR Reasoning Free-text
    document.getElementById("llm-atr-reasoning").innerText = interLog.atr_response || "No analysis text logged.";
}

/**
 * PixiJS Simulator Rendering
 */
let ready = false;

function updateReplaySpeed(speed){
    speed = Math.min(speed, 1);
    speed = Math.max(speed, 0.01);
    controls.replaySpeed = speed;
    replayControlDom.value = speed * 100;
    replaySpeedDom.innerHTML = speed.toFixed(2);
}

function initCanvas() {
    app = new Application({
        width: nodeCanvas.offsetWidth,
        height: nodeCanvas.offsetHeight,
        transparent: false,
        backgroundColor: BACKGROUND_COLOR
    });

    nodeCanvas.appendChild(app.view);
    app.view.classList.add("d-none");

    renderer = app.renderer;
    renderer.interactive = true;
    renderer.autoResize = true;

    renderer.resize(nodeCanvas.offsetWidth, nodeCanvas.offsetHeight);
    app.ticker.add(run);
}

function showCanvas() {
    document.getElementById("spinner").classList.add("d-none");
    app.view.classList.remove("d-none");
}

function hideCanvas() {
    document.getElementById("spinner").classList.remove("d-none");
    app.view.classList.add("d-none");
}

function drawRoadnet() {
    if (simulatorContainer) {
        simulatorContainer.destroy(true);
    }
    app.stage.removeChildren();
    viewport = new Viewport.Viewport({
        screenWidth: nodeCanvas.offsetWidth,
        screenHeight: nodeCanvas.offsetHeight,
        interaction: app.renderer.plugins.interaction
    });
    viewport
        .drag()
        .pinch()
        .wheel()
        .decelerate();
    app.stage.addChild(viewport);
    simulatorContainer = new Container();
    viewport.addChild(simulatorContainer);

    roadnet = simulation.static;
    nodes = [];
    edges = [];
    trafficLightsG = {};
    intersectionIds = [];

    for (let i = 0, len = roadnet.nodes.length;i < len;++i) {
        node = roadnet.nodes[i];
        node.point = new Point(transCoord(node.point));
        nodes[node.id] = node;
    }

    for (let i = 0, len = roadnet.edges.length;i < len;++i) {
        edge = roadnet.edges[i];
        edge.from = nodes[edge.from];
        edge.to = nodes[edge.to];
        for (let j = 0, len = edge.points.length;j < len;++j) {
            edge.points[j] = new Point(transCoord(edge.points[j]));
        }
        edges[edge.id] = edge;
    }

    // Draw Map Graphics
    trafficLightContainer = new ParticleContainer(MAX_TRAFFIC_LIGHT_NUM, {tint: true});
    let mapContainer = new Container();
    simulatorContainer.addChild(mapContainer);

    // Populate dropdown with intersections
    let interSelector = document.getElementById("inter-log-selector");
    interSelector.innerHTML = '<option value="" disabled selected>Select intersection node...</option>';

    for (nodeId in nodes) {
        if (!nodes[nodeId].virtual) {
            let nodeGraphics = new Graphics();
            mapContainer.addChild(nodeGraphics);
            drawNode(nodes[nodeId], nodeGraphics);
            
            // Add to selections
            intersectionIds.push(nodeId);
            let option = document.createElement("option");
            option.value = nodeId;
            option.text = nodeId;
            interSelector.appendChild(option);
        }
    }
    
    // Choose first intersection as default
    if (intersectionIds.length > 0) {
        selectIntersection(intersectionIds[0]);
    }

    for (edgeId in edges) {
        let edgeGraphics = new Graphics();
        mapContainer.addChild(edgeGraphics);
        drawEdge(edges[edgeId], edgeGraphics);
    }
    
    let bounds = simulatorContainer.getBounds();
    simulatorContainer.pivot.set(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    simulatorContainer.position.set(renderer.width / 2, renderer.height / 2);
    simulatorContainer.addChild(trafficLightContainer);

    // Settings for Cars
    TURN_SIGNAL_LENGTH = CAR_LENGTH;
    TURN_SIGNAL_WIDTH  = CAR_WIDTH / 2;

    var carG = new Graphics();
    carG.lineStyle(0);
    carG.beginFill(0xFFFFFF, 0.8);
    carG.drawRect(0, 0, CAR_LENGTH, CAR_WIDTH);

    let carTexture = renderer.generateTexture(carG);

    let signalG = new Graphics();
    signalG.beginFill(TURN_SIGNAL_COLOR, 0.7).drawRect(0,0, TURN_SIGNAL_LENGTH, TURN_SIGNAL_WIDTH)
           .drawRect(0, 3 * CAR_WIDTH - TURN_SIGNAL_WIDTH, TURN_SIGNAL_LENGTH, TURN_SIGNAL_WIDTH).endFill();
    let turnSignalTexture = renderer.generateTexture(signalG);

    let signalLeft = new Texture(turnSignalTexture, new Rectangle(0, 0, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    let signalStraight = new Texture(turnSignalTexture, new Rectangle(0, CAR_WIDTH, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    let signalRight = new Texture(turnSignalTexture, new Rectangle(0, CAR_WIDTH * 2, TURN_SIGNAL_LENGTH, CAR_WIDTH));
    turnSignalTextures = [signalLeft, signalStraight, signalRight];

    carPool = [];
    carContainer = new Container();
    turnSignalContainer = new ParticleContainer(NUM_CAR_POOL, {rotation: true, tint: true});
    simulatorContainer.addChild(carContainer);
    simulatorContainer.addChild(turnSignalContainer);
    
    for (let i = 0, len = NUM_CAR_POOL;i < len;++i) {
        let car = new Sprite(carTexture);
        let signal = new Sprite(turnSignalTextures[1]);
        car.anchor.set(1, 0.5);

        if (debugMode) {
            car.interactive = true;
            car.on('mouseover', function () {
                car.alpha = 0.8;
            });
            car.on('mouseout', function () {
                car.alpha = 1;
            });
        }
        
        signal.anchor.set(1, 0.5);
        carPool.push([car, signal]);
    }
    
    showCanvas();
    return true;
}

initCanvas();

function transCoord(point) {
    return [point[0], -point[1]];
}

PIXI.Graphics.prototype.drawLine = function(pointA, pointB) {
    this.moveTo(pointA.x, pointA.y);
    this.lineTo(pointB.x, pointB.y);
}

PIXI.Graphics.prototype.drawDashLine = function(pointA, pointB, dash = 16, gap = 8) {
    let direct = pointA.directTo(pointB);
    let distance = pointA.distanceTo(pointB);

    let currentPoint = pointA;
    let currentDistance = 0;
    let length;
    let finish = false;
    while (true) {
        this.moveTo(currentPoint.x, currentPoint.y);
        if (currentDistance + dash >= distance) {
            length = distance - currentDistance;
            finish = true;
        } else {
            length = dash
        }
        currentPoint = currentPoint.moveAlong(direct, length);
        this.lineTo(currentPoint.x, currentPoint.y);
        if (finish) break;
        currentDistance += length;

        if (currentDistance + gap >= distance) {
            break;
        } else {
            currentPoint = currentPoint.moveAlong(direct, gap);
            currentDistance += gap;
        }
    }
};

function drawNode(node, graphics) {
    graphics.beginFill(LANE_COLOR);
    let outline = node.outline;
    for (let i = 0 ; i < outline.length ; i+=2) {
        outline[i+1] = -outline[i+1];
        if (i == 0)
            graphics.moveTo(outline[i], outline[i+1]);
        else
            graphics.lineTo(outline[i], outline[i+1]);
    }
    graphics.endFill();

    graphics.hitArea = new PIXI.Polygon(outline);
    graphics.interactive = true;
    graphics.on("mouseover", function () {
        graphics.alpha = 0.5;
    });
    graphics.on("mouseout", function () {
        graphics.alpha = 1;
    });
    graphics.on("click", function () {
        selectIntersection(node.id);
    });
}

function drawEdge(edge, graphics) {
    let from = edge.from;
    let to = edge.to;
    let points = edge.points;

    let pointA, pointAOffset, pointB, pointBOffset;
    let prevPointBOffset = null;

    let roadWidth = 0;
    edge.laneWidths.forEach(function(l){
        roadWidth += l;
    }, 0);

    let coords = [], coords1 = [];

    for (let i = 1;i < points.length;++i) {
        if (i == 1){
            pointA = points[0].moveAlongDirectTo(points[1], from.virtual ? 0 : from.width);
            pointAOffset = points[0].directTo(points[1]).rotate(ROTATE);
        } else {
            pointA = points[i-1];
            pointAOffset = prevPointBOffset;
        }
        if (i == points.length - 1) {
            pointB = points[i].moveAlongDirectTo(points[i-1], to.virtual ? 0 : to.width);
            pointBOffset = points[i-1].directTo(points[i]).rotate(ROTATE);
        } else {
            pointB = points[i];
            pointBOffset = points[i-1].directTo(points[i+1]).rotate(ROTATE);
        }
        prevPointBOffset = pointBOffset;

        lightG = new Graphics();
        lightG.lineStyle(TRAFFIC_LIGHT_WIDTH, 0xFFFFFF);
        lightG.drawLine(new Point(0, 0), new Point(1, 0));
        lightTexture = renderer.generateTexture(lightG);

        // Draw Traffic Lights
        if (i == points.length-1 && !to.virtual) {
            edgeTrafficLights = [];
            prevOffset = offset = 0;
            for (lane = 0;lane < edge.nLane;++lane) {
                offset += edge.laneWidths[lane];
                var light = new Sprite(lightTexture);
                light.anchor.set(0, 0.5);
                light.scale.set(offset - prevOffset, 1);
                point_ = pointB.moveAlong(pointBOffset, prevOffset);
                light.position.set(point_.x, point_.y);
                light.rotation = pointBOffset.getAngleInRadians();
                edgeTrafficLights.push(light);
                prevOffset = offset;
                trafficLightContainer.addChild(light);
            }
            trafficLightsG[edge.id] = edgeTrafficLights;
        }

        // Draw Roads
        graphics.lineStyle(LANE_BORDER_WIDTH, LANE_BORDER_COLOR, 1);
        graphics.drawLine(pointA, pointB);

        pointA1 = pointA.moveAlong(pointAOffset, roadWidth);
        pointB1 = pointB.moveAlong(pointBOffset, roadWidth);

        graphics.lineStyle(0);
        graphics.beginFill(LANE_COLOR);

        coords = coords.concat([pointA.x, pointA.y, pointB.x, pointB.y]);
        coords1 = coords1.concat([pointA1.y, pointA1.x, pointB1.y, pointB1.x]);

        graphics.drawPolygon([pointA.x, pointA.y, pointB.x, pointB.y, pointB1.x, pointB1.y, pointA1.x, pointA1.y]);
        graphics.endFill();

        offset = 0;
        for (let lane = 0, len = edge.nLane-1;lane < len;++lane) {
            offset += edge.laneWidths[lane];
            graphics.lineStyle(LANE_BORDER_WIDTH, LANE_INNER_COLOR);
            graphics.drawDashLine(pointA.moveAlong(pointAOffset, offset), pointB.moveAlong(pointBOffset, offset), LANE_DASH, LANE_GAP);
        }
    }
}

function run(delta) {
    let redraw = false;

    if (ready && (!controls.paused || redraw)) {
        try {
            drawStep(cnt);
        } catch (e) {
            console.error("Error occurred when drawing step:", e);
            ready = false;
        }
        if (!controls.paused) {
            frameElapsed += 1;
            if (frameElapsed >= 1 / controls.replaySpeed ** 2) {
                cnt += 1;
                frameElapsed = 0;
                if (cnt == totalStep) cnt = 0;
            }
        }
    }
}

function _statusToColor(status) {
    switch (status) {
        case 'r':
            return LIGHT_RED;
        case 'g':
            return LIGHT_GREEN;
        default:
            return 0x374151;  
    }
}

function stringHash(str) {
    let hash = 0;
    let p = 127, p_pow = 1;
    let m = 1e9 + 9;
    for (let i = 0; i < str.length; i++) {
        hash = (hash + str.charCodeAt(i) * p_pow) % m;
        p_pow = (p_pow * p) % m;
    }
    return hash;
}

function drawStep(step) {
    let [carLogs, tlLogs] = logs[step].split(';');

    tlLogs = tlLogs.split(',');
    carLogs = carLogs.split(',');
    
    let tlLog, tlEdge, tlStatus;
    for (let i = 0, len = tlLogs.length;i < len;++i) {
        tlLog = tlLogs[i].split(' ');
        tlEdge = tlLog[0];
        tlStatus = tlLog.slice(1);
        for (let j = 0, len = tlStatus.length;j < len;++j) {
            if (trafficLightsG[tlEdge] && trafficLightsG[tlEdge][j]) {
                trafficLightsG[tlEdge][j].tint = _statusToColor(tlStatus[j]);
                if (tlStatus[j] == 'i' ) {
                    trafficLightsG[tlEdge][j].alpha = 0;
                } else {
                    trafficLightsG[tlEdge][j].alpha = 1;
                }
            }
        }
    }

    carContainer.removeChildren();
    turnSignalContainer.removeChildren();
    
    let carLog, position, length, width;
    for (let i = 0, len = carLogs.length - 1;i < len;++i) {
        carLog = carLogs[i].split(' ');
        position = transCoord([parseFloat(carLog[0]), parseFloat(carLog[1])]);
        length = parseFloat(carLog[5]);
        width = parseFloat(carLog[6]);
        
        if (carPool[i]) {
            carPool[i][0].position.set(position[0], position[1]);
            carPool[i][0].rotation = 2*Math.PI - parseFloat(carLog[2]);
            carPool[i][0].name = carLog[3];
            let carColorId = stringHash(carLog[3]) % CAR_COLORS_NUM;
            carPool[i][0].tint = CAR_COLORS[carColorId];
            carPool[i][0].width = length;
            carPool[i][0].height = width;
            carContainer.addChild(carPool[i][0]);

            let laneChange = parseInt(carLog[4]) + 1;
            carPool[i][1].position.set(position[0], position[1]);
            carPool[i][1].rotation = carPool[i][0].rotation;
            carPool[i][1].texture = turnSignalTextures[laneChange];
            carPool[i][1].width = length;
            carPool[i][1].height = width;
            turnSignalContainer.addChild(carPool[i][1]);
        }
    }
    
    nodeCarNum.innerText = carLogs.length - 1;
    nodeTotalStep.innerText = totalStep;
    nodeCurrentStep.innerText = cnt + 1;
    nodeProgressPercentage.innerText = (cnt / totalStep * 100).toFixed(2) + "%";
    
    // Sync reasoning sidebar text
    updateReasoningUI(step);
}

// Playback slider & buttons
document.getElementById("slow-btn")?.addEventListener("click", function() {
    updateReplaySpeed(controls.replaySpeed - 0.1);
});

document.getElementById("fast-btn")?.addEventListener("click", function() {
    updateReplaySpeed(controls.replaySpeed + 0.1);
});

updateReplaySpeed(0.5);

replayControlDom.addEventListener('input', function(e){
    updateReplaySpeed(replayControlDom.value / 100);
});

document.addEventListener('keydown', function(e) {
    if (e.keyCode == P || e.keyCode == SPACE) {
        togglePause();
        e.preventDefault();
    } else if (e.keyCode == ONE) {
        updateReplaySpeed(Math.max(controls.replaySpeed / 1.5, controls.replaySpeedMin));
    } else if (e.keyCode == TWO) {
        updateReplaySpeed(Math.min(controls.replaySpeed * 1.5, controls.replaySpeedMax));
    } else if (e.keyCode == LEFT_BRACKET) {
        stepBackward();
    } else if (e.keyCode == RIGHT_BRACKET) {
        stepForward();
    } else {
        keyDown.add(e.keyCode);
    }
});

document.addEventListener('keyup', (e) => keyDown.delete(e.keyCode));

nodeCanvas.addEventListener('dblclick', function(e){
    togglePause();
});

pauseButton.addEventListener('click', function(e){
    togglePause();
});

function togglePause() {
    controls.paused = !controls.paused;
    if (controls.paused) {
        playPauseIcon.className = "fas fa-play";
    } else {
        playPauseIcon.className = "fas fa-pause";
    }
}

function stepForward() {
    if (ready) {
        cnt = (cnt + 1) % totalStep;
        drawStep(cnt);
    }
}

function stepBackward() {
    if (ready) {
        cnt = (cnt - 1) % totalStep;
        cnt = (cnt + totalStep) % totalStep;
        drawStep(cnt);
    }
}

// Tab Switching logic for Left Panel
const tabReplayLink = document.getElementById("tab-replay-link");
const tabSimulateLink = document.getElementById("tab-simulate-link");
const tabReplayContent = document.getElementById("tab-replay-content");
const tabSimulateContent = document.getElementById("tab-simulate-content");

tabReplayLink.addEventListener("click", (e) => {
    e.preventDefault();
    tabReplayLink.classList.add("active", "text-success");
    tabReplayLink.classList.remove("text-muted");
    tabSimulateLink.classList.remove("active", "text-success");
    tabSimulateLink.classList.add("text-muted");
    
    tabReplayContent.classList.remove("d-none");
    tabSimulateContent.classList.add("d-none");
});

tabSimulateLink.addEventListener("click", (e) => {
    e.preventDefault();
    tabSimulateLink.classList.add("active", "text-success");
    tabSimulateLink.classList.remove("text-muted");
    tabReplayLink.classList.remove("active", "text-success");
    tabReplayLink.classList.add("text-muted");
    
    tabSimulateContent.classList.remove("d-none");
    tabReplayContent.classList.add("d-none");
});

// ResizeObserver to handle canvas resizing dynamically
const resizeObserver = new ResizeObserver(entries => {
    for (let entry of entries) {
        let width = entry.contentRect.width;
        let height = entry.contentRect.height;
        if (renderer && viewport) {
            renderer.resize(width, height);
            viewport.resize(width, height);
            
            // Re-center map if loaded
            if (ready && simulatorContainer) {
                let bounds = simulatorContainer.getBounds();
                simulatorContainer.pivot.set(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
                simulatorContainer.position.set(width / 2, height / 2);
            }
        }
    }
});
if (nodeCanvas) {
    resizeObserver.observe(nodeCanvas);
}

// Initial calls
loadRunsList();
startPollingStatus();

// Keep local clock updated
setInterval(() => {
    let now = new Date();
    document.getElementById("local-time").innerText = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}, 1000);
