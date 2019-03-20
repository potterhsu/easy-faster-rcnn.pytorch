const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const URI = "ws://127.0.0.1:8765";

let video = document.getElementById("video");
let videoFile = document.getElementById("video-file");
let outputCanvas = document.getElementById("output-canvas");
let URL = window.URL || window.webkitURL;
let videoCapture = null;
let rgbFrame = null;
let rgbaFrame = null;
let websocket;
let shouldContinue = false;
let sn;
let period;

let Module = {
    locateFile: function (name) {
        let files = {
            "opencv_js.wasm": "/opencv/opencv_js.wasm"
        };
        return files[name]
    },
    preRun: [() => {}],
    postRun: [
        init
    ]
};

window.addEventListener("load", function() {
    websocket = new WebSocket(URI);
    websocket.onopen = function(event) {
        console.log("CONNECTED");
    };
    websocket.onclose = function(event) {
        console.log("DISCONNECTED");
    };
    websocket.onmessage = function(event) {
        onResult(event.data);
    };
    websocket.onerror = function(event) {
        console.log("ERROR: " + event.data);
    };
}, false);

window.addEventListener("unload", function() {
    websocket.close();
}, false);

function init() {
    videoCapture = new cv.VideoCapture(video);
    rgbaFrame = new cv.Mat(VIDEO_HEIGHT, VIDEO_WIDTH, cv.CV_8UC4);
    rgbFrame = new cv.Mat(VIDEO_HEIGHT, VIDEO_WIDTH, cv.CV_8UC3);
}

async function start() {
    if (video.paused) {
        switch (document.querySelector('input[name="source"]:checked').value) {
            case "camera":
                video.srcObject = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: {
                            exact: VIDEO_WIDTH
                        },
                        height: {
                            exact: VIDEO_HEIGHT
                        }
                    },
                    audio: false
                });
                break;
            case "file":
                video.src = URL.createObjectURL(videoFile.files[0]);
                break;
            default:
                break;
        }

        video.play();
        shouldContinue = true;
        sn = 1;
        period = parseInt(document.getElementById("period").value);
        requestAnimationFrame(onFrame);
    }
}

function stop() {
    video.pause();
    video.srcObject = null;
    video.src = "";
    shouldContinue = false;
}

function onFrame() {
    if (!shouldContinue)
        return;

    if (websocket.readyState === 1) {
        videoCapture.read(rgbaFrame);  // format RGBA
        cv.cvtColor(rgbaFrame, rgbFrame, cv.COLOR_RGBA2RGB);

        if (sn % period === 0) {
            websocket.send(rgbFrame.data);
        }

        sn += 1;
    }

    requestAnimationFrame(onFrame);
}

function onResult(message) {
    let results = JSON.parse(message);
    for (let i in results) {
        let result = results[i];
        let point1 = new cv.Point(result.left, result.top);
        let point2 = new cv.Point(result.right, result.bottom);
        cv.rectangle(rgbFrame, point1, point2, [0, 255, 0, 255]);
        cv.putText(rgbFrame, result.category, point1, 1, 1, [0, 255, 0, 255]);
    }
    cv.imshow(outputCanvas, rgbFrame);
}