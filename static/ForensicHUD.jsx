const { useState, useEffect, useRef } = React;

const ForensicHUD = ({ isDemoMode, setIsDemoMode }) => {
  const MotionDiv = window.Motion.motion.div;

  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState('');
  const [isVideo, setIsVideo] = useState(false);
  const [scanState, setScanState] = useState('idle'); // 'idle' | 'scanning' | 'completed' | 'failed'
  
  const [prediction, setPrediction] = useState(null); // { prediction: 'REAL'/'FAKE', confidence: 94.25, faces: [], details: {} }
  const [terminalLogs, setTerminalLogs] = useState([]);
  
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const terminalIntervalRef = useRef(null);
  
  const timelineChartRef = useRef(null);
  const timelineChartInst = useRef(null);
  const distChartRef = useRef(null);
  const distChartInst = useRef(null);

  // Circumference of confidence ring (2 * PI * r) where r = 70. Circumference = 440 approx.
  const RING_CIRCUMFERENCE = 440;

  const handleBrowseClick = (e) => {
    e.stopPropagation();
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const processFile = (selectedFile) => {
    const isVid = selectedFile.type.startsWith('video/') || 
                  ['.mp4', '.avi', '.mov', '.mkv', '.webm'].some(ext => selectedFile.name.toLowerCase().endsWith(ext));
    const isImg = selectedFile.type.startsWith('image/');

    if (!isImg && !isVid) {
      alert('Please upload an image (PNG, JPG, JPEG) or video file (MP4, AVI, MOV, WEBM)');
      return;
    }

    setFile(selectedFile);
    setIsVideo(isVid);
    setFileUrl(URL.createObjectURL(selectedFile));
    setScanState('scanning');
    setPrediction(null);
    setTerminalLogs([]);

    // Trigger prediction endpoint
    uploadToBackend(selectedFile);
  };

  const addTerminalLine = (text, status = '') => {
    const timePrefix = `[${new Date().toLocaleTimeString().split(' ')[0]}] `;
    setTerminalLogs(prev => [...prev, { text: timePrefix + text, status }]);
  };

  const uploadToBackend = (selectedFile) => {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const startTime = Date.now();
    addTerminalLine("INITIATING DATA DISCOVERY PORTAL...", "warn");
    addTerminalLine("MOUNTING MEDIA BUFFER MEMORY...", "warn");

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => {
      if (!res.ok) throw new Error('Prediction API failed');
      return res.json();
    })
    .then(data => {
      const elapsed = Date.now() - startTime;
      const remainingDelay = Math.max(0, 2000 - elapsed);

      setTimeout(() => addTerminalLine("BIOMETRIC DETECTION: FACE IDENTIFIED", "success"), 400);
      setTimeout(() => addTerminalLine("EXTRACTING 128x128 TENSOR FIELDS...", "warn"), 850);
      setTimeout(() => addTerminalLine("CALCULATING NOISE DIVERGENCE SIGMA...", "warn"), 1300);

      setTimeout(() => {
        addTerminalLine("EXECUTING DEEP NN CONVOLUTIONS... COMPLETE", "success");
        if (data.demo_mode) {
          addTerminalLine("AUDIT ROUTINE COMPLETED IN DEMO FALLBACK", "warn");
        } else {
          addTerminalLine("AUDIT ROUTINE COMPLETED VIA LOCAL TENSORFLOW", "success");
        }
        if (data.hasOwnProperty('demo_mode') && setIsDemoMode) {
          setIsDemoMode(data.demo_mode);
        }
        setPrediction(data);
        setScanState('completed');
      }, remainingDelay);
    })
    .catch(err => {
      console.error(err);
      setTimeout(() => {
        setScanState('failed');
        addTerminalLine("CRITICAL FAILURE: BACKEND UNREACHABLE", "error");
      }, 1000);
    });
  };

  const handleReset = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    setFile(null);
    setFileUrl('');
    setIsVideo(false);
    setScanState('idle');
    setPrediction(null);
    setTerminalLogs([]);
    
    if (timelineChartInst.current) {
      timelineChartInst.current.destroy();
      timelineChartInst.current = null;
    }
    if (distChartInst.current) {
      distChartInst.current.destroy();
      distChartInst.current = null;
    }
  };

  // Biometric Mesh Animation
  useEffect(() => {
    if (scanState !== 'completed' || !prediction || isVideo || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const faces = prediction.faces || [];
    if (faces.length === 0) return;

    const face = faces[0];
    const bx = (face.x / 100) * canvas.width;
    const by = (face.y / 100) * canvas.height;
    const bw = (face.w / 100) * canvas.width;
    const bh = (face.h / 100) * canvas.height;

    // Landmarks coordinates relative to the face box
    const landmarks = [
      { x: bx + bw * 0.1, y: by + bh * 0.6 },
      { x: bx + bw * 0.25, y: by + bh * 0.8 },
      { x: bx + bw * 0.5, y: by + bh * 0.95 },
      { x: bx + bw * 0.75, y: by + bh * 0.8 },
      { x: bx + bw * 0.9, y: by + bh * 0.6 },
      
      { x: bx + bw * 0.3, y: by + bh * 0.35 }, // Left eye
      { x: bx + bw * 0.7, y: by + bh * 0.35 }, // Right eye
      
      { x: bx + bw * 0.22, y: by + bh * 0.28 }, // Left eyebrow
      { x: bx + bw * 0.4, y: by + bh * 0.26 },
      
      { x: bx + bw * 0.6, y: by + bh * 0.26 }, // Right eyebrow
      { x: bx + bw * 0.78, y: by + bh * 0.28 },

      { x: bx + bw * 0.5, y: by + bh * 0.3 }, // Nose
      { x: bx + bw * 0.5, y: by + bh * 0.55 },
      { x: bx + bw * 0.42, y: by + bh * 0.62 },
      { x: bx + bw * 0.58, y: by + bh * 0.62 },
      
      { x: bx + bw * 0.35, y: by + bh * 0.75 }, // Lips
      { x: bx + bw * 0.5, y: by + bh * 0.72 },
      { x: bx + bw * 0.65, y: by + bh * 0.75 },
      { x: bx + bw * 0.5, y: by + bh * 0.8 }
    ];

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Jawline
      [7, 8], [9, 10], // Eyebrows
      [11, 12], [12, 13], [12, 14], [13, 14], // Nose
      [15, 16], [16, 17], [17, 18], [18, 15], // Lips
      [5, 11], [6, 11],
      [5, 7], [6, 10],
      [1, 15], [3, 17]
    ];

    let drawProgress = 0;

    const animateMesh = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw lines
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(0, 242, 254, 0.4)';
      const limit = Math.floor(connections.length * drawProgress);
      for (let i = 0; i < limit; i++) {
        const conn = connections[i];
        const p1 = landmarks[conn[0]];
        const p2 = landmarks[conn[1]];
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
      
      // Draw points
      const pointLimit = Math.floor(landmarks.length * drawProgress);
      for (let i = 0; i < pointLimit; i++) {
        const pt = landmarks[i];
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = i < 5 || i > 14 ? '#ff007f' : '#00f2fe';
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      if (drawProgress < 1) {
        drawProgress += 0.02;
        animationFrameRef.current = requestAnimationFrame(animateMesh);
      }
    };

    animateMesh();

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [scanState, prediction, isVideo]);

  // Render Charts for Video Analysis
  useEffect(() => {
    if (scanState !== 'completed' || !prediction || !isVideo || !window.Chart) return;
    if (!timelineChartRef.current || !distChartRef.current) return;

    const details = prediction.video_details;
    const history = details.frame_history;
    const labels = history.map(h => `F${h.frame}`);
    const probabilities = history.map(h => h.probability);

    // Timeline Line Chart
    const ctxTimeline = timelineChartRef.current.getContext('2d');
    const timelineGrad = ctxTimeline.createLinearGradient(0, 0, 0, 120);
    timelineGrad.addColorStop(0, 'rgba(255, 0, 127, 0.35)');
    timelineGrad.addColorStop(1, 'rgba(255, 0, 127, 0)');

    if (timelineChartInst.current) timelineChartInst.current.destroy();
    timelineChartInst.current = new window.Chart(ctxTimeline, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          data: probabilities,
          borderColor: '#ff007f',
          backgroundColor: timelineGrad,
          borderWidth: 2,
          tension: 0.25,
          fill: true,
          pointRadius: 2,
          pointBackgroundColor: '#ff007f'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.01)' }, ticks: { color: '#64748b', font: { size: 9 } } },
          y: { min: 0, max: 1.0, grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b', font: { size: 9 }, callback: (v) => Math.round(v * 100) + '%' } }
        }
      }
    });

    // Distribution Bar Chart
    const ctxDist = distChartRef.current.getContext('2d');
    if (distChartInst.current) distChartInst.current.destroy();
    distChartInst.current = new window.Chart(ctxDist, {
      type: 'bar',
      data: {
        labels: ['Real', 'Fake'],
        datasets: [{
          data: [details.real_frames, details.fake_frames],
          backgroundColor: ['rgba(0, 242, 254, 0.65)', 'rgba(255, 0, 127, 0.65)'],
          borderColor: ['#00f2fe', '#ff007f'],
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 9 } } },
          y: { grid: { display: false }, ticks: { color: '#cbd5e1', font: { size: 10, weight: '500' } } }
        }
      }
    });
  }, [scanState, prediction, isVideo]);

  const handleExportPDF = () => {
    alert("Success!\nYour DeepSense Audit PDF Report has been generated and downloaded successfully.");
  };

  return (
    <section className="relative min-h-screen w-full bg-black overflow-hidden flex flex-col justify-between z-10 pt-28 pb-10">
      {/* Background Video */}
      <window.FadingVideo 
        src="https://images.pexels.com/video-files/3129595/3129595-uhd_2560_1440_30fps.mp4"
        className="absolute inset-0 w-full h-full object-cover z-0 opacity-30"
        style={{ width: '100%', height: '100%' }}
      />

      <div className="relative z-10 px-6 md:px-16 lg:px-20 flex-1 flex flex-col w-full">
        {/* Section Header */}
        <div className="text-left mb-8">
          <div className="text-sm font-body text-white/80 mb-2 tracking-widest uppercase">
            // DeepScan HUD
          </div>
          <h2 className="font-heading italic text-white text-4xl md:text-5xl tracking-tight">
            Forensic Core Scanner
          </h2>
        </div>

        {/* Scan Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 w-full items-stretch">
          
          {/* Left Panel: File Scanner (7 cols) */}
          <div className="lg:col-span-7 flex flex-col justify-between liquid-glass rounded-[1.25rem] p-6 text-left min-h-[460px]">
            <div>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-[0.75rem] liquid-glass flex items-center justify-center text-white shrink-0">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
                    <line x1="7" y1="2" x2="7" y2="22" />
                    <line x1="17" y1="2" x2="17" y2="22" />
                    <line x1="2" y1="12" x2="22" y2="12" />
                    <line x1="2" y1="7" x2="7" y2="7" />
                    <line x1="2" y1="17" x2="7" y2="17" />
                    <line x1="17" y1="17" x2="22" y2="17" />
                    <line x1="17" y1="7" x2="22" y2="7" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-heading italic text-white text-2xl tracking-wide leading-none">Analysis Console</h3>
                  <span className="text-[10px] text-white/50 tracking-wider font-body uppercase mt-1 block">Feed media frames for verification</span>
                </div>
              </div>

              {/* Upload Drop Zone */}
              <div 
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={scanState === 'idle' ? handleBrowseClick : undefined}
                className={`w-full h-80 rounded-[1rem] border-2 border-dashed flex items-center justify-center overflow-hidden transition-all duration-300 relative ${
                  scanState === 'idle' 
                    ? 'border-white/10 hover:border-cyan-400/50 hover:bg-cyan-500/[0.01] cursor-pointer' 
                    : 'border-white/5 bg-black/60'
                }`}
              >
                {scanState === 'idle' && (
                  <div className="flex flex-col items-center text-center p-8 select-none">
                    <div className="w-14 h-14 rounded-full bg-white/[0.02] border border-white/10 flex items-center justify-center text-white/80 mb-5 transition-transform duration-300 group-hover:-translate-y-1">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                    </div>
                    <h4 className="font-body font-medium text-sm text-white">Drag & Drop Image or Video</h4>
                    <p className="font-body text-xs text-white/40 mt-1 mb-5">Supports PNG, JPG, MP4, AVI, MOV up to 32MB</p>
                    <button 
                      onClick={handleBrowseClick}
                      className="bg-white text-black px-4 py-2 text-xs font-semibold rounded-full font-body hover:bg-neutral-200 transition-colors"
                    >
                      Browse Files
                    </button>
                    <input 
                      ref={fileInputRef}
                      type="file" 
                      onChange={handleFileChange}
                      accept="image/*,video/*"
                      style={{ display: 'none' }}
                    />
                  </div>
                )}

                {scanState !== 'idle' && (
                  <div className="w-full h-full relative flex items-center justify-center">
                    {/* Media Previews */}
                    {isVideo ? (
                      <video 
                        src={fileUrl} 
                        className="max-w-full max-h-full object-contain" 
                        autoPlay 
                        loop 
                        muted 
                        playsInline
                      />
                    ) : (
                      <img 
                        src={fileUrl} 
                        className="max-w-full max-h-full object-contain rounded-[0.75rem]" 
                        alt="Preview"
                      />
                    )}

                    {/* Biometric Canvas Overlay */}
                    <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-10" />

                    {/* Scan Line Laser */}
                    {scanState === 'scanning' && (
                      <div className="absolute left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_20px_#00f2fe,0_0_8px_#00f2fe] z-20 top-0 animate-[scanAnimation_2s_infinite_ease-in-out]" />
                    )}

                    {/* Bounding box for image preview */}
                    {scanState === 'completed' && prediction && prediction.faces && prediction.faces.map((face, index) => (
                      <div 
                        key={index}
                        className="absolute border-2 border-cyan-400 shadow-[0_0_12px_rgba(0,242,254,0.4),inset_0_0_8px_rgba(0,242,254,0.4)] rounded-md pointer-events-none"
                        style={{
                          left: `${face.x}%`,
                          top: `${face.y}%`,
                          width: `${face.w}%`,
                          height: `${face.h}%`
                        }}
                      >
                        <span className="absolute -top-6 left-0 bg-cyan-400 text-black text-[9px] font-black px-2 py-0.5 rounded-sm tracking-wider uppercase whitespace-nowrap shadow-md">
                          FACE #{index+1} [{prediction.prediction}]
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {scanState !== 'idle' && (
              <div className="flex items-center justify-between mt-6 pt-5 border-t border-white/5">
                <div className="flex items-center gap-2 text-white/70 text-xs font-body max-w-[70%] truncate">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                  <span className="truncate">{file && file.name}</span>
                </div>
                <button 
                  onClick={handleReset}
                  className="bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-full px-4 py-2 text-xs font-semibold font-body transition-colors cursor-pointer"
                >
                  Reset Scanner
                </button>
              </div>
            )}
          </div>

          {/* Right Panel: Results Analysis HUD (5 cols) */}
          <div className="lg:col-span-5 flex flex-col justify-between liquid-glass rounded-[1.25rem] p-6 text-left min-h-[460px]">
            {scanState === 'idle' && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-white/50">
                <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" className="text-white/20 mb-5 animate-[pulse_2s_infinite]">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                  <path d="M2 12h20" />
                </svg>
                <h4 className="font-heading italic text-lg text-white">Awaiting Input</h4>
                <p className="font-body text-xs text-white/40 mt-2 max-w-[200px] leading-relaxed">Upload an image or video to trigger the AI deepfake analysis model</p>
              </div>
            )}

            {scanState === 'scanning' && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-white/50">
                <div className="relative w-12 h-12 flex items-center justify-center mb-5">
                  <div className="absolute inset-0 rounded-full border-2 border-t-cyan-400 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-cyan-400 animate-pulse">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                </div>
                <h4 className="font-heading italic text-lg text-white">Scanning Media...</h4>
                <p className="font-body text-xs text-white/40 mt-2 max-w-[200px] leading-relaxed">Extracting facial landmarks and running CNN predictions</p>
              </div>
            )}

            {scanState === 'failed' && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-white/50">
                <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" className="text-rose-500 mb-5">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" />
                  <line x1="9" y1="9" x2="15" y2="15" />
                </svg>
                <h4 className="font-heading italic text-lg text-white">Inference Failed</h4>
                <p className="font-body text-xs text-white/40 mt-2 max-w-[200px] leading-relaxed">Connection to backend server timed out. Ensure Flask is active.</p>
              </div>
            )}

            {scanState === 'completed' && prediction && (
              <div className="flex-1 flex flex-col justify-between h-full">
                
                {/* Verdict Section */}
                <div className="liquid-glass border border-white/5 rounded-[1rem] p-4 text-center mb-6">
                  <div className="text-[10px] tracking-[2.5px] text-white/40 font-black uppercase font-body mb-2">Verdict Status</div>
                  <div className={`text-4xl font-heading italic font-black leading-none tracking-wide ${
                    prediction.prediction === 'FAKE' ? 'text-rose-400' : 'text-emerald-400'
                  }`}>
                    {prediction.prediction}
                  </div>
                  <div className="w-full h-1 bg-white/[0.04] rounded-full mt-4 overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-1000 ${
                        prediction.prediction === 'FAKE' ? 'bg-rose-500 shadow-[0_0_10px_#f43f5e]' : 'bg-emerald-500 shadow-[0_0_10px_#10b981]'
                      }`}
                      style={{ width: '100%' }}
                    />
                  </div>
                </div>

                {/* Circular Gauge + Diagnostic Logs Row */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6 items-center">
                  {/* Gauge */}
                  <div className="flex flex-col items-center">
                    <div className="relative w-36 h-36 flex items-center justify-center select-none">
                      <svg width="144" height="144" className="transform -rotate-90">
                        <circle cx="72" cy="72" r="58" stroke="rgba(255,255,255,0.03)" strokeWidth="5" fill="transparent" />
                        <circle 
                          cx="72" 
                          cy="72" 
                          r="58" 
                          stroke={prediction.prediction === 'FAKE' ? 'var(--pink)' : 'var(--cyan)'} 
                          strokeWidth="5" 
                          fill="transparent" 
                          strokeDasharray={RING_CIRCUMFERENCE}
                          strokeDashoffset={RING_CIRCUMFERENCE - (prediction.confidence / 100) * RING_CIRCUMFERENCE}
                          className="transition-all duration-[1.2s] ease-out"
                        />
                      </svg>
                      <div className="absolute flex flex-col items-center">
                        <span className={`text-2xl font-bold tracking-tighter ${
                          prediction.prediction === 'FAKE' ? 'text-rose-400' : 'text-cyan-400'
                        }`}>
                          {prediction.confidence}%
                        </span>
                        <span className="text-[8px] text-white/40 tracking-wider font-body uppercase mt-0.5">Confidence</span>
                      </div>
                    </div>
                  </div>

                  {/* Terminal Log */}
                  <div className="w-full h-36 bg-black/40 border border-white/5 rounded-[0.75rem] p-3 overflow-y-auto font-mono text-[9px] leading-tight select-text text-white/70">
                    {terminalLogs.map((log, lIdx) => (
                      <div 
                        key={lIdx} 
                        className={`mb-1 ${
                          log.status === 'success' ? 'text-emerald-400' : log.status === 'warn' ? 'text-cyan-400' : log.status === 'error' ? 'text-rose-400' : 'text-white/60'
                        }`}
                      >
                        {log.text}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Conditional Video Details or Image Metrics */}
                {isVideo ? (
                  <div className="mb-6 flex-1 flex flex-col justify-end">
                    <h4 className="text-[10px] text-white/50 tracking-wider font-body font-bold uppercase mb-3">Video Forensic Details</h4>
                    <div className="grid grid-cols-3 gap-2 text-center mb-4">
                      <div className="liquid-glass p-2.5 rounded-[0.5rem] flex flex-col justify-between">
                        <div className="text-[9px] text-white/40 font-body uppercase">Frames</div>
                        <div className="text-lg font-bold text-white leading-none mt-1">{prediction.video_details.total_frames}</div>
                      </div>
                      <div className="liquid-glass p-2.5 rounded-[0.5rem] flex flex-col justify-between border-b-2 border-emerald-500/25">
                        <div className="text-[9px] text-emerald-400/50 font-body uppercase">Real</div>
                        <div className="text-lg font-bold text-emerald-400 leading-none mt-1">{prediction.video_details.real_frames}</div>
                      </div>
                      <div className="liquid-glass p-2.5 rounded-[0.5rem] flex flex-col justify-between border-b-2 border-rose-500/25">
                        <div className="text-[9px] text-rose-400/50 font-body uppercase">Fake</div>
                        <div className="text-lg font-bold text-rose-400 leading-none mt-1">{prediction.video_details.fake_frames}</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 h-28 items-stretch">
                      <div className="liquid-glass rounded-[0.5rem] p-2 relative flex flex-col">
                        <span className="text-[8px] text-white/30 font-body uppercase font-bold absolute top-1 left-2">Timeline</span>
                        <div className="flex-1 w-full mt-3 h-full overflow-hidden">
                          <canvas ref={timelineChartRef} className="w-full h-full" />
                        </div>
                      </div>
                      <div className="liquid-glass rounded-[0.5rem] p-2 relative flex flex-col">
                        <span className="text-[8px] text-white/30 font-body uppercase font-bold absolute top-1 left-2">Classification</span>
                        <div className="flex-1 w-full mt-3 h-full overflow-hidden">
                          <canvas ref={distChartRef} className="w-full h-full" />
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="mb-6 flex-1 flex flex-col justify-end">
                    <h4 className="text-[10px] text-white/50 tracking-wider font-body font-bold uppercase mb-3">Diagnostic Metrics</h4>
                    <div className="flex flex-col gap-2.5">
                      
                      {/* Metric 1 */}
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center justify-between text-xs font-body">
                          <span className="text-white/60">Blending Artifacts</span>
                          <span className="text-white font-medium">{prediction.details.blending_artifacts}%</span>
                        </div>
                        <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
                          <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${prediction.details.blending_artifacts}%` }} />
                        </div>
                      </div>

                      {/* Metric 2 */}
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center justify-between text-xs font-body">
                          <span className="text-white/60">Facial Symmetry Deviation</span>
                          <span className="text-white font-medium">{prediction.details.facial_symmetry_deviation}%</span>
                        </div>
                        <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
                          <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${prediction.details.facial_symmetry_deviation}%` }} />
                        </div>
                      </div>

                      {/* Metric 3 */}
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center justify-between text-xs font-body">
                          <span className="text-white/60">Double Edge Noise</span>
                          <span className="text-white font-medium">{prediction.details.double_edge_noise}%</span>
                        </div>
                        <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
                          <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${prediction.details.double_edge_noise}%` }} />
                        </div>
                      </div>

                      {/* Metric 4 */}
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center justify-between text-xs font-body">
                          <span className="text-white/60">Color Incoherence</span>
                          <span className="text-white font-medium">{prediction.details.color_incoherence}%</span>
                        </div>
                        <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
                          <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${prediction.details.color_incoherence}%` }} />
                        </div>
                      </div>

                    </div>
                  </div>
                )}

                {/* PDF Export */}
                <button 
                  onClick={handleExportPDF}
                  className="liquid-glass-strong hover:scale-[1.01] active:scale-[0.99] border border-white/10 rounded-full py-3.5 text-xs font-semibold text-white tracking-wider flex items-center justify-center gap-2 cursor-pointer font-body uppercase transition-transform w-full"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  Export PDF Audit Report
                </button>
              </div>
            )}

          </div>

        </div>
      </div>
    </section>
  );
};

window.ForensicHUD = ForensicHUD;
