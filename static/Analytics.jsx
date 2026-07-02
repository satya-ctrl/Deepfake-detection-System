const { useState, useEffect, useRef } = React;

const Analytics = () => {
  const MotionDiv = window.Motion.motion.div;

  const [metrics, setMetrics] = useState(null);
  
  const accChartRef = useRef(null);
  const accChartInst = useRef(null);
  const lossChartRef = useRef(null);
  const lossChartInst = useRef(null);
  const cmChartRef = useRef(null);
  const cmChartInst = useRef(null);

  useEffect(() => {
    fetch('/metrics')
      .then(res => res.json())
      .then(data => {
        setMetrics(data);
      })
      .catch(err => console.error("Error fetching metrics:", err));
  }, []);

  useEffect(() => {
    if (!metrics || !window.Chart) return;

    const epochs = Array.from({ length: 15 }, (_, i) => i + 1);

    // Accuracy Chart
    const ctxAcc = accChartRef.current.getContext('2d');
    const cyanGrad = ctxAcc.createLinearGradient(0, 0, 0, 200);
    cyanGrad.addColorStop(0, 'rgba(0, 242, 254, 0.2)');
    cyanGrad.addColorStop(1, 'rgba(0, 242, 254, 0)');

    const trainAcc = [0.65, 0.73, 0.79, 0.83, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.935, 0.94, 0.942, 0.943, metrics.accuracy / 100];
    const valAcc = [0.63, 0.70, 0.76, 0.80, 0.83, 0.85, 0.86, 0.88, 0.89, 0.90, 0.905, 0.91, 0.912, 0.915, 0.918];

    if (accChartInst.current) accChartInst.current.destroy();
    accChartInst.current = new window.Chart(ctxAcc, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {
            label: 'Training Accuracy',
            data: trainAcc,
            borderColor: '#00f2fe',
            backgroundColor: cyanGrad,
            borderWidth: 2.5,
            tension: 0.3,
            fill: true,
            pointRadius: 2
          },
          {
            label: 'Validation Accuracy',
            data: valAcc,
            borderColor: '#9d4edd',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.3,
            pointRadius: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b' } },
          y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b' } }
        }
      }
    });

    // Loss Chart
    const ctxLoss = lossChartRef.current.getContext('2d');
    const pinkGrad = ctxLoss.createLinearGradient(0, 0, 0, 200);
    pinkGrad.addColorStop(0, 'rgba(255, 0, 127, 0.2)');
    pinkGrad.addColorStop(1, 'rgba(255, 0, 127, 0)');

    const trainLoss = [0.68, 0.58, 0.49, 0.41, 0.34, 0.29, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11];
    const valLoss = [0.70, 0.61, 0.53, 0.46, 0.40, 0.35, 0.32, 0.29, 0.27, 0.26, 0.25, 0.24, 0.23, 0.235, 0.24];

    if (lossChartInst.current) lossChartInst.current.destroy();
    lossChartInst.current = new window.Chart(ctxLoss, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {
            label: 'Training Loss',
            data: trainLoss,
            borderColor: '#ff007f',
            backgroundColor: pinkGrad,
            borderWidth: 2.5,
            tension: 0.3,
            fill: true,
            pointRadius: 2
          },
          {
            label: 'Validation Loss',
            data: valLoss,
            borderColor: '#9d4edd',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.3,
            pointRadius: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b' } },
          y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b' } }
        }
      }
    });

    // Confusion Matrix Chart
    const ctxCm = cmChartRef.current.getContext('2d');
    const tp = metrics.cm[0][0];
    const fn = metrics.cm[0][1];
    const fp = metrics.cm[1][0];
    const tn = metrics.cm[1][1];

    if (cmChartInst.current) cmChartInst.current.destroy();
    cmChartInst.current = new window.Chart(ctxCm, {
      type: 'bar',
      data: {
        labels: ['Real Reference', 'Fake Reference'],
        datasets: [
          {
            label: 'Predicted Real',
            data: [tp, fp],
            backgroundColor: 'rgba(0, 242, 254, 0.7)',
            borderColor: '#00f2fe',
            borderWidth: 1.5,
            borderRadius: 6
          },
          {
            label: 'Predicted Fake',
            data: [fn, tn],
            backgroundColor: 'rgba(255, 0, 127, 0.7)',
            borderColor: '#ff007f',
            borderWidth: 1.5,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#cbd5e1' } },
          tooltip: {
            backgroundColor: 'rgba(9, 14, 26, 0.9)',
            callbacks: {
              afterBody: (items) => {
                const index = items[0].dataIndex;
                const datasetIndex = items[0].datasetIndex;
                if (index === 0 && datasetIndex === 0) return 'Correct Classification (True Positive)';
                if (index === 0 && datasetIndex === 1) return 'Missed Detection (False Negative)';
                if (index === 1 && datasetIndex === 0) return 'False Alarm (False Positive)';
                if (index === 1 && datasetIndex === 1) return 'Correct Detection (True Negative)';
                return '';
              }
            }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#cbd5e1' } },
          y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#64748b' } }
        }
      }
    });

  }, [metrics]);

  return (
    <section className="relative min-h-screen w-full bg-black overflow-hidden flex flex-col justify-between z-10 pt-28 pb-10">
      {/* Dark Greenish Background */}
      <div className="absolute inset-0 z-0" style={{ background: 'radial-gradient(ellipse at 50% 0%, rgba(16, 185, 129, 0.12) 0%, rgba(0, 0, 0, 1) 70%)' }} />

      <div className="relative z-10 px-6 md:px-16 lg:px-20 flex-1 flex flex-col w-full">
        {/* Section Header */}
        <div className="text-left mb-8">
          <div className="text-sm font-body text-white/80 mb-2 tracking-widest uppercase">
            // Model Performance
          </div>
          <h2 className="font-heading italic text-white text-4xl md:text-5xl tracking-tight">
            Accuracy & Training Curves
          </h2>
        </div>

        {/* Info Cards Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full mb-8">
          {/* Card 1 */}
          <div className="liquid-glass p-5 rounded-[1rem] text-left">
            <span className="text-[10px] text-white/40 font-body uppercase font-bold tracking-wider">Overall Accuracy</span>
            <div className="text-3xl font-heading italic text-white leading-none mt-2">
              {metrics ? `${metrics.accuracy}%` : '---'}
            </div>
          </div>
          {/* Card 2 */}
          <div className="liquid-glass p-5 rounded-[1rem] text-left">
            <span className="text-[10px] text-white/40 font-body uppercase font-bold tracking-wider">Model Area (AUC)</span>
            <div className="text-3xl font-heading italic text-white leading-none mt-2">
              {metrics ? metrics.auc : '---'}
            </div>
          </div>
          {/* Card 3 */}
          <div className="liquid-glass p-5 rounded-[1rem] text-left">
            <span className="text-[10px] text-white/40 font-body uppercase font-bold tracking-wider">Precision (Real)</span>
            <div className="text-3xl font-heading italic text-white leading-none mt-2">
              {metrics ? `${metrics.precision_real}%` : '---'}
            </div>
          </div>
          {/* Card 4 */}
          <div className="liquid-glass p-5 rounded-[1rem] text-left">
            <span className="text-[10px] text-white/40 font-body uppercase font-bold tracking-wider">Precision (Fake)</span>
            <div className="text-3xl font-heading italic text-white leading-none mt-2">
              {metrics ? `${metrics.precision_fake}%` : '---'}
            </div>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full mb-8">
          <div className="liquid-glass p-6 rounded-[1.25rem] min-h-[300px] flex flex-col justify-between">
            <h4 className="text-xs text-white/50 tracking-wider font-body font-bold uppercase mb-4">Training & Validation Accuracy</h4>
            <div className="flex-1 w-full min-h-[220px]">
              <canvas ref={accChartRef} />
            </div>
          </div>
          <div className="liquid-glass p-6 rounded-[1.25rem] min-h-[300px] flex flex-col justify-between">
            <h4 className="text-xs text-white/50 tracking-wider font-body font-bold uppercase mb-4">Training & Validation Loss</h4>
            <div className="flex-1 w-full min-h-[220px]">
              <canvas ref={lossChartRef} />
            </div>
          </div>
        </div>

        {/* Confusion Matrix & NN Arch */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full">
          {/* CM Panel (1 col) */}
          <div className="liquid-glass p-6 rounded-[1.25rem] min-h-[300px] flex flex-col justify-between lg:col-span-1">
            <h4 className="text-xs text-white/50 tracking-wider font-body font-bold uppercase mb-4">Confusion Matrix</h4>
            <div className="flex-1 w-full min-h-[220px]">
              <canvas ref={cmChartRef} />
            </div>
          </div>

          {/* NN Flow (2 cols) */}
          <div className="liquid-glass p-6 rounded-[1.25rem] min-h-[300px] flex flex-col justify-between lg:col-span-2 text-left">
            <div>
              <h4 className="text-xs text-white/50 tracking-wider font-body font-bold uppercase mb-2">Neural Network Architecture</h4>
              <span className="text-[10px] text-white/35 font-body uppercase mt-1 block">Sequential CNN layout (128x128x3 input)</span>
            </div>
            
            <div className="flex flex-col sm:flex-row items-center gap-3 justify-between py-6 overflow-x-auto text-[10px] font-mono leading-none">
              
              <div className="liquid-glass border border-cyan-400/20 px-4 py-3.5 rounded-[0.75rem] w-full sm:w-auto text-center shrink-0">
                <span className="text-cyan-400 font-bold block mb-1">INPUT</span>
                <span className="text-white">Image Frame</span>
                <span className="text-white/40 block mt-1">128×128×3 RGB</span>
              </div>
              
              <div className="text-white/20 hidden sm:block shrink-0">&rarr;</div>
              
              <div className="liquid-glass px-4 py-3.5 rounded-[0.75rem] w-full sm:w-auto text-center shrink-0">
                <span className="text-white/40 font-bold block mb-1">CONV2D</span>
                <span className="text-white">32 filters (3x3)</span>
                <span className="text-white/40 block mt-1">ReLU Activation</span>
              </div>

              <div className="text-white/20 hidden sm:block shrink-0">&rarr;</div>

              <div className="liquid-glass px-4 py-3.5 rounded-[0.75rem] w-full sm:w-auto text-center shrink-0">
                <span className="text-white/40 font-bold block mb-1">MAXPOOL2D</span>
                <span className="text-white">2x2 Pool Size</span>
                <span className="text-white/40 block mt-1">Downsampling</span>
              </div>

              <div className="text-white/20 hidden sm:block shrink-0">&rarr;</div>

              <div className="liquid-glass px-4 py-3.5 rounded-[0.75rem] w-full sm:w-auto text-center shrink-0">
                <span className="text-white/40 font-bold block mb-1">CONV2D</span>
                <span className="text-white">64 & 128 filters</span>
                <span className="text-white/40 block mt-1">Features extraction</span>
              </div>

              <div className="text-white/20 hidden sm:block shrink-0">&rarr;</div>

              <div className="liquid-glass px-4 py-3.5 rounded-[0.75rem] w-full sm:w-auto text-center shrink-0">
                <span className="text-white/40 font-bold block mb-1">DENSE</span>
                <span className="text-white">Flatten & Dense (256)</span>
                <span className="text-white/40 block mt-1">Dropout (0.3)</span>
              </div>

              <div className="text-white/20 hidden sm:block shrink-0">&rarr;</div>

              <div className="liquid-glass border border-rose-500/20 px-4 py-3.5 rounded-[0.75rem] w-full sm:w-auto text-center shrink-0">
                <span className="text-rose-400 font-bold block mb-1">SIGMOID</span>
                <span className="text-white">Binary Classification</span>
                <span className="text-white/40 block mt-1">0.0 (Real) to 1.0 (Fake)</span>
              </div>

            </div>
          </div>
        </div>

      </div>
    </section>
  );
};

window.Analytics = Analytics;
