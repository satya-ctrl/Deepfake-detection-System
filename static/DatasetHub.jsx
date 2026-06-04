const DatasetHub = () => {
  const MotionDiv = window.Motion.motion.div;

  return (
    <section className="relative min-h-screen w-full bg-black overflow-hidden flex flex-col justify-between z-10 pt-28 pb-10">
      {/* Background Video */}
      <window.FadingVideo 
        src="https://images.pexels.com/video-files/3129595/3129595-uhd_2560_1440_30fps.mp4"
        className="absolute inset-0 w-full h-full object-cover z-0 opacity-30"
        style={{ width: '100%', height: '100%' }}
      />

      <div className="relative z-10 px-6 md:px-16 lg:px-20 flex-1 flex flex-col w-full text-left">
        {/* Section Header */}
        <div className="mb-8">
          <div className="text-sm font-body text-white/80 mb-2 tracking-widest uppercase">
            // Dataset Onboarding
          </div>
          <h2 className="font-heading italic text-white text-4xl md:text-5xl tracking-tight">
            Expand & Train Your Model
          </h2>
        </div>

        {/* Intro */}
        <div className="liquid-glass p-6 rounded-[1.25rem] w-full mb-8">
          <h3 className="font-heading italic text-2xl text-white mb-2">Biometric Diversity Training</h3>
          <p className="font-body text-sm text-white/70 leading-relaxed max-w-4xl">
            To make your deepfake detector highly robust, you need to train it on a larger diversity of images. We have configured the <strong>140k Real & Fake Faces</strong> dataset to fit your existing pipeline perfectly. Follow the three steps below to expand the dataset, run GPU training, and deploy the new weights locally.
          </p>
        </div>

        {/* 3 Step Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
          
          {/* Step 1 */}
          <div className="liquid-glass p-6 rounded-[1.25rem] flex flex-col justify-between min-h-[380px]">
            <div>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-[0.75rem] liquid-glass flex items-center justify-center text-white shrink-0">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </div>
                <h4 className="font-heading italic text-xl text-white">1. Download Dataset</h4>
              </div>
              <p className="font-body text-xs text-white/75 leading-relaxed mb-6">
                We created an automated Kaggle downloading script. In your project root terminal, execute:
              </p>
              <div className="bg-black/50 border border-white/10 rounded-[0.5rem] p-3 font-mono text-xs text-cyan-400 select-all mb-4">
                python download_dataset.py
              </div>
              <p className="font-body text-[11px] text-white/40 leading-snug">
                This utility will authenticate with your Kaggle API, download the 1.2 GB dataset, and structure files to match the expected training directories automatically.
              </p>
            </div>
          </div>

          {/* Step 2 */}
          <div className="liquid-glass p-6 rounded-[1.25rem] flex flex-col justify-between min-h-[380px]">
            <div>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-[0.75rem] liquid-glass flex items-center justify-center text-white shrink-0">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                </div>
                <h4 className="font-heading italic text-xl text-white">2. GPU Acceleration</h4>
              </div>
              <p className="font-body text-xs text-white/75 leading-relaxed mb-4">
                Since training 140,000 images on local CPU takes a long time, we recommend training in Google Colab using a free GPU:
              </p>
              <ol className="list-decimal pl-4 font-body text-[11px] text-white/60 leading-relaxed flex flex-col gap-2">
                <li>Compress the downloaded <code>Dataset/</code> folder into a ZIP named <strong><code>Dataset.zip</code></strong>.</li>
                <li>Upload <code>Dataset.zip</code> to your main Google Drive directory.</li>
                <li>Open <code>Deepfake.ipynb</code> in Google Colab and run it to mount your Drive and extract the ZIP.</li>
              </ol>
            </div>
          </div>

          {/* Step 3 */}
          <div className="liquid-glass p-6 rounded-[1.25rem] flex flex-col justify-between min-h-[380px]">
            <div>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-[0.75rem] liquid-glass flex items-center justify-center text-white shrink-0">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                  </svg>
                </div>
                <h4 className="font-heading italic text-xl text-white">3. Link Local Model</h4>
              </div>
              <p className="font-body text-xs text-white/75 leading-relaxed mb-4">
                Append and execute this cell at the end of your Colab notebook to download the trained weights file:
              </p>
              <div className="bg-black/50 border border-white/10 rounded-[0.5rem] p-3 font-mono text-[10px] text-cyan-400 select-all mb-4 leading-normal whitespace-pre">
{`model.save("deepfake_model.h5")
from google.colab import files
files.download("deepfake_model.h5")`}
              </div>
              <p className="font-body text-[11px] text-white/40 leading-snug">
                Place <strong><code>deepfake_model.h5</code></strong> inside the project's root folder and restart the Flask server. The status indicator will transition to <strong>LIVE AI MODE</strong>!
              </p>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

window.DatasetHub = DatasetHub;
