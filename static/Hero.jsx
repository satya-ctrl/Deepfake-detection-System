const Hero = ({ setActiveTab }) => {
  const MotionDiv = window.Motion.motion.div;

  return (
    <section id="hero" className="relative min-h-screen w-full bg-black overflow-hidden flex flex-col justify-between z-10">
      {/* Background Video */}
      <window.FadingVideo 
        src="https://images.pexels.com/video-files/3130284/3130284-uhd_2560_1440_30fps.mp4"
        className="absolute left-1/2 top-0 -translate-x-1/2 object-cover object-top z-0 opacity-50"
        style={{ width: '120%', height: '120%' }}
      />

      {/* Hero Content */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center pt-32 pb-12 px-4 text-center">
        
        {/* Badge */}
        <MotionDiv
          initial={{ filter: 'blur(10px)', opacity: 0, y: 20 }}
          animate={{ filter: 'blur(0px)', opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.4 }}
          className="liquid-glass rounded-full p-1.5 flex items-center gap-3 pr-4 mb-6 max-w-lg select-none"
        >
          <span className="bg-white text-black px-3 py-1 text-xs font-semibold rounded-full tracking-wide">
            SYSTEM ACTIVE
          </span>
          <span className="text-sm text-white/90 font-body font-light tracking-wide">
            Secure Forensic Verification & Deepfake Auditing
          </span>
        </MotionDiv>

        {/* Headline */}
        <div className="w-full max-w-4xl px-2 flex flex-col items-center gap-2">
          <window.BlurText 
            text="Deep Sense"
            className="text-6xl md:text-7xl lg:text-[5.5rem] font-heading italic text-white leading-[0.8] tracking-[-4px]"
          />
          <window.BlurText 
            text="A Deepfake Detection Platform"
            className="text-4xl md:text-5xl lg:text-[3rem] font-heading italic text-white/95 leading-[0.8] tracking-[-2px]"
          />
        </div>

        {/* Subheading */}
        <MotionDiv
          initial={{ filter: 'blur(10px)', opacity: 0, y: 20 }}
          animate={{ filter: 'blur(0px)', opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.8 }}
          className="mt-6 text-sm md:text-base text-white/80 max-w-2xl font-body font-light leading-relaxed px-4"
        >
          Analyze media files for pixel inconsistencies, double-edge lighting anomalies, and blending artifacts. Access state-of-the-art forensic analysis in real time.
        </MotionDiv>

        {/* CTAs */}
        <MotionDiv
          initial={{ filter: 'blur(10px)', opacity: 0, y: 20 }}
          animate={{ filter: 'blur(0px)', opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 1.1 }}
          className="flex flex-wrap items-center justify-center gap-6 mt-8"
        >
          <button 
            onClick={() => setActiveTab('detector')}
            className="liquid-glass-strong rounded-full px-6 py-3 text-sm font-medium text-white flex items-center gap-2 hover:scale-[1.03] transition-transform duration-200 active:scale-95 cursor-pointer"
          >
            Launch Scanner HUD
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
              <path d="M7 17L17 7" />
              <path d="M7 7h10v10" />
            </svg>
          </button>
        </MotionDiv>

        {/* Stats Row */}
        <MotionDiv
          initial={{ filter: 'blur(10px)', opacity: 0, y: 20 }}
          animate={{ filter: 'blur(0px)', opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 1.3 }}
          className="flex flex-col sm:flex-row items-stretch justify-center gap-4 mt-12 w-full"
        >
          {/* Card 1 */}
          <div className="liquid-glass p-5 w-full sm:w-[220px] rounded-[1.25rem] flex flex-col justify-between text-left mx-auto sm:mx-0">
            <div className="mb-8">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-white shrink-0">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            </div>
            <div>
              <div className="text-4xl font-heading italic text-white tracking-[-1px] leading-none">94.25%</div>
              <div className="text-xs text-white/75 font-body font-light mt-2">CNN Verification Precision</div>
            </div>
          </div>

          {/* Card 2 */}
          <div className="liquid-glass p-5 w-full sm:w-[220px] rounded-[1.25rem] flex flex-col justify-between text-left mx-auto sm:mx-0">
            <div className="mb-8">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-white shrink-0">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <div>
              <div className="text-4xl font-heading italic text-white tracking-[-1px] leading-none">140K+</div>
              <div className="text-xs text-white/75 font-body font-light mt-2">Trained Dataset Library</div>
            </div>
          </div>
        </MotionDiv>
      </div>

      {/* Partners */}
      <MotionDiv
        initial={{ filter: 'blur(10px)', opacity: 0, y: 20 }}
        animate={{ filter: 'blur(0px)', opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: 'easeOut', delay: 1.4 }}
        className="relative z-10 flex flex-col items-center gap-4 pb-12"
      >
        <div className="liquid-glass rounded-full px-4 py-1.5 text-[11px] font-medium text-white/80 tracking-wide uppercase font-body select-none">
          Powered by modern deep learning & neural network frameworks
        </div>
        <div className="flex flex-wrap items-center justify-center gap-6 md:gap-12 text-white/95 font-heading italic text-2xl md:text-3xl tracking-tight select-none">
          <span>Python</span>
          <span className="text-white/20 text-sm font-normal font-body">·</span>
          <span>TensorFlow</span>
          <span className="text-white/20 text-sm font-normal font-body">·</span>
          <span>Keras</span>
          <span className="text-white/20 text-sm font-normal font-body">·</span>
          <span>Flask</span>
          <span className="text-white/20 text-sm font-normal font-body">·</span>
          <span>React</span>
          <span className="text-white/20 text-sm font-normal font-body">·</span>
          <span>Kaggle</span>
        </div>
      </MotionDiv>
    </section>
  );
};

window.Hero = Hero;
