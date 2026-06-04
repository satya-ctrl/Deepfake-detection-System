const { useRef, useEffect } = React;

const FadingVideo = ({ src, className, style }) => {
  const videoRef = useRef(null);
  const rafIdRef = useRef(null);
  const timeoutIdRef = useRef(null);
  const fadingOutRef = useRef(false);

  const FADE_MS = 500;
  const FADE_OUT_LEAD = 0.55;

  const fadeTo = (targetOpacity, duration) => {
    const video = videoRef.current;
    if (!video) return;

    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
    }

    const startOpacity = parseFloat(video.style.opacity) || 0;
    const opacityDiff = targetOpacity - startOpacity;
    if (opacityDiff === 0) return;

    const startTime = performance.now();

    const animate = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const currentOpacity = startOpacity + opacityDiff * progress;
      video.style.opacity = currentOpacity.toString();

      if (progress < 1) {
        rafIdRef.current = requestAnimationFrame(animate);
      } else {
        rafIdRef.current = null;
      }
    };

    rafIdRef.current = requestAnimationFrame(animate);
  };

  const handleLoadedData = () => {
    const video = videoRef.current;
    if (!video) return;

    video.style.opacity = '0';
    video.play().then(() => {
      fadeTo(1, FADE_MS);
    }).catch(e => {
      console.log("Play failed on loadeddata:", e);
    });
  };

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (!video || fadingOutRef.current) return;

    const duration = video.duration;
    const currentTime = video.currentTime;

    if (duration && (duration - currentTime <= FADE_OUT_LEAD) && (duration - currentTime > 0)) {
      fadingOutRef.current = true;
      fadeTo(0, FADE_MS);
    }
  };

  const handleEnded = () => {
    const video = videoRef.current;
    if (!video) return;

    video.style.opacity = '0';

    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
    }

    timeoutIdRef.current = setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.currentTime = 0;
        videoRef.current.play().then(() => {
          fadingOutRef.current = false;
          fadeTo(1, FADE_MS);
        }).catch(e => {
          console.log("Play failed on loop restart:", e);
        });
      }
    }, 100);
  };

  useEffect(() => {
    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
      if (timeoutIdRef.current) {
        clearTimeout(timeoutIdRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
      if (timeoutIdRef.current) clearTimeout(timeoutIdRef.current);
      fadingOutRef.current = false;
      video.style.opacity = '0';
      video.load();
    }
  }, [src]);

  return (
    <video
      ref={videoRef}
      className={className}
      style={{ ...style, opacity: 0 }}
      autoPlay
      muted
      playsInline
      preload="auto"
      onLoadedData={handleLoadedData}
      onTimeUpdate={handleTimeUpdate}
      onEnded={handleEnded}
    >
      <source src={src} type="video/mp4" />
    </video>
  );
};

window.FadingVideo = FadingVideo;
