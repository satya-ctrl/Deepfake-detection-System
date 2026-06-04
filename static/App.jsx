const { useState, useEffect } = React;

// Suppress benign Framer Motion / key warnings
const originalError = console.error;
console.error = (...args) => {
  if (
    args[0] &&
    typeof args[0] === 'string' &&
    (args[0].includes('Framer Motion') || args[0].includes('key') || args[0].includes('React does not recognize'))
  ) {
    return;
  }
  originalError(...args);
};

const App = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [isDemoMode, setIsDemoMode] = useState(true);
  const [showToast, setShowToast] = useState(true);

  // Check initial demo mode from database status or history
  useEffect(() => {
    fetch('/history')
      .then(res => res.json())
      .then(data => {
        if (data && data.length > 0) {
          // If there are entries, take the demo mode of the latest entry if present
          if (data[0].details && data[0].details.hasOwnProperty('demo_mode')) {
            setIsDemoMode(data[0].details.demo_mode);
          }
        }
      })
      .catch(err => console.log("Init status check:", err));
  }, []);

  return (
    <div className="relative bg-black w-full min-h-screen">
      {/* Fixed Navbar */}
      <window.Navbar activeTab={activeTab} setActiveTab={setActiveTab} isDemoMode={isDemoMode} />

      {/* Render Active Tab */}
      {activeTab === 'home' && <window.Hero setActiveTab={setActiveTab} />}
      {activeTab === 'detector' && <window.ForensicHUD isDemoMode={isDemoMode} setIsDemoMode={setIsDemoMode} />}
      {activeTab === 'analytics' && <window.Analytics />}
      {activeTab === 'dataset' && <window.DatasetHub />}
      {activeTab === 'history' && <window.History setActiveTab={setActiveTab} />}

      {/* Onboarding Alert for Demo Mode */}
      {isDemoMode && showToast && activeTab !== 'home' && (
        <div className="fixed bottom-4 right-4 z-50 bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-[1rem] p-4 max-w-sm flex items-start gap-3 shadow-lg backdrop-blur-md">
          <div className="shrink-0 w-8 h-8 rounded-full bg-rose-500/20 flex items-center justify-center text-rose-400">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </div>
          <div>
            <h4 className="text-xs font-bold font-body text-white uppercase">Running in Demonstration Mode</h4>
            <p className="text-[11px] font-body font-light text-white/70 mt-1 leading-relaxed">
              The app is using simulated predictions because <code>deepfake_model.h5</code> was not detected in the root folder. Go to the <strong>Dataset Hub</strong> tab to learn how to train and link your neural network model!
            </p>
          </div>
          <button 
            onClick={() => setShowToast(false)} 
            className="text-white/40 hover:text-white text-sm shrink-0 font-bold font-body cursor-pointer self-start"
          >
            &times;
          </button>
        </div>
      )}
    </div>
  );
};

// Mount the React Application
const rootElement = document.getElementById('root');
const root = ReactDOM.createRoot(rootElement);
root.render(<App />);

window.App = App;
