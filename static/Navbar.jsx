const Navbar = ({ activeTab, setActiveTab, isDemoMode }) => {
  return (
    <nav className="fixed top-4 left-0 right-0 w-full z-50 flex items-center justify-between px-8 lg:px-16">
      {/* Left: Brand Logo */}
      <div 
        onClick={() => setActiveTab('home')}
        className="w-12 h-12 rounded-full liquid-glass flex items-center justify-center text-white font-heading italic text-[1.75rem] leading-none select-none cursor-pointer hover:scale-105 transition-transform duration-200"
      >
        D
      </div>

      {/* Center: Navigation Pill */}
      <div className="flex items-center gap-1 liquid-glass rounded-full px-1.5 py-1.5">
        <button 
          onClick={() => setActiveTab('home')}
          className={`px-3.5 py-2 text-xs md:text-sm font-medium font-body rounded-full transition-all duration-200 ${
            activeTab === 'home' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white'
          }`}
        >
          Home
        </button>
        <button 
          onClick={() => setActiveTab('detector')}
          className={`px-3.5 py-2 text-xs md:text-sm font-medium font-body rounded-full transition-all duration-200 ${
            activeTab === 'detector' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white'
          }`}
        >
          DeepScan HUD
        </button>
        <button 
          onClick={() => setActiveTab('analytics')}
          className={`px-3.5 py-2 text-xs md:text-sm font-medium font-body rounded-full transition-all duration-200 ${
            activeTab === 'analytics' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white'
          }`}
        >
          Model Analytics
        </button>
        <button 
          onClick={() => setActiveTab('dataset')}
          className={`px-3.5 py-2 text-xs md:text-sm font-medium font-body rounded-full transition-all duration-200 ${
            activeTab === 'dataset' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white'
          }`}
        >
          Dataset Hub
        </button>
        <button 
          onClick={() => setActiveTab('history')}
          className={`px-3.5 py-2 text-xs md:text-sm font-medium font-body rounded-full transition-all duration-200 ${
            activeTab === 'history' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white'
          }`}
        >
          Audit History
        </button>
      </div>

      {/* Right: Status Pill */}
      <div className={`px-4 py-2 text-xs font-semibold rounded-full font-body select-none flex items-center gap-2 ${
        isDemoMode ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
      }`}>
        <span className={`w-2 h-2 rounded-full ${isDemoMode ? 'bg-rose-400' : 'bg-emerald-400'}`}></span>
        <span className="hidden sm:inline">{isDemoMode ? 'DEMO MODE' : 'LIVE AI MODE'}</span>
        <span className="sm:hidden">{isDemoMode ? 'DEMO' : 'LIVE'}</span>
      </div>
    </nav>
  );
};

window.Navbar = Navbar;
