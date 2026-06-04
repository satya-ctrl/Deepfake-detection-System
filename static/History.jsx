const { useState, useEffect } = React;

const History = ({ setActiveTab }) => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = () => {
    setLoading(true);
    fetch('/history')
      .then(res => res.json())
      .then(data => {
        setLogs(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching history:", err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleClearAll = () => {
    if (!confirm("Are you sure you want to clear all audit logs from the database?")) return;
    fetch('/history/clear', { method: 'POST' })
      .then(res => res.json())
      .then(() => {
        setLogs([]);
      })
      .catch(err => console.error("Error clearing logs:", err));
  };

  const handleDelete = (id) => {
    if (!confirm("Are you sure you want to delete this log entry?")) return;
    fetch(`/history/delete/${id}`, { method: 'POST' })
      .then(res => res.json())
      .then(() => {
        setLogs(prev => prev.filter(log => log.id !== id));
      })
      .catch(err => console.error("Error deleting log:", err));
  };

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
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
          <div>
            <div className="text-sm font-body text-white/80 mb-2 tracking-widest uppercase">
              // Database Audit Logs
            </div>
            <h2 className="font-heading italic text-white text-4xl md:text-5xl tracking-tight">
              Forensic Scan History
            </h2>
          </div>
          
          {logs.length > 0 && (
            <button 
              onClick={handleClearAll}
              className="bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20 rounded-full px-5 py-2.5 text-xs font-semibold font-body uppercase tracking-wider flex items-center gap-2 cursor-pointer transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                <line x1="10" y1="11" x2="10" y2="17" />
                <line x1="14" y1="11" x2="14" y2="17" />
              </svg>
              Clear Audit Logs
            </button>
          )}
        </div>

        {/* Database Table Panel */}
        <div className="liquid-glass rounded-[1.25rem] p-6 w-full min-h-[350px] flex flex-col">
          {loading ? (
            <div className="flex-1 flex flex-col items-center justify-center text-white/40 text-sm font-body gap-3">
              <div className="w-8 h-8 rounded-full border-2 border-t-cyan-400 border-r-transparent border-b-transparent border-l-transparent animate-spin" />
              Retrieving database logs...
            </div>
          ) : logs.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-white/40">
              <div className="w-14 h-14 rounded-full bg-white/[0.02] border border-white/10 flex items-center justify-center text-white/20 mb-5">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <h4 className="font-heading italic text-lg text-white">No Records Identified</h4>
              <p className="font-body text-xs text-white/40 mt-1 max-w-[280px] leading-relaxed">
                Scan an image or video inside the DeepScan HUD to record forensic predictions in the local database.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto w-full">
              <table className="w-full text-left font-body text-xs text-white/80 border-collapse">
                <thead>
                  <tr className="border-b border-white/5 text-white/40 font-bold uppercase tracking-wider text-[10px]">
                    <th className="py-4 px-4">Timestamp</th>
                    <th className="py-4 px-4">File Checked</th>
                    <th className="py-4 px-4">Media Type</th>
                    <th className="py-4 px-4">AI Verdict</th>
                    <th className="py-4 px-4">Confidence</th>
                    <th className="py-4 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((audit) => {
                    const cleanName = audit.filename.substring(audit.filename.indexOf('_') + 1);
                    return (
                      <tr key={audit.id} className="border-b border-white/5 hover:bg-white/[0.01] transition-colors">
                        <td className="py-4 px-4 text-white/40 font-mono text-[10px] whitespace-nowrap">{audit.timestamp}</td>
                        <td className="py-4 px-4 font-medium max-w-[180px] truncate" title={cleanName}>{cleanName}</td>
                        <td className="py-4 px-4 whitespace-nowrap">
                          {audit.media_type === 'video' ? (
                            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 text-[10px] font-semibold border border-purple-500/20">
                              VIDEO
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px] font-semibold border border-blue-500/20">
                              IMAGE
                            </span>
                          )}
                        </td>
                        <td className="py-4 px-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold ${
                            audit.prediction === 'FAKE' 
                              ? 'bg-rose-500/15 text-rose-400 border border-rose-500/25' 
                              : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/25'
                          }`}>
                            {audit.prediction}
                          </span>
                        </td>
                        <td className="py-4 px-4 font-mono font-bold whitespace-nowrap">{audit.confidence}%</td>
                        <td className="py-4 px-4 text-right whitespace-nowrap">
                          <button 
                            onClick={() => handleDelete(audit.id)}
                            className="p-2 text-rose-400/60 hover:text-rose-400 hover:bg-rose-500/10 rounded-full transition-colors cursor-pointer inline-flex items-center justify-center"
                            title="Delete Log"
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <polyline points="3 6 5 6 21 6" />
                              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                            </svg>
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

window.History = History;
