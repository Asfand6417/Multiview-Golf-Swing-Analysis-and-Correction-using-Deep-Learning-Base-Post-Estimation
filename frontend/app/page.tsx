"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import {
  ActiveTab,
  SwingSession
} from '../types';
import Sidebar from '../components/Sidebar';
import Header from '../components/Header';
import WorkspaceSynchronizer from '../components/WorkspaceSynchronizer';
import SkeletalAnalysis from '../components/SkeletalAnalysis';
import HistoryDashboard from '../components/HistoryDashboard';
import PipelineControl from '../components/PipelineControl';

import {
  Sparkles,
  ChevronRight,
  Send,
  HelpCircle,
  FileText,
  MessageSquare,
  BookOpen
} from 'lucide-react';

export default function App() {
  // Navigation State
  const [activeTab, setActiveTab] = useState<ActiveTab>('analyze');
  const [activeSection, setActiveSection] = useState<string>('sync'); // sync, biometrics, video, reports
  const [selectedSampleName, setSelectedSampleName] = useState<string>("Front_Swinger_PRO_3203.mp4");

  // 2D overlay state from the synchronizer. Metric-grade 3D analysis is disabled until DLT is wired.
  const [liveMetrics, setLiveMetrics] = useState({
    swayProxy: 0.0,
    kneeAngle: 155.0,
    mpjpe: null as number | null,
    fps: 0,
    poseStatus: '2D Overlay',
    phase: 'Setup',
    issues: [] as string[]
  });

  // Coach Advisory Panel active states
  const [isCoachOpen, setIsCoachOpen] = useState<boolean>(false);
  const [coachQuery, setCoachQuery] = useState<string>("");
  const [coachChat, setCoachChat] = useState<Array<{ sender: 'user' | 'ai'; text: string }>>([
    {
      sender: 'ai',
      text: "Hi Alex. The current workspace can show a 2D pose overlay, but calibrated 3D DLT analysis is not connected yet. I can help interpret what is available without presenting fake precision."
    }
  ]);
  const [isCoachTyping, setIsCoachTyping] = useState<boolean>(false);

  // Load selected swing session back into the synchronizer
  const handleLoadSession = (session: SwingSession) => {
    // switch tab to analyze & workspace
    setActiveTab('analyze');
    setActiveSection('sync');
    setSelectedSampleName(session.location === 'Augusta National' ? 'Front_Angle_Rory_McIlroy.mp4' : 'Front_Swinger_PRO_3203.mp4');
  };

  // Reset or start new setup session
  const handleNewSession = () => {
    setActiveTab('analyze');
    setActiveSection('sync');
    setSelectedSampleName("Draft_New_Capture_Front_1.mp4");
    setLiveMetrics({ swayProxy: 0, kneeAngle: 0, mpjpe: null, fps: 0, poseStatus: '', phase: '', issues: [] });
    alert("New empty session workspace configured. User-uploaded 3D analysis remains disabled until calibrated camera matrices are available.");
  };

  // Chat responses are limited to 2D overlay guidance until calibrated DLT is available.
  const handleCoachSubmit = (e: any) => {
    e.preventDefault();
    if (!coachQuery.trim()) return;

    const userMsg = coachQuery;
    setCoachChat((prev) => [...prev, { sender: 'user', text: userMsg }]);
    setCoachQuery("");
    setIsCoachTyping(true);

    setTimeout(() => {
      let aiText = "Interesting query! To optimize this specific component of your golf swing, consider maintaining horizontal pelvis flex locked and holding weight preloaded on your trailing arch.";
      
      const normalized = userMsg.toLowerCase();
      if (normalized.includes('sway') || normalized.includes('lateral')) {
        const isSwayBad = liveMetrics.swayProxy > 2.5;
        aiText = `Based on the 2D overlay, your lateral head movement proxy is ${liveMetrics.swayProxy.toFixed(1)}% of frame width. ${isSwayBad ? 'This exceeds the provisional 2D threshold.' : 'This is within the provisional 2D threshold.'} Metric-grade 3D confirmation still requires calibration.`;
      } else if (normalized.includes('smash') || normalized.includes('factor')) {
        aiText = "Smash factor is the ratio of ball speed to clubhead speed. Prioritize striking inside the absolute sweet-spot. Avoid shifting your sternum forward before the impact plane.";
      } else if (normalized.includes('knee') || normalized.includes('posture')) {
        aiText = `The 2D overlay estimates your lead knee angle at ${liveMetrics.kneeAngle.toFixed(1)} degrees. ${liveMetrics.kneeAngle < 155 ? 'That suggests stable flex in the current view.' : 'That suggests possible early extension in the current view.'} Treat this as 2D guidance, not DLT-validated analysis.`;
      } else if (normalized.includes('drill') || normalized.includes('exercise')) {
        aiText = "I can suggest general drills from the 2D overlay, but the app is no longer presenting drill recommendations as DLT-validated until the real pipeline is connected.";
      }

      setCoachChat((prev) => [...prev, { sender: 'ai', text: aiText }]);
      setIsCoachTyping(false);
    }, 1200);
  };

  // Direct workspace sections mapping
  const renderWorkspaceContent = () => {
    if (activeTab === 'analyze') {
      if (activeSection === 'biometrics') {
        return <SkeletalAnalysis />;
      } else if (activeSection === 'video') {
        // Render simple elegant video catalog layout
        return (
          <div id="video-catalog-view" className="flex-1 overflow-y-auto p-8 bg-surface-base">
            <span className="text-xs uppercase font-mono tracking-widest text-brand-secondary/40">DIGITAL ASSETS LOCKER</span>
            <h2 className="text-2xl font-bold font-sans text-white tracking-tight mt-1 mb-6">Imported Video Files</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {[
                { name: 'Front_Swinger_PRO_3203.mp4', size: '42.5 MB', duration: '6.2s', angle: 'Front' },
                { name: 'Side_Swinger_PRO_3203.mp4', size: '41.1 MB', duration: '6.2s', angle: 'Side' },
                { name: 'Front_Angle_Rory_McIlroy.mp4', size: '18.9 MB', duration: '4.8s', angle: 'Front' },
                { name: 'Side_Angle_Rory_McIlroy.mp4', size: '19.4 MB', duration: '4.8s', angle: 'Side' }
              ].map((vid, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    setSelectedSampleName(vid.name);
                    setActiveSection('sync');
                  }}
                  className="bg-surface-low border border-white/5 hover:border-brand-primary/30 p-4 rounded-lg cursor-pointer transition flex flex-col justify-between"
                >
                  <div className="w-full aspect-video bg-[#0d0d0d] rounded-md border border-white/5 mb-3 flex items-center justify-center">
                    <span className="text-[10px] font-mono text-brand-secondary/35 uppercase tracking-widest">MP4 Video Clip</span>
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white truncate">{vid.name}</h4>
                    <div className="flex justify-between text-[10px] font-mono text-brand-secondary/40 mt-1 uppercase">
                      <span>{vid.duration} • {vid.angle} View</span>
                      <span className="text-brand-primary">{vid.size}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      } else if (activeSection === 'reports') {
        return <PipelineControl />;
      } else {
        return (
          <WorkspaceSynchronizer
            selectedSampleName={selectedSampleName}
            setSelectedSampleName={setSelectedSampleName}
            onMetricsUpdate={(newMetrics) => setLiveMetrics(newMetrics)}
          />
        );
      }
    } else if (activeTab === 'history') {
      return <HistoryDashboard onLoadSession={handleLoadSession} />;
    } else if (activeTab === 'insights') {
      return <PipelineControl />;
    }
    return null;
  };

  return (
    <div
      id="app-root-container"
      className="flex h-screen w-screen overflow-hidden bg-surface-lowest text-white font-sans selection:bg-brand-primary selection:text-black relative"
    >
      {/* 1. Left Vertical Options Sidebar */}
      <Sidebar
        activeSection={activeSection}
        setActiveSection={setActiveSection}
        onNewSession={handleNewSession}
      />

      {/* 2. Main Workspace Layout */}
      <div id="workspace-layout-wrapper" className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        
        {/* Horizontal Navigation Header */}
        <Header
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          activeSection={activeSection}
          setActiveSection={setActiveSection}
        />

        {/* Dynamic Inner Dashboard Page Render */}
        <main id="workspace-main-content" className="flex-1 min-h-0 flex flex-col bg-surface-base relative">
          {renderWorkspaceContent()}
        </main>
      </div>

      {/* Floating Gemini Pro-Coach Coach Sidebar Toggle */}
      <button
        id="btn-coach-toggle"
        onClick={() => setIsCoachOpen(!isCoachOpen)}
        className="fixed bottom-6 right-6 z-40 bg-brand-primary hover:bg-brand-primary-dim text-black font-semibold rounded-full p-4 flex items-center gap-2 shadow-2xl transition duration-300 hover:scale-105 active:scale-95 cursor-pointer"
        title="Ask Gemini Coach"
      >
        <Sparkles className="w-5 h-5 text-black fill-black" />
        <span className="text-xs font-bold tracking-tight uppercase select-none">Ask Gemini Coach</span>
      </button>

      {/* Chat sliding panel */}
      {isCoachOpen && (
        <div
          id="coach-chat-drawer"
          className="fixed inset-y-0 right-0 w-80 md:w-96 bg-[#161616] border-l border-white/10 z-50 flex flex-col justify-between shadow-2xl transition-all duration-300 transform translate-x-0"
        >
          {/* Drawer Header */}
          <div className="p-4 border-b border-white/5 bg-[#121212] flex justify-between items-center">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-brand-primary fill-brand-primary" />
              <div>
                <h3 className="text-sm font-bold tracking-tight text-white uppercase font-sans">Gemini Swing Coach</h3>
                <span className="text-[9px] font-mono tracking-widest text-[#859585]">2D OVERLAY ASSISTANT</span>
              </div>
            </div>
            <button
              onClick={() => setIsCoachOpen(false)}
              className="text-brand-secondary hover:text-white text-xs cursor-pointer px-2 py-1"
            >
              ✕
            </button>
          </div>

          {/* Messages Lists container */}
          <div className="flex-1 p-4 overflow-y-auto space-y-4 text-xs">
            {coachChat.map((msg, mIdx) => (
              <div
                key={mIdx}
                className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded p-3 leading-relaxed flex flex-col gap-1 text-left ${
                    msg.sender === 'user'
                      ? 'bg-brand-primary/10 text-white border border-brand-primary/20'
                      : 'bg-surface-lowest text-brand-secondary border border-white/5'
                  }`}
                >
                  <span className="text-[8px] font-mono text-brand-secondary/40 font-bold uppercase select-none">
                    {msg.sender === 'user' ? 'ALEX MERCER (YOU)' : 'GEMINI AI COACH'}
                  </span>
                  <p className="font-sans text-[11px] leading-relaxed select-text">{msg.text}</p>
                </div>
              </div>
            ))}
            
            {isCoachTyping && (
              <div className="flex items-center gap-2 text-brand-secondary/40 font-mono text-[9px] pl-1 select-none">
                <span className="animate-bounce">•</span>
                <span className="animate-bounce delay-100">•</span>
                <span className="animate-bounce delay-200">•</span>
                <span>Gemini Coach is tracking indices...</span>
              </div>
            )}
          </div>

          {/* Quick FAQ tag shortcuts */}
          <div className="p-3 bg-[#111] border-t border-white/5 flex flex-wrap gap-1.5 justify-center">
	            {[
	              "Why lateral head sway?",
	              "Explain smash factor",
	              "Extend shoulder rotation"
	            ].map((faq, fIdx) => (
              <button
                key={fIdx}
                onClick={() => setCoachQuery(faq)}
                className="text-[9px] font-sans bg-surface-lowest hover:bg-white/5 text-brand-secondary hover:text-white border border-white/5 rounded px-2.5 py-1 text-center transition tracking-tight whitespace-nowrap cursor-pointer"
              >
                {faq}
              </button>
            ))}
          </div>

          {/* Form Submit Input */}
          <form onSubmit={handleCoachSubmit} className="p-4 border-t border-white/5 bg-[#121212] flex gap-2">
            <input
              type="text"
              value={coachQuery}
              onChange={(e) => setCoachQuery(e.target.value)}
              placeholder="Ask coach about setup sway, smash factor..."
              className="flex-1 bg-surface-lowest border border-white/5 text-xs rounded-md px-3 py-2 text-white placeholder-brand-secondary/35 focus:outline-none focus:border-brand-primary/40 font-sans"
            />
            <button
              type="submit"
              className="bg-brand-primary hover:bg-brand-primary-dim text-black font-semibold rounded-md p-2 flex items-center justify-center cursor-pointer transition"
            >
              <Send className="w-4 h-4 text-black" />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
