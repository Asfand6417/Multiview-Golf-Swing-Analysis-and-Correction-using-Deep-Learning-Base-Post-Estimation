"use client";
import React, { useEffect, useRef } from 'react';

export default function FineTuningPage() {
    const panelsRef = useRef<(HTMLDivElement | null)[]>([]);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent, panel: HTMLDivElement) => {
            const rect = panel.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            panel.style.setProperty('--mouse-x', `${x}px`);
            panel.style.setProperty('--mouse-y', `${y}px`);
        };

        const panels = panelsRef.current.filter(Boolean);
        panels.forEach(panel => {
            if (panel) panel.addEventListener('mousemove', (e) => handleMouseMove(e, panel));
        });

        return () => {
            panels.forEach(panel => {
                if (panel) panel.removeEventListener('mousemove', (e) => handleMouseMove(e as MouseEvent, panel));
            });
        };
    }, []);

    const setPanelRef = (index: number) => (el: HTMLDivElement | null) => {
        panelsRef.current[index] = el;
    };

    return (
        <div className="bg-[#0A0A0A] text-[#e5e2e1] font-sans min-h-screen overflow-x-hidden selection:bg-[#75ff9e] selection:text-[#00210b]">
            <style jsx global>{`
                .glass-panel {
                    background: rgba(26, 28, 30, 0.4);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    position: relative;
                }
                .glass-panel::before {
                    content: '';
                    position: absolute;
                    top: var(--mouse-y);
                    left: var(--mouse-x);
                    transform: translate(-50%, -50%);
                    width: 300px;
                    height: 300px;
                    background: radial-gradient(circle, rgba(117,255,158,0.05) 0%, transparent 70%);
                    opacity: 0;
                    transition: opacity 0.3s;
                    pointer-events: none;
                    z-index: 0;
                }
                .glass-panel:hover::before {
                    opacity: 1;
                }
                .performance-glow {
                    box-shadow: 0 0 20px rgba(117, 255, 158, 0.15);
                }
                .skeletal-connector {
                    stroke: #75ff9e;
                    stroke-width: 1.5;
                    stroke-linecap: round;
                    opacity: 0.6;
                }
                .landmark-point {
                    fill: #75ff9e;
                    filter: drop-shadow(0 0 2px #75ff9e);
                }
                @keyframes pulse-dot {
                    0% { transform: scale(1); opacity: 0.8; }
                    50% { transform: scale(1.5); opacity: 1; }
                    100% { transform: scale(1); opacity: 0.8; }
                }
                .active-dot {
                    animation: pulse-dot 2s infinite ease-in-out;
                }
            `}</style>

            {/* TopNavBar */}
            <nav className="flex justify-between items-center px-6 py-4 w-full sticky top-0 z-50 bg-[#131313]/80 backdrop-blur-xl border-b border-white/10 h-16">
                <div className="flex items-center gap-8">
                    <a href="/" className="text-[24px] text-[#75ff9e] tracking-tighter italic font-black uppercase font-inter hover:opacity-80 transition-opacity" style={{ fontStyle: 'normal' }}>
                        PRO SWING AI
                    </a>
                    <div className="hidden md:flex items-center gap-6">
                        <a className="text-[12px] font-medium font-mono hover:text-[#e5e2e1] transition-colors duration-200 text-[#a0a0a0]" href="/acquisition">Acquisition</a>
                        <a className="text-[12px] font-medium font-mono hover:text-[#e5e2e1] transition-colors duration-200 text-[#a0a0a0]" href="/splitting">Preprocessing</a>
                        <a className="text-[12px] font-medium font-mono text-[#75ff9e] border-b-2 border-[#75ff9e] pb-2 font-bold" href="/finetuning">Fine-Tuning</a>
                        <a className="text-[12px] font-medium font-mono hover:text-[#e5e2e1] transition-colors duration-200 text-[#a0a0a0]" href="/analysis">Reconstruction</a>
                        <a className="text-[12px] font-medium font-mono hover:text-[#e5e2e1] transition-colors duration-200 text-[#a0a0a0]" href="#">Analysis</a>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <button className="p-2 hover:bg-white/5 rounded-full transition-all text-[#a0a0a0]">
                        <span className="material-symbols-outlined">notifications</span>
                    </button>
                    <button className="p-2 hover:bg-white/5 rounded-full transition-all text-[#a0a0a0]">
                        <span className="material-symbols-outlined">settings</span>
                    </button>
                    <div className="h-8 w-8 rounded-full bg-[#75ff9e]/20 flex items-center justify-center border border-[#75ff9e]/40">
                        <span className="material-symbols-outlined text-[#75ff9e] text-[20px]">account_circle</span>
                    </div>
                </div>
            </nav>

            {/* Sidebar & Main Content Wrapper */}
            <div className="flex min-h-[calc(100vh-64px)]">
                
                {/* SideNavBar */}
                <aside className="hidden md:flex flex-col h-[calc(100vh-64px)] w-64 fixed left-0 top-16 bg-[#1c1b1b] border-r border-white/10 py-8 px-4 space-y-6 z-40">
                    <div className="px-2">
                        <h2 className="text-[18px] font-bold text-[#75ff9e] uppercase tracking-wider font-inter">PIPELINE CTRL</h2>
                        <p className="text-[12px] font-medium font-mono opacity-60 text-[#a0a0a0]">V3.4 Stable</p>
                    </div>
                    <div className="space-y-1">
                        <button onClick={() => window.location.href='/history'} className="w-full flex items-center gap-3 px-3 py-3 hover:bg-white/5 rounded-lg group transition-all text-[#a0a0a0]">
                            <span className="material-symbols-outlined text-[20px] group-hover:text-[#75ff9e]">video_library</span>
                            <span className="text-[12px] font-medium font-mono">Recent Captures</span>
                        </button>
                        <button className="w-full flex items-center gap-3 px-3 py-3 hover:bg-white/5 rounded-lg group transition-all text-[#a0a0a0]">
                            <span className="material-symbols-outlined text-[20px] group-hover:text-[#75ff9e]">batch_prediction</span>
                            <span className="text-[12px] font-medium font-mono">Batch Process</span>
                        </button>
                        <button onClick={() => window.location.href='/analysis'} className="w-full flex items-center gap-3 px-3 py-3 bg-[#75ff9e]/10 text-[#75ff9e] border-r-4 border-[#75ff9e] rounded-l-lg transition-all translate-x-1">
                            <span className="material-symbols-outlined text-[20px]">monitoring</span>
                            <span className="text-[12px] font-medium font-mono font-bold">Model Metrics</span>
                        </button>
                        <button className="w-full flex items-center gap-3 px-3 py-3 hover:bg-white/5 rounded-lg group transition-all text-[#a0a0a0]">
                            <span className="material-symbols-outlined text-[20px] group-hover:text-[#75ff9e]">ios_share</span>
                            <span className="text-[12px] font-medium font-mono">Export</span>
                        </button>
                    </div>
                    
                    <div className="pt-4 border-t border-white/5 px-2">
                        <div className="p-3 bg-[#131313] rounded-lg border border-white/5 mb-4">
                            <div className="flex justify-between items-end mb-1">
                                <span className="text-[10px] uppercase text-[#a0a0a0] font-mono">Annotated Vids</span>
                                <span className="text-[20px] font-bold font-mono text-[#75ff9e]">1,400</span>
                            </div>
                            <div className="h-1 bg-white/10 w-full rounded-full overflow-hidden">
                                <div className="h-full bg-[#75ff9e] w-[85%] shadow-[0_0_8px_#75ff9e]"></div>
                            </div>
                        </div>
                        <div className="p-3 bg-[#131313] rounded-lg border border-white/5">
                            <div className="flex justify-between items-end mb-1">
                                <span className="text-[10px] uppercase text-[#a0a0a0] font-mono">Keypoints</span>
                                <span className="text-[20px] font-bold font-mono text-[#c6c6c9]">25</span>
                            </div>
                            <div className="h-1 bg-white/10 w-full rounded-full overflow-hidden">
                                <div className="h-full bg-[#c6c6c9] w-[40%]"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div className="mt-auto space-y-1">
                        <button className="w-full flex items-center gap-3 px-3 py-3 hover:bg-white/5 rounded-lg transition-all text-[#a0a0a0]">
                            <span className="material-symbols-outlined text-[20px]">menu_book</span>
                            <span className="text-[12px] font-medium font-mono">Docs</span>
                        </button>
                        <button onClick={() => window.location.href='/login'} className="w-full flex items-center gap-3 px-3 py-3 text-[#ffb4ab]/80 hover:bg-[#ffb4ab]/5 rounded-lg transition-all">
                            <span className="material-symbols-outlined text-[20px]">logout</span>
                            <span className="text-[12px] font-medium font-mono">Logout</span>
                        </button>
                    </div>
                </aside>

                {/* Main Content */}
                <main className="ml-0 md:ml-64 pt-8 px-6 pb-12 w-full max-w-[1440px]">
                    
                    {/* Header Section */}
                    <header className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
                        <div>
                            <div className="flex items-center gap-2 text-[#75ff9e] mb-1">
                                <span className="material-symbols-outlined text-[18px]">tune</span>
                                <span className="text-[12px] font-medium font-mono uppercase tracking-widest font-bold">STEP 03 / 05</span>
                            </div>
                            <h1 className="text-[32px] md:text-[40px] font-bold text-[#e5e2e1] font-inter tracking-tight">ML Pipeline Fine-Tuning</h1>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="flex items-center gap-3 bg-[#131313]/60 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full">
                                <div className="flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-[#75ff9e] active-dot"></span>
                                    <span className="text-[12px] font-medium font-mono uppercase text-[#a0a0a0]">GPU Status:</span>
                                    <span className="text-[12px] font-medium font-mono text-[#75ff9e]">NVIDIA A100 Active</span>
                                </div>
                            </div>
                        </div>
                    </header>

                    {/* Bento Grid Layout */}
                    <div className="grid grid-cols-12 gap-6">
                        
                        {/* Left Column: Model Visualization */}
                        <section className="col-span-12 lg:col-span-7 space-y-6">
                            
                            {/* 1. Pose Detection Model section (Base) */}
                            <div ref={setPanelRef(0)} className="glass-panel p-6 rounded-xl relative overflow-hidden group">
                                <div className="relative z-10">
                                    <div className="absolute top-0 right-0 p-4 opacity-20 pointer-events-none group-hover:opacity-40 transition-opacity">
                                        <span className="material-symbols-outlined text-[80px]">accessibility_new</span>
                                    </div>
                                    <div className="flex justify-between items-start mb-6">
                                        <div>
                                            <span className="text-[12px] font-medium font-mono text-[#b0c8ff] mb-1 block">Component 01</span>
                                            <h3 className="text-[20px] font-bold font-inter text-[#e5e2e1]">Pose Detection Model (Base)</h3>
                                        </div>
                                        <span className="bg-[#002e6a] text-[#75ff9e] font-mono text-[10px] font-bold tracking-widest px-2 py-1 rounded">STABLE</span>
                                    </div>
                                    <div className="aspect-video bg-[#0e0e0e] rounded-lg border border-white/5 flex items-center justify-center overflow-hidden relative">
                                        
                                        {/* Simulated Skeletal Visualization */}
                                        <svg className="w-full h-full max-w-[400px]" viewBox="0 0 200 300">
                                            <defs>
                                                <pattern height="20" id="grid" patternUnits="userSpaceOnUse" width="20">
                                                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5"></path>
                                                </pattern>
                                            </defs>
                                            <rect fill="url(#grid)" height="100%" width="100%"></rect>
                                            <g className="skeleton-base">
                                                <line className="skeletal-connector" x1="100" x2="100" y1="50" y2="120"></line>
                                                <line className="skeletal-connector" x1="100" x2="140" y1="70" y2="80"></line>
                                                <line className="skeletal-connector" x1="100" x2="60" y1="70" y2="80"></line>
                                                <line className="skeletal-connector" x1="140" x2="150" y1="80" y2="130"></line>
                                                <line className="skeletal-connector" x1="100" x2="120" y1="120" y2="200"></line>
                                                <line className="skeletal-connector" x1="100" x2="80" y1="120" y2="200"></line>
                                                <circle className="landmark-point" cx="100" cy="50" r="4"></circle>
                                                <circle className="landmark-point" cx="140" cy="80" r="3"></circle>
                                                <circle className="landmark-point" cx="60" cy="80" r="3"></circle>
                                                <circle className="landmark-point" cx="100" cy="120" r="3"></circle>
                                            </g>
                                        </svg>
                                        <div className="absolute bottom-4 left-4 text-[10px] font-medium font-mono text-[#bacbb9]/40 tracking-widest">
                                            BASE TOPOLOGY: GLOBAL SKELETAL MESH
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* 2. Regular Landmark Model section (Processing) */}
                            <div ref={setPanelRef(1)} className="glass-panel p-6 rounded-xl">
                                <div className="relative z-10">
                                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                                        <div>
                                            <span className="text-[12px] font-medium font-mono text-[#75ff9e] mb-1 block">Component 02</span>
                                            <h3 className="text-[20px] font-bold font-inter text-[#e5e2e1]">Regular Landmark Model (Processing)</h3>
                                        </div>
                                        <div className="flex items-center bg-[#2a2a2a] p-1 rounded-lg">
                                            <button className="px-4 py-1.5 text-[12px] font-medium font-mono rounded hover:text-[#e5e2e1] transition-all text-[#a0a0a0]">Lite 1</button>
                                            <button className="px-4 py-1.5 text-[12px] font-medium font-mono rounded bg-[#75ff9e] text-[#00210b] font-bold shadow-lg shadow-[#75ff9e]/20">Full 2</button>
                                            <button className="px-4 py-1.5 text-[12px] font-medium font-mono rounded hover:text-[#e5e2e1] transition-all text-[#a0a0a0]">Heavy 3</button>
                                        </div>
                                    </div>
                                    <div className="aspect-video bg-[#0e0e0e] rounded-lg border border-white/10 flex items-center justify-center overflow-hidden relative">
                                        <div className="grid grid-cols-11 grid-rows-3 gap-2 p-8 opacity-40">
                                            {Array.from({ length: 33 }).map((_, i) => (
                                                <div key={i} className="w-2 h-2 rounded-full bg-[#75ff9e] shadow-[0_0_5px_#75ff9e]"></div>
                                            ))}
                                        </div>
                                        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 200 100">
                                            <path d="M 20 50 Q 100 0 180 50" fill="none" stroke="#75ff9e" strokeDasharray="2,2" strokeWidth="0.5"></path>
                                            <path d="M 20 60 Q 100 100 180 60" fill="none" opacity="0.3" stroke="#75ff9e" strokeDasharray="2,2" strokeWidth="0.5"></path>
                                        </svg>
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <span className="text-[40px] font-bold font-mono text-[#75ff9e]/10 tracking-[0.5em] pointer-events-none select-none">33 x 3 MESH</span>
                                        </div>
                                        <div className="absolute top-4 right-4 text-[10px] font-mono text-[#75ff9e] flex gap-2">
                                            <span>X: [0.42..0.98]</span>
                                            <span>Y: [0.11..0.56]</span>
                                        </div>
                                    </div>
                                    <div className="mt-4 flex gap-4 overflow-x-auto pb-2 custom-scrollbar">
                                        <div className="min-w-[120px] bg-white/5 p-3 rounded-lg border border-white/5">
                                            <p className="text-[10px] font-mono uppercase mb-1 text-[#a0a0a0]">Density</p>
                                            <p className="text-[14px] font-bold font-mono text-[#e5e2e1]">HIGH (33 pt)</p>
                                        </div>
                                        <div className="min-w-[120px] bg-white/5 p-3 rounded-lg border border-white/5">
                                            <p className="text-[10px] font-mono uppercase mb-1 text-[#a0a0a0]">Smoothing</p>
                                            <p className="text-[14px] font-bold font-mono text-[#e5e2e1]">Kalman Filter</p>
                                        </div>
                                        <div className="min-w-[120px] bg-white/5 p-3 rounded-lg border border-white/5">
                                            <p className="text-[10px] font-mono uppercase mb-1 text-[#a0a0a0]">Sampling</p>
                                            <p className="text-[14px] font-bold font-mono text-[#e5e2e1]">120 FPS</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </section>

                        {/* Right Column: Stats, Config & Log */}
                        <section className="col-span-12 lg:col-span-5 space-y-6">
                            
                            {/* 3. Fine-Tuned Detection output card */}
                            <div ref={setPanelRef(2)} className="glass-panel p-6 rounded-xl border border-[#75ff9e]/40 performance-glow">
                                <div className="relative z-10">
                                    <div className="flex items-center gap-3 mb-6">
                                        <span className="material-symbols-outlined text-[#75ff9e]">auto_fix_high</span>
                                        <h3 className="text-[20px] font-bold font-inter text-[#e5e2e1]">Output Configuration</h3>
                                    </div>
                                    <div className="space-y-4 mb-8">
                                        <div className="flex items-start gap-4 p-3 bg-white/5 rounded-lg border border-white/5">
                                            <span className="material-symbols-outlined text-[#75ff9e] text-[20px]">check_circle</span>
                                            <div className="flex flex-col">
                                                <span className="text-[12px] font-medium font-mono text-[#e5e2e1]">Pose Detection Model Integration</span>
                                                <span className="text-[11px] font-inter text-[#a0a0a0]">Synchronized with Base-V3 mesh topology</span>
                                            </div>
                                        </div>
                                        <div className="flex items-start gap-4 p-3 bg-white/5 rounded-lg border border-white/5">
                                            <span className="material-symbols-outlined text-[#75ff9e] text-[20px]">check_circle</span>
                                            <div className="flex flex-col">
                                                <span className="text-[12px] font-medium font-mono text-[#e5e2e1]">Regular Landmark Mapping</span>
                                                <span className="text-[11px] font-inter text-[#a0a0a0]">33-point dense cloud projection enabled</span>
                                            </div>
                                        </div>
                                        <div className="flex items-start gap-4 p-3 bg-white/5 rounded-lg border border-white/5">
                                            <span className="material-symbols-outlined text-[#75ff9e] text-[20px]">check_circle</span>
                                            <div className="flex flex-col">
                                                <span className="text-[12px] font-medium font-mono text-[#e5e2e1]">2D Pose Estimation Overlay</span>
                                                <span className="text-[11px] font-inter text-[#a0a0a0]">Coordinate space normalization completed</span>
                                            </div>
                                        </div>
                                        <div className="flex items-start gap-4 p-3 bg-white/5 rounded-lg border border-white/5">
                                            <span className="material-symbols-outlined text-[#75ff9e] text-[20px]">check_circle</span>
                                            <div className="flex flex-col">
                                                <span className="text-[12px] font-medium font-mono text-[#e5e2e1]">Optimized Complexity Level 2</span>
                                                <span className="text-[11px] font-inter text-[#a0a0a0]">Balanced for real-time mobile inference</span>
                                            </div>
                                        </div>
                                    </div>
                                    <button onClick={() => window.location.href='/analysis'} className="w-full py-4 bg-[#75ff9e] text-[#00210b] font-bold font-inter tracking-wider rounded-lg flex items-center justify-center gap-2 hover:brightness-110 active:scale-[0.98] transition-all group shadow-[0_0_15px_rgba(117,255,158,0.2)]">
                                        <span className="material-symbols-outlined group-hover:rotate-180 transition-transform duration-500">sync</span>
                                        <span>INITIATE FINE-TUNING</span>
                                    </button>
                                </div>
                            </div>

                            {/* Accuracy Metrics Panel */}
                            <div className="grid grid-cols-2 gap-4">
                                <div ref={setPanelRef(3)} className="glass-panel p-4 rounded-xl border border-white/5">
                                    <div className="relative z-10 w-full">
                                        <div className="flex items-center gap-2 mb-2 opacity-60">
                                            <span className="material-symbols-outlined text-[16px]">target</span>
                                            <span className="text-[10px] uppercase font-bold text-[#e5e2e1] font-mono tracking-widest">Accuracy</span>
                                        </div>
                                        <div className="flex items-end gap-1">
                                            <span className="text-[24px] font-bold font-mono text-[#75ff9e]">98.4</span>
                                            <span className="text-[12px] font-medium font-mono text-[#e5e2e1] mb-1 opacity-40">%</span>
                                        </div>
                                        <div className="mt-2 h-0.5 bg-white/5 w-full">
                                            <div className="h-full bg-[#75ff9e] w-[98%] shadow-[0_0_5px_#75ff9e]"></div>
                                        </div>
                                    </div>
                                </div>
                                <div ref={setPanelRef(4)} className="glass-panel p-4 rounded-xl border border-white/5">
                                    <div className="relative z-10 w-full">
                                        <div className="flex items-center gap-2 mb-2 opacity-60">
                                            <span className="material-symbols-outlined text-[16px]">bolt</span>
                                            <span className="text-[10px] uppercase font-bold text-[#e5e2e1] font-mono tracking-widest">Latency</span>
                                        </div>
                                        <div className="flex items-end gap-1">
                                            <span className="text-[24px] font-bold font-mono text-[#c6c6c9]">12</span>
                                            <span className="text-[12px] font-medium font-mono text-[#e5e2e1] mb-1 opacity-40">ms</span>
                                        </div>
                                        <div className="mt-2 h-0.5 bg-white/5 w-full">
                                            <div className="h-full bg-[#c6c6c9] w-[15%]"></div>
                                        </div>
                                    </div>
                                </div>
                                <div ref={setPanelRef(5)} className="glass-panel p-4 rounded-xl border border-white/5 col-span-2">
                                    <div className="relative z-10">
                                        <div className="flex justify-between items-center mb-3">
                                            <div className="flex items-center gap-2 opacity-60">
                                                <span className="material-symbols-outlined text-[16px]">database</span>
                                                <span className="text-[10px] uppercase font-bold font-mono tracking-widest text-[#e5e2e1]">Dataset</span>
                                            </div>
                                            <span className="text-[10px] font-medium font-mono text-[#e5e2e1] bg-white/10 px-2 py-0.5 rounded">Training Ready</span>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-[20px] font-bold font-mono text-white">1,400 Videos</span>
                                            <div className="flex -space-x-2">
                                                <div className="w-8 h-8 rounded-full border-2 border-[#131313] bg-[#353534] overflow-hidden">
                                                    <img className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDImwp7ljwMJ8fnyKMCcqKjzth5NmadzAjF33oRa_WMmgUs6d2zuyuYykGCgk5ODpZuRIjESIPXrZ_HgjgF07ao89_WiaUc9A_eG8CRjsR43lcaz9Lmi0KkQY6nuxUw2xm1udrh57WSYf8i6qEVgF-_cN_Lsv44UfeLftdQeT5nnSk98tlrfbfiiVmVuhGSE0-tCME2mOPDY1GkaEosby60agSsxSoMWdcy2Lq_3qkT1BBq7dOFlv1oEA-xJITCtGUxeizUBtSdci5_" />
                                                </div>
                                                <div className="w-8 h-8 rounded-full border-2 border-[#131313] bg-[#353534] overflow-hidden">
                                                    <img className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCV-xxELAeZDZTHRGL3-0KDKALr7-vlU5k0lEOgWj6UUmxKDE8XUHcnbyWNPIz2IsaPWuagGS26a_6G64dn9GTbggCr5tV9b4RrThmtF-n7p46KJRCEJ3cRXSWqVniFOvY95pyb79wvjPXahL_Ik7AOf6jsEQOJnD2PCR06XNn9mgHZlcWuc2_nqt1mBU30DR4b2eWH8_tZtB2VkKDrW7sVYsDe1pPjKKAVl2AE8L6kJ-JUgwS7mRbkgEaB2BkbbOnuMY0EF0o50e8P" />
                                                </div>
                                                <div className="w-8 h-8 rounded-full border-2 border-[#131313] bg-[#75ff9e]/20 flex items-center justify-center text-[10px] font-bold font-mono text-[#75ff9e]">
                                                    +1k
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Fine-Tuning Log */}
                            <div ref={setPanelRef(6)} className="glass-panel p-4 rounded-xl border border-white/5 h-48 overflow-hidden relative">
                                <div className="relative z-10 w-full h-full flex flex-col">
                                    <div className="flex justify-between items-center mb-4 shrink-0">
                                        <span className="text-[10px] uppercase font-bold font-mono tracking-widest text-[#e5e2e1] opacity-60">Fine-Tuning Log</span>
                                        <span className="flex gap-1">
                                            <span className="w-1.5 h-1.5 rounded-full bg-[#75ff9e]/20"></span>
                                            <span className="w-1.5 h-1.5 rounded-full bg-[#75ff9e]"></span>
                                            <span className="w-1.5 h-1.5 rounded-full bg-[#75ff9e]/20"></span>
                                        </span>
                                    </div>
                                    <div className="space-y-2 font-mono text-[10px] opacity-80 overflow-y-auto max-h-32 custom-scrollbar text-[#e5e2e1] flex-1 pb-8">
                                        <p className="text-[#75ff9e]">&gt; Initializing tensor kernels...</p>
                                        <p>&gt; Loading 1,400 annotated golf sequences</p>
                                        <p>&gt; Baseline loss: 0.8423</p>
                                        <p>&gt; Optimized landmarks: 33 joints identified</p>
                                        <p className="text-[#c6c6c9]">&gt; Epoch 12/50: loss=0.214 | acc=0.91</p>
                                        <p className="text-[#c6c6c9]">&gt; Epoch 28/50: loss=0.082 | acc=0.96</p>
                                        <p className="animate-pulse">&gt; Processing batch 42...</p>
                                    </div>
                                    <div className="absolute bottom-0 left-0 right-0 h-10 bg-gradient-to-t from-[rgba(26,28,30,0.8)] to-transparent pointer-events-none"></div>
                                </div>
                            </div>

                        </section>
                    </div>
                </main>
            </div>
        </div>
    );
}
