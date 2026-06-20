"use client";
import React, { useEffect, useRef } from 'react';

export default function ReconstructionPage() {
    const coordRefX = useRef<HTMLSpanElement>(null);
    const coordRefY = useRef<HTMLSpanElement>(null);
    const coordRefZ = useRef<HTMLSpanElement>(null);

    // Micro-interaction: Update coordinate numbers subtly
    useEffect(() => {
        const interval = setInterval(() => {
            const targets = [coordRefX.current, coordRefY.current, coordRefZ.current];
            const target = targets[Math.floor(Math.random() * targets.length)];
            
            if (target) {
                const current = parseFloat(target.innerText);
                if (!isNaN(current)) {
                    const variation = (Math.random() - 0.5) * 0.005;
                    target.innerText = (current + variation).toFixed(3);
                }
            }
        }, 800);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="bg-[#131313] text-[#e5e2e1] font-sans min-h-screen overflow-x-hidden selection:bg-[#75ff9e] selection:text-[#00210b]">
            <style jsx global>{`
                .glass-panel {
                    background: rgba(26, 28, 30, 0.4);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .grid-bg {
                    background-image: radial-gradient(rgba(117, 255, 158, 0.05) 1px, transparent 1px);
                    background-size: 24px 24px;
                }
                .blueprint-line {
                    stroke-dasharray: 4;
                    animation: dash 20s linear infinite;
                }
                @keyframes dash {
                    to { stroke-dashoffset: -1000; }
                }
                .glow-green {
                    box-shadow: 0 0 15px rgba(117, 255, 158, 0.3);
                }
                .scanline {
                    width: 100%;
                    height: 2px;
                    background: linear-gradient(to bottom, rgba(117, 255, 158, 0.2), transparent);
                    position: absolute;
                    animation: scan 4s linear infinite;
                }
                @keyframes scan {
                    from { top: 0%; }
                    to { top: 100%; }
                }
            `}</style>
            
            <div className="grid-bg min-h-screen">
                
                {/* TopNavBar */}
                <header className="bg-[#131313]/80 backdrop-blur-xl border-b border-white/10 flex justify-between items-center px-6 py-4 w-full sticky top-0 z-50 h-16">
                    <div className="flex items-center gap-8">
                        <h1 className="text-[24px] text-[#75ff9e] tracking-tighter italic font-black uppercase font-inter"><span style={{ fontStyle: 'normal' }}>PRO SWING AI</span></h1>
                        <nav className="hidden md:flex items-center space-x-6">
                            <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono" href="/acquisition">Acquisition</a>
                            <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono" href="/splitting">Preprocessing</a>
                            <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono" href="/finetuning">Fine-Tuning</a>
                            <a className="text-[#75ff9e] border-b-2 border-[#75ff9e] pb-1 font-bold text-[12px] font-mono" href="/reconstruction">Reconstruction</a>
                            <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono" href="/analysis">Analysis</a>
                        </nav>
                    </div>
                    <div className="flex items-center space-x-4">
                        <button className="p-2 hover:bg-white/5 rounded-full transition-all duration-150">
                            <span className="material-symbols-outlined text-[#75ff9e]">notifications</span>
                        </button>
                        <button className="p-2 hover:bg-white/5 rounded-full transition-all duration-150">
                            <span className="material-symbols-outlined text-[#75ff9e]">settings</span>
                        </button>
                        <div className="h-8 w-8 rounded-full overflow-hidden border border-[#75ff9e]/30">
                            <img alt="Profile" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuB0-HgtlsfauZPFFDLurT3SCQ-6_b97iZGrgLyFjgLM0ead39B4h6qtLYXk0OGgyuqKuxhdWkMKGZlaWLN5hwHNZgQacnKj5QGGRSOh7CpfWOglywfCeMcRCtbdW9E1CnhycFz-26cK-TQD4oZuLP3Sw9tIhAB0Trr7g4vUDlhEK67Yx234LPOM6qahoPZ9h0dLypZrgoCSdjdJFMu2z2IzZkPhP_xpJ-YXZzGfltorsNiPtunrnIpZq54FlcSKIMorjV2B13ZtoM2P" />
                        </div>
                    </div>
                </header>

                {/* Sidebar */}
                <aside className="hidden lg:flex flex-col h-[calc(100vh-64px)] w-64 fixed left-0 top-[64px] py-8 px-4 space-y-6 bg-[#1c1b1b] border-r border-white/10 z-40">
                    <div className="space-y-1">
                        <span className="text-[12px] font-medium font-mono text-[#bacbb9] px-4">PIPELINE CTRL</span>
                        <div className="px-4 py-1 flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-[#75ff9e] animate-pulse"></div>
                            <span className="font-mono text-[12px] text-[#75ff9e] font-bold">V3.4 Stable</span>
                        </div>
                    </div>
                    <nav className="flex-1 space-y-2">
                        <button onClick={() => window.location.href='/history'} className="w-full flex items-center gap-3 px-4 py-3 text-[#bacbb9] hover:bg-white/5 rounded-lg transition-all">
                            <span className="material-symbols-outlined">video_library</span>
                            <span className="text-[12px] font-medium font-mono">Recent Captures</span>
                        </button>
                        <button className="w-full flex items-center gap-3 px-4 py-3 text-[#bacbb9] hover:bg-white/5 rounded-lg transition-all">
                            <span className="material-symbols-outlined">settings_suggest</span>
                            <span className="text-[12px] font-medium font-mono">Batch Process</span>
                        </button>
                        <button onClick={() => window.location.href='/analysis'} className="w-full flex items-center gap-3 px-4 py-3 bg-[#75ff9e]/10 text-[#75ff9e] border-r-4 border-[#75ff9e] transition-all translate-x-1 rounded-l-lg">
                            <span className="material-symbols-outlined">monitoring</span>
                            <span className="text-[12px] font-medium font-mono font-bold">Model Metrics</span>
                        </button>
                        <button className="w-full flex items-center gap-3 px-4 py-3 text-[#bacbb9] hover:bg-white/5 rounded-lg transition-all">
                            <span className="material-symbols-outlined">ios_share</span>
                            <span className="text-[12px] font-medium font-mono">Export</span>
                        </button>
                    </nav>
                    <div className="pt-4 border-t border-white/10 space-y-2">
                        <button onClick={() => window.location.href='/'} className="w-full bg-[#75ff9e] text-[#00210b] py-3 rounded-lg font-bold hover:scale-[0.98] transition-transform">
                            New Session
                        </button>
                    </div>
                </aside>

                {/* Main Content Canvas */}
                <main className="lg:ml-64 p-6 md:p-10 max-w-screen-2xl mx-auto space-y-6">
                    {/* Header Section */}
                    <section className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[#75ff9e]/20 pb-4">
                        <div>
                            <span className="text-[12px] font-medium font-mono text-[#75ff9e] tracking-widest uppercase">STEP 04/05</span>
                            <h2 className="text-[32px] font-bold text-[#e5e2e1] font-inter">3D Reconstruction Dashboard</h2>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="glass-panel px-4 py-2 flex items-center gap-3 rounded-lg">
                                <div className="w-3 h-3 bg-[#ffb4ab] rounded-full animate-pulse"></div>
                                <span className="text-[20px] font-bold font-mono text-[#ffb4ab]">LIVE FEED SYNCING</span>
                            </div>
                        </div>
                    </section>

                    <div className="grid grid-cols-12 gap-6">
                        
                        {/* 1. DLT Transformation Engine (Bento Main) */}
                        <div className="col-span-12 xl:col-span-8 flex flex-col gap-6">
                            <div className="glass-panel rounded-xl p-6 relative overflow-hidden h-[400px]">
                                <div className="flex justify-between items-center mb-8 relative z-10">
                                    <h3 className="text-[12px] font-medium font-mono text-[#bacbb9] flex items-center gap-2">
                                        <span className="material-symbols-outlined text-[18px]">hub</span>
                                        DLT TRANSFORMATION ENGINE
                                    </h3>
                                    <span className="text-[14px] font-bold font-mono text-[#75ff9e] uppercase tracking-widest bg-[#75ff9e]/10 px-3 py-1 rounded border border-[#75ff9e]/30">8-Point Calibration Active</span>
                                </div>

                                {/* Flow Visualization */}
                                <div className="relative h-full flex items-center justify-between px-12 pb-12">
                                    {/* Camera Source 1 */}
                                    <div className="flex flex-col items-center gap-4 z-10">
                                        <div className="w-32 h-20 glass-panel rounded border-[#75ff9e]/40 flex items-center justify-center relative group">
                                            <div className="scanline"></div>
                                            <span className="material-symbols-outlined text-[#75ff9e] text-3xl">videocam</span>
                                            <div className="absolute -bottom-8 text-[10px] text-[#bacbb9] font-mono">CAM_01_REAR</div>
                                        </div>
                                        <div className="text-[12px] font-mono text-[#75ff9e]/60 font-bold">60 FPS / 4K</div>
                                    </div>

                                    {/* Central Logic Block */}
                                    <div className="relative z-10">
                                        <div className="w-40 h-40 rounded-full border-2 border-[#75ff9e]/20 flex items-center justify-center animate-[spin_20s_linear_infinite]">
                                            <div className="w-32 h-32 rounded-full border border-dashed border-[#75ff9e]/40 flex items-center justify-center"></div>
                                        </div>
                                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                                            <span className="font-mono text-[#75ff9e] text-2xl font-black">DLT</span>
                                            <span className="text-[10px] text-[#bacbb9] tracking-[4px] font-mono">LOGIC</span>
                                        </div>
                                        {/* Floating Particles */}
                                        <div className="absolute -top-4 -right-4 w-2 h-2 bg-[#75ff9e] rounded-full blur-[1px] animate-pulse"></div>
                                    </div>

                                    {/* Camera Source 2 */}
                                    <div className="flex flex-col items-center gap-4 z-10">
                                        <div className="w-32 h-20 glass-panel rounded border-[#75ff9e]/40 flex items-center justify-center relative group">
                                            <div className="scanline"></div>
                                            <span className="material-symbols-outlined text-[#75ff9e] text-3xl">videocam</span>
                                            <div className="absolute -bottom-8 text-[10px] text-[#bacbb9] font-mono">CAM_02_SIDE</div>
                                        </div>
                                        <div className="text-[12px] font-mono text-[#75ff9e]/60 font-bold">60 FPS / 4K</div>
                                    </div>

                                    {/* Background SVG Lines */}
                                    <svg className="absolute inset-0 w-full h-full pointer-events-none" preserveAspectRatio="none">
                                        <path className="blueprint-line stroke-[#75ff9e]/30 fill-none" d="M 128,150 L 350,150" strokeWidth="1"></path>
                                        <path className="blueprint-line stroke-[#75ff9e]/30 fill-none" d="M 572,150 L 350,150" strokeWidth="1"></path>
                                    </svg>
                                </div>
                                <div className="absolute bottom-4 left-6 text-[10px] font-mono text-[#bacbb9] opacity-40">
                                    MAT[H]: [1.023, 0.002, 0.493; 0.223, 0.998, 0.042; 0.000, 0.000, 1.000]
                                </div>
                            </div>

                            {/* 2. Metadata Display */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div className="glass-panel p-5 rounded-lg border-l-4 border-l-[#75ff9e]">
                                    <label className="font-mono text-[11px] text-[#75ff9e] uppercase tracking-wider mb-2 block font-bold">Focal Length</label>
                                    <div className="flex items-baseline gap-2">
                                        <span className="text-[24px] font-bold font-mono text-[#e5e2e1]">35.404</span>
                                        <span className="text-[#bacbb9] text-[14px]">mm</span>
                                    </div>
                                </div>
                                <div className="glass-panel p-5 rounded-lg border-l-4 border-l-[#75ff9e]">
                                    <label className="font-mono text-[11px] text-[#75ff9e] uppercase tracking-wider mb-2 block font-bold">Distortion (K1)</label>
                                    <div className="flex items-baseline gap-2">
                                        <span className="text-[24px] font-bold font-mono text-[#e5e2e1]">0.000</span>
                                        <span className="text-[#bacbb9] text-[9px] px-2 py-0.5 bg-[#75ff9e]/10 text-[#75ff9e] border border-[#75ff9e]/20 rounded font-mono font-bold">CORRECTED</span>
                                    </div>
                                </div>
                                <div className="glass-panel p-5 rounded-lg border-l-4 border-l-[#75ff9e] bg-[#75ff9e]/[0.02]">
                                    <label className="font-mono text-[11px] text-[#75ff9e] uppercase tracking-wider mb-2 block font-bold">Coordinate Arrays</label>
                                    <div className="space-y-1">
                                        <div className="flex justify-between font-mono text-[10px]">
                                            <span className="text-[#bacbb9]">CAM 1:</span>
                                            <span className="text-[#75ff9e] font-bold">[42.1, 88.3, 12.0]</span>
                                        </div>
                                        <div className="flex justify-between font-mono text-[10px]">
                                            <span className="text-[#bacbb9]">CAM 2:</span>
                                            <span className="text-[#75ff9e] font-bold">[39.8, 86.5, 11.4]</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Sidebar Panels (Col-4) */}
                        <div className="col-span-12 xl:col-span-4 space-y-6">
                            
                            {/* 3. 3D Space-Coord [M] */}
                            <div className="glass-panel rounded-xl p-6">
                                <h3 className="text-[12px] font-medium font-mono text-[#bacbb9] mb-6 flex items-center justify-between">
                                    3D SPACE-COORD [M]
                                    <span className="material-symbols-outlined text-[#75ff9e] text-[18px]">view_in_ar</span>
                                </h3>
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="font-mono text-[#75ff9e] font-bold">X-AXIS</span>
                                        <div className="flex-1 mx-4 h-[1px] bg-white/10"></div>
                                        <span ref={coordRefX} className="text-[20px] font-bold font-mono text-[#e5e2e1]">1.229</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="font-mono text-[#75ff9e] font-bold">Y-AXIS</span>
                                        <div className="flex-1 mx-4 h-[1px] bg-white/10"></div>
                                        <span ref={coordRefY} className="text-[20px] font-bold font-mono text-[#e5e2e1]">0.817</span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="font-mono text-[#75ff9e] font-bold">Z-AXIS</span>
                                        <div className="flex-1 mx-4 h-[1px] bg-white/10"></div>
                                        <span ref={coordRefZ} className="text-[20px] font-bold font-mono text-[#e5e2e1]">2.444</span>
                                    </div>
                                </div>
                            </div>

                            {/* 4. Precision Panel */}
                            <div className="bg-[#75ff9e]/5 border border-[#75ff9e]/20 rounded-xl p-6 relative overflow-hidden">
                                <div className="absolute -right-4 -top-4 opacity-10">
                                    <span className="material-symbols-outlined text-[120px]" style={{ fontVariationSettings: "'FILL' 1" }}>error_med</span>
                                </div>
                                <h3 className="text-[12px] font-medium font-mono text-[#75ff9e] mb-2 uppercase tracking-tighter">Precision Metric</h3>
                                <div className="flex flex-col">
                                    <span className="text-[48px] font-bold text-[#75ff9e] font-inter">1.2mm</span>
                                    <span className="text-[12px] font-medium font-mono text-[#75ff9e]/60">Ultra Precision Threshold</span>
                                </div>
                                <div className="mt-4 flex gap-1">
                                    <div className="h-1 flex-1 bg-[#75ff9e] shadow-[0_0_5px_#75ff9e]"></div>
                                    <div className="h-1 flex-1 bg-[#75ff9e] shadow-[0_0_5px_#75ff9e]"></div>
                                    <div className="h-1 flex-1 bg-[#75ff9e] shadow-[0_0_5px_#75ff9e]"></div>
                                    <div className="h-1 flex-1 bg-[#75ff9e] shadow-[0_0_5px_#75ff9e]"></div>
                                    <div className="h-1 flex-1 bg-[#75ff9e]/20"></div>
                                </div>
                            </div>

                            {/* 5. Transition Pipeline */}
                            <div className="glass-panel rounded-xl p-6">
                                <h3 className="text-[12px] font-medium font-mono text-[#bacbb9] mb-6 uppercase">Pipeline Progress</h3>
                                <div className="space-y-6 relative">
                                    <div className="absolute left-[13px] top-2 bottom-2 w-[1px] bg-white/10"></div>
                                    
                                    {/* Step 01 */}
                                    <div className="flex gap-4 relative">
                                        <div className="w-[28px] h-[28px] rounded-full bg-[#75ff9e] flex items-center justify-center z-10">
                                            <span className="material-symbols-outlined text-[16px] text-[#00210b] font-bold">check</span>
                                        </div>
                                        <div>
                                            <p className="text-[12px] font-medium font-mono text-[#e5e2e1]">01 2D Keypoint Detection</p>
                                            <p className="text-[11px] text-[#75ff9e] uppercase font-bold font-mono pt-1">Complete</p>
                                        </div>
                                    </div>
                                    
                                    {/* Step 02 */}
                                    <div className="flex gap-4 relative">
                                        <div className="w-[28px] h-[28px] rounded-full bg-[#131313] border border-[#75ff9e] flex items-center justify-center z-10">
                                            <div className="w-1.5 h-1.5 bg-[#75ff9e] rounded-full animate-ping"></div>
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex justify-between items-start">
                                                <p className="text-[12px] font-medium font-mono text-[#e5e2e1]">02 Triangulation (DLT)</p>
                                                <span className="text-[11px] font-mono text-[#75ff9e] font-bold">88%</span>
                                            </div>
                                            <div className="w-full h-1 bg-white/10 mt-2 rounded-full overflow-hidden">
                                                <div className="h-full bg-[#75ff9e] w-[88%] shadow-[0_0_8px_#75ff9e]"></div>
                                            </div>
                                            <p className="text-[10px] font-mono text-[#bacbb9] mt-1">Processing Vector Fields...</p>
                                        </div>
                                    </div>
                                    
                                    {/* Step 03 */}
                                    <div className="flex gap-4 relative opacity-40">
                                        <div className="w-[28px] h-[28px] rounded-full bg-[#131313] border border-white/20 flex items-center justify-center z-10">
                                            <span className="material-symbols-outlined text-[14px]">lock</span>
                                        </div>
                                        <div>
                                            <p className="text-[12px] font-medium font-mono text-[#e5e2e1]">03 3D Skeleton-based Pose</p>
                                            <p className="text-[11px] font-mono text-[#bacbb9] uppercase pt-1 tracking-wider">Queued</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Primary CTA */}
                            <button onClick={() => window.location.href='/analysis'} className="w-full py-5 bg-[#75ff9e] text-[#00210b] font-black text-[16px] uppercase tracking-widest rounded-xl glow-green hover:brightness-110 transition-all flex items-center justify-center gap-3 group active:scale-[0.98]">
                                ANALYZE 3D MODEL
                                <span className="material-symbols-outlined group-hover:translate-x-2 transition-transform">arrow_forward</span>
                            </button>
                        </div>
                    </div>

                    {/* Detail Grid Footer (Geometric Aesthetic) */}
                    <div className="grid grid-cols-12 gap-6 pb-12">
                        {/* 3D Skeleton Visualization */}
                        <div className="col-span-12 md:col-span-6 glass-panel rounded-xl p-6 h-64 relative overflow-hidden group border-[#75ff9e]/20">
                            <img className="absolute inset-0 w-full h-full object-cover opacity-40 group-hover:scale-105 transition-all duration-700" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAEZYQpv4oobVQJdYkm3PAcmtwWS7Lnyy_Qb4eL8fv_s_H1Muln-xF8p7oWVVz_XWSN_6Trxf-Rxd9zsdw0v0TnzXT5q9omsMMrh0r4qqu7A9ZYhDU0KiWU6zTazBwfX7dBysdwqPjwg9v2cfxxtI1083rG6nniz4yADQu4hgm0P-LhfqReQ2TPLGTniMAddugeV68ZP9alxXhAwj7TAhLzmLnwrSmI-JFVYjbmxXwYhPlF5Er0W3dkSDOjqCrA7osM64AJw8MPqNiD" />
                            <div className="absolute inset-0 bg-gradient-to-t from-[rgba(19,19,19,0.9)] to-transparent opacity-60"></div>
                            <div className="relative z-10 h-full flex flex-col justify-between">
                                <span className="text-[12px] font-medium font-mono text-[#75ff9e] flex items-center gap-2">
                                    <span className="material-symbols-outlined text-[16px]">accessibility_new</span>
                                    3D SKELETON RECONSTRUCTION
                                </span>
                                <div className="p-4 bg-black/60 backdrop-blur-md border border-white/10 w-fit rounded-lg">
                                    <p className="text-[12px] font-mono text-[#75ff9e] font-bold">LATENCY: 12ms</p>
                                    <p className="text-[12px] font-mono text-[#bacbb9]">STABILITY: 99.8%</p>
                                </div>
                            </div>
                        </div>
                        
                        <div className="col-span-12 md:col-span-6 glass-panel rounded-xl p-6 h-64 relative flex items-center justify-center border border-[#3b4a3d]">
                            <div className="text-center space-y-4">
                                <div className="w-16 h-16 rounded-full border border-[#75ff9e]/40 mx-auto flex items-center justify-center animate-pulse glow-green">
                                    <span className="material-symbols-outlined text-[#75ff9e] text-3xl">data_exploration</span>
                                </div>
                                <h4 className="text-[24px] font-bold text-[#e5e2e1] font-inter">Ready for Biometric Analysis</h4>
                                <p className="text-[16px] text-[#bacbb9] max-w-sm mx-auto leading-relaxed">Reconstruction complete for frame range 0-144. Kinematic sequencing engine is ready.</p>
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
