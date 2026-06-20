"use client";
import React, { useEffect } from 'react';
import Head from 'next/head';

export default function SplittingPage() {
    
    // Progress bar micro-interaction simulation
    useEffect(() => {
        const interval = setInterval(() => {
            const pulseBar = document.getElementById('progress-bar-primary');
            if (pulseBar && Math.random() > 0.8) {
                const currentWidth = parseFloat(pulseBar.style.width) || 84;
                const newWidth = Math.min(84.9, currentWidth + (Math.random() * 0.05));
                pulseBar.style.width = newWidth + '%';
            }
        }, 2000);
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
                .glow-green {
                    box-shadow: 0 0 20px rgba(0, 230, 118, 0.15);
                }
                .active-glow {
                    box-shadow: 0 0 15px rgba(117, 255, 158, 0.4);
                }
                .step-inactive { opacity: 0.6; }
                .step-active { color: #75ff9e; position: relative; }
                .step-active::after {
                    content: '';
                    position: absolute;
                    bottom: -8px;
                    left: 0;
                    right: 0;
                    height: 2px;
                    background: #75ff9e;
                }
            `}</style>

            {/* TopNavBar */}
            <nav className="flex justify-between items-center px-6 py-4 w-full sticky top-0 z-50 bg-[#131313]/80 backdrop-blur-xl border-b border-white/10 h-16">
                <a href="/" className="text-[24px] text-[#75ff9e] tracking-tight font-bold font-inter hover:opacity-80 transition-opacity">PRO SWING AI</a>
                <div className="hidden md:flex items-center space-x-12">
                    <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono flex items-center gap-1" href="/acquisition">
                        <span className="text-[10px] opacity-60">01</span> Acquisition
                    </a>
                    <a className="step-active font-bold text-[12px] font-mono flex items-center gap-1" href="/splitting">
                        <span className="text-[10px]">02</span> Preprocessing
                    </a>
                    <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono flex items-center gap-1" href="#">
                        <span className="text-[10px] opacity-60">03</span> Fine-Tuning
                    </a>
                    <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono flex items-center gap-1" href="/analysis">
                        <span className="text-[10px] opacity-60">04</span> Reconstruction
                    </a>
                    <a className="text-[#bacbb9] hover:text-[#e5e2e1] transition-colors duration-200 text-[12px] font-medium font-mono flex items-center gap-1" href="#">
                        <span className="text-[10px] opacity-60">05</span> Analysis
                    </a>
                </div>
                <div className="flex items-center space-x-4">
                    <button className="material-symbols-outlined text-[#bacbb9] hover:text-[#75ff9e] transition-all">notifications</button>
                    <button className="material-symbols-outlined text-[#bacbb9] hover:text-[#75ff9e] transition-all">settings</button>
                    <button className="material-symbols-outlined text-[#e5e2e1] transition-all">account_circle</button>
                </div>
            </nav>

            {/* Sidebar & Main Content Wrapper */}
            <div className="flex min-h-[calc(100vh-64px)]">
                {/* SideNavBar */}
                <aside className="hidden lg:flex flex-col h-[calc(100vh-64px)] w-64 fixed left-0 top-16 bg-[#1c1b1b] border-r border-[#3b4a3d] py-8 px-4 space-y-6 z-40">
                    <div className="mb-8">
                        <div className="text-[20px] font-bold text-[#75ff9e] uppercase font-inter">PIPELINE CTRL</div>
                        <div className="text-[#bacbb9] text-xs opacity-60 font-mono">V3.4 Stable</div>
                    </div>
                    <nav className="flex-1 space-y-2">
                        <button onClick={() => window.location.href='/history'} className="flex items-center w-full px-4 py-3 text-[#bacbb9] hover:bg-white/5 transition-all group rounded-lg">
                            <span className="material-symbols-outlined mr-3">video_library</span>
                            <span className="text-[12px] font-medium font-mono">Recent Captures</span>
                        </button>
                        <button className="flex items-center w-full px-4 py-3 bg-[#75ff9e]/10 text-[#75ff9e] border-r-4 border-[#75ff9e] transition-all rounded-l-lg">
                            <span className="material-symbols-outlined mr-3" style={{ fontVariationSettings: "'FILL' 1" }}>settings_suggest</span>
                            <span className="text-[12px] font-medium font-mono">Batch Process</span>
                        </button>
                        <button onClick={() => window.location.href='/analysis'} className="flex items-center w-full px-4 py-3 text-[#bacbb9] hover:bg-white/5 transition-all group rounded-lg">
                            <span className="material-symbols-outlined mr-3">monitoring</span>
                            <span className="text-[12px] font-medium font-mono">Model Metrics</span>
                        </button>
                        <button className="flex items-center w-full px-4 py-3 text-[#bacbb9] hover:bg-white/5 transition-all group rounded-lg">
                            <span className="material-symbols-outlined mr-3">ios_share</span>
                            <span className="text-[12px] font-medium font-mono">Export</span>
                        </button>
                    </nav>
                    <button onClick={() => window.location.href='/'} className="w-full bg-[#75ff9e] text-[#00210b] py-3 font-bold uppercase tracking-wider text-xs active:scale-95 transition-all rounded-lg shadow-[0_0_15px_rgba(117,255,158,0.2)]">
                        New Session
                    </button>
                    <div className="pt-6 border-t border-white/5 space-y-2">
                        <button className="flex items-center w-full px-4 py-2 text-[#bacbb9] hover:text-[#e5e2e1] text-xs font-mono">
                            <span className="material-symbols-outlined mr-2 text-sm">menu_book</span> Docs
                        </button>
                        <button onClick={() => window.location.href='/login'} className="flex items-center w-full px-4 py-2 text-[#bacbb9] hover:text-[#ffb4ab] text-xs font-mono transition-colors">
                            <span className="material-symbols-outlined mr-2 text-sm">logout</span> Logout
                        </button>
                    </div>
                </aside>

                {/* Main Canvas */}
                <main className="flex-1 lg:ml-64 p-6 md:p-10 max-w-[1440px] mx-auto">
                    {/* Page Header */}
                    <header className="mb-10 flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
                        <div>
                            <span className="text-[#75ff9e] text-sm uppercase tracking-widest block mb-1 font-mono font-medium">STEP 02/05</span>
                            <h1 className="text-[40px] font-bold leading-tight mb-2 font-inter text-[#e5e2e1]">Preprocessing & Splitting</h1>
                            <p className="text-[#bacbb9] max-w-2xl text-[16px] font-inter leading-relaxed">Refining multi-view golf swing captures into synchronized datasets for deep skeletal reconstruction. Current node: <span className="text-[#75ff9e] font-bold">Training Set Distribution</span>.</p>
                        </div>
                        <div className="flex gap-4">
                            <div className="glass-panel p-4 flex flex-col items-center min-w-[120px] rounded-lg">
                                <span className="text-[10px] font-medium font-mono text-[#bacbb9] mb-1 uppercase tracking-wider">Annotated Vids</span>
                                <span className="text-[20px] font-bold font-mono text-[#75ff9e]">1,400</span>
                            </div>
                            <div className="glass-panel p-4 flex flex-col items-center min-w-[120px] rounded-lg">
                                <span className="text-[10px] font-medium font-mono text-[#bacbb9] mb-1 uppercase tracking-wider">KeyPoints/Shot</span>
                                <span className="text-[20px] font-bold font-mono text-[#75ff9e]">25</span>
                            </div>
                        </div>
                    </header>

                    {/* Grid Layout */}
                    <div className="grid grid-cols-12 gap-6">
                        
                        {/* Dataset Distribution & Task Queue (Left Side) */}
                        <section className="col-span-12 lg:col-span-8 space-y-6">
                            
                            {/* Data Splitting Logic */}
                            <div className="glass-panel p-8 rounded-xl relative overflow-hidden group">
                                <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
                                    <span className="material-symbols-outlined text-[120px]">hub</span>
                                </div>
                                <div className="flex items-center justify-between mb-8">
                                    <h2 className="text-[20px] font-bold text-[#e5e2e1] font-inter flex items-center">
                                        <span className="material-symbols-outlined mr-3 text-[#75ff9e]">schema</span>
                                        Data Splitting Logic
                                    </h2>
                                    <span className="bg-[#75ff9e]/20 text-[#75ff9e] px-3 py-1 text-[10px] font-bold tracking-widest uppercase rounded border border-[#75ff9e]/30">Processing Active</span>
                                </div>

                                {/* Split Visualization (Standardized) */}
                                <div className="relative py-4 px-2">
                                    <div className="flex gap-4 items-stretch mb-8">
                                        <div className="flex-1 glass-panel p-6 border-[#75ff9e]/30 relative active-glow rounded-xl">
                                            <div className="flex justify-between items-end mb-3">
                                                <span className="text-[10px] font-medium font-mono uppercase tracking-widest text-[#75ff9e]">Training Set</span>
                                                <span className="text-3xl font-bold font-inter text-[#e5e2e1]">80<span className="text-sm font-normal">%</span></span>
                                            </div>
                                            <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                                                <div className="bg-[#75ff9e] h-full w-[80%] glow-green"></div>
                                            </div>
                                        </div>
                                        <div className="w-[30%] glass-panel p-6 border-white/5 bg-white/5 rounded-xl">
                                            <div className="flex justify-between items-end mb-3">
                                                <span className="text-[10px] font-medium font-mono uppercase tracking-widest text-[#bacbb9]">Validation</span>
                                                <span className="text-2xl font-bold font-mono text-[#e5e2e1]">20<span className="text-sm font-normal">%</span></span>
                                            </div>
                                            <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                                                <div className="bg-[#bacbb9] h-full w-[20%] opacity-50"></div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    {/* Detailed Branching */}
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-4 bg-white/5 border border-white/5 rounded-lg flex items-center justify-between">
                                            <div>
                                                <span className="text-[10px] text-[#bacbb9] uppercase block font-mono">Fine-Tuning Vids</span>
                                                <span className="text-xl font-bold font-inter text-[#e5e2e1]">60 <span className="text-xs font-mono font-normal text-[#859585]">FILES</span></span>
                                            </div>
                                            <span className="text-[#75ff9e] font-bold font-mono text-sm">70%</span>
                                        </div>
                                        <div className="p-4 bg-white/5 border border-white/5 rounded-lg flex items-center justify-between opacity-60">
                                            <div>
                                                <span className="text-[10px] text-[#bacbb9] uppercase block font-mono">Control Group</span>
                                                <span className="text-xl font-bold font-inter text-[#e5e2e1]">40 <span className="text-xs font-mono font-normal text-[#859585]">FILES</span></span>
                                            </div>
                                            <span className="text-[#bacbb9] font-bold font-mono text-sm">30%</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="mt-8 pt-8 border-t border-white/5 flex gap-12">
                                    <div>
                                        <div className="text-[10px] text-[#bacbb9] uppercase mb-1 tracking-widest font-mono">Methodology</div>
                                        <div className="text-sm font-semibold font-inter text-[#e5e2e1]">Random Seed Stratification</div>
                                    </div>
                                    <div>
                                        <div className="text-[10px] text-[#bacbb9] uppercase mb-1 tracking-widest font-mono">Integrity</div>
                                        <div className="text-sm font-semibold text-[#75ff9e] font-inter">Verified Hash (SHA-256)</div>
                                    </div>
                                </div>
                            </div>

                            {/* Task Queue Progress (Standardized) */}
                            <div className="glass-panel p-8 rounded-xl">
                                <div className="flex items-center justify-between mb-8">
                                    <h2 className="text-[20px] font-bold text-[#e5e2e1] flex items-center uppercase tracking-tight font-inter">
                                        <span className="material-symbols-outlined mr-3 text-[#75ff9e]">list_alt</span>
                                        Task Queue Status
                                    </h2>
                                </div>
                                <div className="space-y-8">
                                    {/* Task 1 */}
                                    <div className="group">
                                        <div className="flex justify-between items-center mb-3">
                                            <span className="flex items-center text-sm font-medium tracking-wide text-[#e5e2e1] font-inter">
                                                <span className="material-symbols-outlined text-[#75ff9e] mr-3 text-lg">sync_alt</span>
                                                Frame Synchronization (Multi-Cam)
                                            </span>
                                            <span className="text-xs font-mono text-[#75ff9e] font-bold">84%</span>
                                        </div>
                                        <div className="w-full h-[6px] bg-white/5 rounded-full overflow-hidden">
                                            <div id="progress-bar-primary" className="h-full bg-[#75ff9e] w-[84%] animate-pulse glow-green transition-all duration-1000"></div>
                                        </div>
                                    </div>
                                    {/* Task 2 */}
                                    <div className="group">
                                        <div className="flex justify-between items-center mb-3">
                                            <span className="flex items-center text-sm font-medium tracking-wide text-[#e5e2e1] font-inter">
                                                <span className="material-symbols-outlined text-[#75ff9e] mr-3 text-lg">align_vertical_bottom</span>
                                                Impact Alignment Metrics
                                            </span>
                                            <span className="text-xs font-mono text-[#75ff9e] font-bold">62%</span>
                                        </div>
                                        <div className="w-full h-[6px] bg-white/5 rounded-full overflow-hidden">
                                            <div className="h-full bg-[#75ff9e] w-[62%] transition-all duration-1000"></div>
                                        </div>
                                    </div>
                                    {/* Task 3 */}
                                    <div className="group">
                                        <div className="flex justify-between items-center mb-3">
                                            <span className="flex items-center text-sm font-medium tracking-wide text-[#e5e2e1] font-inter">
                                                <span className="material-symbols-outlined text-[#75ff9e] mr-3 text-lg">check_circle</span>
                                                Keypoint Extraction Pipeline
                                            </span>
                                            <span className="text-xs font-mono text-[#75ff9e] font-bold">100%</span>
                                        </div>
                                        <div className="w-full h-[6px] bg-white/5 rounded-full overflow-hidden">
                                            <div className="h-full bg-[#75ff9e] w-[100%] transition-all duration-1000 shadow-[0_0_8px_#75ff9e]"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </section>

                        {/* Right Sidebar Details (Right Side) */}
                        <section className="col-span-12 lg:col-span-4 space-y-6">
                            
                            {/* Dataset Overview */}
                            <div className="glass-panel p-6 rounded-xl border-l-4 border-l-[#75ff9e]">
                                <h3 className="text-[10px] font-medium font-mono uppercase tracking-widest text-[#75ff9e] mb-4">GolfSwing Dataset V1</h3>
                                <div className="aspect-video bg-[#353534] mb-6 relative rounded-lg overflow-hidden border border-[#3b4a3d]">
                                    <img alt="Swing Analysis Ground Truth" className="w-full h-full object-cover opacity-60 grayscale hover:grayscale-0 transition-all duration-700" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAZra_mDeO6xRaPwOgkHDu6IDe2mmvaFE4u_szvkuAvQdxkBrrF_ACSJ9pF7_emkjyEVaom-2e7nCuLco40DSoooKHL7EWazjUxu3SykG62oWwCT2gVvhiyINHqzS76oFKIlRfFguCFJWyR5Nqaom6wSxPrvs-MPNNrb3GZKkEeMlFRDd8VABWkm0fpBgeSLZnvWL-r4rCv_TmpUotIO3CalX3o7hbZ8JJw1x-cyaV-AHG9d3CT5OBux4S_-NPfZQaARh5SdcYVPykg" />
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <span className="bg-[#131313]/80 backdrop-blur-md px-4 py-2 rounded-full text-[10px] uppercase font-bold font-mono tracking-widest border border-white/10 text-[#75ff9e]">3D Ground Truth Active</span>
                                    </div>
                                </div>
                                <ul className="space-y-4 text-sm font-medium font-inter">
                                    <li className="flex justify-between items-center text-[#e5e2e1]">
                                        <span className="text-[#bacbb9] text-[12px] font-mono">Views:</span>
                                        <span>Front / Side-On</span>
                                    </li>
                                    <li className="flex justify-between items-center text-[#e5e2e1]">
                                        <span className="text-[#bacbb9] text-[12px] font-mono">KeyPoints:</span>
                                        <span className="font-mono">25 COCO-Format</span>
                                    </li>
                                    <li className="flex justify-between items-center text-[#e5e2e1]">
                                        <span className="text-[#bacbb9] text-[12px] font-mono">Precision:</span>
                                        <span className="text-[#75ff9e] font-bold">Sub-mm Accuracy</span>
                                    </li>
                                </ul>
                            </div>

                            {/* Pipeline Architecture: Model Config */}
                            <div className="glass-panel p-6 rounded-xl space-y-6">
                                <div className="flex items-center space-x-3 text-[#e5e2e1]">
                                    <span className="material-symbols-outlined text-[#75ff9e]">memory</span>
                                    <span className="text-[20px] font-bold uppercase tracking-tight font-inter">Model Config</span>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-4 bg-[#201f1f] rounded-lg border border-[#3b4a3d] shadow-inner">
                                        <span className="text-[10px] text-[#bacbb9] uppercase block mb-1 font-mono tracking-wider">Architecture</span>
                                        <span className="text-sm font-bold text-[#e5e2e1]">HRNet + MPJPE</span>
                                    </div>
                                    <div className="p-4 bg-[#201f1f] rounded-lg border border-[#3b4a3d] shadow-inner">
                                        <span className="text-[10px] text-[#bacbb9] uppercase block mb-1 font-mono tracking-wider">Complexity</span>
                                        <span className="text-sm font-bold text-[#e5e2e1]">Lvl 3 (Heavy)</span>
                                    </div>
                                </div>
                            </div>

                            {/* CTA Section */}
                            <div className="pt-4">
                                <button className="w-full py-5 bg-[#75ff9e] text-[#00210b] rounded-xl font-black tracking-[0.2em] text-sm flex items-center justify-center group hover:bg-[#00e475] transition-all active:scale-[0.98] shadow-[0_0_20px_rgba(117,255,158,0.2)] hover:shadow-[0_0_30px_rgba(117,255,158,0.4)]">
                                    INITIATE MODEL TRAINING
                                    <span className="material-symbols-outlined ml-3 transition-transform group-hover:translate-x-2">arrow_forward</span>
                                </button>
                                <p className="text-center text-[10px] text-[#bacbb9] mt-4 uppercase font-mono tracking-tighter opacity-80">
                                    Approx. Training Time: 4h 12m (RTX 4090 x4 Cluster)
                                </p>
                            </div>
                        </section>
                    </div>

                    {/* Global Footer Status */}
                    <footer className="mt-20 border-t border-white/5 pt-8 pb-12 flex flex-col md:flex-row justify-between items-center gap-6 text-[11px] text-[#bacbb9] uppercase tracking-widest font-mono">
                        <div className="flex flex-col sm:flex-row items-center space-y-4 sm:space-y-0 sm:space-x-8">
                            <span className="flex items-center font-bold text-[#75ff9e]"><span className="w-2 h-2 rounded-full bg-[#75ff9e] mr-2 glow-green animate-pulse"></span> SYSTEM READY</span>
                            <span>LATENCY: 12ms</span>
                            <span>ENCRYPTION: AES-256</span>
                        </div>
                        <div className="font-mono opacity-60">
                            Build: psa_v3.4.0_stable_901
                        </div>
                    </footer>
                </main>
            </div>

            {/* Background Decoration */}
            <div className="fixed bottom-12 right-12 pointer-events-none opacity-10 hidden 2xl:block z-0">
                <div className="w-72 h-72 border-[1px] border-[#75ff9e]/40 rounded-full animate-[spin_30s_linear_infinite] flex items-center justify-center">
                    <div className="w-48 h-48 border-[1px] border-[#75ff9e]/20 rounded-full animate-[spin_20s_linear_infinite_reverse]"></div>
                </div>
                <div className="absolute inset-0 flex items-center justify-center">
                    <span className="material-symbols-outlined text-[#75ff9e] text-5xl">slow_motion_video</span>
                </div>
            </div>

        </div>
    );
}
