"use client";
import React, { useEffect, useRef, useState } from 'react';

export default function AnalysisPage() {
    const [isPlaying, setIsPlaying] = useState(false);
    const panelRefs = useRef<(HTMLDivElement | null)[]>([]);

    useEffect(() => {
        // Subtle HUD parallax effect
        const handleMouseMove = (e: MouseEvent) => {
            const x = (e.clientX / window.innerWidth - 0.5) * 10;
            const y = (e.clientY / window.innerHeight - 0.5) * 10;
            panelRefs.current.forEach((el, index) => {
                if (el) {
                    const factor = (index + 1) * 0.2;
                    el.style.transform = `translate(${x * factor}px, ${y * factor}px)`;
                }
            });
        };
        document.addEventListener('mousemove', handleMouseMove);
        return () => document.removeEventListener('mousemove', handleMouseMove);
    }, []);

    const setRef = (index: number) => (el: HTMLDivElement | null) => {
        panelRefs.current[index] = el;
    };

    return (
        <div className="bg-[#131313] text-[#e5e2e1] font-sans min-h-screen flex flex-col overflow-hidden selection:bg-[#75ff9e] selection:text-[#00210b]">
            <style jsx global>{`
                .glass-panel {
                    background: rgba(26, 28, 30, 0.4);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .material-symbols-outlined {
                    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
                    vertical-align: middle;
                }
                .performance-glow {
                    box-shadow: 0 0 15px rgba(117, 255, 158, 0.2);
                }
                .grid-bg {
                    background-image: 
                        linear-gradient(to right, rgba(0, 230, 118, 0.1) 1px, transparent 1px),
                        linear-gradient(to bottom, rgba(0, 230, 118, 0.1) 1px, transparent 1px);
                    background-size: 40px 40px;
                }
                .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
                .custom-scrollbar::-webkit-scrollbar-track { background: #131313; }
                .custom-scrollbar::-webkit-scrollbar-thumb { background: #353534; border-radius: 10px; }
            `}</style>

            {/* TopAppBar */}
            <header className="flex justify-between items-center w-full px-4 h-16 z-50 bg-[#131313]/80 backdrop-blur-md border-b border-[#3b4a3d] fixed top-0 left-0">
                <a href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
                    <span className="material-symbols-outlined text-[#75ff9e]">analytics</span>
                    <h1 className="text-[32px] font-bold text-[#75ff9e] tracking-tight font-inter">PRO SWING AI</h1>
                </a>
                <nav className="hidden md:flex items-center gap-8">
                    <a className="text-[#75ff9e] font-medium font-mono text-[12px] transition-colors hover:text-[#00e475]" href="/">Analyze</a>
                    <a className="text-[#bacbb9] font-medium font-mono text-[12px] transition-colors hover:text-[#75ff9e]" href="/history">History</a>
                    <a className="text-[#bacbb9] font-medium font-mono text-[12px] transition-colors hover:text-[#75ff9e]" href="#">Insights</a>
                </nav>
                <div className="flex items-center gap-4">
                    <button className="material-symbols-outlined text-[#bacbb9] hover:bg-[#353534]/50 p-2 rounded-full transition-colors">settings</button>
                    <div className="w-10 h-10 rounded-full border border-[#75ff9e]/20 overflow-hidden">
                        <img alt="User Profile" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAx0ksRGQGAv1FxdIriK2IGHYDSVI_qiDWlty3OnCz3Y4lT_vDDpVBoiMFyCIORShghXUDpK1BmlsEMuNgXjx-4T8SDxR8u6NfVWEgmzvJ9ZBVqufvv3ju6KW8CfhyqVlXGvIF8By8lpc1WDfR881q0SP1cHir1WXXr9wi5vm-GVNyeGsYBg1A1ikupC6-Hvgslr_2iGli4XGt5B_7NCYLrM4EiwI4h-QmDyzXHRthebMP6_whwVcAF1KJ5up1S1zlLqpPAgw6i0r81" />
                    </div>
                </div>
            </header>

            {/* Main Viewport Area */}
            <main className="flex-grow pt-16 flex flex-col md:flex-row overflow-hidden">
                
                {/* 3D Analysis Viewport */}
                <section className="relative flex-grow flex flex-col bg-[#0e0e0e] overflow-hidden group">
                    <div className="relative flex-grow overflow-hidden flex items-center justify-center">
                        <div className="absolute inset-0 grid-bg opacity-30"></div>
                        <img alt="3D Swing Reconstruction" className="relative z-10 w-full h-full object-contain mix-blend-screen opacity-90" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDnV_3AxtCi8STKeL6GGkBong0SrJsKJXP7uxXjPcrx4vcipb0iZUdXAwbyTejHO6ycU1-ZBdh6VoTmjKPKVWZ1_5RarmCAMdZ0CejFL0BMdRf6LQFXwK7ORMvzTyS4ECwygcwyDLXzh5qTAHkZqCdA6IZNv20ZmK9Pq12f2J78NmDd1ZYC_KOH2Kmrkx5Ie_ozN2e0_7B9pRl592naMPGsPW-GSCCFbBOQburlGdBy2i79Q1qSkYFfFQ9pMJUgHIMx3yMf3busC2gc" />
                        
                        {/* HUD Overlays */}
                        <div className="absolute inset-0 pointer-events-none z-20">
                            {/* Floating Rotation Speed */}
                            <div ref={setRef(0)} className="absolute top-12 left-12 glass-panel p-6 rounded-2xl border-l-4 border-l-[#75ff9e] performance-glow transition-transform ease-out duration-75">
                                <p className="text-[10px] text-[#bacbb9] uppercase tracking-widest font-bold mb-1 font-mono">Rotation Speed</p>
                                <div className="flex items-baseline gap-2">
                                    <span className="text-[48px] font-bold text-white leading-none">102</span>
                                    <span className="text-[#75ff9e] font-bold">MPH</span>
                                </div>
                            </div>
                            
                            {/* Impact Point Marker */}
                            <div className="absolute top-1/2 left-2/3 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-2">
                                <div className="w-4 h-4 rounded-full border-2 border-[#75ff9e] animate-ping"></div>
                                <div className="w-2 h-2 rounded-full bg-[#75ff9e] absolute top-1 left-1"></div>
                                <div ref={setRef(1)} className="glass-panel px-2 py-1 rounded text-[10px] text-[#75ff9e] font-bold uppercase tracking-tighter mt-4 transition-transform ease-out duration-100">Impact Point</div>
                            </div>

                            {/* Perspective Indicator */}
                            <div className="absolute top-12 right-12 flex flex-col gap-2">
                                <button className="glass-panel p-3 rounded-full text-[#e5e2e1] hover:text-[#75ff9e] transition-colors pointer-events-auto">
                                    <span className="material-symbols-outlined">videocam</span>
                                </button>
                                <button className="glass-panel p-3 rounded-full text-[#e5e2e1] hover:text-[#75ff9e] transition-colors pointer-events-auto">
                                    <span className="material-symbols-outlined">3d_rotation</span>
                                </button>
                            </div>
                        </div>

                        {/* Bottom Playback Controls */}
                        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-[90%] md:w-[600px] z-30">
                            <div className="glass-panel p-4 rounded-2xl flex flex-col gap-4 shadow-xl">
                                <div className="flex items-center gap-4">
                                    <button 
                                        onClick={() => setIsPlaying(!isPlaying)}
                                        className="w-10 h-10 flex items-center justify-center text-white hover:text-[#75ff9e] transition-colors"
                                    >
                                        <span className="material-symbols-outlined text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>
                                            {isPlaying ? 'pause' : 'play_arrow'}
                                        </span>
                                    </button>
                                    <div className="flex-grow h-1.5 bg-[#353534] rounded-full relative overflow-hidden cursor-pointer group/scrub">
                                        <div className="absolute inset-0 bg-[#75ff9e]/20 w-3/4"></div>
                                        <div className="absolute top-0 bottom-0 left-[75%] w-0.5 bg-white shadow-[0_0_8px_white]"></div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button className="material-symbols-outlined text-[#bacbb9] hover:text-white">chevron_left</button>
                                        <span className="text-[11px] font-mono text-[#bacbb9]">142/300</span>
                                        <button className="material-symbols-outlined text-[#bacbb9] hover:text-white">chevron_right</button>
                                    </div>
                                </div>
                                <div className="flex items-center justify-between px-1">
                                    <div className="flex items-center gap-4">
                                        <button className="text-[10px] font-bold text-[#bacbb9] hover:text-[#75ff9e] transition-colors uppercase font-mono">0.25x</button>
                                        <button className="text-[10px] font-bold text-[#75ff9e] uppercase font-mono">1.0x</button>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-[#75ff9e] animate-pulse"></div>
                                        <span className="text-[10px] font-bold text-[#75ff9e] uppercase tracking-widest font-mono">Live AI Analysis</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Swing Phase Navigation */}
                    <div className="h-28 bg-[#0e0e0e] border-t border-[#3b4a3d] px-8 flex items-center overflow-x-auto gap-4 custom-scrollbar shrink-0">
                        {/* Setup */}
                        <button className="flex-shrink-0 w-32 flex flex-col gap-1 group">
                            <div className="aspect-video bg-[#201f1f] rounded-lg border border-[#3b4a3d] overflow-hidden relative group-hover:border-[#75ff9e]/50 transition-all">
                                <img alt="Setup" className="w-full h-full object-cover opacity-40 group-hover:opacity-60 transition-opacity" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAUT5pGvVExIiTPaXKXm9wTjJu-MQM5o186maKIdSJIXFRoxCiYvjmVXr32HRXKRgV0OD3ikQEOJQOwMq8R4mNDkE74sFgDycSP1FbdWIEx3KCeILEenLHkMN7jSQrVeopT15vR3uxPUaa-93Ea1G_2BxB6NKQV88YqHQQJbIuV6RXQipmv4-QqZR1r8xsF9mcOyXt7MeMe2ZD2NnAt-lzMYEqmZdvCZl7L_S_3-dgEo7oC1peZLR11zhKxx2hOPARBgro5rYJ9EU7U" />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                                <span className="absolute bottom-1 left-2 text-[9px] font-bold text-white uppercase tracking-tighter">Setup</span>
                            </div>
                            <div className="h-0.5 w-full bg-[#3b4a3d] rounded-full"></div>
                        </button>
                        {/* Backswing */}
                        <button className="flex-shrink-0 w-32 flex flex-col gap-1 group">
                            <div className="aspect-video bg-[#201f1f] rounded-lg border border-[#3b4a3d] overflow-hidden relative group-hover:border-[#75ff9e]/50 transition-all">
                                <img alt="Backswing" className="w-full h-full object-cover opacity-40 group-hover:opacity-60 transition-opacity" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDzOgbGd0UCKZo2KHQ9WhHAhGxVjmv7_4BcA-Bx3tfLRI45yKUxoGJYURTrpRKufHWAxs6U1sQuiafiyuzlmbcU_YZShVea7EjtwcLX9vjrNjt-Yt94Ucc_jc-2o3pXinxJi2FRxSmdIbPQnTD4c0f0RFNJlrsKtiBJE9b1yVFjc3JRQ6LLay5qJLBrldjdAYOi9gWRaHizpIgjJrrM1dNPDJkIo3aZ45W1XUrn0GQHrLTcYb9kmw17e3Js1htQvGImzP1X8MUI9y53" />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                                <span className="absolute bottom-1 left-2 text-[9px] font-bold text-white uppercase tracking-tighter">Backswing</span>
                            </div>
                            <div className="h-0.5 w-full bg-[#3b4a3d] rounded-full"></div>
                        </button>
                        {/* Downswing (Active) */}
                        <button className="flex-shrink-0 w-32 flex flex-col gap-1 group">
                            <div className="aspect-video bg-[#201f1f] rounded-lg border-2 border-[#75ff9e] overflow-hidden relative performance-glow">
                                <img alt="Downswing" className="w-full h-full object-cover opacity-80" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCR7kbm269yZTzJ5FubsK6jWTeqpF0xgp7hZ0vRa_1kbWraBoCB3E79IoDuL-oVxnMHg97KCyWFQD3gz7K1rBZ9q47uNEyDK6sqozqlD-kMRZGSA6nGctXBKgRE1cTE44zImj0AQXcAfCmqwI7-IKZ-fHHKJdLq0SYHlMv8nTvc33_CO5JbDAdYC5TLBocX44LLas1C8kfToTcflEw62vbrylJiALYWOpJXVu5fFgE00QZSH-vSsmSErn6K0t4mC4GVfvLFJL3dzA4y" />
                                <div className="absolute inset-0 bg-[#75ff9e]/20"></div>
                                <div className="absolute inset-0 bg-gradient-to-t from-[#75ff9e]/60 to-transparent"></div>
                                <span className="absolute bottom-1 left-2 text-[9px] font-bold text-white uppercase tracking-tighter">Downswing</span>
                            </div>
                            <div className="h-0.5 w-full bg-[#75ff9e] rounded-full shadow-[0_0_5px_#75ff9e]"></div>
                        </button>
                        {/* Impact */}
                        <button className="flex-shrink-0 w-32 flex flex-col gap-1 group">
                            <div className="aspect-video bg-[#201f1f] rounded-lg border border-[#3b4a3d] overflow-hidden relative group-hover:border-[#75ff9e]/50 transition-all">
                                <img alt="Impact" className="w-full h-full object-cover opacity-40 group-hover:opacity-60 transition-opacity" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCAzSM3XFk_6ym-U0Aymj7qV13kSPbGC2zmfv1-J0wM4OIHYvkiISAtI1Z1a7Nw1FC30LaYEc3i7EHG1npvxmeraY3CGTMqQS_bwQ3ORCe_nv7KuDEmsusmnLV-gSQQU_QWuLXTHMCfg3p3znh_F3koFgKJ_xmbareYjagb-J5LFiEGA6OUZmPtxeAEJREXSPDz7bZbUQt9MOTCQQ9Cp1vGz7v1fz3g71JGsmHxBy5wGg7z-0IFkvSCF0vkNgi3cRGJ3r0Z5ZfL90y8" />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                                <span className="absolute bottom-1 left-2 text-[9px] font-bold text-white uppercase tracking-tighter">Impact</span>
                            </div>
                            <div className="h-0.5 w-full bg-[#3b4a3d] rounded-full"></div>
                        </button>
                        {/* Finish */}
                        <button className="flex-shrink-0 w-32 flex flex-col gap-1 group">
                            <div className="aspect-video bg-[#201f1f] rounded-lg border border-[#3b4a3d] overflow-hidden relative group-hover:border-[#75ff9e]/50 transition-all">
                                <img alt="Finish" className="w-full h-full object-cover opacity-40 group-hover:opacity-60 transition-opacity" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDcneDoHwFMp-Ju5AIbhhiWdHby6y9qvU-IhbcnkHK1Vb6WohefIOUgbzYcAAdiOMytDRoS6PbIOq11FyWDnL1izqId9xpQrDOBgCQkBDT-Z9E97bvs8k7xBDTKGfKUIb6xvYFhJkBDOwhMoDCbsSBvjJNnf7gSaHF8xIMq4wxcUEaR7iuhDH4HtEwafR384gfMdydUyYjxzOsBBVv2ICtvK9LHuOYWSgi7zOdN-ws08h-Jrp4U0LxWl_pStqGBkHQ-cWlilr2R6x_8" />
                                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                                <span className="absolute bottom-1 left-2 text-[9px] font-bold text-white uppercase tracking-tighter">Finish</span>
                            </div>
                            <div className="h-0.5 w-full bg-[#3b4a3d] rounded-full"></div>
                        </button>
                    </div>
                </section>

                {/* Sidebar Analysis */}
                <aside className="w-full md:w-[400px] bg-[#1c1b1b] border-l border-[#3b4a3d] flex flex-col overflow-y-auto custom-scrollbar p-6 gap-8 shrink-0">
                    
                    {/* System Precision */}
                    <section>
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-[24px] font-semibold text-[#e5e2e1]">System Precision</h2>
                            <span className="bg-[#00e676]/10 text-[#75ff9e] text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-widest border border-[#75ff9e]/20">Live AI</span>
                        </div>
                        <div className="bg-[#201f1f] border border-[#3b4a3d] rounded-xl p-5 flex flex-col gap-3">
                            <div className="flex justify-between items-center">
                                <span className="text-[#bacbb9] text-sm font-medium">MPJPE Metric</span>
                                <span className="text-[#75ff9e] font-bold font-mono">4.2mm</span>
                            </div>
                            <div className="w-full bg-[#353534] h-1.5 rounded-full overflow-hidden">
                                <div className="bg-[#75ff9e] h-full w-[92%] shadow-[0_0_8px_#75ff9e]"></div>
                            </div>
                            <p className="text-[11px] text-[#bacbb9] italic">High confidence pose estimation achieved via multi-view synthesis.</p>
                        </div>
                    </section>

                    {/* Biomechanical Issues */}
                    <section>
                        <h2 className="text-[24px] font-semibold text-[#e5e2e1] mb-5">Biomechanical Issues</h2>
                        <div className="flex flex-col gap-3">
                            <div className="bg-[#93000a]/10 border border-[#ffb4ab]/20 rounded-xl p-5 flex gap-4 items-start shadow-sm">
                                <span className="material-symbols-outlined text-[#ffb4ab]">warning</span>
                                <div>
                                    <p className="font-bold text-[#e5e2e1] text-sm mb-1">Excessive Head Lateral</p>
                                    <p className="text-xs text-[#bacbb9] leading-relaxed">Head moved 4.2cm off-axis during backswing, leading to inconsistent strike point.</p>
                                </div>
                            </div>
                            <div className="bg-[#b0c8ff]/10 border border-[#dae4ff]/20 rounded-xl p-5 flex gap-4 items-start shadow-sm">
                                <span className="material-symbols-outlined text-[#dae4ff]">info</span>
                                <div>
                                    <p className="font-bold text-[#e5e2e1] text-sm mb-1">Shoulder Rotation Limit</p>
                                    <p className="text-xs text-[#bacbb9] leading-relaxed">Reached 88° rotation. Ideal target for your profile is 92°.</p>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* AI Pro-Tips */}
                    <section className="mt-auto">
                        <div className="bg-[#353534] rounded-2xl p-6 relative overflow-hidden group">
                            <div className="absolute -right-4 -top-4 w-24 h-24 bg-[#75ff9e] opacity-5 rounded-full blur-2xl group-hover:opacity-10 transition-opacity"></div>
                            <div className="relative z-10">
                                <div className="flex items-center gap-2 mb-4">
                                    <span className="material-symbols-outlined text-[#75ff9e]" style={{ fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
                                    <span className="text-[12px] font-medium text-[#75ff9e] uppercase tracking-wider font-mono">AI Pro-Tips</span>
                                </div>
                                <p className="text-[#e5e2e1] text-sm leading-relaxed mb-6 italic">"Focus on keeping your lead knee stable during the initial takeaway to prevent the lateral head shift detected."</p>
                                <button className="w-full py-3 bg-[#75ff9e] text-[#00210b] font-bold rounded-lg text-sm shadow-[0_0_15px_rgba(117,255,158,0.2)] hover:shadow-[0_0_20px_rgba(117,255,158,0.4)] transition-all">
                                    View Drill Video
                                </button>
                            </div>
                        </div>
                    </section>

                </aside>

            </main>

            {/* Bottom Nav (Mobile Only) */}
            <footer className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-4 pb-4 pt-2 bg-[#201f1f] rounded-t-xl shadow-[0_-10px_20px_rgba(0,0,0,0.5)] border-t border-[#3b4a3d]">
                <button onClick={() => window.location.href = '/'} className="flex flex-col items-center justify-center text-[#75ff9e] bg-[#00e676]/10 rounded-full px-4 py-1 active:scale-90 transition-transform duration-200">
                    <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>add_circle</span>
                    <span className="text-[12px] font-medium font-mono pt-1">Analyze</span>
                </button>
                <button onClick={() => window.location.href = '/history'} className="flex flex-col items-center justify-center text-[#bacbb9] hover:text-[#75ff9e] transition-colors active:scale-90 duration-200">
                    <span className="material-symbols-outlined">history</span>
                    <span className="text-[12px] font-medium font-mono pt-1">History</span>
                </button>
                <button className="flex flex-col items-center justify-center text-[#bacbb9] hover:text-[#75ff9e] transition-colors active:scale-90 duration-200">
                    <span className="material-symbols-outlined">query_stats</span>
                    <span className="text-[12px] font-medium font-mono pt-1">Insights</span>
                </button>
            </footer>

        </div>
    );
}
