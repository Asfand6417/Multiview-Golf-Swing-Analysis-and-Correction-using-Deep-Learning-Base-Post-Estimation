"use client";
import React, { useEffect, useRef } from 'react';
import Head from 'next/head';

export default function HistoryPage() {
    const cardsRef = useRef<(HTMLDivElement | null)[]>([]);

    useEffect(() => {
        // Ambient Micro-interaction logic
        const handleMouseMove = (e: MouseEvent, card: HTMLDivElement) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        };

        const cards = cardsRef.current.filter(Boolean);
        cards.forEach((card) => {
            if (card) {
                card.addEventListener('mousemove', (e) => handleMouseMove(e, card));
            }
        });

        return () => {
            cards.forEach((card) => {
                if (card) {
                    card.removeEventListener('mousemove', (e) => handleMouseMove(e as MouseEvent, card));
                }
            });
        };
    }, []);

    const setCardRef = (index: number) => (el: HTMLDivElement | null) => {
        cardsRef.current[index] = el;
    };

    return (
        <div className="bg-[#0A0A0A] text-[#e5e2e1] font-sans min-h-screen custom-scrollbar flex flex-col selection:bg-[#75ff9e] selection:text-[#00210b]">
            <style jsx global>{`
                .glass-card {
                    background: rgba(26, 28, 30, 0.6);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    position: relative;
                    overflow: hidden;
                }
                .glass-card::before {
                    content: '';
                    position: absolute;
                    top: var(--mouse-y);
                    left: var(--mouse-x);
                    transform: translate(-50%, -50%);
                    width: 300px;
                    height: 300px;
                    background: radial-gradient(circle, rgba(117,255,158,0.1) 0%, transparent 70%);
                    opacity: 0;
                    transition: opacity 0.3s;
                    pointer-events: none;
                }
                .glass-card:hover::before {
                    opacity: 1;
                }
                .performance-glow {
                    box-shadow: 0 0 20px rgba(117, 255, 158, 0.15);
                }
                .material-symbols-outlined {
                    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
                    vertical-align: middle;
                }
                .custom-scrollbar::-webkit-scrollbar { width: 4px; }
                .custom-scrollbar::-webkit-scrollbar-track { background: #131313; }
                .custom-scrollbar::-webkit-scrollbar-thumb { background: #353534; border-radius: 2px; }
                .chart-path {
                    stroke-dasharray: 1000;
                    stroke-dashoffset: 1000;
                    animation: dash 3s ease-in-out forwards;
                }
                @keyframes dash { to { stroke-dashoffset: 0; } }
            `}</style>

            {/* TopAppBar */}
            <header className="flex justify-between items-center w-full px-10 h-16 z-50 bg-[#131313]/80 backdrop-blur-md border-b border-[#3b4a3d] fixed top-0">
                <a href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
                    <span className="material-symbols-outlined text-[#75ff9e] text-2xl">analytics</span>
                    <h1 className="text-[32px] font-bold text-[#75ff9e] tracking-tight font-inter">PRO SWING AI</h1>
                </a>
                <nav className="hidden md:flex gap-8 items-center">
                    <a className="text-[#bacbb9] text-[12px] font-medium font-mono hover:text-[#75ff9e] transition-colors" href="/">Analyze</a>
                    <a className="text-[#75ff9e] text-[12px] font-medium font-mono flex items-center justify-center bg-[#00e676]/10 rounded-full px-4 py-1" href="/history">History</a>
                    <a className="text-[#bacbb9] text-[12px] font-medium font-mono hover:text-[#75ff9e] transition-colors" href="#">Insights</a>
                </nav>
                <div className="flex items-center gap-4">
                    <button className="material-symbols-outlined p-2 rounded-full hover:bg-[#353534]/50 transition-colors text-[#bacbb9]">search</button>
                    <div className="w-10 h-10 rounded-full border border-[#75ff9e]/30 overflow-hidden cursor-pointer active:scale-95 duration-100">
                        <img alt="User profile avatar" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAuuxlzmarSyg6yl8DfbdThlpUBlELZGIXIk9vOdArClwaTrK1M1Y6g5mafatyyN1Ck3sSZo2qo4JCashICIT-4dzKXwcDn7IvlvdObKlZmy-ZZwC_t1zeHFvJA6j8sFxFvUr3R2z9hM1nkkWImUCYqS5gr01iO3LeOIonEY_6L9GMK31mj2lSfMNxF6fnd_eJ_cj-zR2sSf3uJ3FlqujAIqj3xZR0fNM2BIhtBSu3Jw86ttd_H_YTwbVaE2uOJKlmJ5A5EEWdiZny7" />
                    </div>
                </div>
            </header>

            <main className="pt-24 pb-12 px-10 max-w-[1440px] mx-auto w-full flex-grow">
                
                {/* Hero Stats Row */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
                    <div ref={setCardRef(0)} className="glass-card p-6 rounded-xl flex flex-col justify-between h-32">
                        <span className="text-[12px] font-medium text-[#bacbb9] uppercase tracking-wider font-mono">Total Analysis</span>
                        <div className="flex items-end justify-between">
                            <span className="text-[48px] font-bold leading-none font-inter text-[#e5e2e1]">1,284</span>
                            <span className="text-[#75ff9e] flex items-center gap-1 text-[12px] font-medium font-mono">
                                <span className="material-symbols-outlined text-sm">trending_up</span> 12%
                            </span>
                        </div>
                    </div>
                    <div ref={setCardRef(1)} className="glass-card p-6 rounded-xl flex flex-col justify-between h-32">
                        <span className="text-[12px] font-medium text-[#bacbb9] uppercase tracking-wider font-mono">Top Score</span>
                        <div className="flex items-end justify-between">
                            <span className="text-[48px] font-bold leading-none font-inter text-[#e5e2e1]">94.8</span>
                            <span className="text-[#dae4ff] text-[12px] font-medium font-mono border border-[#dae4ff]/30 bg-[#dae4ff]/10 px-2 py-0.5 rounded">Elite Rank</span>
                        </div>
                    </div>
                    <div ref={setCardRef(2)} className="glass-card p-6 rounded-xl flex flex-col justify-between h-32">
                        <span className="text-[12px] font-medium text-[#bacbb9] uppercase tracking-wider font-mono">Avg Smash Factor</span>
                        <div className="flex items-end justify-between">
                            <span className="text-[48px] font-bold leading-none font-inter text-[#e5e2e1]">1.48</span>
                            <span className="text-[#75ff9e] text-[12px] font-medium font-mono">Optimal</span>
                        </div>
                    </div>
                    <div ref={setCardRef(3)} className="glass-card p-6 rounded-xl border-[#75ff9e]/20 bg-[#75ff9e]/5 performance-glow flex flex-col justify-between h-32">
                        <span className="text-[12px] font-medium text-[#75ff9e] uppercase tracking-wider font-mono">Session Streak</span>
                        <div className="flex items-end justify-between">
                            <span className="text-[48px] font-bold leading-none font-inter text-[#75ff9e]">12</span>
                            <span className="material-symbols-outlined text-[#75ff9e] text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>bolt</span>
                        </div>
                    </div>
                </div>

                {/* Precision Trend Chart Section */}
                <section className="mb-10">
                    <div ref={setCardRef(4)} className="glass-card rounded-xl p-8">
                        <div className="flex justify-between items-center mb-8">
                            <div>
                                <h2 className="text-[32px] font-bold text-[#e5e2e1] mb-1 font-inter">Precision Trend</h2>
                                <p className="text-[16px] text-[#bacbb9] font-inter">Mean Per Joint Position Error (MPJPE) tracked over last 30 days.</p>
                            </div>
                            <div className="flex gap-2">
                                <button className="px-4 py-2 rounded-lg bg-[#2a2a2a] text-[#e5e2e1] text-[12px] font-mono hover:bg-[#353534] transition-colors">7D</button>
                                <button className="px-4 py-2 rounded-lg bg-[#75ff9e] text-[#00210b] text-[12px] font-mono font-bold">30D</button>
                                <button className="px-4 py-2 rounded-lg bg-[#2a2a2a] text-[#e5e2e1] text-[12px] font-mono hover:bg-[#353534] transition-colors">90D</button>
                            </div>
                        </div>
                        <div className="relative h-[400px] w-full mt-4">
                            {/* Mock Chart SVG */}
                            <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 1000 400">
                                {/* Grid Lines */}
                                <line stroke="#353534" strokeWidth="1" x1="0" x2="1000" y1="100" y2="100"></line>
                                <line stroke="#353534" strokeWidth="1" x1="0" x2="1000" y1="200" y2="200"></line>
                                <line stroke="#353534" strokeWidth="1" x1="0" x2="1000" y1="300" y2="300"></line>
                                {/* Trend Line */}
                                <path className="chart-path" d="M0,350 L100,320 L200,340 L300,280 L400,250 L500,200 L600,180 L700,120 L800,100 L900,110 L1000,80" fill="none" stroke="#75ff9e" strokeWidth="4"></path>
                                {/* Area Gradient */}
                                <path d="M0,350 L100,320 L200,340 L300,280 L400,250 L500,200 L600,180 L700,120 L800,100 L900,110 L1000,80 V400 H0 Z" fill="url(#gradient-primary)" fillOpacity="0.1"></path>
                                <defs>
                                    <linearGradient id="gradient-primary" x1="0%" x2="0%" y1="0%" y2="100%">
                                        <stop offset="0%" style={{ stopColor: "#75ff9e", stopOpacity: 1 }}></stop>
                                        <stop offset="100%" style={{ stopColor: "#75ff9e", stopOpacity: 0 }}></stop>
                                    </linearGradient>
                                </defs>
                            </svg>
                            {/* Data Labels */}
                            <div className="absolute bottom-0 left-0 w-full flex justify-between px-2 pt-4 border-t border-[#3b4a3d] text-[#bacbb9] text-[12px] font-mono">
                                <span>Oct 1</span>
                                <span>Oct 7</span>
                                <span>Oct 14</span>
                                <span>Oct 21</span>
                                <span>Oct 28</span>
                                <span>Today</span>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Recent Sessions Section */}
                <section>
                    <div className="flex justify-between items-center mb-6">
                        <h2 className="text-[32px] font-bold text-[#e5e2e1] font-inter">Recent Sessions</h2>
                        <button className="text-[#75ff9e] text-[12px] font-mono flex items-center gap-2 hover:underline">
                            View All History <span className="material-symbols-outlined text-sm">chevron_right</span>
                        </button>
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                        
                        {/* Session Card 1 */}
                        <div ref={setCardRef(5)} className="glass-card rounded-xl overflow-hidden group cursor-pointer hover:border-[#75ff9e]/50 transition-all duration-300">
                            <div className="relative h-48">
                                <img className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBaObFNKhZql1uw4XFVzV__CCau-SmETkzh6fJ3RjfdqKRt1EPgOKKdx_FYqhkeURSMEAhAeOIfW6BhO8S0PBlXPmBBWJEurQ_8Dtb2XuSVoLfla2aLnLgoxYH6Cuh70d04Ik61LO94XXXFO7Q06SNWDl8hG-KA3BhJ3JEozfyELm3V0kgbgLzuWhYtYlu3fXFFbAZfTp9GmyeQ_kN0fUDkgz-x-V0k47J-ZA8IbpuSU6B2ZkVqvlCNYvdeTkAZ-asKipEOwM5roDXt" />
                                <div className="absolute top-4 left-4 px-3 py-1 bg-[#131313]/80 backdrop-blur-md rounded-full text-[#75ff9e] text-[12px] font-mono font-bold border border-[#75ff9e]/30">92 Score</div>
                                <div className="absolute inset-0 bg-gradient-to-t from-[#131313] to-transparent opacity-60"></div>
                            </div>
                            <div className="p-6">
                                <div className="flex justify-between items-start mb-4">
                                    <div>
                                        <h3 className="text-[20px] font-bold text-[#e5e2e1] font-mono">Augusta National</h3>
                                        <p className="text-[12px] text-[#bacbb9] font-mono mt-1">Today, 08:30 AM • 42 Swings</p>
                                    </div>
                                    <span className="material-symbols-outlined text-[#bacbb9] group-hover:text-[#75ff9e] transition-colors">more_vert</span>
                                </div>
                                <div className="grid grid-cols-2 gap-4 border-t border-[#3b4a3d] pt-4">
                                    <div>
                                        <p className="text-[12px] text-[#bacbb9] uppercase font-mono mb-1">Smash Factor</p>
                                        <p className="text-[20px] font-bold text-[#75ff9e] font-mono">1.51</p>
                                    </div>
                                    <div>
                                        <p className="text-[12px] text-[#bacbb9] uppercase font-mono mb-1">Launch Angle</p>
                                        <p className="text-[20px] font-bold text-[#e5e2e1] font-mono">12.4°</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Session Card 2 */}
                        <div ref={setCardRef(6)} className="glass-card rounded-xl overflow-hidden group cursor-pointer hover:border-[#75ff9e]/50 transition-all duration-300">
                            <div className="relative h-48">
                                <img className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBi50JOigxCIZWGfblUt3_RfIAgyyX98j0KC-_rfwYtLCPm0fwgSozCM2124yTS8MgECNNI79omr9Ykk94AcgGpFGAJHU-E8Hr_7dLqE_cTi6y3d98VT4STsyH65oHQhAeb9Dcz2nUBPi3uH6AdLzt88Zh-5KRg4poegWthA7mev73lJY5tyCGQxackNnAXeVwr9jZOpZqNscvPPLkwPId_62kyQO2L6xlcUfiaqjk2Yo_-BH16OL0o7_cgjeIlsPKAyprrkv0u2KW_" />
                                <div className="absolute top-4 left-4 px-3 py-1 bg-[#131313]/80 backdrop-blur-md rounded-full text-[#75ff9e] text-[12px] font-mono font-bold border border-[#75ff9e]/30">88 Score</div>
                                <div className="absolute inset-0 bg-gradient-to-t from-[#131313] to-transparent opacity-60"></div>
                            </div>
                            <div className="p-6">
                                <div className="flex justify-between items-start mb-4">
                                    <div>
                                        <h3 className="text-[20px] font-bold text-[#e5e2e1] font-mono">Precision Lab</h3>
                                        <p className="text-[12px] text-[#bacbb9] font-mono mt-1">Oct 26, 04:15 PM • 120 Swings</p>
                                    </div>
                                    <span className="material-symbols-outlined text-[#bacbb9] group-hover:text-[#75ff9e] transition-colors">more_vert</span>
                                </div>
                                <div className="grid grid-cols-2 gap-4 border-t border-[#3b4a3d] pt-4">
                                    <div>
                                        <p className="text-[12px] text-[#bacbb9] uppercase font-mono mb-1">Smash Factor</p>
                                        <p className="text-[20px] font-bold text-[#75ff9e] font-mono">1.45</p>
                                    </div>
                                    <div>
                                        <p className="text-[12px] text-[#bacbb9] uppercase font-mono mb-1">Launch Angle</p>
                                        <p className="text-[20px] font-bold text-[#e5e2e1] font-mono">11.8°</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Session Card 3 */}
                        <div ref={setCardRef(7)} className="glass-card rounded-xl overflow-hidden group cursor-pointer hover:border-[#75ff9e]/50 transition-all duration-300">
                            <div className="relative h-48">
                                <img className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDDUwQLnTW06HWHuCpWP4WTIbYyfLvzjCkf7vhgN0KWG9EzfpasChh0knk0Ju_OcjaosbLy8gYWrHQN8d2oe9n0TbJBuV2_rPFhBS4_2d8j7UU6CQx8h3svQUObJbmiAu6PtkhSqvX1vklqrPPH4-_j9Tzmwyk62jWMYl59qvbejLpSNiP9Wtwl8sVlFa3RlQHrpCjn49PfTFtLTFc96JUm3hQihKHPMqz9pEmDVEksbB8qjBu6lPlsL1j1unQlQ2VoJeukUSwQD1vT" />
                                <div className="absolute top-4 left-4 px-3 py-1 bg-[#131313]/80 backdrop-blur-md rounded-full text-[#75ff9e] text-[12px] font-mono font-bold border border-[#75ff9e]/30">95 Score</div>
                                <div className="absolute inset-0 bg-gradient-to-t from-[#131313] to-transparent opacity-60"></div>
                            </div>
                            <div className="p-6">
                                <div className="flex justify-between items-start mb-4">
                                    <div>
                                        <h3 className="text-[20px] font-bold text-[#e5e2e1] font-mono">Pebble Beach</h3>
                                        <p className="text-[12px] text-[#bacbb9] font-mono mt-1">Oct 24, 10:00 AM • 68 Swings</p>
                                    </div>
                                    <span className="material-symbols-outlined text-[#bacbb9] group-hover:text-[#75ff9e] transition-colors">more_vert</span>
                                </div>
                                <div className="grid grid-cols-2 gap-4 border-t border-[#3b4a3d] pt-4">
                                    <div>
                                        <p className="text-[12px] text-[#bacbb9] uppercase font-mono mb-1">Smash Factor</p>
                                        <p className="text-[20px] font-bold text-[#75ff9e] font-mono">1.54</p>
                                    </div>
                                    <div>
                                        <p className="text-[12px] text-[#bacbb9] uppercase font-mono mb-1">Launch Angle</p>
                                        <p className="text-[20px] font-bold text-[#e5e2e1] font-mono">13.1°</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                </section>
            </main>

            {/* Footer */}
            <footer className="mt-20 border-t border-[#3b4a3d] bg-[#201f1f] py-12 px-10 shrink-0">
                <div className="max-w-[1440px] mx-auto flex flex-col md:flex-row justify-between items-start gap-10">
                    <div className="max-w-xs">
                        <a href="/" className="flex items-center gap-2 mb-4 hover:opacity-80 transition-opacity">
                            <span className="material-symbols-outlined text-[#75ff9e]">analytics</span>
                            <span className="text-[24px] font-bold text-[#75ff9e] font-inter tracking-tight">PRO SWING AI</span>
                        </a>
                        <p className="text-[16px] text-[#bacbb9] font-inter leading-relaxed">Elite performance tracking powered by advanced kinematic analysis and real-time AI computer vision.</p>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-16">
                        <div>
                            <h4 className="text-[12px] font-medium text-[#e5e2e1] uppercase mb-6 tracking-widest font-mono">Navigation</h4>
                            <ul className="space-y-4 text-[12px] font-medium text-[#bacbb9] font-mono">
                                <li><a className="hover:text-[#75ff9e] transition-colors" href="/">Dashboard</a></li>
                                <li><a className="hover:text-[#75ff9e] transition-colors" href="/analysis">Swing Analysis</a></li>
                                <li><a className="hover:text-[#75ff9e] transition-colors" href="/history">Performance History</a></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="text-[12px] font-medium text-[#e5e2e1] uppercase mb-6 tracking-widest font-mono">Support</h4>
                            <ul className="space-y-4 text-[12px] font-medium text-[#bacbb9] font-mono">
                                <li><a className="hover:text-[#75ff9e] transition-colors" href="#">Technical Docs</a></li>
                                <li><a className="hover:text-[#75ff9e] transition-colors" href="#">Contact Pro-Team</a></li>
                                <li><a className="hover:text-[#75ff9e] transition-colors" href="#">System Status</a></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="text-[12px] font-medium text-[#e5e2e1] uppercase mb-6 tracking-widest font-mono">Connect</h4>
                            <div className="flex gap-4">
                                <button className="material-symbols-outlined text-[#bacbb9] hover:text-[#75ff9e] transition-colors">share</button>
                                <button className="material-symbols-outlined text-[#bacbb9] hover:text-[#75ff9e] transition-colors">rss_feed</button>
                                <button className="material-symbols-outlined text-[#bacbb9] hover:text-[#75ff9e] transition-colors">mail</button>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="max-w-[1440px] mx-auto mt-12 pt-8 border-t border-[#3b4a3d] flex flex-col md:flex-row justify-between text-[#bacbb9] text-[12px] font-medium font-mono opacity-50">
                    <span>© 2024 Kinetic Elite. All Rights Reserved.</span>
                    <div className="flex gap-6 mt-4 md:mt-0">
                        <a className="hover:text-white" href="#">Privacy Policy</a>
                        <a className="hover:text-white" href="#">Terms of Service</a>
                    </div>
                </div>
            </footer>
        </div>
    );
}
