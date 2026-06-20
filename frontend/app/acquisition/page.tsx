"use client";
import React, { useEffect, useRef } from 'react';
import Head from 'next/head';

export default function AcquisitionPage() {
    const [frontVideo, setFrontVideo] = React.useState<File | null>(null);
    const [sideVideo, setSideVideo] = React.useState<File | null>(null);
    
    // Connected engine state 
    const [isProcessing, setIsProcessing] = React.useState<boolean>(false);
    const [progress, setProgress] = React.useState<number>(0);

    // Simple micro-interaction for the "spin" animation fallback
    useEffect(() => {
        const spinIcon = document.getElementById('sync-spinner');
        let rotation = 0;
        const interval = setInterval(() => {
            if (spinIcon) {
                rotation += 45;
                spinIcon.style.transform = `rotate(${rotation}deg)`;
                spinIcon.style.transition = 'transform 0.5s ease-in-out';
            }
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    const handleFrontDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.currentTarget.classList.remove('bg-[#75ff9e]/5', 'border-[#75ff9e]');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFrontVideo(e.dataTransfer.files[0]);
        }
    };

    const handleSideDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.currentTarget.classList.remove('bg-[#75ff9e]/5', 'border-[#75ff9e]');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setSideVideo(e.dataTransfer.files[0]);
        }
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.currentTarget.classList.add('bg-[#75ff9e]/5', 'border-[#75ff9e]');
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.currentTarget.classList.remove('bg-[#75ff9e]/5', 'border-[#75ff9e]');
    };

    // Engine Connection Handler for Dual-Camera 3D approach
    const handleBeginProcessing = async () => {
        if (!frontVideo || !sideVideo) {
            alert("Please upload both Front and Side view videos to enable accurate 3D spatial reconstruction.");
            return;
        }

        setIsProcessing(true);
        setProgress(5);

        // Visual loading simulation for the pipeline steps 
        const progressInterval = setInterval(() => {
            setProgress((p) => Math.min(p + 4, 90));
        }, 500);

        try {
            const formData = new FormData();
            formData.append('front_video', frontVideo);
            formData.append('side_video', sideVideo);

            // Connects directly to the FastAPI server running the MediaPipe Dual-View ML Engine
            const response = await fetch('http://localhost:8000/process-dual', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                console.warn("Backend warning: Proceeding in visual/standalone mode for demo.");
            }

            clearInterval(progressInterval);
            setProgress(100);
            
            // Route to Preprocessing / Splitting stage automatically
            setTimeout(() => {
                window.location.href = '/splitting'; 
            }, 500);
            
        } catch (err) {
            console.error(err);
            alert("Could not connect to Python engine on port 8000. Start backend to process real video. Proceeding in UI demo mode.");
            clearInterval(progressInterval);
            setProgress(100);
            setTimeout(() => {
                window.location.href = '/splitting';
            }, 800);
        }
    };

    return (
        <div className="bg-[#131313] text-[#e5e2e1] font-sans min-h-screen overflow-x-hidden selection:bg-[#75ff9e] selection:text-[#00210b]">
            <style jsx global>{`
                .glass-overlay {
                    background: rgba(26, 28, 30, 0.4);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .material-symbols-outlined {
                    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
                    vertical-align: middle;
                }
                ::-webkit-scrollbar { width: 6px; }
                ::-webkit-scrollbar-track { background: #131313; }
                ::-webkit-scrollbar-thumb { background: #353534; border-radius: 10px; }
                @keyframes pulse-primary {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.6; }
                }
                .animate-status-pulse {
                    animation: pulse-primary 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                }
                .no-scrollbar::-webkit-scrollbar {
                    display: none;
                }
                .no-scrollbar {
                    -ms-overflow-style: none;  /* IE and Edge */
                    scrollbar-width: none;  /* Firefox */
                }
            `}</style>

            {/* TopAppBar */}
            <header className="flex justify-between items-center w-full px-6 h-16 z-50 bg-[#131313]/80 backdrop-blur-md fixed top-0 border-b border-[#3b4a3d]">
                <a href="/" className="flex items-center gap-4 hover:opacity-80 transition-opacity">
                    <span className="material-symbols-outlined text-[#75ff9e]">analytics</span>
                    <h1 className="text-[32px] font-bold text-[#75ff9e] tracking-tight font-inter">PRO SWING AI</h1>
                </a>
                <div className="hidden md:flex items-center gap-8">
                    <div onClick={() => window.location.href = '/'} className="cursor-pointer flex items-center gap-2 text-[#75ff9e] bg-[#00e676]/10 rounded-full px-4 py-1 hover:bg-[#00e676]/20 transition-colors">
                        <span className="material-symbols-outlined text-sm">add_circle</span>
                        <span className="text-[12px] font-medium font-mono uppercase tracking-wider">Analyze</span>
                    </div>
                    <div onClick={() => window.location.href = '/history'} className="flex items-center gap-2 text-[#bacbb9] hover:text-[#75ff9e] transition-colors cursor-pointer">
                        <span className="material-symbols-outlined text-sm">history</span>
                        <span className="text-[12px] font-medium font-mono uppercase tracking-wider">History</span>
                    </div>
                    <div className="flex items-center gap-2 text-[#bacbb9] hover:text-[#75ff9e] transition-colors cursor-pointer">
                        <span className="material-symbols-outlined text-sm">query_stats</span>
                        <span className="text-[12px] font-medium font-mono uppercase tracking-wider">Insights</span>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <div className="text-right hidden sm:block">
                        <p className="text-[12px] font-medium font-mono text-[#e5e2e1]">Pro Analyst</p>
                        <p className="text-[10px] font-medium font-mono text-[#75ff9e]">ELITE MEMBER</p>
                    </div>
                    <img alt="User Profile" className="w-10 h-10 rounded-full border border-[#3b4a3d] object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDFzm0qSUdCQpPvoqCCrZTtzW8xjdhP5ubyCUslQ8WlvOmslQjQpNes0PBLkvvvxB-ybrkNvMl44eoCah4Vc-dhwE467t7NEVVwyNr4IrT53-6e0pH0ceONH0SA1mRKvYFN4mVMc2Ld_Ov4SnRek0_WMe-dSt_T-RLzFx53Tm_Q2Ec5Wsw_3Nf0Qtm8s4tzpexnrAoA-F965IVmA_5uceHjiSvplexUblje7IGRc7GQmX-onJRzSsCA_08tDi7CG8K30F5jYTBsVAYR" />
                </div>
            </header>

            <main className="pt-24 pb-32 px-6 md:px-10 max-w-[1440px] mx-auto">
                
                {/* Step Indicator */}
                <div className="mb-10">
                    <div className="flex items-center gap-2 text-[#75ff9e] mb-2">
                        <span className="text-[12px] font-medium font-mono font-bold tracking-[0.2em] uppercase">
                            <span style={{ color: "rgb(0, 230, 118)", fontSize: "14px", letterSpacing: "2.8px", textTransform: "none", backgroundColor: "rgba(0, 230, 118, 0.1)" }}>STEP 01 / 05</span>
                        </span>
                        <div className="h-[1px] w-12 bg-[#75ff9e]"></div>
                    </div>
                    <h2 className="text-[48px] leading-[56px] font-bold text-[#e5e2e1] mb-8 font-inter tracking-tight">Data Acquisition</h2>
                    
                    <div className="flex gap-4 overflow-x-auto pb-4 no-scrollbar">
                        <div className="flex-none flex items-center gap-3 px-6 py-3 bg-[#00e676]/20 border border-[#75ff9e]/50 rounded-lg">
                            <span className="w-6 h-6 flex items-center justify-center rounded-full bg-[#75ff9e] text-[#131313] font-bold text-xs">01</span>
                            <span className="text-[12px] font-medium font-mono text-[#75ff9e] whitespace-nowrap">ACQUISITION</span>
                        </div>
                        <div className="flex-none flex items-center gap-3 px-6 py-3 bg-[#201f1f] border border-[#3b4a3d] rounded-lg opacity-60">
                            <span className="w-6 h-6 flex items-center justify-center rounded-full bg-[#3b4a3d] text-[#e5e2e1] font-bold text-xs">02</span>
                            <span className="text-[12px] font-medium font-mono text-[#e5e2e1] whitespace-nowrap">POSE DETECTION</span>
                        </div>
                        <div className="flex-none flex items-center gap-3 px-6 py-3 bg-[#201f1f] border border-[#3b4a3d] rounded-lg opacity-60">
                            <span className="w-6 h-6 flex items-center justify-center rounded-full bg-[#3b4a3d] text-[#e5e2e1] font-bold text-xs">03</span>
                            <span className="text-[12px] font-medium font-mono text-[#e5e2e1] whitespace-nowrap">2D-3D MODEL</span>
                        </div>
                        <div className="flex-none flex items-center gap-3 px-6 py-3 bg-[#201f1f] border border-[#3b4a3d] rounded-lg opacity-60">
                            <span className="w-6 h-6 flex items-center justify-center rounded-full bg-[#3b4a3d] text-[#e5e2e1] font-bold text-xs">04</span>
                            <span className="text-[12px] font-medium font-mono text-[#e5e2e1] whitespace-nowrap">SEGMENTATION</span>
                        </div>
                        <div className="flex-none flex items-center gap-3 px-6 py-3 bg-[#201f1f] border border-[#3b4a3d] rounded-lg opacity-60">
                            <span className="w-6 h-6 flex items-center justify-center rounded-full bg-[#3b4a3d] text-[#e5e2e1] font-bold text-xs">05</span>
                            <span className="text-[12px] font-medium font-mono text-[#e5e2e1] whitespace-nowrap">ANALYSIS</span>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                    
                    {/* Stats Sidebar */}
                    <div className="lg:col-span-3 space-y-6">
                        <div className="bg-[#201f1f] p-6 border border-[#3b4a3d] rounded-lg relative overflow-hidden group">
                            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                                <span className="material-symbols-outlined text-5xl">video_library</span>
                            </div>
                            <p className="text-[12px] font-medium font-mono text-[#859585] mb-1 uppercase tracking-wider">Annotated Dataset</p>
                            <h3 className="text-[48px] font-bold leading-none text-[#75ff9e] font-inter">1,400</h3>
                            <p className="text-[12px] font-medium font-mono text-[#bacbb9] mt-2">Verified Swing Videos</p>
                            <div className="mt-4 flex items-center gap-2 text-[10px] text-[#75ff9e]">
                                <span className="material-symbols-outlined text-[14px]">check_circle</span>
                                <span>DIVERSE LIGHTING MODELS ENABLED</span>
                            </div>
                        </div>

                        <div className="bg-[#201f1f] p-6 border border-[#3b4a3d] rounded-lg relative overflow-hidden group">
                            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                                <span className="material-symbols-outlined text-5xl">rebase_edit</span>
                            </div>
                            <p className="text-[12px] font-medium font-mono text-[#859585] mb-1 uppercase tracking-wider">Landmark Accuracy</p>
                            <h3 className="text-[48px] font-bold leading-none text-[#e5e2e1] font-inter">98.4%</h3>
                            <div className="mt-2 w-full bg-[#353534] h-1 rounded-full overflow-hidden">
                                <div className="bg-[#75ff9e] h-full w-[98.4%]"></div>
                            </div>
                            <p className="text-[12px] font-medium font-mono text-[#bacbb9] mt-2">Active Landmark Detection</p>
                        </div>

                        <div className="bg-[#201f1f] p-6 border border-[#3b4a3d] rounded-lg">
                            <h4 className="text-[12px] font-medium font-mono text-[#75ff9e] mb-4 uppercase tracking-widest border-b border-[#3b4a3d] pb-2">Technical Metadata</h4>
                            <div className="space-y-4">
                                <div className="flex justify-between items-end">
                                    <div>
                                        <p className="text-[10px] text-[#859585] uppercase font-bold">Focal Length [F]</p>
                                        <p className="text-[20px] font-bold font-mono text-[#e5e2e1]">35.2 mm</p>
                                    </div>
                                    <span className="material-symbols-outlined text-[#859585]">settings_input_component</span>
                                </div>
                                <div className="flex justify-between items-end">
                                    <div>
                                        <p className="text-[10px] text-[#859585] uppercase font-bold">Distortion [K]</p>
                                        <p className="text-[20px] font-bold font-mono text-[#e5e2e1]">0.0024</p>
                                    </div>
                                    <span className="material-symbols-outlined text-[#859585]">blur_on</span>
                                </div>
                                <div className="flex justify-between items-end">
                                    <div>
                                        <p className="text-[10px] text-[#859585] uppercase font-bold">Sensor ID</p>
                                        <p className="text-[20px] font-bold font-mono text-[#e5e2e1]">GH-X829</p>
                                    </div>
                                    <span className="material-symbols-outlined text-[#859585]">fingerprint</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Workspace */}
                    <div className="lg:col-span-9 space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            
                            {/* Front View (Live/Upload) */}
                            <div className="space-y-3">
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-2">
                                        <span className={`w-2 h-2 rounded-full ${frontVideo ? 'bg-[#75ff9e] animate-status-pulse' : 'bg-[#859585]'}`}></span>
                                        <span className="text-[12px] font-medium font-mono text-[#e5e2e1] uppercase font-bold tracking-tight">Front View Camera</span>
                                    </div>
                                    <span className={`text-[10px] font-medium font-mono px-2 py-0.5 border rounded ${frontVideo ? 'text-[#75ff9e] border-[#75ff9e]/30' : 'text-[#859585] border-[#859585]'}`}>
                                        {frontVideo ? 'CAM_01_ACTIVE' : 'WAITING_FOR_INPUT'}
                                    </span>
                                </div>
                                
                                <label className="block w-full h-full cursor-pointer">
                                    <input type="file" className="hidden" accept="video/*" onChange={(e) => { if(e.target.files?.[0]) setFrontVideo(e.target.files[0]); }} />
                                    <div 
                                        className={`aspect-video w-full relative rounded-lg border-2 ${frontVideo ? 'border-[#3b4a3d]' : 'border-dashed border-[#3b4a3d]'} bg-[#0e0e0e] overflow-hidden group hover:border-[#75ff9e]/50 transition-all flex flex-col items-center justify-center`}
                                        onDrop={handleFrontDrop}
                                        onDragOver={handleDragOver}
                                        onDragLeave={handleDragLeave}
                                    >
                                        {frontVideo ? (
                                            <video 
                                                src={URL.createObjectURL(frontVideo)} 
                                                className="absolute inset-0 w-full h-full object-cover" 
                                                autoPlay 
                                                loop 
                                                muted 
                                                playsInline 
                                            />
                                        ) : (
                                            <img alt="Front View Golf Swing" className="absolute inset-0 w-full h-full object-cover opacity-40 group-hover:opacity-60 transition-opacity" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAxbEWQMM7MqnKA2-l1lNZ8juCxla3PxVjDW2n-Nfp4g6u21UInVi1dV4e6DimRGkX9IBhw5HhRo-LsgK_wXr_irwy8Z0Ihn9RmOo20a8Fc9qM9Nzemt38VWZYwK5uEb5voPn2SVYVxPmyiLImtZehU1_udmAMxgKw6DIkMzvhC60YlGWnX4ozkdBicEg1Ful86YoLvVoVP1nBdFkQfO0PaM7zdxtCaN8-CQyeMQbU10AZazRX3RTKibiKz8tAVCCg_drML8HhvtG_I" />
                                        )}
                                        
                                        <div className={`absolute inset-0 flex flex-col justify-between p-4 pointer-events-none transition-all ${frontVideo ? 'opacity-0 group-hover:opacity-100 bg-[#131313]/60' : ''}`}>
                                            {!frontVideo && (
                                                <div className="flex flex-col items-center justify-center h-full w-full gap-4 opacity-0 group-hover:opacity-100 transition-opacity group-hover:scale-105">
                                                    <div className="w-16 h-16 rounded-full bg-[#353534] flex items-center justify-center text-[#bacbb9]">
                                                        <span className="material-symbols-outlined text-3xl">cloud_upload</span>
                                                    </div>
                                                    <div className="text-center bg-[#131313]/80 p-2 rounded">
                                                        <p className="text-[16px] font-bold font-mono text-[#e5e2e1]">Drop Front View MOV/MP4</p>
                                                    </div>
                                                </div>
                                            )}
                                            
                                            {frontVideo && (
                                                <>
                                                    <div className="flex justify-between">
                                                        <div className="w-8 h-8 border-t-2 border-l-2 border-[#75ff9e]/40"></div>
                                                        <div className="w-8 h-8 border-t-2 border-r-2 border-[#75ff9e]/40"></div>
                                                    </div>
                                                    <div className="glass-overlay p-4 rounded-lg self-center flex items-center gap-4">
                                                        <div className="flex flex-col items-center">
                                                            <span className="text-[10px] text-[#75ff9e] font-bold">FPS</span>
                                                            <span className="text-[20px] font-bold font-mono">240</span>
                                                        </div>
                                                        <div className="w-[1px] h-8 bg-[#3b4a3d]"></div>
                                                        <div className="flex flex-col items-center">
                                                            <span className="text-[10px] text-[#75ff9e] font-bold">BIT</span>
                                                            <span className="text-[20px] font-bold font-mono">10-b</span>
                                                        </div>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <div className="w-8 h-8 border-b-2 border-l-2 border-[#75ff9e]/40"></div>
                                                        <div className="w-8 h-8 border-b-2 border-r-2 border-[#75ff9e]/40"></div>
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                </label>
                            </div>

                            {/* Side View (Upload) */}
                            <div className="space-y-3">
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-2">
                                        <span className={`w-2 h-2 rounded-full ${sideVideo ? 'bg-[#75ff9e] animate-status-pulse' : 'bg-[#859585]'}`}></span>
                                        <span className="text-[12px] font-medium font-mono text-[#e5e2e1] uppercase font-bold tracking-tight">Upload Side View</span>
                                    </div>
                                    <span className={`text-[10px] font-medium font-mono px-2 py-0.5 border rounded ${sideVideo ? 'text-[#75ff9e] border-[#75ff9e]/30' : 'text-[#859585] border-[#859585]'}`}>
                                        {sideVideo ? 'CAM_02_ACTIVE' : 'WAITING_FOR_INPUT'}
                                    </span>
                                </div>
                                <label className="block w-full h-full cursor-pointer">
                                    <input type="file" className="hidden" accept="video/*" onChange={(e) => { if(e.target.files?.[0]) setSideVideo(e.target.files[0]); }} />
                                    <div 
                                        className={`aspect-video w-full relative rounded-lg border-2 ${sideVideo ? 'border-[#3b4a3d]' : 'border-dashed border-[#3b4a3d]'} bg-[#1c1b1b] flex flex-col items-center justify-center hover:border-[#75ff9e]/50 hover:bg-[#2a2a2a] transition-colors group overflow-hidden`}
                                        onDrop={handleSideDrop}
                                        onDragOver={handleDragOver}
                                        onDragLeave={handleDragLeave}
                                    >
                                        {sideVideo ? (
                                            <video 
                                                src={URL.createObjectURL(sideVideo)} 
                                                className="absolute inset-0 w-full h-full object-cover" 
                                                autoPlay 
                                                loop 
                                                muted 
                                                playsInline 
                                            />
                                        ) : (
                                            <div className="flex flex-col items-center gap-4 group-hover:scale-105 transition-transform p-4">
                                                <div className="w-16 h-16 rounded-full bg-[#353534] flex items-center justify-center text-[#bacbb9]">
                                                    <span className="material-symbols-outlined text-3xl">cloud_upload</span>
                                                </div>
                                                <div className="text-center">
                                                    <p className="text-[20px] font-bold font-mono text-[#e5e2e1] mb-1">Drop Side View MOV/MP4</p>
                                                    <p className="text-[12px] font-medium font-mono text-[#859585]">Minimum 120 FPS Recommended</p>
                                                </div>
                                            </div>
                                        )}
                                        
                                        {sideVideo && (
                                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#131313]/60 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                                                <div className="glass-overlay p-4 rounded-full mb-4">
                                                    <span className="material-symbols-outlined text-[#75ff9e] text-4xl">swap_horiz</span>
                                                </div>
                                                <p className="text-[16px] font-bold font-mono text-[#e5e2e1]">Click to replace Side View</p>
                                            </div>
                                        )}
                                    </div>
                                </label>
                            </div>
                        </div>

                        {/* Preprocessing Pipeline */}
                        <div className="bg-[#2a2a2a] p-6 border border-[#3b4a3d] rounded-lg">
                            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                                <div>
                                    <h4 className="text-[24px] font-bold text-[#e5e2e1] font-inter">Preprocessing Pipeline</h4>
                                    <p className="text-[12px] font-medium font-mono text-[#859585]">Synchronizing multi-view temporal data clusters</p>
                                </div>
                                <div className="flex items-center gap-4 w-full md:w-auto">
                                    <div className="text-right">
                                        <p className="text-[10px] text-[#75ff9e] uppercase font-black">Current Status</p>
                                        <p className="text-[12px] font-medium font-mono text-[#e5e2e1] uppercase">{isProcessing ? 'UPLOADING TO ENGINE' : 'FRAME SYNCHRONIZATION'}</p>
                                    </div>
                                    <div className="w-16 h-16 rounded-full border-4 border-[#353534] flex items-center justify-center relative">
                                        <svg className="absolute inset-0 w-full h-full -rotate-90">
                                            <circle className="text-[#75ff9e]" cx="28" cy="28" fill="none" r="28" stroke="currentColor" strokeDasharray="176" strokeDashoffset={176 - (176 * (isProcessing ? progress : 0)) / 100} strokeWidth="4" style={{ transformOrigin: 'center', transform: 'scale(0.875)' }}></circle>
                                        </svg>
                                        <span className="text-sm font-bold font-mono text-[#75ff9e]">{isProcessing ? `${progress}%` : '--'}</span>
                                    </div>
                                </div>
                            </div>
                            <div className="space-y-6">
                                <div className="relative w-full h-1.5 bg-[#353534] rounded-full overflow-hidden">
                                    <div className="absolute h-full bg-[#75ff9e] transition-all duration-300" style={{ width: `${isProcessing ? progress : 0}%` }}>
                                        <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className={`p-4 rounded border transition-all ${isProcessing ? 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-100' : 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-40'}`}>
                                        <p className="text-[10px] text-[#859585] uppercase mb-2">Temporal Alignment</p>
                                        <div className="flex items-center gap-2 text-[#75ff9e]">
                                            <span className="material-symbols-outlined text-sm">check_circle</span>
                                            <span className="text-xs font-bold font-mono uppercase">Complete</span>
                                        </div>
                                    </div>
                                    <div className={`p-4 rounded border transition-all ${progress > 25 ? 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-100' : 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-40'}`}>
                                        <p className="text-[10px] text-[#859585] uppercase mb-2">Color Calibration</p>
                                        <div className="flex items-center gap-2 text-[#75ff9e]">
                                            <span className="material-symbols-outlined text-sm">check_circle</span>
                                            <span className="text-xs font-bold font-mono uppercase">Complete</span>
                                        </div>
                                    </div>
                                    <div className={`p-4 rounded border relative transition-all ${progress > 50 ? 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-100' : 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-40'}`}>
                                        <p className="text-[10px] text-[#859585] uppercase mb-2">Frame Extraction</p>
                                        <div className="flex items-center gap-2 text-[#e5e2e1] animate-status-pulse">
                                            <span id="sync-spinner" className="material-symbols-outlined text-sm">sync</span>
                                            <span className="text-xs font-bold font-mono uppercase">Processing...</span>
                                        </div>
                                    </div>
                                    <div className={`p-4 rounded border transition-all ${progress > 75 ? 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-100' : 'bg-[#201f1f] border-[#3b4a3d]/50 opacity-40'}`}>
                                        <p className="text-[10px] text-[#859585] uppercase mb-2">Landmark Mapping</p>
                                        <div className="flex items-center gap-2 text-[#859585]">
                                            <span className="material-symbols-outlined text-sm">pending</span>
                                            <span className="text-xs font-bold font-mono uppercase">Queued</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Final Actions */}
                        <div className="flex flex-col md:flex-row justify-end items-center gap-6 pt-4">
                            <button className="w-full md:w-auto px-8 py-4 border border-[#3b4a3d] text-[#bacbb9] text-[12px] font-medium font-mono rounded uppercase tracking-widest hover:bg-[#353534]/50 transition-colors">
                                Save Workspace Session
                            </button>
                            <button 
                                onClick={handleBeginProcessing} 
                                disabled={isProcessing}
                                className={`w-full md:w-auto px-10 py-4 bg-[#75ff9e] text-[#003918] text-[24px] font-bold font-inter rounded-lg shadow-[0_0_20px_rgba(117,255,158,0.3)] hover:scale-[1.02] active:scale-95 transition-all flex items-center justify-center gap-3 ${isProcessing ? 'opacity-80' : ''}`}
                            >
                                <span className={`material-symbols-outlined ${isProcessing ? 'animate-spin' : ''}`}>{isProcessing ? 'sync' : 'play_arrow'}</span>
                                {isProcessing ? 'PROCESSING VIEWS...' : 'BEGIN PRE-PROCESSING'}
                            </button>
                        </div>

                    </div>
                </div>
            </main>

            {/* Bottom Navigation (Mobile Only) */}
            <nav className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-4 pb-4 pt-2 bg-[#201f1f] border-t border-[#3b4a3d] shadow-lg">
                <div onClick={() => window.location.href = '/'} className="flex flex-col items-center justify-center text-[#75ff9e] bg-[#00e676]/10 rounded-full px-4 py-1 cursor-pointer active:scale-95">
                    <span className="material-symbols-outlined">add_circle</span>
                    <span className="text-[12px] font-medium font-mono">Analyze</span>
                </div>
                <div onClick={() => window.location.href = '/history'} className="flex flex-col items-center justify-center text-[#bacbb9] cursor-pointer active:scale-95">
                    <span className="material-symbols-outlined">history</span>
                    <span className="text-[12px] font-medium font-mono">History</span>
                </div>
                <div className="flex flex-col items-center justify-center text-[#bacbb9] cursor-pointer">
                    <span className="material-symbols-outlined">query_stats</span>
                    <span className="text-[12px] font-medium font-mono">Insights</span>
                </div>
            </nav>

        </div>
    );
}
