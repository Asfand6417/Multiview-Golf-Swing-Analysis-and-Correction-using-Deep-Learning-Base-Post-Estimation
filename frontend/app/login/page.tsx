"use client";
import React, { useEffect, useRef } from 'react';

export default function LoginPage() {
    const cardRef = useRef<HTMLDivElement>(null);

    // Atmospheric mouse movement effect for 3D card tilt
    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!cardRef.current) return;
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;
            
            const rotateX = (y - 0.5) * 4;
            const rotateY = (x - 0.5) * -4;
            
            cardRef.current.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        };

        document.addEventListener('mousemove', handleMouseMove);
        return () => document.removeEventListener('mousemove', handleMouseMove);
    }, []);

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        window.location.href = '/';
    };

    return (
        <div className="min-h-screen flex items-center justify-center relative bg-[#0A0A0A] text-[#e5e2e1] overflow-hidden font-sans selection:bg-[#75ff9e] selection:text-[#00210b]">
            
            {/* Embedded styles for exact matching */}
            <style jsx global>{`
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Geist:wght@400;500;600&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
                
                .material-symbols-outlined {
                    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
                }
                .glass-panel {
                    background: rgba(26, 28, 30, 0.6);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .performance-glow {
                    box-shadow: 0 0 20px rgba(117, 255, 158, 0.2);
                }
                .performance-glow:hover {
                    box-shadow: 0 0 30px rgba(117, 255, 158, 0.4);
                }
                input:-webkit-autofill,
                input:-webkit-autofill:hover, 
                input:-webkit-autofill:focus, 
                input:-webkit-autofill:active{
                    -webkit-box-shadow: 0 0 0 30px #050505 inset !important;
                    -webkit-text-fill-color: #e5e2e1 !important;
                }
            `}</style>

            {/* Cinematic Background */}
            <div className="absolute inset-0 z-0">
                <img 
                    alt="Golf course at dusk" 
                    className="w-full h-full object-cover opacity-40" 
                    src="https://lh3.googleusercontent.com/aida-public/AB6AXuCsN8gQdAZqJm-9sWOZbn0RxxRAGFEi5w7-MTCngF9OWaYEIC52qgPQJ43Y4bcAFc84RP48lZRI9f-G8JDZnL789Q4_5D9QgY0WstdRwPBJbcZNrp3AgL-JL2Rljypyx3BYHoQJ540c5ZDRJMhCQjzM5IUGFDj76BooJVq-k1gAWEKi8y5GSM-FxYQTdwX9aG2Kos4McmTnM4tVjTNAX43LdY3wEENF1Ip4DXkD82dlvrPuFO4wZffjwDU4E-3dniZZXZf6dIIFPU0i"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0A0A0A] via-transparent to-[#0A0A0A]/60"></div>
            </div>

            {/* Login Container */}
            <main className="relative z-10 w-full max-w-[480px] px-4 md:px-0">
                
                {/* Branding Header */}
                <div className="text-center mb-8">
                    <div className="flex items-center justify-center gap-3 mb-2">
                        <span className="material-symbols-outlined text-[#75ff9e] text-4xl" style={{ fontVariationSettings: "'FILL' 1" }}>analytics</span>
                        <h1 className="text-[48px] font-bold leading-[56px] tracking-tighter text-[#75ff9e] font-inter">PRO SWING AI</h1>
                    </div>
                    <p className="text-[12px] font-medium leading-[16px] text-[#bacbb9] tracking-widest uppercase font-mono">Elite Athletic Analysis Engine</p>
                </div>

                {/* Login Card */}
                <div 
                    ref={cardRef} 
                    className="glass-panel rounded-xl p-8 shadow-2xl transition-transform duration-100 ease-out" 
                    style={{ transform: 'perspective(1000px) rotateX(0deg) rotateY(0deg)' }}
                >
                    <h2 className="text-[32px] font-semibold leading-[40px] text-[#e5e2e1] mb-6 text-center font-inter">Welcome Back</h2>
                    
                    <form className="space-y-6" onSubmit={handleLogin}>
                        
                        {/* Email Field */}
                        <div className="space-y-2 group">
                            <label className="text-[12px] font-medium leading-[16px] text-[#859585] tracking-wider uppercase ml-1 font-mono transition-colors group-focus-within:text-[#75ff9e]" htmlFor="email">Email Address</label>
                            <div className="relative">
                                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-[#859585] transition-colors group-focus-within:text-[#75ff9e]">mail</span>
                                <input 
                                    className="w-full bg-[#050505] border-b border-[#3b4a3d] focus:border-[#75ff9e] focus:ring-0 text-[#e5e2e1] py-4 pl-12 pr-4 transition-all duration-300 outline-none" 
                                    id="email" 
                                    placeholder="pro@athlete.com" 
                                    type="email" 
                                    required
                                />
                            </div>
                        </div>

                        {/* Password Field */}
                        <div className="space-y-2 group">
                            <div className="flex justify-between items-center px-1">
                                <label className="text-[12px] font-medium leading-[16px] text-[#859585] tracking-wider uppercase font-mono transition-colors group-focus-within:text-[#75ff9e]" htmlFor="password">Password</label>
                                <a className="text-[12px] font-medium leading-[16px] text-[#75ff9e] hover:underline transition-all font-mono" href="#">Forgot?</a>
                            </div>
                            <div className="relative">
                                <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-[#859585] transition-colors group-focus-within:text-[#75ff9e]">lock</span>
                                <input 
                                    className="w-full bg-[#050505] border-b border-[#3b4a3d] focus:border-[#75ff9e] focus:ring-0 text-[#e5e2e1] py-4 pl-12 pr-4 transition-all duration-300 outline-none" 
                                    id="password" 
                                    placeholder="••••••••" 
                                    type="password"
                                    required 
                                />
                                <button className="absolute right-4 top-1/2 -translate-y-1/2 text-[#859585] hover:text-[#e5e2e1] transition-colors" type="button">
                                    <span className="material-symbols-outlined">visibility</span>
                                </button>
                            </div>
                        </div>

                        {/* Primary CTA */}
                        <button 
                            className="w-full bg-[#75ff9e] text-[#003918] font-bold py-4 rounded-lg flex items-center justify-center gap-2 group performance-glow transition-all active:scale-95 duration-200" 
                            type="submit"
                        >
                            <span className="text-[20px] leading-[28px] font-semibold font-inter">Start Analyzing</span>
                            <span className="material-symbols-outlined transition-transform group-hover:translate-x-1">arrow_forward</span>
                        </button>

                        {/* Divider */}
                        <div className="flex items-center gap-4 py-2">
                            <div className="h-[1px] flex-1 bg-[#3b4a3d]"></div>
                            <span className="text-[12px] font-medium leading-[16px] text-[#859585] font-mono tracking-widest">OR ACCESS WITH</span>
                            <div className="h-[1px] flex-1 bg-[#3b4a3d]"></div>
                        </div>

                        {/* Social/Biometric Logins */}
                        <div className="grid grid-cols-2 gap-4">
                            <button className="flex items-center justify-center gap-3 border border-[#3b4a3d] hover:bg-[#353534]/30 py-3 rounded-lg transition-all active:scale-95 duration-200" type="button">
                                <span className="material-symbols-outlined text-[#e5e2e1]">fingerprint</span>
                                <span className="text-[12px] font-medium leading-[16px] text-[#e5e2e1] font-mono">Biometrics</span>
                            </button>
                            <button className="flex items-center justify-center gap-3 border border-[#3b4a3d] hover:bg-[#353534]/30 py-3 rounded-lg transition-all active:scale-95 duration-200" type="button">
                                <svg className="w-5 h-5 fill-current text-[#e5e2e1]" viewBox="0 0 24 24">
                                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"></path>
                                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"></path>
                                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"></path>
                                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.66l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"></path>
                                </svg>
                                <span className="text-[12px] font-medium leading-[16px] text-[#e5e2e1] font-mono">Google</span>
                            </button>
                        </div>
                    </form>
                    
                    <p className="text-center mt-8 text-[12px] font-medium leading-[16px] text-[#859585] font-mono">
                        New to the platform? <a className="text-[#75ff9e] font-bold hover:underline" href="#">Apply for Access</a>
                    </p>
                </div>

                {/* Footer Tech Info */}
                <div className="mt-8 flex justify-between items-center px-4">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-[#75ff9e] animate-pulse"></span>
                        <span className="text-[10px] text-[#859585] uppercase tracking-widest font-mono">Neural Engine Online</span>
                    </div>
                    <span className="text-[10px] text-[#859585] uppercase tracking-widest font-mono">Ver 4.2.0-Pro</span>
                </div>
            </main>

            {/* Subtle Lighting Effects */}
            <div className="fixed top-0 right-0 w-[500px] h-[500px] bg-[#75ff9e]/5 rounded-full blur-[120px] pointer-events-none"></div>
            <div className="fixed bottom-0 left-0 w-[400px] h-[400px] bg-[#0050ad]/5 rounded-full blur-[100px] pointer-events-none"></div>
        </div>
    );
}
