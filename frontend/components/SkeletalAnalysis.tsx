"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect } from 'react';
import {
  Activity,
  Award,
  ShieldAlert,
  Info,
  CheckCircle2,
  Minimize,
  Maximize,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  HelpCircle,
  Video,
  ExternalLink,
  Flame,
  UserCheck,
  ChevronRight,
  TrendingDown,
  Volume2
} from 'lucide-react';

interface BiomechanicalIssue {
  id: string;
  type: 'critical' | 'stable' | 'optimal';
  title: string;
  rule: string;
  description: string;
  injuryRisk: string;
  recommendation: string;
}

interface SkeletalAnalysisProps {
  metrics?: {
    swayCm: number;
    kneeAngle: number;
    mpjpe: number;
    fps: number;
    poseStatus: string;
    phase: string;
    issues: string[];
  };
}

export default function SkeletalAnalysis({ metrics = { swayCm: 0, kneeAngle: 155, mpjpe: 4.2, fps: 60, poseStatus: 'GOOD Pose', phase: 'Setup', issues: [] } }: SkeletalAnalysisProps) {
  const [activePhase, setActivePhase] = useState<'setup' | 'backswing' | 'downswing' | 'impact' | 'finish'>('downswing');
  const [rotationSpeed, setRotationSpeed] = useState<number>(102);
  const [frameIndex, setFrameIndex] = useState<number>(142);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [showDrillModal, setShowDrillModal] = useState<boolean>(false);
  const [selectedJoint, setSelectedJoint] = useState<string | null>(null);

  // Stats multiplier based on rotational speed slider
  const speedRatio = rotationSpeed / 102;

  // Live frame progression simulation
  useEffect(() => {
    let timer: any = null;
    if (isPlaying) {
      timer = setInterval(() => {
        setFrameIndex((prev) => {
          if (prev >= 300) {
            return 1;
          }
          return prev + 1;
        });
      }, 35 / playbackSpeed);
    }
    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed]);

  const phases = [
    { id: 'setup', label: 'SETUP', img: 'https://images.unsplash.com/photo-1593111774240-d529f12cf4bb?auto=format&fit=crop&q=80&w=150' },
    { id: 'backswing', label: 'BACKSWING', img: 'https://images.unsplash.com/photo-1535131749006-b7f58c99034b?auto=format&fit=crop&q=80&w=150' },
    { id: 'downswing', label: 'DOWNSWING', img: 'https://images.unsplash.com/photo-1471958680802-1345a694ba6d?auto=format&fit=crop&q=80&w=150' },
    { id: 'impact', label: 'IMPACT', img: 'https://images.unsplash.com/photo-1592919505780-303950717480?auto=format&fit=crop&q=80&w=150' },
    { id: 'finish', label: 'FINISH', img: 'https://images.unsplash.com/photo-1587174486073-ae5e5cff23aa?auto=format&fit=crop&q=80&w=150' }
  ];

  // Dynamic Array that reacts to actual MediaPipe Data Tracking values
  const biomechanicalIssues: BiomechanicalIssue[] = [
    {
      id: 'issue-1',
      type: metrics.issues.includes("head_sway") ? 'critical' : 'optimal',
      title: 'Head Lateral Sway',
      rule: 'Head lateral shift < 2.5cm',
      description: `Head moved ${metrics.swayCm.toFixed(1)}cm off-axis during swing segment. ${metrics.issues.includes("head_sway") ? 'Leads to inconsistent strike point.' : 'Excellent stability.'}`,
      injuryRisk: metrics.issues.includes("head_sway") ? 'MEDIUM - Lumbar Strain risk' : 'MINIMAL',
      recommendation: 'Imagine a physical vertical glass plane on the right ear at setup. Hold your core fixed.'
    },
    {
      id: 'issue-2',
      type: metrics.issues.includes("early_extension") ? 'critical' : 'stable',
      title: 'Lead Knee Flex',
      rule: 'θ < 155° at impact point',
      description: `Target angular threshold monitored. Angle currently registers at ${metrics.kneeAngle.toFixed(1)}°.`,
      injuryRisk: metrics.issues.includes("early_extension") ? 'MODERATE - Joint Hyperextension' : 'LOW',
      recommendation: metrics.issues.includes("early_extension") ? 'Do not stand up at impact. Maintain loaded knee flex.' : 'Maintain current lead side weight preloading transition!'
    },
    {
      id: 'issue-3',
      type: metrics.issues.includes("torso_rotation") ? 'critical' : 'stable',
      title: 'Torso Rotation Separation',
      rule: 'Shoulder/Hip rotation separation',
      description: `Insufficient shoulder/hip torque at impact detected. ${metrics.issues.includes("torso_rotation") ? 'Loss of kinetic sequence power.' : 'Good core engagement.'}`,
      injuryRisk: metrics.issues.includes("torso_rotation") ? 'HIGH - Compensatory pivot' : 'LOW',
      recommendation: 'Target stretching exercises focusing on T-spine segment extension for 10 minutes prior to tee time.'
    }
  ];

  return (
    <div id="biometrics-root" className="flex-1 overflow-y-auto p-8 bg-surface-base grid grid-cols-1 xl:grid-cols-12 gap-8 select-none">
      
      {/* LEFT SECTION: IMPOSING 3D SKELETAL VIDEO GRID PLAYGROUND (8-Cols) */}
      <div className="xl:col-span-8 flex flex-col gap-6">
        
        {/* Viewport Card */}
        <div id="skeletal-canvas-view" className="relative bg-[#0b0b0b] border border-white/5 rounded-lg overflow-hidden flex-1 flex flex-col min-h-[500px]">
          
          {/* Top Overlays */}
          <div className="absolute top-4 left-4 z-10 flex gap-3">
            
            {/* Rotation speed overlay badge */}
            <div className="bg-black/85 backdrop-blur-xl border border-white/10 rounded px-4 py-3 shadow-[0_4px_24px_rgba(0,0,0,0.8)]">
              <span className="text-[9px] font-mono uppercase tracking-widest text-[#859585]">ROTATION SPEED</span>
              <div className="flex items-baseline gap-1 mt-0.5">
                <span id="rotation-speed-value" className="text-3xl font-bold font-sans text-brand-primary tracking-tight">
                  {rotationSpeed}
                </span>
                <span className="text-xs font-mono font-semibold text-brand-primary/80">MPH</span>
              </div>
              <div className="mt-2 w-28">
                <input
                  type="range"
                  min="80"
                  max="135"
                  value={rotationSpeed}
                  onChange={(e) => setRotationSpeed(Number(e.target.value))}
                  className="w-full accent-brand-primary cursor-pointer"
                  title="Adjust Speed"
                />
              </div>
            </div>
          </div>

          <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
            <button
              onClick={() => alert("Simulated viewport layout adjustment")}
              className="w-9 h-9 bg-black/85 border border-white/10 rounded-full flex items-center justify-center text-white hover:bg-brand-primary hover:text-black transition"
              title="Camera Toggles"
            >
              <Video className="w-4 h-4" />
            </button>
            <button
              onClick={() => {
                setRotationSpeed(105);
                setFrameIndex(142);
                setIsPlaying(false);
              }}
              className="w-9 h-9 bg-black/85 border border-white/10 rounded-full flex items-center justify-center text-white hover:text-brand-primary transition"
              title="Reset Sandbox"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Core glow interactive schematic SVG of the Golfer Skeleton */}
          <div className="flex-1 flex items-center justify-center relative p-8">
            
            {/* Visual indicators of selected hot zones */}
            <div className="absolute inset-0 bg-radial-gradient from-brand-primary/5 to-transparent pointer-events-none" />

            <svg
              viewBox="0 0 500 500"
              className="w-full max-w-[450px] aspect-square transition-all duration-300 drop-shadow-[0_0_25px_rgba(0,228,255,0.15)]"
            >
              {/* Ground alignment matrix plate */}
              <ellipse cx="250" cy="450" rx="140" ry="25" fill="none" stroke="#75ff9e" strokeWidth="1.5" strokeDasharray="6 3" opacity="0.4" />
              <ellipse cx="250" cy="450" rx="120" ry="20" fill="none" stroke="#75ff9e" strokeWidth="2" opacity="0.6" />
              <line x1="250" y1="450" x2="250" y2="280" stroke="rgba(117,255,158,0.15)" strokeWidth="1" strokeDasharray="4 4" />

              {/* Grid backdrop */}
              <line x1="50" y1="250" x2="450" y2="250" stroke="rgba(255,255,255,0.02)" strokeWidth="1" />
              <line x1="250" y1="50" x2="250" y2="450" stroke="rgba(255,255,255,0.02)" strokeWidth="1" />

              {/* SKELETAL LINES */}
              {/* Spinal column */}
              <path d="M 230 110 Q 235 150 245 220" fill="none" stroke="#00e4ff" strokeWidth="4.5" strokeLinecap="round" />
              {/* Shoulders */}
              <line x1="190" y1="140" x2="270" y2="135" stroke="#00e4ff" strokeWidth="4.5" strokeLinecap="round" />
              
              {/* Lead left arm (extended at impact swing) */}
              <line x1="190" y1="140" x2="160" y2="210" stroke="#00e4ff" strokeWidth="4" />
              <line x1="160" y1="210" x2="120" y2="280" stroke="#00e4ff" strokeWidth="4" />
              
              {/* Trailing arm */}
              <line x1="270" y1="135" x2="230" y2="200" stroke="#00e4ff" strokeWidth="3" opacity="0.8" />
              <line x1="230" y1="200" x2="120" y2="280" stroke="#00e4ff" strokeWidth="3" opacity="0.8" />

              {/* Pelvis hips line */}
              <line x1="210" y1="250" x2="280" y2="245" stroke="#00e4ff" strokeWidth="4.5" />

              {/* Left leg */}
              <line x1="210" y1="250" x2="228" y2="340" stroke="#00e4ff" strokeWidth="4" />
              <line x1="228" y1="340" x2="235" y2="440" stroke="#00e4ff" strokeWidth="4" />

              {/* Right leg */}
              <line x1="280" y1="245" x2="295" y2="335" stroke="#00e4ff" strokeWidth="3.5" opacity="0.9" />
              <line x1="295" y1="335" x2="270" y2="430" stroke="#1c1b1b" strokeWidth="3.5" strokeDasharray="3 3" />

              {/* Club shaft line in fluorescent Performance Green */}
              <line x1="120" y1="280" x2="60" y2="390" stroke="#75ff9e" strokeWidth="5.5" strokeLinecap="round" className="drop-shadow-[0_0_8px_#75ff9e]" />

              {/* Dynamic trajectory motion trace lines */}
              <path d="M 60 390 A 250 250 0 0 1 180 140" fill="none" stroke="#75ff9e" strokeWidth="1.5" strokeDasharray="5 5" opacity="0.35" />
              <circle cx="60" cy="390" r="8" fill="#75ff9e" />

              {/* INTERACTIVE JOINTS NODES */}
              {/* Head */}
              <circle
                cx="225" cy="85" r="14"
                fill={selectedJoint === 'head' ? '#75ff9e' : '#00e4ff'}
                fillOpacity="0.2"
                stroke={selectedJoint === 'head' ? '#75ff9e' : '#00e4ff'}
                strokeWidth="2.5"
                className="cursor-pointer hover:scale-125 transition-transform"
                onClick={() => setSelectedJoint(selectedJoint === 'head' ? null : 'head')}
              />
              <circle cx="225" cy="85" r="3" fill="#ffffff" />

              {/* Shoulder center */}
              <circle
                cx="230" cy="138" r="7"
                fill="#00e4ff"
                className="cursor-pointer hover:scale-125 transition-transform"
                onClick={() => setSelectedJoint('shoulders')}
              />

              {/* Arms/hands */}
              <circle
                cx="120" cy="280" r="9"
                fill="#75ff9e"
                stroke="#fff"
                strokeWidth="1.5"
                className="cursor-pointer hover:scale-125 transition"
                onClick={() => setSelectedJoint('arms')}
              />

              {/* Knee joints */}
              <circle
                cx="228" cy="340" r="7"
                fill={selectedJoint === 'knee' ? '#75ff9e' : '#00e4ff'}
                stroke="#ffffff"
                strokeWidth="1"
                className="cursor-pointer hover:scale-125 transition"
                onClick={() => setSelectedJoint(selectedJoint === 'knee' ? null : 'knee')}
              />

              {/* Alignment annotation labels in space */}
              <text x="50" y="375" fill="#75ff9e" fontSize="10" fontFamily="monospace" fontWeight="bold">IMPACT POINT</text>
              <line x1="120" y1="280" x2="160" y2="280" stroke="#75ff9e" strokeWidth="1" strokeDasharray="2 2" />
              
              <text x="250" y="340" fill="#00e4ff" fontSize="10" fontFamily="monospace">KNEE θ = 148°</text>
              <text x="255" y="132" fill="#00e4ff" fontSize="10" fontFamily="monospace">SHOULDERS 88°</text>

              {/* Highlight dynamic joint stats on select */}
              {selectedJoint && (
                <g transform="translate(300, 60)">
                  <rect width="180" height="75" rx="4" fill="#1c1b1b" stroke="#00e4ff" strokeWidth="1" opacity="0.95" />
                  <text x="12" y="22" fill="#75ff9e" fontSize="11" fontFamily="sans-serif" fontWeight="bold">
                    {selectedJoint.toUpperCase()} PROFILE
                  </text>
                  <text x="12" y="42" fill="#e5e2e1" fontSize="9" fontFamily="monospace">
                    Offset: +0.02cm (Stable)
                  </text>
                  <text x="12" y="58" fill="#e5e2e1" fontSize="9" fontFamily="monospace">
                    Force Load: {(120 * speedRatio).toFixed(0)}N / optimal
                  </text>
                </g>
              )}
            </svg>

            {/* Float visual alerts */}
            <div id="float-impact-tag" className="absolute bottom-16 left-1/2 -translate-x-1/2 bg-[#201f1f] border border-brand-primary/20 rounded py-1 px-3 text-[10px] font-mono tracking-wider text-brand-primary font-bold shadow-md uppercase">
              Impact Point Correct
            </div>
          </div>

          {/* Frame/Scrutiny loop controllers */}
          <div id="canvas-video-scrub-bar" className="bg-surface-low border-t border-white/5 p-4 flex flex-col md:flex-row items-center gap-4">
            
            {/* Play/Pause Button */}
            <div className="flex items-center gap-2.5">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-8 h-8 rounded-full bg-surface-highest hover:bg-white/10 text-white flex items-center justify-center cursor-pointer transition"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 fill-white" /> : <Play className="w-3.5 h-3.5 fill-white ml-0.5" />}
              </button>
              <span className="text-[10px] font-mono text-brand-secondary/40">SPEED:</span>
              <div className="flex bg-[#121212] p-0.5 rounded border border-white/5 text-[10px] font-mono">
                {[0.25, 0.5, 1.0].map((spd) => (
                  <button
                    key={spd}
                    onClick={() => setPlaybackSpeed(spd)}
                    className={`px-2 py-1 rounded transition ${
                      playbackSpeed === spd ? 'bg-brand-primary text-black font-semibold' : 'text-brand-secondary hover:text-white'
                    }`}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>

            {/* Precision range slider Progress timeline */}
            <div className="flex-1 w-full flex items-center gap-2">
              <span className="text-[9px] font-mono text-brand-secondary/40 font-bold">1</span>
              <input
                type="range"
                min="1"
                max="300"
                value={frameIndex}
                onChange={(e) => setFrameIndex(Number(e.target.value))}
                className="flex-1 accent-brand-primary cursor-pointer"
              />
              <span className="text-[9px] font-mono text-brand-secondary/40 font-bold">300</span>
            </div>

            {/* Frame readout status */}
            <div className="flex items-center gap-4">
              <span className="text-xs font-mono font-bold text-white whitespace-nowrap">
                {frameIndex} <span className="text-brand-secondary/40">/ 300</span>
              </span>
              <div className="flex items-center gap-1.5 bg-brand-primary/10 text-brand-primary border border-brand-primary/25 rounded-full px-2.5 py-0.5 text-[9px] font-mono font-bold tracking-wider uppercase">
                <span className="w-1.5 h-1.5 bg-brand-primary rounded-full animate-pulse" />
                Live AI Analysis
              </div>
            </div>
          </div>
        </div>

        {/* Phase selector thumbnails */}
        <div id="biomechanics-phase-strip" className="grid grid-cols-5 gap-3">
          {phases.map((ph) => {
            const isActive = activePhase === ph.id;
            return (
              <button
                key={ph.id}
                onClick={() => {
                  setActivePhase(ph.id as any);
                  setFrameIndex(ph.id === 'setup' ? 1 : ph.id === 'backswing' ? 75 : ph.id === 'downswing' ? 142 : ph.id === 'impact' ? 220 : 285);
                }}
                className={`relative bg-[#161616] rounded overflow-hidden aspect-video text-left transition duration-200 border cursor-pointer active:scale-95 group ${
                  isActive
                    ? 'border-brand-primary shadow-[0_4px_12px_rgba(117,255,158,0.2)]'
                    : 'border-white/5 opacity-50 hover:opacity-100 hover:border-white/20'
                }`}
              >
                <img
                  src={ph.img}
                  alt={ph.label}
                  referrerPolicy="no-referrer"
                  className="absolute inset-0 w-full h-full object-cover mix-blend-overlay brightness-75 group-hover:scale-105 transition duration-300"
                />
                
                {/* Text identifier bottom-aligned */}
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black to-black/10 p-2">
                  <span className={`text-[9px] font-mono tracking-wider font-bold block ${isActive ? 'text-brand-primary' : 'text-brand-secondary'}`}>
                    {ph.label}
                  </span>
                </div>

                {isActive && (
                  <span className="absolute top-2 right-2 w-1.5 h-1.5 bg-brand-primary rounded-full" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* RIGHT SECTION: POWERFUL COCKPIT BIOMETRIC DATA READOUTS (4-Cols) */}
      <div className="xl:col-span-4 flex flex-col gap-5">
        
        {/* Metric panel 1: System Precision with error rate (Image 2) */}
        <div id="precision-dashboard-card" className="bg-surface-low border border-white/5 rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585]">System Precision</span>
            <span className={`text-[9px] font-mono rounded-xs px-1.5 uppercase font-bold ${metrics.mpjpe > 8.0 ? 'bg-brand-error/15 text-brand-error' : 'bg-brand-primary/15 text-brand-primary'}`}>
                {metrics.poseStatus}
            </span>
          </div>

          <div className="flex justify-between items-baseline mb-2">
            <span className="text-xs text-brand-secondary/60 font-sans">MPJPE Metric</span>
            <span id="metric-mpjpe" className={`text-xl font-mono font-extrabold ${metrics.mpjpe > 8.0 ? 'text-[#ffb4ab]' : 'text-[#75ff9e]'}`}>
              {metrics.mpjpe.toFixed(2)}mm
            </span>
          </div>

          {/* Thin progress gradient track reacts to MPJPE error dynamically */}
          <div className="w-full bg-[#111] h-1.5 rounded-full overflow-hidden border border-white/5">
            <div className={`h-full transition-all ${metrics.mpjpe > 8.0 ? 'bg-brand-error' : 'bg-brand-primary'}`} style={{ width: `${Math.max(10, 100 - (metrics.mpjpe * 4))}%` }} />
          </div>
          <span className="text-[9px] font-mono text-brand-secondary/40 mt-1 block">
            {metrics.fps} FPS Capture. Phase locked at {metrics.phase}.
          </span>
        </div>

        {/* Metric panel 2: Neural network Model process mapping Pipeline (Image 2) */}
        <div id="model-pipeline-card" className="bg-surface-low border border-white/5 rounded-lg p-4">
          <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585] block mb-3">Model Pipeline Nodes</span>
          
          <div className="grid grid-cols-3 gap-1 divide-x divide-white/5 text-center">
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold">01 / INPUT</div>
              <div className="text-xs font-semibold text-white mt-0.5">Dual-Vid</div>
            </div>
            <div className="p-1.5 bg-brand-primary/5">
              <div className="text-[10px] font-mono text-brand-primary/80 font-semibold uppercase">02 / Split</div>
              <div className="text-xs font-bold text-brand-primary mt-0.5">80/20 STRAT</div>
            </div>
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold">03 / RECON</div>
              <div className="text-xs font-semibold text-white mt-0.5">3D Skeleton</div>
            </div>
          </div>
        </div>

        {/* Classifier Ensemble Table */}
        <div id="classifier-ensemble-card" className="bg-surface-low border border-white/5 rounded-lg p-4">
          <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585] block mb-3">Classifier Ensemble</span>
          <div className="flex flex-col gap-2 text-xs">
            <div className="flex justify-between items-center py-1 border-b border-white/5">
              <span className="font-mono text-brand-secondary/60">SVM (Support Vector Machine)</span>
              <span className="font-mono font-bold text-brand-primary">{ (0.942 * speedRatio).toFixed(3) }</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-white/5">
              <span className="font-mono text-brand-secondary/60">LR (Logistic Regression)</span>
              <span className="font-mono font-semibold text-brand-secondary/80">0.891</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="font-mono text-brand-secondary/60">LSTM (Neural Network Recurrent)</span>
              <span className="font-mono font-bold text-brand-primary">0.958</span>
            </div>
          </div>
        </div>

        {/* Metric Performance indicators */}
        <div id="accuracy-scores-grid" className="grid grid-cols-2 gap-3">
          {[
            { label: 'ACCURACY', val: '96.4%' },
            { label: 'PRECISION', val: '95.2%' },
            { label: 'RECALL', val: '94.8%' },
            { label: 'F1-SCORE', val: '95.0%' }
          ].map((item, idx) => (
            <div key={idx} className="bg-surface-low border border-white/5 rounded-lg p-3">
              <span className="text-[9px] font-mono text-brand-secondary/40 font-semibold tracking-wider uppercase block">{item.label}</span>
              <h5 className="text-[15px] font-bold font-mono text-white mt-0.5">{item.val}</h5>
            </div>
          ))}
        </div>

        {/* Biomechanical Issues Feed List */}
        <div id="biomechanical-list-card" className="bg-surface-low border border-white/5 rounded-lg p-4 flex flex-col gap-3">
          <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585] block mb-1">
            Biomechanical Diagnostics
          </span>
          <div className="flex flex-col gap-3">
            {biomechanicalIssues.map((issue) => {
              const imgIconColor =
                issue.type === 'critical'
                  ? 'bg-brand-error/10 text-brand-error border border-brand-error/20'
                  : issue.type === 'stable'
                  ? 'bg-brand-blue/10 text-brand-blue border border-brand-blue/20'
                  : 'bg-brand-primary/10 text-brand-primary border border-brand-primary/20';

              return (
                <div
                  key={issue.id}
                  className="border border-white/5 rounded bg-surface-lowest/70 p-3 hover:bg-surface-lowest transition duration-200"
                >
                  <div className="flex justify-between items-start gap-2">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${issue.type === 'critical' ? 'bg-brand-error' : issue.type === 'stable' ? 'bg-brand-blue' : 'bg-brand-primary'}`} />
                      <h6 className="text-xs font-bold text-white font-sans">{issue.title}</h6>
                    </div>
                    <span className="text-[8px] font-mono bg-white/10 text-brand-secondary px-1 py-0.5 rounded uppercase">
                      {issue.type}
                    </span>
                  </div>
                  
                  <span className="text-[9px] font-mono text-brand-secondary/40 block mt-1 uppercase">
                    {issue.rule}
                  </span>
                  
                  <p className="text-xs text-brand-secondary mt-1.5 leading-relaxed">
                    {issue.description}
                  </p>
                  
                  {issue.type === 'critical' && (
                    <div className="mt-2 bg-brand-error/5 border-l-2 border-brand-error p-1.5 rounded-r">
                      <span className="text-[8px] font-mono text-brand-error block tracking-widest font-bold uppercase">
                        INJURY RISK / ALERT RISK
                      </span>
                      <span className="text-[10px] text-white font-sans block mt-0.5">
                        {issue.injuryRisk}
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* AI Pro-Tips Card with Video popup trigger */}
        <div id="pro-tips-card" className="bg-brand-primary/5 border border-brand-primary/10 rounded-lg p-5">
          <div className="flex items-center gap-2 text-brand-primary mb-3">
            <Sparkles className="w-4 h-4" />
            <span className="text-[10px] font-mono tracking-widest font-extrabold uppercase">AI PRO-COACH TIPS</span>
          </div>

          <p className="text-xs text-brand-secondary leading-relaxed font-sans italic">
            "Focus on keeping your lead knee stable and preloading 55% weight during the initial takeaway to prevent the lateral head shift shift detected at transition segment."
          </p>

          <button
            onClick={() => setShowDrillModal(true)}
            className="w-full bg-brand-primary hover:bg-brand-primary-dim text-black font-semibold text-xs py-2.5 px-4 rounded-md transition duration-200 flex items-center justify-center gap-2 mt-4 cursor-pointer"
          >
            <span>View Drill Video</span>
          </button>
        </div>

      </div>

      {/* Interactive Coaching Modal with real golf biomechanics guidance video */}
      {showDrillModal && (
        <div id="drill-dialog-overlay" className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
          <div className="bg-surface-low border border-white/10 max-w-lg w-full rounded-lg p-6 shadow-2xl relative">
            <button
              onClick={() => setShowDrillModal(false)}
              className="absolute top-4 right-4 text-brand-secondary hover:text-white text-sm cursor-pointer"
            >
              ✕
            </button>
            <div className="flex items-center gap-2 text-brand-primary mb-3.5">
              <Sparkles className="w-5 h-5" />
              <h3 className="text-sm font-bold tracking-tight uppercase font-mono">EXERCISE: LATERAL TETHER ROTATION DRILL</h3>
            </div>
            
            <p className="text-xs text-brand-secondary mb-4 leading-relaxed">
              Designed specifically to target thoracic spine rotation while maintaining hip stabilizers locked to prevent head sway.
            </p>

            <div className="relative aspect-video rounded-md overflow-hidden bg-black/95 border border-white/5 mb-4 flex items-center justify-center">
              {/* Image representing training video */}
              <img
                src="https://images.unsplash.com/photo-1592919505780-303950717480?auto=format&fit=crop&q=80&w=480"
                alt="Drill Preview"
                referrerPolicy="no-referrer"
                className="absolute inset-0 w-full h-full object-cover opacity-60 mix-blend-overlay"
              />
              <div className="w-14 h-14 bg-brand-primary rounded-full flex items-center justify-center text-black cursor-pointer shadow-lg hover:scale-110 transition animate-pulse">
                <Play className="w-5 h-5 fill-black ml-1 text-black" />
              </div>
              <div className="absolute bottom-3 left-3 bg-black/80 px-2 py-0.5 rounded text-[10px] font-mono text-brand-primary">
                02:15 LENGTH • HD TRAINER
              </div>
            </div>

            <div className="bg-surface-lowest rounded p-3 text-[11px] text-brand-secondary flex flex-col gap-1">
              <span className="font-mono text-white text-[10px] uppercase font-bold text-brand-primary">Instructions:</span>
              <span>1. Secure dynamic bands around target lead knee hip-plane.</span>
              <span>2. Anchor hands to core belt level at 45 degree tilt angle.</span>
              <span>3. Replicate 12 transition movements in high cadence without moving head.</span>
            </div>

            <div className="mt-5 flex justify-end gap-2.5">
              <button
                onClick={() => setShowDrillModal(false)}
                className="bg-surface-highest hover:bg-white/10 text-white font-semibold text-xs px-4 py-2.5 rounded transition cursor-pointer"
              >
                Close Guidance
              </button>
              <button
                onClick={() => {
                  alert("Successfully added this exercise to your drill logs!");
                  setShowDrillModal(false);
                }}
                className="bg-brand-primary hover:bg-brand-primary-dim text-black font-semibold text-xs px-4 py-2.5 rounded transition cursor-pointer"
              >
                Add to Drill Logs
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
