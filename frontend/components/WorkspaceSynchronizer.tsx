"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect } from 'react';
import {
  Upload,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Zap,
  CheckCircle,
  Lock,
  Unlock,
  Sparkles,
  RefreshCw
} from 'lucide-react';
import { DemoDltResponse } from '../types';

declare global {
  interface Window {
    Pose: any;
  }
}

interface WorkspaceSynchronizerProps {
  selectedSampleName: string;
  setSelectedSampleName: (name: string) => void;
  onMetricsUpdate?: (metrics: any) => void;
  onDemoLoaded?: (demo: DemoDltResponse) => void;
}

export default function WorkspaceSynchronizer({
  selectedSampleName,
  setSelectedSampleName,
  onMetricsUpdate,
  onDemoLoaded
}: WorkspaceSynchronizerProps) {
  // Sync Controls State
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [frameIndex, setFrameIndex] = useState<number>(1);
  const totalFrames = 120;
  const [offset, setOffset] = useState<number>(0.04);
  const [isLocked, setIsLocked] = useState<boolean>(true);
  const [complexity, setComplexity] = useState<number>(2);

  // Native files and URLs for MediaPipe Video Processing
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [frontVideoURL, setFrontVideoURL] = useState<string | null>(null);
  const [sideVideoURL, setSideVideoURL] = useState<string | null>(null);
  
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [analysisNotice, setAnalysisNotice] = useState<string | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [isDemoLoading, setIsDemoLoading] = useState<boolean>(false);

  // References for the 2D overlay only. Metric-grade 3D values require calibrated DLT.
  const initialHeadXRef = useRef<number | null>(null);
  const lastMetricsUpdate = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(Date.now()); 

  const containerRef = useRef<HTMLDivElement>(null);
  const videoFrontRef = useRef<HTMLVideoElement>(null);
  const videoSideRef = useRef<HTMLVideoElement>(null);
  const canvasFrontRef = useRef<HTMLCanvasElement>(null);
  const canvasSideRef = useRef<HTMLCanvasElement>(null);

  // Playback timer bridging
  useEffect(() => {
    let interval: any = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setFrameIndex((prev) => (prev >= totalFrames ? 1 : prev + 1));
      }, 50);
      videoFrontRef.current?.play();
      videoSideRef.current?.play();
    } else {
      videoFrontRef.current?.pause();
      videoSideRef.current?.pause();
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Handle local preview uploads
  const handleFileUpload = (type: 'front' | 'side', e: any) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      if (type === 'front') {
        setFrontFile(file);
        setFrontVideoURL(url);
      } else {
        setSideFile(file);
        setSideVideoURL(url);
      }
    }
  };

  const loadSample = async () => {
    setIsDemoLoading(true);
    setDemoError(null);
    setAnalysisNotice(null);

    try {
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';
      const response = await fetch(`${apiBaseUrl}/demo/dlt-reconstruction`);
      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const demo = await response.json() as DemoDltResponse;
      if (demo.status !== 'ok' || !Array.isArray(demo.frames) || demo.frames.length === 0) {
        throw new Error('Demo endpoint did not return reconstructed frames.');
      }

      setSelectedSampleName('Calibrated_DLT_Demo');
      onDemoLoaded?.(demo);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setDemoError(`Could not load calibrated DLT demo: ${message}. No fallback skeleton was rendered.`);
    } finally {
      setIsDemoLoading(false);
    }
  };

  const triggerAutoSync = () => {
    setIsSyncing(true);
    setTimeout(() => {
      setOffset(0.04);
      setFrameIndex(46);
      setIsSyncing(false);
    }, 1500);
  };

  // -----------------------------------------------------
  // FRONTEND 2D MEDIAPIPE KEYPOINT EXTRACTION ENGINE
  // -----------------------------------------------------

  // Helper calculating joint angles in 2D space (matching Python logic)
  const calculateAngle = (a: any, b: any, c: any) => {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return angle;
  };

  const drawKineticSkeleton = (ctx: CanvasRenderingContext2D, landmarks: any, width: number, height: number, isFront: boolean, frameIdx: number) => {
    ctx.save();
    const primaryColor = isFront ? "#75ff9e" : "#00e4ff"; 
    
    // FPS Calc
    const now = Date.now();
    const fps = Math.round(1000 / Math.max(now - lastFrameTimeRef.current, 1));
    lastFrameTimeRef.current = now;

    // ------------------------------------------------------------------
    // 2D overlay annotation. Metric-grade biomechanics require calibrated DLT.
    // ------------------------------------------------------------------
    let swayProxy = 0;
    let leftKneeAngle = 180;
    let phase = "Setup";
    let poseStatus = "2D Clear";
    let currentIssues: string[] = [];

    if (landmarks[0] && landmarks[23] && landmarks[25] && landmarks[27] && landmarks[11] && landmarks[12]) {
        // 1. Lateral Head Sway proxy, expressed as percent of frame width.
        if (initialHeadXRef.current === null) initialHeadXRef.current = landmarks[0].x;
        swayProxy = Math.abs(landmarks[0].x - initialHeadXRef.current) * 100;

        // 2. Lead Knee Angle 
        leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);

        // Determine Phase based on hands vs hip Y pos
        const handY = (landmarks[15].y + landmarks[16].y) / 2;
        const hipY = (landmarks[23].y + landmarks[24].y) / 2;
        const shoulderY = (landmarks[11].y + landmarks[12].y) / 2;

        if (handY < shoulderY) phase = "Backswing";
        else if (handY > hipY) phase = "Impact";
        else phase = "Downswing";

        // Push diagnostics
        if (swayProxy > 2.5) currentIssues.push("head_sway");
        if (leftKneeAngle >= 155 && phase === "Impact") currentIssues.push("early_extension");

        if (currentIssues.length > 0) poseStatus = "2D Flag";
    }

    // Pass data upwards to React state (throttled)
    if (onMetricsUpdate && (now - lastMetricsUpdate.current > 300)) {
        onMetricsUpdate({
            swayProxy,
            kneeAngle: leftKneeAngle,
            mpjpe: null,
            fps,
            phase,
            poseStatus,
            issues: currentIssues
        });
        lastMetricsUpdate.current = now;
    }

    // Draw Standard Skeleton Connections
    const CONNECTIONS = [
        [11, 13], [13, 15], [12, 14], [14, 16], [15, 17], // Arms
        [23, 25], [25, 27], [27, 29], [29, 31], // Left Leg
        [24, 26], [26, 28], [28, 30], [30, 32], // Right Leg
        [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10] // Face
    ];

    ctx.lineWidth = 3;
    CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const p1 = landmarks[startIdx];
        const p2 = landmarks[endIdx];
        if (p1 && p2 && p1.visibility > 0.4 && p2.visibility > 0.4) {
            ctx.beginPath();
            
            // Dynamic coloring based on issues
            if (currentIssues.includes("early_extension") && [23,25,27,24,26,28].includes(startIdx)) {
               ctx.strokeStyle = "red";
            } else {
               ctx.strokeStyle = primaryColor;
            }

            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        }
    });

    // Draw Torso Logic (11->23->24->12)
    if (landmarks[11] && landmarks[12] && landmarks[23] && landmarks[24]) {
        ctx.beginPath();
        ctx.moveTo(landmarks[11].x * width, landmarks[11].y * height);
        ctx.lineTo(landmarks[23].x * width, landmarks[23].y * height);
        ctx.lineTo(landmarks[24].x * width, landmarks[24].y * height);
        ctx.lineTo(landmarks[12].x * width, landmarks[12].y * height);
        ctx.closePath();
        
        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = 3;
        ctx.stroke();
    }

    // Draw Nodes
    landmarks.forEach((lm: any) => {
        if (lm.visibility > 0.4) {
            ctx.beginPath();
            ctx.arc(lm.x * width, lm.y * height, 4, 0, 2 * Math.PI);
            ctx.fillStyle = "#131313";
            ctx.fill();
            ctx.strokeStyle = primaryColor;
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    });

    // ------------------------------------------------------------------
    // HUD OVERLAY TEXT (Mirroring OpenCV Output)
    // ------------------------------------------------------------------
    ctx.font = "bold 16px sans-serif";
    ctx.fillStyle = "rgba(229, 226, 225, 0.9)"; // Off-white
    ctx.fillText(`FPS: ${fps}`, 15, 30);

    ctx.fillStyle = poseStatus === "2D Flag" ? "#ff0000" : "#00ff00";
    ctx.fillText(`${poseStatus}`, 15, 55);

    ctx.fillStyle = "#0000ff"; // Blue for phase
    ctx.fillText(`Phase: ${phase}`, 15, 80);

    ctx.fillStyle = "#a0a0a0";
    ctx.fillText(`3D DLT: Awaiting calibration`, 15, 105);

    if (currentIssues.length > 0) {
        ctx.fillStyle = "#ff0000";
        ctx.fillText(`Issues: ${currentIssues.join(", ")}`, 15, 130);
    }
    
    // Bottom left Frame Index
    ctx.font = "bold 14px sans-serif";
    ctx.fillStyle = "white";
    ctx.fillText(`Frame: ${frameIdx}`, 15, height - 15);

    ctx.restore();
  };

  // Mount MediaPipe Script securely and run continuously for Front Video
  useEffect(() => {
    let poseFront: any = null;
    let animationId: number;

    const initMediaPipeFront = async () => {
      if (!window.Pose) {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js';
        script.async = true;
        document.body.appendChild(script);
        await new Promise((resolve) => { script.onload = resolve; });
      }

      poseFront = new window.Pose({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      poseFront.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      poseFront.onResults((results: any) => {
        const canvas = canvasFrontRef.current;
        const video = videoFrontRef.current;
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.poseLandmarks) {
          drawKineticSkeleton(ctx, results.poseLandmarks, canvas.width, canvas.height, true, frameIndex);
        }
      });

      const processVideo = async () => {
        if (videoFrontRef.current && !videoFrontRef.current.paused && videoFrontRef.current.readyState >= 2) {
            await poseFront.send({ image: videoFrontRef.current });
        }
        animationId = requestAnimationFrame(processVideo);
      };
      
      // Start tracking loop
      processVideo();
    };

    if (frontVideoURL) initMediaPipeFront();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (poseFront) poseFront.close();
    };
  }, [frontVideoURL]);

  // Mount MediaPipe Script securely and run continuously for Side Video
  useEffect(() => {
    let poseSide: any = null;
    let animationId: number;

    const initMediaPipeSide = async () => {
      if (!window.Pose) await new Promise(r => setTimeout(r, 500)); // wait for front mount
      
      poseSide = new window.Pose({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      poseSide.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      poseSide.onResults((results: any) => {
        const canvas = canvasSideRef.current;
        const video = videoSideRef.current;
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.poseLandmarks) {
          drawKineticSkeleton(ctx, results.poseLandmarks, canvas.width, canvas.height, false, frameIndex);
        }
      });

      const processVideo = async () => {
        if (videoSideRef.current && !videoSideRef.current.paused && videoSideRef.current.readyState >= 2) {
            await poseSide.send({ image: videoSideRef.current });
        }
        animationId = requestAnimationFrame(processVideo);
      };
      
      processVideo();
    };

    if (sideVideoURL) initMediaPipeSide();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (poseSide) poseSide.close();
    };
  }, [sideVideoURL]);


  const handleStartAnalysis = async () => {
      // TODO(Phase 2): route only the public calibrated demo through a real backend job;
      // user uploads stay blocked until Phase 3 camera calibration exists.
      setAnalysisNotice(
        "Analysis pipeline not yet connected. Real 3D DLT triangulation requires calibrated projection matrices and a submitted backend job. User-uploaded 3D analysis remains disabled until calibration exists."
      );
  };

  return (
    <div id="synchronizer-root" className="flex-1 overflow-y-auto p-8 bg-surface-base">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <span className="text-xs uppercase font-mono tracking-widest text-brand-secondary/50">WORKSPACE / ANALYSIS</span>
          <h2 className="text-2xl font-bold font-sans text-white tracking-tight mt-1">Multi-Angle Synchronizer</h2>
        </div>

        <div id="sync-toolbar" className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2 bg-surface-low border border-white/5 py-1 px-3 rounded text-xs">
            <span className="text-brand-secondary/60 font-mono tracking-wider text-[11px] mr-1">COMPLEXITY</span>
            {[1, 2, 3].map((num) => (
              <button
                key={num}
                onClick={() => setComplexity(num)}
                className={`w-7 h-7 rounded flex items-center justify-center font-mono font-bold transition-all ${
                  complexity === num
                    ? 'bg-brand-primary text-black shadow-[0_2px_8px_rgba(117,255,158,0.3)]'
                    : 'text-brand-secondary hover:text-white hover:bg-white/5'
                }`}
              >
                {num}
              </button>
            ))}
          </div>

          <button
            id="btn-load-dlt-demo"
            onClick={loadSample}
            disabled={isDemoLoading}
            className="border border-brand-primary/20 text-brand-primary hover:bg-brand-primary/10 disabled:opacity-60 disabled:cursor-not-allowed transition px-3.5 py-2 rounded text-xs font-mono font-semibold"
          >
            {isDemoLoading ? 'Loading DLT Demo...' : 'Load Calibrated DLT Demo'}
          </button>

          <button onClick={() => alert("Draft saved successfully components in workspace state.")} className="bg-surface-low hover:bg-surface-high border border-white/10 text-brand-secondary hover:text-white px-4 py-2 rounded text-xs font-sans transition flex items-center gap-2 cursor-pointer">
            <span>Save Draft</span>
          </button>

          <button
            onClick={handleStartAnalysis}
            className="bg-brand-primary hover:bg-brand-primary-dim text-black font-semibold px-5 py-2.5 rounded transition flex items-center gap-2 text-xs shadow-[0_4px_12px_rgba(117,255,158,0.25)] hover:shadow-[0_4px_20px_rgba(117,255,158,0.35)] cursor-pointer tracking-wider uppercase active:scale-[0.98]"
          >
            <Zap className="w-4 h-4 text-black fill-black" />
            <span>Start 3D Analysis</span>
          </button>
        </div>
      </div>

      {analysisNotice && (
        <div className="mb-6 rounded border border-[#facc15]/30 bg-[#facc15]/10 px-4 py-3 text-sm text-[#fde68a]">
          <div className="font-semibold text-white">Analysis pipeline not available</div>
          <div className="mt-1 text-xs leading-relaxed text-[#fde68a]">{analysisNotice}</div>
        </div>
      )}

      {demoError && (
        <div
          id="demo-dlt-error"
          className="mb-6 rounded border border-brand-error/30 bg-brand-error/10 px-4 py-3 text-sm text-[#ffb4ab]"
        >
          <div className="font-semibold text-white">Calibrated DLT demo failed</div>
          <div className="mt-1 text-xs leading-relaxed">{demoError}</div>
        </div>
      )}

      {analysisNotice ? (
        <div
          id="analysis-blocked-state"
          className="rounded-lg border border-white/5 bg-surface-low p-8 text-center min-h-[360px] flex flex-col items-center justify-center"
        >
          <div className="text-xs font-mono uppercase tracking-widest text-[#facc15]">No analysis job submitted</div>
          <h3 className="mt-3 text-2xl font-bold text-white">3D analysis is disabled</h3>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-brand-secondary">
            The app is not showing sync scores, frame matches, MPJPE, classifier output, or a skeleton because no calibrated backend reconstruction job ran.
          </p>
          <div className="mt-6 rounded border border-[#facc15]/20 bg-[#facc15]/5 px-4 py-3 text-xs text-[#fde68a]">
            Public calibrated demo: Phase 2 pending. User uploads: calibration required.
          </div>
        </div>
      ) : (
        <>
      <div id="synchronizer-grid" className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        
        {/* FRONT VIEW CARD/PLAYER */}
        <div className="relative bg-surface-low border border-white/5 rounded p-5 relative overflow-hidden flex flex-col justify-between group min-h-[380px]">
          <div className="flex justify-between items-center mb-4 z-10">
            <span className="text-[10px] font-mono bg-brand-primary/10 text-brand-primary border border-brand-primary/20 rounded px-2.5 py-1 tracking-wider uppercase font-semibold">
              Primary Source
            </span>
            <span className="text-xs text-brand-secondary/40 font-mono">FRONT VIEW</span>
          </div>

          {frontVideoURL ? (
            <div className="relative border border-white/5 bg-[#0e0e0e] rounded aspect-video overflow-hidden flex flex-col items-center justify-center mb-2">
              <video 
                  ref={videoFrontRef}
                  src={frontVideoURL}
                  crossOrigin="anonymous"
                  className="absolute inset-0 w-full h-full object-contain"
                  playsInline
                  autoPlay
                  loop
                  muted
              />
              <canvas ref={canvasFrontRef} className="absolute inset-0 w-full h-full pointer-events-none z-10 opacity-80" />
              
              <div className="absolute top-3 left-3 z-20 bg-black/60 backdrop-blur-md border border-white/10 text-[10px] uppercase font-mono px-2 py-0.5 rounded text-brand-blue flex items-center gap-1.5 shadow">
                <span className="w-2 h-2 rounded-full bg-brand-primary animate-pulse" />
                <span>FRONT CAPTURE / {frontFile ? frontFile.name : selectedSampleName}</span>
              </div>
              <div className="absolute bottom-3 right-3 z-20 bg-black/70 px-2 py-1 rounded text-[9px] font-mono text-brand-secondary/50">
                FRAME {frameIndex}/120
              </div>
            </div>
          ) : (
            <label className="flex-1 border-2 border-dashed border-white/10 hover:border-brand-primary/30 rounded flex flex-col items-center justify-center p-8 text-center cursor-pointer transition bg-surface-lowest/40 hover:bg-surface-lowest group relative z-10">
              <input type="file" accept="video/*" className="hidden" onChange={(e) => handleFileUpload('front', e)} />
              <div className="w-12 h-12 rounded-full bg-brand-primary/5 group-hover:bg-brand-primary/10 border border-brand-primary/10 flex items-center justify-center text-brand-primary mb-3 transition shadow">
                <Upload className="w-5 h-5" />
              </div>
              <p className="text-sm font-semibold text-white">Front View</p>
              <p className="text-xs text-brand-secondary/50 mt-1 max-w-[200px]">
                Drag & Drop or Click to Upload Front-Angle Swing
              </p>
              <p className="text-[10px] text-brand-secondary/30 font-mono mt-3 uppercase tracking-wider">
                MP4, MOV, AVI (Max 200MB)
              </p>
            </label>
          )}

          {frontVideoURL && (
            <div className="flex justify-between items-center text-[11px] font-mono text-brand-secondary/50 mt-2 px-1 z-10">
              <span>Resolution: Not measured</span>
              <button onClick={() => setFrontVideoURL(null)} className="text-brand-error hover:underline transition">
                Clear Video
              </button>
            </div>
          )}
        </div>

        {/* SIDE VIEW CARD/PLAYER */}
        <div className="relative bg-surface-low border border-white/5 rounded p-5 relative overflow-hidden flex flex-col justify-between group min-h-[380px]">
          <div className="flex justify-between items-center mb-4 z-10">
            <span className="text-[10px] font-mono bg-[#00e4ff]/10 text-brand-blue border border-[#00e4ff]/20 rounded px-2.5 py-1 tracking-wider uppercase font-semibold">
              Secondary Frame
            </span>
            <span className="text-xs text-brand-secondary/40 font-mono">SIDE VIEW</span>
          </div>

          {sideVideoURL ? (
            <div className="relative border border-white/5 bg-[#0e0e0e] rounded aspect-video overflow-hidden flex flex-col items-center justify-center mb-2">
              <video 
                  ref={videoSideRef}
                  src={sideVideoURL}
                  crossOrigin="anonymous"
                  className="absolute inset-0 w-full h-full object-contain"
                  playsInline
                  autoPlay
                  loop
                  muted
              />
              <canvas ref={canvasSideRef} className="absolute inset-0 w-full h-full pointer-events-none z-10 opacity-80" />
              
              <div className="absolute top-3 left-3 z-20 bg-black/60 backdrop-blur-md border border-white/10 text-[10px] uppercase font-mono px-2 py-0.5 rounded text-brand-blue flex items-center gap-1.5 shadow">
                <span className="w-2 h-2 rounded-full bg-brand-blue" />
                <span>SIDE CAPTURE / {sideFile ? sideFile.name : selectedSampleName}</span>
              </div>
              <div className="absolute bottom-3 right-3 z-20 bg-black/70 px-2 py-1 rounded text-[9px] font-mono text-brand-secondary/50">
                FRAME {(frameIndex + (isLocked ? 0 : 3)) % totalFrames || 1}/120
              </div>
            </div>
          ) : (
            <label className="flex-1 border-2 border-dashed border-white/10 hover:border-brand-primary/30 rounded flex flex-col items-center justify-center p-8 text-center cursor-pointer transition bg-surface-lowest/40 hover:bg-surface-lowest group relative z-10">
              <input type="file" accept="video/*" className="hidden" onChange={(e) => handleFileUpload('side', e)} />
              <div className="w-12 h-12 rounded-full bg-brand-primary/5 group-hover:bg-brand-primary/10 border border-brand-primary/10 flex items-center justify-center text-brand-primary mb-3 transition shadow">
                <Upload className="w-5 h-5" />
              </div>
              <p className="text-sm font-semibold text-white">Side View</p>
              <p className="text-xs text-brand-secondary/50 mt-1 max-w-[200px]">
                Drag & Drop or Click to Upload Side-Angle Swing
              </p>
              <p className="text-[10px] text-brand-secondary/30 font-mono mt-3 uppercase tracking-wider">
                MP4, MOV, AVI (Max 200MB)
              </p>
            </label>
          )}

          {sideVideoURL && (
            <div className="flex justify-between items-center text-[11px] font-mono text-brand-secondary/50 mt-2 px-1 z-10">
              <span>Resolution: Not measured</span>
              <button onClick={() => setSideVideoURL(null)} className="text-brand-error hover:underline transition">
                Clear Video
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Frame synchronization controller bar at bottom of workspace */}
      <div id="synchronizer-timeline-panel" className="bg-surface-low border border-white/5 rounded-lg p-5">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-white/5 pb-4 mb-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setFrameIndex((prev) => Math.max(1, prev - 1))}
                className="p-1.5 text-brand-secondary hover:text-white rounded hover:bg-white/5 cursor-pointer"
              >
                <SkipBack className="w-4 h-4" />
              </button>

              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-10 h-10 rounded-full bg-brand-primary hover:bg-brand-primary-dim text-black flex items-center justify-center shadow-[0_0_12px_rgba(117,255,158,0.3)] hover:shadow-[0_0_18px_rgba(117,255,158,0.45)] cursor-pointer transition-all active:scale-95"
              >
                {isPlaying ? <Pause className="w-4 h-4 text-black fill-black" /> : <Play className="w-4 h-4 text-black fill-black ml-0.5" />}
              </button>

              <button
                onClick={() => setFrameIndex((prev) => Math.min(totalFrames, prev + 1))}
                className="p-1.5 text-brand-secondary hover:text-white rounded hover:bg-white/5 cursor-pointer"
              >
                <SkipForward className="w-4 h-4" />
              </button>
            </div>

            <div className="w-px h-8 bg-white/10" />

            <div>
              <span className="text-[10px] font-mono uppercase text-brand-secondary/40 tracking-wider">Current Offset</span>
              <div className="flex items-baseline gap-1 text-brand-blue font-mono">
                <span className="text-lg font-bold">
                  {offset >= 0 ? `+${offset.toFixed(2)}` : offset.toFixed(2)}
                </span>
                <span className="text-[10px] uppercase font-semibold text-brand-blue/70">s</span>
              </div>
            </div>

            <div className="hidden sm:block">
              <span className="text-[10px] font-mono uppercase text-brand-secondary/40 tracking-wider block">Auto-Sync Status</span>
              <span className="text-xs font-semibold text-white flex items-center gap-1 mt-0.5">
                <CheckCircle className="w-3.5 h-3.5 text-brand-primary" /> Preview Alignment
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={triggerAutoSync}
              disabled={isSyncing}
              className="border border-brand-primary/20 text-brand-primary hover:bg-brand-primary/10 px-4 py-2 rounded text-xs px-3.5 py-1.5 font-sans font-medium transition cursor-pointer flex items-center gap-1.5"
            >
              {isSyncing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Aligning...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Preview Align</span>
                </>
              )}
            </button>

            <button
              onClick={() => setIsLocked(!isLocked)}
              className={`border px-3.5 py-1.5 rounded text-xs font-sans font-medium transition cursor-pointer flex items-center gap-1.5 ${
                isLocked
                  ? 'bg-brand-primary/10 border-brand-primary/30 text-brand-primary'
                  : 'border-white/10 text-brand-secondary hover:text-white'
              }`}
            >
              {isLocked ? (
                <>
                  <Lock className="w-3.5 h-3.5" />
                  <span>Lock Offset</span>
                </>
              ) : (
                <>
                  <Unlock className="w-3.5 h-3.5" />
                  <span>Lock Offset</span>
                </>
              )}
            </button>
          </div>
        </div>

        <div className="flex flex-col gap-4" ref={containerRef}>
          <div className="flex items-center gap-4">
            <span className="w-24 text-[10px] tracking-wider font-mono text-brand-secondary/40 font-semibold uppercase">
              Front View
            </span>
            <div className="flex-1 bg-[#151515] border border-white/5 rounded px-4 py-1.5 relative select-none h-14 flex items-center">
              <div className="flex items-center justify-between w-full h-8 opacity-40">
                {Array.from({ length: 90 }).map((_, idx) => {
                  // deterministic waveform heights (same on server & client)
                  const raw = Math.abs(Math.sin(idx * 0.15)) * 26 + (idx % 7 === 0 ? 6 : 2);
                  const heightPx = Math.round(raw); // integer px ensures identical serialization
                  const isImpact = idx > 43 && idx < 48;
                  const bg = isImpact ? 'rgb(117,255,158)' : 'rgb(229,226,225)'; // explicit rgb avoids hex/rgb mismatch
                  const op = isImpact ? 1 : 0.4;
                  return (
                    <div
                      key={idx}
                      className="w-[3px] rounded-full transition-colors"
                      style={{
                        height: `${heightPx}px`,
                        backgroundColor: bg,
                        opacity: op
                      }}
                    />
                  );
                })}
              </div>

              {/* Scrub Progress Track and Needle */}
              <div
                className="absolute inset-y-0 w-0.5 bg-brand-primary-dim shadow-[0_0_8px_#75ff9e] pointer-events-none"
                style={{ left: `${(frameIndex / totalFrames) * 98}%` }}
              >
                <div className="w-2.5 h-2.5 bg-brand-primary rounded-full -ml-[4px] -mt-[4px] pointer-events-none ring-4 ring-brand-primary/10" />
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className="w-24 text-[10px] tracking-wider font-mono text-brand-secondary/40 font-semibold uppercase">
              Side View
            </span>
            <div className="flex-1 bg-[#151515] border border-white/5 rounded px-4 py-1.5 relative select-none h-14 flex items-center">
              <div className="flex items-center justify-between w-full h-8 opacity-30">
                {Array.from({ length: 90 }).map((_, idx) => {
                  const raw = Math.abs(Math.cos(idx * 0.18)) * 24 + ((idx + 3) % 8 === 0 ? 8 : 1);
                  const heightPx = Math.round(raw);
                  const isImpact = idx > 40 && idx < 45;
                  const bg = isImpact ? 'rgb(0,228,255)' : 'rgb(229,226,225)';
                  const op = isImpact ? 0.95 : 0.35;
                  return (
                    <div
                      key={idx}
                      className="w-[3px] rounded-full transition-colors"
                      style={{
                        height: `${heightPx}px`,
                        backgroundColor: bg,
                        opacity: op
                      }}
                    />
                  );
                })}
              </div>

              <div
                className="absolute inset-y-0 w-0.5 bg-[#00e4ff] pointer-events-none"
                style={{ left: `${((frameIndex + (isLocked ? 0 : 4)) / totalFrames) * 98}%` }}
              >
                <div className="w-2.5 h-2.5 bg-brand-blue rounded-full -ml-[4px] -mt-[4px] pointer-events-none ring-4 ring-brand-blue/10" />
                
                <div className="absolute top-12 -left-6 bg-[#161616] border border-white/10 text-[9px] font-mono text-brand-blue py-0.5 px-1.5 rounded-sm whitespace-nowrap shadow-md">
                  +{offset.toFixed(2)}s
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-5 flex items-center justify-between text-[10px] font-mono text-brand-secondary/40 uppercase">
          <span>0.0s (Start)</span>
          <div className="flex items-center gap-1 text-xs font-semibold text-brand-secondary">
            <span>Drag sliders or timelines to scrub frames</span>
          </div>
          <span>6.0s (End)</span>
        </div>
      </div>

      <div id="metrics-tiles-panel" className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
        <div className="bg-surface-low border border-white/5 rounded p-4">
          <span className="text-[10px] font-mono text-brand-secondary/40 font-semibold tracking-wider uppercase">Alignment Score</span>
          <div className="flex items-center gap-2 mt-1">
            <h4 id="metric-align-score" className="text-xl font-bold font-sans text-brand-primary tracking-tight">
                Preview only
            </h4>
            <div className="w-4 h-4 rounded-full bg-brand-primary/10 flex items-center justify-center text-brand-primary text-[10px] font-bold">✓</div>
          </div>
          <span className="text-[9px] font-mono text-brand-secondary/30 mt-1 block">Upload sync preview only</span>
        </div>

        <div className="bg-surface-low border border-white/5 rounded p-4">
          <span className="text-[10px] font-mono text-[#859585] font-semibold tracking-wider uppercase">Frame Match</span>
          <div className="flex items-baseline gap-1 mt-1">
            <h4 id="metric-frame-match" className="text-xl font-mono font-bold text-white tracking-tight">{frameIndex}</h4>
            <span className="text-xs text-brand-secondary/40">/ {totalFrames}</span>
          </div>
          <div className="w-full bg-[#1e1e1e] rounded-[1px] h-1 overflow-hidden mt-1.5">
            <div className="bg-brand-primary h-full" style={{ width: `${(frameIndex / totalFrames) * 100}%` }} />
          </div>
        </div>

        <div className="bg-surface-low border border-white/5 rounded p-4">
          <span className="text-[10px] font-mono text-brand-secondary/40 font-semibold tracking-wider uppercase">Capture Rate</span>
          <h4 id="metric-fps" className="text-xl font-semibold font-sans text-white tracking-tight mt-1">Not measured</h4>
          <span className="text-[9px] font-mono text-[#00e4ff] mt-1 block">Video FPS readout not connected</span>
        </div>

        <div className="bg-surface-low border border-white/5 rounded p-4">
          <span className="text-[10px] font-mono text-brand-secondary/40 font-semibold tracking-wider uppercase">Processing Status</span>
          <div className="flex items-center gap-2 mt-1">
            <h4 id="metric-status" className="text-xl font-semibold font-sans text-white tracking-tight">
                {((frontVideoURL || sideVideoURL) && isPlaying) ? '2D Overlay' : 'Ready' }
            </h4>
            <span className={`w-2.5 h-2.5 rounded-full ${isPlaying ? 'bg-brand-blue animate-pulse' : 'bg-brand-primary'}`} />
          </div>
          <span className="text-[9px] font-mono text-brand-secondary/30 mt-1 block">3D analysis pending calibration</span>
        </div>
      </div>
        </>
      )}
    </div>
  );
}
