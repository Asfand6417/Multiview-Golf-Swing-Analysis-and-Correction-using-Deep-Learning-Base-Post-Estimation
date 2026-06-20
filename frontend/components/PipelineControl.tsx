"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import {
  Brain,
  Sliders,
  Play,
  Activity,
  CheckCircle,
  FileCode,
  Layers,
  Sparkles,
  Search,
  Zap,
  Cpu,
  RefreshCw,
  Terminal,
  Database
} from 'lucide-react';
import { PipelineStep } from '../types';

export default function PipelineControl() {
  const [trainSplit, setTrainSplit] = useState<number>(80); // 80% split default
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [progressVal, setProgressVal] = useState<number>(0);
  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const terminalBottomRef = useRef<HTMLDivElement>(null);

  // Translate split percentage into count values
  const totalSwingSamples = 1400;
  const trainCount = Math.round(totalSwingSamples * (trainSplit / 100));
  const valCount = totalSwingSamples - trainCount;

  // Pipeline Status list mapped visually
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([
    {
      id: 'step-1',
      number: '01',
      name: 'Multi-Angle Acquisition',
      description: 'Acquiring paired dual-view front and side HD video recordings at 120 FPS high rate constraint.',
      status: 'completed'
    },
    {
      id: 'step-2',
      number: '02',
      name: 'Dynamic Temporal Alignment',
      description: 'Translating sub-frame audio waveform impact levels matching offsets perfectly.',
      status: 'completed'
    },
    {
      id: 'step-3',
      number: '03',
      name: 'Classifier Weight Fine-Tuning',
      description: 'Backpropagating CNN convolutional landmark checkpoints over biomechanical reference swing datasets.',
      status: 'active'
    },
    {
      id: 'step-4',
      number: '04',
      name: '3D Point Spatial Reconstruction',
      description: 'Recompiling 2D coordinates into a 3D biomechanical joint skeletal orientation cloud.',
      status: 'pending'
    },
    {
      id: 'step-5',
      number: '05',
      name: 'Diagnostic Rule Evaluation',
      description: 'Parsing angle limits (theta shoulders, head sway, knees) matching biomechanical profiles.',
      status: 'pending'
    }
  ]);

  // Terminal log simulation upon model execution
  useEffect(() => {
    if (isTraining) {
      const messages = [
        "Initializing pipeline optimizer wrapper on CUDA device [id:0 Gforce]...",
        "Loading Convolutional Pose Checkpoints: ResNet50_Kinematic_Weights.bin.",
        "Compiling dataset categories: Setup, Backswing, Downswing, Impact, Finish...",
        `Stratified split configuration applied: ${trainSplit}% Train set / ${100 - trainSplit}% Validation set`,
        `Selected: ${trainCount} training inputs, ${valCount} validations.`,
        "Preloading local memory cache buffer blocks... Ready.",
        "----------------------------------------------------------------",
        "Epoch 1/5 - Loss: 0.8412 - Val Loss: 0.9012 - Accuracy: 84.5%",
        "Epoch 2/5 - Loss: 0.6214 - Val Loss: 0.6124 - Accuracy: 89.2%",
        "Epoch 3/5 - Loss: 0.4518 - Val Loss: 0.4901 - Accuracy: 92.4%",
        "Epoch 4/5 - Loss: 0.3121 - Val Loss: 0.3802 - Accuracy: 94.8%",
        "Epoch 5/5 - Loss: 0.1982 - Val Loss: 0.2841 - Accuracy: 96.4%",
        "----------------------------------------------------------------",
        "✓ Backpropagation optimize vector successful.",
        "✓ Classifier weights compiled: SVM (0.942), LSTM (0.958).",
        "✓ 3D Spatial landmark model fine-tuning successfully finalized!"
      ];

      setTerminalLogs([]);
      let logIdx = 0;
      const logInterval = setInterval(() => {
        if (logIdx < messages.length) {
          setTerminalLogs((prev) => [...prev, messages[logIdx]]);
          logIdx++;
          setProgressVal((prev) => Math.min(100, Math.round((logIdx / messages.length) * 100)));
        } else {
          clearInterval(logInterval);
          setIsTraining(false);
          // Complete step 3
          setPipelineSteps((prev) =>
            prev.map((step) =>
              step.id === 'step-3'
                ? { ...step, status: 'completed' }
                : step.id === 'step-4'
                ? { ...step, status: 'active' }
                : step
            )
          );
        }
      }, 500);

      return () => clearInterval(logInterval);
    }
  }, [isTraining]);

  useEffect(() => {
    if (terminalBottomRef.current) {
      terminalBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalLogs]);

  const initiateTraining = () => {
    setIsTraining(true);
    setProgressVal(0);
  };

  return (
    <div id="pipeline-root" className="flex-1 overflow-y-auto p-8 bg-surface-base">
      
      {/* Visual Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <span className="text-xs uppercase font-mono tracking-widest text-[#859585]">NEURAL NETWORK FINE-TUNING Cockpit</span>
          <h2 className="text-2xl font-bold text-white tracking-tight mt-1">Model Pipeline Control</h2>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 bg-brand-primary/10 border border-brand-primary/20 text-brand-primary rounded-full px-3 py-1 text-[11px] font-mono">
            <Cpu className="w-3.5 h-3.5 animate-pulse" />
            <span>CUDA GRID ACTIVE</span>
          </div>
        </div>
      </div>

      {/* Grid: 2 columns split */}
      <div id="pipeline-grid" className="grid grid-cols-1 xl:grid-cols-12 gap-8 mb-8">
        
        {/* LEFT COMPONENT: STRAW SPLIT SETTING & DENSE CHART CATEGORIES */}
        <div className="xl:col-span-5 flex flex-col gap-6">
          
          {/* Split Configuration Card */}
          <div id="split-card" className="bg-surface-low border border-white/5 rounded-lg p-5">
            <div className="flex justify-between items-center mb-3">
              <span className="text-[10px] font-mono text-brand-secondary/40 font-bold uppercase tracking-wider">
                Split Configuration
              </span>
              <span className="text-[9px] font-mono text-brand-primary uppercase bg-brand-primary/10 border border-brand-primary/10 rounded px-1.5">
                PRO STRATIFIED
              </span>
            </div>

            <p className="text-xs text-brand-secondary/50 mb-4 font-sans">
              Separate your {totalSwingSamples} verified visual annotations into Training datasets and Validation blocks to prevent optimizer overfitting.
            </p>

            {/* Config Split Value Slider */}
            <div className="flex justify-between text-mono text-xs mb-3">
              <span className="text-white font-bold">Train: {trainSplit}%</span>
              <span className="text-brand-blue font-bold">Val: {100 - trainSplit}%</span>
            </div>

            <div className="mb-4">
              <input
                type="range"
                min="50"
                max="95"
                step="5"
                value={trainSplit}
                onChange={(e) => setTrainSplit(Number(e.target.value))}
                className="w-full h-2 bg-[#121212] rounded-lg appearance-none cursor-pointer accent-brand-primary border border-white/5"
              />
            </div>

            {/* Split counts outputs details */}
            <div className="grid grid-cols-2 gap-3 text-xs border-t border-white/5 pt-4">
              <div>
                <span className="text-brand-secondary/40 font-mono uppercase text-[9px]">Training Set Inputs</span>
                <span className="text-lg font-bold text-[#e5e2e1] block mt-0.5">{trainCount} Swings</span>
              </div>
              <div>
                <span className="text-brand-secondary/40 font-mono uppercase text-[9px]">Validation Set Clips</span>
                <span className="text-lg font-bold text-brand-blue block mt-0.5">{valCount} Swings</span>
              </div>
            </div>
          </div>

          {/* Slices category data counts */}
          <div id="dataset-category-card" className="bg-surface-low border border-white/5 rounded-lg p-5">
            <span className="text-[10px] font-mono text-brand-secondary/40 font-bold uppercase tracking-wider block mb-3">
              Dataset Category distribution
            </span>

            {/* Grid distribution visualization */}
            <div className="flex flex-col gap-4">
              {[
                { category: 'Setup Node', count: 280 },
                { category: 'Backswing Pivot', count: 310 },
                { category: 'Downswing Peak', count: 320 },
                { category: 'Impact Sync', count: 290 },
                { category: 'Finish Posture', count: 200 }
              ].map((item, idx) => {
                const percentageOfMax = (item.count / 320) * 100;
                // Calculate dynamic split lengths inside the bars
                const trainBarWidth = percentageOfMax * (trainSplit / 100);
                const valBarWidth = percentageOfMax * ((100 - trainSplit) / 100);

                return (
                  <div key={idx} className="flex flex-col gap-1 text-[11px]">
                    <div className="flex justify-between font-mono">
                      <span className="text-white font-semibold">{item.category}</span>
                      <span className="text-brand-secondary/40">{item.count} samples</span>
                    </div>

                    {/* Coupled coupled colored split bars */}
                    <div className="w-full bg-[#111] h-3 rounded-[2px] overflow-hidden flex border border-white/5">
                      {/* Train block */}
                      <div
                        className="bg-brand-primary transition-all duration-300 h-full hover:brightness-110"
                        style={{ width: `${trainBarWidth}%` }}
                        title={`Train set: ${Math.round(item.count * (trainSplit / 100))} samples`}
                      />
                      {/* Val block */}
                      <div
                        className="bg-brand-blue transition-all duration-300 h-full hover:brightness-110"
                        style={{ width: `${valBarWidth}%` }}
                        title={`Validation set: ${Math.round(item.count * ((100 - trainSplit) / 100))} samples`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Color key guide footnotes */}
            <div className="flex gap-4 mt-5 text-[10px] font-mono text-brand-secondary/40 uppercase justify-center border-t border-white/5 pt-3">
              <span className="flex items-center gap-2">
                <span className="w-3 h-1.5 bg-brand-primary rounded" /> Training Class
              </span>
              <span className="flex items-center gap-2">
                <span className="w-3 h-1.5 bg-brand-blue rounded" /> Validation Class
              </span>
            </div>
          </div>

        </div>

        {/* RIGHT COMPONENT: DETAILED SYSTEM STEPS PROGRESS & TERMINAL LOGS */}
        <div className="xl:col-span-7 flex flex-col gap-6">

          {/* Linear pipeline step lists */}
          <div id="pipeline-steps-container" className="bg-surface-low border border-white/5 rounded-lg p-5">
            <span className="text-[10px] font-mono text-brand-secondary/40 font-bold uppercase tracking-wider block mb-4">
              Linear Calibration Stages
            </span>

            <div className="flex flex-col gap-4">
              {pipelineSteps.map((step) => (
                <div
                  key={step.id}
                  className={`flex gap-4 p-3 rounded border transition duration-150 ${
                    step.status === 'active'
                      ? 'bg-brand-primary/[0.03] border-brand-primary/30 shadow-[0_2px_12px_rgba(117,255,158,0.04)] animate-pulse'
                      : step.status === 'completed'
                      ? 'bg-surface-lowest/70 border-white/5'
                      : 'bg-surface-lowest/30 border-white/5 opacity-50'
                  }`}
                >
                  {/* Phase number */}
                  <div className={`w-8 h-8 rounded-full border flex items-center justify-center font-mono font-bold text-xs ${
                    step.status === 'completed'
                      ? 'bg-brand-primary/10 border-brand-primary text-brand-primary'
                      : step.status === 'active'
                      ? 'bg-brand-tertiary/10 border-brand-tertiary text-brand-tertiary'
                      : 'bg-none border-white/10 text-brand-secondary/40'
                  }`}>
                    {step.status === 'completed' ? '✓' : step.number}
                  </div>

                  {/* Step parameters */}
                  <div className="flex-1 text-left">
                    <div className="flex justify-between items-baseline">
                      <h4 className="text-xs font-bold text-white font-sans">{step.name}</h4>
                      <span className={`text-[9px] font-mono uppercase font-bold px-1.5 rounded ${
                        step.status === 'completed'
                          ? 'bg-brand-primary/10 text-brand-primary'
                          : step.status === 'active'
                          ? 'bg-brand-tertiary/20 text-brand-tertiary animate-pulse'
                          : 'bg-white/5 text-brand-secondary/30'
                      }`}>
                        {step.status}
                      </span>
                    </div>
                    <p className="text-[11px] text-brand-secondary/60 mt-1 leading-relaxed">
                      {step.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Action Trigger training model & Terminal simulation */}
          <div id="trigger-training-card" className="bg-surface-low border border-white/5 rounded-lg p-5">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h4 className="text-sm font-bold text-white font-sans">Pipeline Fine-Tuning</h4>
                <p className="text-xs text-brand-secondary/40 mt-0.5">
                  Optimize ConvNet deep tracking parameters on golf-domain targets.
                </p>
              </div>

              {/* Initiate Fine-Tuning grand trigger button */}
              <button
                onClick={initiateTraining}
                disabled={isTraining}
                className={`bg-brand-primary hover:bg-brand-primary-dim text-black font-semibold px-4 py-2.5 rounded text-xs transition flex items-center gap-2 shadow-[0_4px_12px_rgba(117,255,158,0.25)] cursor-pointer ${
                  isTraining ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isTraining ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-black" />
                    <span>Tuning Weights...</span>
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 text-black" />
                    <span>Initiate Model Training</span>
                  </>
                )}
              </button>
            </div>

            {/* Active Training progress tracking */}
            {(isTraining || progressVal > 0) && (
              <div className="mb-4">
                <div className="flex justify-between items-center text-[10px] font-mono text-brand-secondary/50 mb-1">
                  <span>OPTIMIZATION PROCESS</span>
                  <span className="text-brand-primary font-bold">{progressVal}%</span>
                </div>
                {/* Thin outline process progress bar */}
                <div className="w-full bg-[#111] h-2 rounded overflow-hidden border border-white/5">
                  <div className="bg-brand-primary-dim h-full transition-all duration-300" style={{ width: `${progressVal}%` }} />
                </div>
              </div>
            )}

            {/* Real Dynamic Log print terminal */}
            <div className="bg-[#0c0c0c] border border-white/10 rounded-md p-4 font-mono text-[11px] text-[#bacbb9] relative min-h-[160px] max-h-[220px] overflow-y-auto">
              <div className="absolute top-3 right-3 flex items-center gap-1.5 text-[9px] font-mono text-brand-secondary/40">
                <Terminal className="w-3.5 h-3.5" />
                <span>SHELL SESSION</span>
              </div>

              {terminalLogs.length > 0 ? (
                <div className="flex flex-col gap-1 text-left">
                  {terminalLogs.map((log, lIdx) => (
                    <div key={lIdx} className="whitespace-pre-wrap leading-relaxed">
                      {log.startsWith('✓') ? (
                        <span className="text-brand-primary">{log}</span>
                      ) : log.includes('Loss:') ? (
                        <span className="text-brand-blue">{log}</span>
                      ) : (
                        <span>{log}</span>
                      )}
                    </div>
                  ))}
                  <div ref={terminalBottomRef} />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center text-center p-8 text-brand-secondary/35">
                  <Activity className="w-8 h-8 text-white/10 mb-2" />
                  <span>Click "Initiate Model Training" to watch the CNN learning-loop compiled.</span>
                </div>
              )}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}