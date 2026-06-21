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
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [progressVal, setProgressVal] = useState<number>(0);
  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const terminalBottomRef = useRef<HTMLDivElement>(null);

  // Pipeline Status list mapped visually
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([
    {
      id: 'step-1',
      number: '01',
      name: 'Multi-Angle Acquisition',
      description: 'User-uploaded front and side videos can be previewed, but calibrated capture is not available.',
      status: 'pending'
    },
    {
      id: 'step-2',
      number: '02',
      name: 'Dynamic Temporal Alignment',
      description: 'Timeline controls are preview-only until a real backend synchronization job exists.',
      status: 'pending'
    },
    {
      id: 'step-3',
      number: '03',
      name: 'Classifier Weight Fine-Tuning',
      description: 'Classifier training is disabled until reconstructed 3D data or documented training data is connected.',
      status: 'pending'
    },
    {
      id: 'step-4',
      number: '04',
      name: '3D Point Spatial Reconstruction',
      description: 'Waiting for Phase 2 to provide calibrated public-dataset projection matrices and 2D points.',
      status: 'pending'
    },
    {
      id: 'step-5',
      number: '05',
      name: 'Diagnostic Rule Evaluation',
      description: 'No diagnostic rule output is shown until it can be derived from real reconstructed coordinates.',
      status: 'pending'
    }
  ]);

  useEffect(() => {
    if (terminalBottomRef.current) {
      terminalBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalLogs]);

  const initiateTraining = () => {
    setIsTraining(false);
    setProgressVal(0);
    setTerminalLogs([
      "Pipeline status check only.",
      "Training is not connected to a real dataset or model job.",
      "No classifier weights, losses, accuracies, MPJPE, or 3D coordinates were computed.",
      "No train/validation split is assigned because no dataset is connected."
    ]);
    setPipelineSteps((prev) => prev.map((step) => ({ ...step, status: 'pending' })));
  };

  return (
    <div id="pipeline-root" className="flex-1 overflow-y-auto p-8 bg-surface-base">
      
      {/* Visual Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <span className="text-xs uppercase font-mono tracking-widest text-[#859585]">PIPELINE STATUS</span>
          <h2 className="text-2xl font-bold text-white tracking-tight mt-1">Model Pipeline Control</h2>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 bg-[#facc15]/10 border border-[#facc15]/20 text-[#facc15] rounded-full px-3 py-1 text-[11px] font-mono">
            <Cpu className="w-3.5 h-3.5" />
            <span>DLT DEMO PENDING</span>
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
                CONFIG ONLY
              </span>
            </div>

            <p className="text-xs text-brand-secondary/50 mb-4 font-sans">
              Configure the future split for public-demo or training datasets. No verified training run is executed from this screen yet.
            </p>

            {/* Config Split Value Slider */}
            <div className="flex justify-between text-mono text-xs mb-3">
              <span className="text-white font-bold">Train: Not assigned</span>
              <span className="text-brand-blue font-bold">Val: Not assigned</span>
            </div>

            <div className="mb-4">
              <div className="w-full h-2 bg-[#121212] rounded-lg border border-white/5 overflow-hidden">
                <div className="h-full w-0 bg-brand-primary" />
              </div>
            </div>

            {/* Split counts outputs details */}
            <div className="grid grid-cols-2 gap-3 text-xs border-t border-white/5 pt-4">
              <div>
                <span className="text-brand-secondary/40 font-mono uppercase text-[9px]">Training Set Inputs</span>
                <span className="text-lg font-bold text-[#e5e2e1] block mt-0.5">No dataset</span>
              </div>
              <div>
                <span className="text-brand-secondary/40 font-mono uppercase text-[9px]">Validation Set Clips</span>
                <span className="text-lg font-bold text-brand-blue block mt-0.5">No clips</span>
              </div>
            </div>
          </div>

          {/* Slices category data counts */}
          <div id="dataset-category-card" className="bg-surface-low border border-white/5 rounded-lg p-5">
            <span className="text-[10px] font-mono text-brand-secondary/40 font-bold uppercase tracking-wider block mb-3">
              Dataset Category Distribution
            </span>

            {/* Grid distribution visualization */}
            <div className="flex flex-col gap-4">
              {[
                'Setup Node',
                'Backswing Pivot',
                'Downswing Peak',
                'Impact Sync',
                'Finish Posture'
              ].map((category, idx) => {
                return (
                  <div key={idx} className="flex flex-col gap-1 text-[11px]">
                    <div className="flex justify-between font-mono">
                      <span className="text-white font-semibold">{category}</span>
                      <span className="text-brand-secondary/40">No samples</span>
                    </div>

                    {/* Coupled coupled colored split bars */}
                    <div className="w-full bg-[#111] h-3 rounded-[2px] overflow-hidden flex border border-white/5">
                      {/* Train block */}
                      <div
                        className="bg-brand-primary transition-all duration-300 h-full hover:brightness-110"
                        style={{ width: '0%' }}
                        title="Train set: no samples"
                      />
                      {/* Val block */}
                      <div
                        className="bg-brand-blue transition-all duration-300 h-full hover:brightness-110"
                        style={{ width: '0%' }}
                        title="Validation set: no samples"
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
                  Disabled until a real dataset, model job, and measured outputs are connected.
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
                    <span>Checking...</span>
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 text-black" />
                    <span>Check Pipeline Status</span>
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
                  {terminalLogs.map((log, lIdx) => {
                    const text = typeof log === 'string' ? log : (log === undefined ? '' : JSON.stringify(log));
                    return (
                      <div key={lIdx} className="whitespace-pre-wrap leading-relaxed">
                        {text.startsWith('✓') ? (
                          <span className="text-brand-primary">{text}</span>
                        ) : (
                          <span>{text}</span>
                        )}
                      </div>
                    );
                  })}
                  <div ref={terminalBottomRef} />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center text-center p-8 text-brand-secondary/35">
                  <Activity className="w-8 h-8 text-white/10 mb-2" />
                  <span>Click "Check Pipeline Status" to see what is and is not connected.</span>
                </div>
              )}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
