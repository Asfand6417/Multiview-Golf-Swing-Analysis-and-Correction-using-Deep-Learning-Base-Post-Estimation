"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Activity,
  CheckCircle2,
  Info,
  ShieldAlert
} from 'lucide-react';

type Coordinate3D = {
  x: number;
  y: number;
  z: number;
};

type TriangulatedFrame = {
  frame: number;
  joints: Coordinate3D[];
};

interface SkeletalAnalysisProps {
  frames3d?: TriangulatedFrame[];
}

export default function SkeletalAnalysis({ frames3d = [] }: SkeletalAnalysisProps) {
  const hasTriangulatedFrames = frames3d.length > 0;

  return (
    <div id="biometrics-root" className="flex-1 overflow-y-auto p-8 bg-surface-base grid grid-cols-1 xl:grid-cols-12 gap-8 select-none">
      <div className="xl:col-span-8 flex flex-col gap-6">
        <div id="skeletal-canvas-view" className="relative bg-[#0b0b0b] border border-white/5 rounded-lg overflow-hidden flex-1 flex flex-col min-h-[500px]">
          <div className="flex-1 flex items-center justify-center p-8">
            {hasTriangulatedFrames ? (
              <div className="w-full max-w-xl rounded border border-brand-primary/20 bg-brand-primary/5 p-6 text-center">
                <CheckCircle2 className="mx-auto mb-4 h-10 w-10 text-brand-primary" />
                <h2 className="text-xl font-bold text-white">Triangulated Coordinate Stream Ready</h2>
                <p className="mt-2 text-sm leading-relaxed text-brand-secondary">
                  A real 3D renderer can mount here when Phase 2 provides frame-by-frame DLT coordinates.
                </p>
              </div>
            ) : (
              <div className="w-full max-w-2xl rounded border border-dashed border-[#facc15]/40 bg-[#facc15]/5 p-8 text-center">
                <ShieldAlert className="mx-auto mb-5 h-12 w-12 text-[#facc15]" />
                <h2 className="text-2xl font-bold tracking-tight text-white">No Triangulated 3D Data</h2>
                <p className="mt-3 text-sm leading-relaxed text-brand-secondary">
                  This screen no longer renders a skeleton unless it receives real frame-by-frame coordinates from calibrated DLT reconstruction.
                </p>
                <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3 text-left">
                  <div className="rounded bg-surface-lowest/80 border border-white/5 p-3">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-brand-secondary/40">User Uploads</div>
                    <div className="mt-1 text-sm font-semibold text-[#facc15]">Calibration required</div>
                  </div>
                  <div className="rounded bg-surface-lowest/80 border border-white/5 p-3">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-brand-secondary/40">Public Demo</div>
                    <div className="mt-1 text-sm font-semibold text-[#facc15]">Phase 2 pending</div>
                  </div>
                  <div className="rounded bg-surface-lowest/80 border border-white/5 p-3">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-brand-secondary/40">Renderer</div>
                    <div className="mt-1 text-sm font-semibold text-brand-secondary">Waiting for data</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div id="canvas-video-scrub-bar" className="bg-surface-low border-t border-white/5 p-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-[10px] font-mono uppercase tracking-wider text-brand-secondary/40">Frame Sequence</div>
                <div className="mt-1 text-sm font-semibold text-white">
                  {hasTriangulatedFrames ? `${frames3d.length} reconstructed frames loaded` : 'Unavailable until DLT runs'}
                </div>
              </div>
              <div className="rounded-full border border-[#facc15]/30 bg-[#facc15]/10 px-3 py-1 text-[10px] font-mono uppercase tracking-wider text-[#facc15]">
                Not live analysis
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="xl:col-span-4 flex flex-col gap-5">
        <div className="bg-surface-low border border-white/5 rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585]">System Precision</span>
            <span className="text-[9px] font-mono rounded-xs px-1.5 uppercase font-bold bg-[#facc15]/15 text-[#facc15]">Unavailable</span>
          </div>
          <div className="flex justify-between items-baseline">
            <span className="text-xs text-brand-secondary/60 font-sans">MPJPE Metric</span>
            <span id="metric-mpjpe" className="text-sm font-mono font-bold text-brand-secondary">Requires ground-truth DLT output</span>
          </div>
          <div className="mt-3 flex items-start gap-2 rounded bg-surface-lowest/70 border border-white/5 p-3">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-[#facc15]" />
            <p className="text-xs leading-relaxed text-brand-secondary">
              MPJPE is hidden until the app can compare real reconstructed 3D joints against a valid reference or documented dataset output.
            </p>
          </div>
        </div>

        <div className="bg-surface-low border border-white/5 rounded-lg p-4">
          <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585] block mb-3">Model Pipeline Nodes</span>
          <div className="grid grid-cols-3 gap-1 divide-x divide-white/5 text-center">
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold">01 / INPUT</div>
              <div className="text-xs font-semibold text-white mt-0.5">Pending</div>
            </div>
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold uppercase">02 / DLT</div>
              <div className="text-xs font-semibold text-brand-secondary mt-0.5">Not run</div>
            </div>
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold">03 / VIEW</div>
              <div className="text-xs font-semibold text-brand-secondary mt-0.5">Empty</div>
            </div>
          </div>
        </div>

        <div className="bg-surface-low border border-white/5 rounded-lg p-4">
          <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585] block mb-3">Classifier Ensemble</span>
          <div className="flex flex-col gap-2 text-xs">
            {['SVM (Support Vector Machine)', 'LR (Logistic Regression)', 'LSTM (Neural Network Recurrent)'].map((label) => (
              <div key={label} className="flex justify-between items-center py-1 border-b border-white/5 last:border-b-0">
                <span className="font-mono text-brand-secondary/60">{label}</span>
                <span className="font-mono font-semibold text-brand-secondary">Not computed</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-surface-low border border-white/5 rounded-lg p-4">
          <div className="flex items-center gap-2 text-[#facc15] mb-3">
            <Activity className="w-4 h-4" />
            <span className="text-[10px] font-mono tracking-widest font-extrabold uppercase">Capability Status</span>
          </div>
          <div className="space-y-3 text-xs leading-relaxed text-brand-secondary">
            <p>
              Phase 1 is active: fake skeletons, fake MPJPE, fake frame counters, and fake classifier values are suppressed.
            </p>
            <p>
              Phase 2 will enable the public pre-calibrated demo path. Phase 3 user uploads remain disabled until real camera calibration exists.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
