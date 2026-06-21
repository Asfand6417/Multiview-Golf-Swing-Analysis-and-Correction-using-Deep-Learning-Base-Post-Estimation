"use client";

import { PhaseSegmentationResponse } from '../types';

interface PhaseTimelineProps {
  phaseSegmentation?: PhaseSegmentationResponse | null;
}

export default function PhaseTimeline({ phaseSegmentation = null }: PhaseTimelineProps) {
  const phases = phaseSegmentation?.phases ?? [];

  return (
    <section className="bg-surface-low border border-white/5 rounded-lg p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#859585]">Phase Timeline</div>
          <h3 className="mt-1 text-sm font-bold text-white">SVM Swing Phase Segmentation</h3>
        </div>
        <span className={`rounded border px-2 py-1 text-[10px] font-mono uppercase ${
          phaseSegmentation?.available
            ? 'border-brand-primary/20 bg-brand-primary/10 text-brand-primary'
            : 'border-[#facc15]/20 bg-[#facc15]/10 text-[#facc15]'
        }`}>
          {phaseSegmentation?.available ? phaseSegmentation.method : 'Unavailable'}
        </span>
      </div>

      {phaseSegmentation?.available && phases.length > 0 ? (
        <div className="mt-4 flex gap-2 overflow-x-auto pb-1">
          {phases.map((phase) => (
            <div
              key={`${phase.frame_index}-${phase.phase}`}
              className="min-w-28 rounded border border-white/5 bg-surface-lowest/70 px-3 py-2"
            >
              <div className="text-[10px] font-mono text-brand-secondary/50">Frame {phase.frame_index}</div>
              <div className="mt-1 text-xs font-semibold text-white">{phase.phase}</div>
              <div className="mt-1 text-[10px] font-mono text-brand-secondary/50">{phase.timestamp.toFixed(3)}s</div>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-4 text-xs leading-relaxed text-brand-secondary">
          {phaseSegmentation?.notes ?? 'SVM unavailable, trained model required.'}
        </p>
      )}
    </section>
  );
}
