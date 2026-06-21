"use client";

import { BiomechanicsResponse } from '../types';

interface BiomechanicsPanelProps {
  biomechanics?: BiomechanicsResponse | null;
}

export default function BiomechanicsPanel({ biomechanics = null }: BiomechanicsPanelProps) {
  const errors = biomechanics?.errors ?? [];

  return (
    <section className="bg-surface-low border border-white/5 rounded-lg p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#859585]">Biomechanics</div>
          <h3 className="mt-1 text-sm font-bold text-white">Rule-Based Corrections</h3>
        </div>
        <span className={`rounded border px-2 py-1 text-[10px] font-mono uppercase ${
          biomechanics?.available
            ? 'border-brand-primary/20 bg-brand-primary/10 text-brand-primary'
            : 'border-[#facc15]/20 bg-[#facc15]/10 text-[#facc15]'
        }`}>
          {biomechanics?.available ? 'Computed' : 'Unavailable'}
        </span>
      </div>

      {biomechanics?.available && errors.length > 0 ? (
        <div className="mt-4 space-y-3">
          {errors.map((error, index) => (
            <div key={`${error.frame_index}-${error.error}-${index}`} className="rounded border border-white/5 bg-surface-lowest/70 p-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs font-semibold text-white">{error.error}</div>
                  <div className="mt-1 text-[10px] font-mono uppercase text-brand-secondary/50">
                    {error.phase} / frame {error.frame_index}
                  </div>
                </div>
                <span className="rounded bg-[#facc15]/10 px-2 py-1 text-[10px] font-mono uppercase text-[#facc15]">
                  {error.severity}
                </span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-brand-secondary">{error.suggestion}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-4 text-xs leading-relaxed text-brand-secondary">
          {biomechanics?.notes ?? 'Biomechanics unavailable until real 3D features and phase labels are available.'}
        </p>
      )}
    </section>
  );
}
