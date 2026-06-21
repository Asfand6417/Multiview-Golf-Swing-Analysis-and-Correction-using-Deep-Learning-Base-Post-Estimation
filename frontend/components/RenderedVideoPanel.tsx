"use client";

import { RenderedVideoResponse } from '../types';

interface RenderedVideoPanelProps {
  renderedVideo?: RenderedVideoResponse | null;
}

export default function RenderedVideoPanel({ renderedVideo = null }: RenderedVideoPanelProps) {
  return (
    <section className="bg-surface-low border border-white/5 rounded-lg p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#859585]">Rendered Output</div>
          <h3 className="mt-1 text-sm font-bold text-white">3D Skeleton MP4</h3>
        </div>
        <span className={`rounded border px-2 py-1 text-[10px] font-mono uppercase ${
          renderedVideo?.available
            ? 'border-brand-primary/20 bg-brand-primary/10 text-brand-primary'
            : 'border-[#facc15]/20 bg-[#facc15]/10 text-[#facc15]'
        }`}>
          {renderedVideo?.available ? 'Available' : 'Unavailable'}
        </span>
      </div>

      {renderedVideo?.available && renderedVideo.path ? (
        <div className="mt-4 rounded border border-white/5 bg-surface-lowest/70 p-3">
          <div className="text-[10px] font-mono uppercase text-brand-secondary/50">Backend path</div>
          <div className="mt-1 break-all text-xs font-mono text-white">{renderedVideo.path}</div>
          <p className="mt-2 text-xs leading-relaxed text-brand-secondary">{renderedVideo.notes}</p>
        </div>
      ) : (
        <p className="mt-4 text-xs leading-relaxed text-brand-secondary">
          {renderedVideo?.notes ?? 'Rendered 3D video unavailable until real triangulated frames are produced.'}
        </p>
      )}
    </section>
  );
}
