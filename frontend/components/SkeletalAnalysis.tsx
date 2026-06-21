"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Info,
  ShieldAlert
} from 'lucide-react';
import { DLTFrame, DLTJoint, DLTReconstructionResponse } from '../types';

interface SkeletalAnalysisProps {
  reconstruction?: DLTReconstructionResponse | null;
  demoReconstruction?: DLTReconstructionResponse | null;
}

const CONNECTIONS: Array<[string, string]> = [
  ['nose', 'left_shoulder'],
  ['nose', 'right_shoulder'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle']
];

type ProjectedJoint = DLTJoint & {
  screenX: number;
  screenY: number;
};

function formatCoordinate(value: number) {
  return value.toFixed(3);
}

function getObservedJoints(frame: DLTFrame | undefined) {
  if (!frame) return [];
  return Object.entries(frame.joints).filter((entry): entry is [string, DLTJoint] => entry[1] !== null);
}

function projectFrame(frame: DLTFrame | undefined) {
  const observedEntries = getObservedJoints(frame);
  if (observedEntries.length === 0) return {};

  const xs = observedEntries.map(([, joint]) => joint.x);
  const ys = observedEntries.map(([, joint]) => joint.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xRange = Math.max(maxX - minX, 0.001);
  const yRange = Math.max(maxY - minY, 0.001);
  const width = 760;
  const height = 420;
  const padding = 56;

  return observedEntries.reduce<Record<string, ProjectedJoint>>((projected, [name, joint]) => {
    projected[name] = {
      ...joint,
      screenX: padding + ((joint.x - minX) / xRange) * (width - padding * 2),
      screenY: height - padding - ((joint.y - minY) / yRange) * (height - padding * 2)
    };
    return projected;
  }, {});
}

export default function SkeletalAnalysis({ reconstruction = null, demoReconstruction = null }: SkeletalAnalysisProps) {
  const activeReconstruction = reconstruction ?? demoReconstruction;
  const isUploadedAnalysis = activeReconstruction?.source === 'uploaded_calibrated_dlt'
    && activeReconstruction !== null
    && 'smoothed_frames' in activeReconstruction;
  const frames = isUploadedAnalysis
    ? activeReconstruction.smoothed_frames
    : activeReconstruction?.frames ?? [];
  const hasTriangulatedFrames = frames.length > 0;
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);

  useEffect(() => {
    setCurrentFrameIndex(0);
  }, [activeReconstruction]);

  const activeFrame = frames[Math.min(currentFrameIndex, Math.max(frames.length - 1, 0))];
  const projectedJoints = useMemo(() => projectFrame(activeFrame), [activeFrame]);
  const observedJoints = getObservedJoints(activeFrame);
  const missingJoints = activeReconstruction?.joint_names.filter((name) => !activeFrame?.joints[name]) ?? [];
  const firstObservedJoint = observedJoints[0];
  const source = activeReconstruction?.source ?? null;
  const sourceLabel = source === 'uploaded_calibrated_dlt'
    ? 'Uploaded calibrated DLT reconstruction'
    : source === 'calibrated_demo'
      ? 'Calibrated DLT demo reconstruction'
      : 'Calibrated DLT reconstruction';
  const sourceBadge = source === 'uploaded_calibrated_dlt' ? 'Uploaded calibrated' : 'Demo only';
  const smoothingStatus = isUploadedAnalysis && 'smoothing' in activeReconstruction
    ? activeReconstruction.smoothing
    : null;

  return (
    <div id="biometrics-root" className="flex-1 overflow-y-auto p-8 bg-surface-base grid grid-cols-1 xl:grid-cols-12 gap-8 select-none">
      <div className="xl:col-span-8 flex flex-col gap-6">
        <div id="skeletal-canvas-view" className="relative bg-[#0b0b0b] border border-white/5 rounded-lg overflow-hidden flex-1 flex flex-col min-h-[500px]">
          <div className="flex-1 flex items-center justify-center p-8">
            {hasTriangulatedFrames ? (
              <div className="w-full">
                <div className="mb-4 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
                  <div>
                    <div id="dlt-demo-label" className="text-[10px] font-mono uppercase tracking-widest text-brand-primary">
                      {sourceLabel}
                    </div>
                    <h2 className="mt-1 text-2xl font-bold text-white">Backend-Triangulated Skeleton</h2>
                  </div>
                  <div className="rounded border border-brand-primary/20 bg-brand-primary/10 px-3 py-2 text-xs font-mono text-brand-primary">
                    Source: {activeReconstruction?.source ?? 'calibrated_demo'}
                  </div>
                </div>

                <div className="rounded-lg border border-white/5 bg-[#050505] p-4">
                  <svg
                    id="dlt-demo-skeleton-svg"
                    className="h-[420px] w-full"
                    viewBox="0 0 760 420"
                    role="img"
                    aria-label="Skeleton rendered from backend DLT coordinates"
                  >
                    <rect x="0" y="0" width="760" height="420" fill="#050505" />
                    {CONNECTIONS.map(([start, end]) => {
                      const startJoint = projectedJoints[start];
                      const endJoint = projectedJoints[end];
                      if (!startJoint || !endJoint) return null;
                      return (
                        <line
                          key={`${start}-${end}`}
                          x1={startJoint.screenX}
                          y1={startJoint.screenY}
                          x2={endJoint.screenX}
                          y2={endJoint.screenY}
                          stroke="#75ff9e"
                          strokeWidth="5"
                          strokeLinecap="round"
                          opacity="0.85"
                        />
                      );
                    })}
                    {Object.entries(projectedJoints).map(([name, joint]) => (
                      <g key={name}>
                        <circle
                          cx={joint.screenX}
                          cy={joint.screenY}
                          r="7"
                          fill="#050505"
                          stroke="#75ff9e"
                          strokeWidth="3"
                        />
                        <title>{`${name}: x=${formatCoordinate(joint.x)}, y=${formatCoordinate(joint.y)}, z=${formatCoordinate(joint.z)}`}</title>
                      </g>
                    ))}
                  </svg>
                </div>
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
                    <div className="mt-1 text-sm font-semibold text-[#facc15]">Load calibrated demo</div>
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
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wider text-brand-secondary/40">Frame Sequence</div>
                  <div id="dlt-demo-frame-count" className="mt-1 text-sm font-semibold text-white">
                  {hasTriangulatedFrames ? `${activeReconstruction?.frame_count ?? frames.length} reconstructed frames loaded` : 'Unavailable until DLT runs'}
                  </div>
                </div>
                <div className="rounded-full border border-[#facc15]/30 bg-[#facc15]/10 px-3 py-1 text-[10px] font-mono uppercase tracking-wider text-[#facc15]">
                  {sourceBadge}
                </div>
              </div>

              {hasTriangulatedFrames && (
                <div>
                  <input
                    id="dlt-demo-frame-scrubber"
                    type="range"
                    min="0"
                    max={Math.max(frames.length - 1, 0)}
                    value={currentFrameIndex}
                    onChange={(event) => setCurrentFrameIndex(Number(event.target.value))}
                    className="w-full accent-brand-primary"
                  />
                  <div className="mt-2 flex justify-between text-[10px] font-mono text-brand-secondary/50">
                    <span>Frame {activeFrame?.frame_index ?? currentFrameIndex}</span>
                    <span>Timestamp {activeFrame?.timestamp ?? 'pending'}s</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="xl:col-span-4 flex flex-col gap-5">
        <div className="bg-surface-low border border-white/5 rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585]">System Precision</span>
            <span className="text-[9px] font-mono rounded-xs px-1.5 uppercase font-bold bg-[#facc15]/15 text-[#facc15]">
              {activeReconstruction?.metrics.mpjpe_available ? 'Available' : 'Unavailable'}
            </span>
          </div>
          <div className="flex justify-between items-baseline">
            <span className="text-xs text-brand-secondary/60 font-sans">MPJPE Metric</span>
            <span id="metric-mpjpe" className="text-sm font-mono font-bold text-brand-secondary">
              {activeReconstruction?.metrics.mpjpe_available && activeReconstruction.metrics.mpjpe_mm !== null
                ? `${activeReconstruction.metrics.mpjpe_mm} mm`
                : 'Requires ground-truth 3D labels'}
            </span>
          </div>
          <div className="mt-3 flex items-start gap-2 rounded bg-surface-lowest/70 border border-white/5 p-3">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-[#facc15]" />
            <p className="text-xs leading-relaxed text-brand-secondary">
              {activeReconstruction?.metrics.notes ?? 'MPJPE is hidden until the app can compare real reconstructed 3D joints against a valid reference or documented dataset output.'}
            </p>
          </div>
        </div>

        <div className="bg-surface-low border border-white/5 rounded-lg p-4">
          <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585] block mb-3">Model Pipeline Nodes</span>
          <div className="grid grid-cols-3 gap-1 divide-x divide-white/5 text-center">
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold">01 / INPUT</div>
              <div className="text-xs font-semibold text-white mt-0.5">{hasTriangulatedFrames ? sourceBadge : 'Pending'}</div>
            </div>
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold uppercase">02 / DLT</div>
              <div className="text-xs font-semibold text-brand-secondary mt-0.5">{hasTriangulatedFrames ? 'Computed' : 'Not run'}</div>
            </div>
            <div className="p-1.5">
              <div className="text-[10px] font-mono text-brand-secondary/40 font-semibold">03 / SMOOTH</div>
              <div className="text-xs font-semibold text-brand-secondary mt-0.5">
                {smoothingStatus ? (smoothingStatus.available ? 'Applied' : 'Skipped') : hasTriangulatedFrames ? 'Raw' : 'Empty'}
              </div>
            </div>
          </div>
          {smoothingStatus && (
            <p className="mt-3 text-[11px] leading-relaxed text-brand-secondary">
              {smoothingStatus.notes}
            </p>
          )}
        </div>

        {hasTriangulatedFrames && (
          <div id="dlt-demo-frame-details" className="bg-surface-low border border-white/5 rounded-lg p-4">
            <span className="text-[10px] font-mono tracking-widest uppercase text-[#859585] block mb-3">Current Frame</span>
            <div className="space-y-2 text-xs text-brand-secondary">
              <div className="flex justify-between border-b border-white/5 pb-2">
                <span>Observed joints</span>
                <span className="font-mono text-white">{observedJoints.length}</span>
              </div>
              <div className="flex justify-between border-b border-white/5 pb-2">
                <span>Unavailable joints</span>
                <span className="font-mono text-[#facc15]">{missingJoints.length}</span>
              </div>
              {firstObservedJoint && (
                <div className="rounded bg-surface-lowest/70 border border-white/5 p-3">
                  <div className="text-[10px] font-mono uppercase text-brand-secondary/40">Sample computed coordinate</div>
                  <div className="mt-1 font-mono text-white">
                    {firstObservedJoint[0]}: {formatCoordinate(firstObservedJoint[1].x)}, {formatCoordinate(firstObservedJoint[1].y)}, {formatCoordinate(firstObservedJoint[1].z)}
                  </div>
                </div>
              )}
              {missingJoints.length > 0 && (
                <p className="text-[11px] leading-relaxed text-[#facc15]">
                  Missing joints are skipped because one or both calibrated 2D observations were unavailable: {missingJoints.join(', ')}.
                </p>
              )}
            </div>
          </div>
        )}

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
              Calibrated DLT rendering is active only for backend-returned reconstructed frames.
            </p>
            <p>
              No classifier output, swing score, speed, or MPJPE is displayed for this demo because those values are not computed here.
            </p>
            {activeReconstruction?.calibration?.notes && (
              <p className="text-[#facc15]">{activeReconstruction.calibration.notes}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
