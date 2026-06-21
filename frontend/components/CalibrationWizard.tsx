"use client";

import { useState } from 'react';
import { CalibrationResponse } from '../types';

interface CalibrationWizardProps {
  onCalibrationReady: (response: CalibrationResponse) => void;
}

export default function CalibrationWizard({ onCalibrationReady }: CalibrationWizardProps) {
  const [frontImages, setFrontImages] = useState<File[]>([]);
  const [sideImages, setSideImages] = useState<File[]>([]);
  const [boardRows, setBoardRows] = useState(6);
  const [boardCols, setBoardCols] = useState(9);
  const [squareSizeMm, setSquareSizeMm] = useState(25);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<CalibrationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submitCalibration = async () => {
    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      if (frontImages.length !== sideImages.length) {
        throw new Error('Front and side checkerboard image counts must match.');
      }

      const formData = new FormData();
      frontImages.forEach((file) => formData.append('front_images', file));
      sideImages.forEach((file) => formData.append('side_images', file));
      formData.append('board_rows', String(boardRows));
      formData.append('board_cols', String(boardCols));
      formData.append('square_size_mm', String(squareSizeMm));

      const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';
      const response = await fetch(`${apiBaseUrl}/calibration/stereo`, {
        method: 'POST',
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Backend returned ${response.status}`);
      }

      const calibration = payload as CalibrationResponse;
      setResult(calibration);
      onCalibrationReady(calibration);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calibration failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="bg-surface-low border border-white/5 rounded-lg p-5 mb-6">
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-5">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-brand-primary">Stereo Calibration</div>
          <h3 className="mt-1 text-lg font-bold text-white">Camera Calibration Workflow</h3>
          <p className="mt-2 max-w-2xl text-xs leading-relaxed text-brand-secondary">
            Upload paired front and side checkerboard images. Uploaded video DLT stays blocked until this returns a real calibration ID.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <label className="flex flex-col gap-1">
            <span className="font-mono text-brand-secondary/60 uppercase">Rows</span>
            <input
              type="number"
              min="1"
              value={boardRows}
              onChange={(event) => setBoardRows(Number(event.target.value))}
              className="bg-surface-lowest border border-white/10 rounded px-3 py-2 text-white"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-mono text-brand-secondary/60 uppercase">Columns</span>
            <input
              type="number"
              min="1"
              value={boardCols}
              onChange={(event) => setBoardCols(Number(event.target.value))}
              className="bg-surface-lowest border border-white/10 rounded px-3 py-2 text-white"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-mono text-brand-secondary/60 uppercase">Square mm</span>
            <input
              type="number"
              min="0.1"
              step="0.1"
              value={squareSizeMm}
              onChange={(event) => setSquareSizeMm(Number(event.target.value))}
              className="bg-surface-lowest border border-white/10 rounded px-3 py-2 text-white"
            />
          </label>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <label className="rounded border border-dashed border-white/10 bg-surface-lowest/50 p-4">
          <span className="block text-[10px] font-mono uppercase tracking-widest text-brand-secondary/60">Front Checkerboard Images</span>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(event) => setFrontImages(Array.from(event.target.files ?? []))}
            className="mt-3 block w-full text-xs text-brand-secondary file:mr-3 file:rounded file:border-0 file:bg-brand-primary file:px-3 file:py-2 file:text-xs file:font-semibold file:text-black"
          />
          <div className="mt-2 text-[11px] text-brand-secondary/60">{frontImages.length} files selected</div>
        </label>

        <label className="rounded border border-dashed border-white/10 bg-surface-lowest/50 p-4">
          <span className="block text-[10px] font-mono uppercase tracking-widest text-brand-secondary/60">Side Checkerboard Images</span>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(event) => setSideImages(Array.from(event.target.files ?? []))}
            className="mt-3 block w-full text-xs text-brand-secondary file:mr-3 file:rounded file:border-0 file:bg-brand-primary file:px-3 file:py-2 file:text-xs file:font-semibold file:text-black"
          />
          <div className="mt-2 text-[11px] text-brand-secondary/60">{sideImages.length} files selected</div>
        </label>
      </div>

      <div className="mt-5 flex flex-col md:flex-row md:items-center gap-4">
        <button
          onClick={submitCalibration}
          disabled={isSubmitting}
          className="bg-brand-primary text-black rounded px-4 py-2 text-xs font-bold uppercase tracking-wider disabled:opacity-60"
        >
          {isSubmitting ? 'Calibrating...' : 'Run Stereo Calibration'}
        </button>

        {result && (
          <div className="text-xs text-brand-secondary">
            <div>
              <span className="font-semibold text-brand-primary">Calibration ready:</span> {result.calibration_id}
              <span className="ml-3">Valid pairs: {result.valid_pair_count}</span>
              <span className="ml-3">Reprojection error: {result.reprojection_error.toFixed(4)}</span>
            </div>
            <div className="mt-1 font-mono text-[10px] uppercase text-brand-secondary/60">
              P_front {result.P_front.length}x{result.P_front[0]?.length ?? 0} / P_side {result.P_side.length}x{result.P_side[0]?.length ?? 0}
            </div>
          </div>
        )}

        {error && (
          <div className="rounded border border-brand-error/30 bg-brand-error/10 px-3 py-2 text-xs text-[#ffb4ab]">
            {error}
          </div>
        )}
      </div>
    </section>
  );
}
