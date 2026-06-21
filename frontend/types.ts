export type ActiveTab = 'analyze' | 'history' | 'insights';

export interface SwingSession {
  id: string;
  location: string;
  date: string;
  time: string;
  clubType: string;
  status?: string;
  thumbnailUrl?: string;
  skeletalOverlayUrl?: string;
  tags?: string[];
}

export interface PipelineStep {
  id: string;
  number: string;
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed';
}

export interface BiomechanicalIssue {
  id: string;
  type: 'critical' | 'stable' | 'optimal';
  title: string;
  rule: string;
  description: string;
  impact?: string;
  injuryRisk?: string;
  target?: string;
  iconName: 'head' | 'arms' | 'knee' | 'hinge';
}

export interface SynchronizerState {
  frontVideoName: string | null;
  sideVideoName: string | null;
  frontVideoUrl: string | null;
  sideVideoUrl: string | null;
  offsetSeconds: number;
  playbackSpeed: number;
  isPlaying: boolean;
  alignScore: number;
  frameIndex: number;
  totalFrames: number;
  isLocked: boolean;
}

export interface DLTJoint {
  x: number;
  y: number;
  z: number;
  observed: boolean;
  smoothed?: boolean;
}

export interface DLTFrame {
  frame_index: number;
  timestamp: number;
  joints: Record<string, DLTJoint | null>;
}

export type DemoDltJoint = DLTJoint;
export type DemoDltFrame = DLTFrame;

export interface CalibrationSummary {
  available: boolean;
  type: string;
  metric_scale: boolean;
  notes: string;
}

export interface DemoDltResponse {
  status: 'ok';
  source: 'calibrated_demo' | string;
  calibration: CalibrationSummary;
  joint_names: string[];
  frame_count: number;
  frames: DemoDltFrame[];
  metrics: {
    mpjpe_mm: number | null;
    mpjpe_available: boolean;
    notes: string;
  };
  notes?: string;
}

export interface CalibrationResponse {
  status: 'ok';
  calibration_id: string;
  valid_pair_count: number;
  reprojection_error: number;
  K_front: number[][];
  D_front: number[][];
  K_side: number[][];
  D_side: number[][];
  R: number[][];
  T: number[][];
  P_front: number[][];
  P_side: number[][];
  calibration: CalibrationSummary;
}

export interface BiomechanicsResponse {
  available: boolean;
  features: FeatureFrame[] | Record<string, number | null> | Array<{
    frame_index: number;
    timestamp: number;
    values: Record<string, number>;
  }>;
  errors?: Array<{
    frame_index: number;
    phase: string;
    error: string;
    severity: string;
    evidence: Record<string, number>;
    suggestion: string;
  }>;
  faults?: Array<{
    id: string;
    label: string;
    measurement: number;
    threshold: number;
    unit: string;
  }>;
  corrections?: string[];
  notes: string;
}

export interface PhaseSegmentationResponse {
  available: boolean;
  method: 'svm' | 'svm_rbf' | null;
  phases: Array<{
    frame_index: number;
    timestamp: number;
    phase: string;
  }>;
  notes: string;
}

export interface RenderedVideoResponse {
  available: boolean;
  path: string | null;
  notes: string;
}

export interface SmoothingResponse {
  available: boolean;
  method: 'savitzky_golay' | null;
  window_length: number | null;
  polyorder: number | null;
  smoothed_joint_values: number;
  notes: string;
}

export interface FeatureFrame {
  frame_index: number;
  timestamp: number;
  features: Record<string, number | null>;
}

export interface CalibratedDLTAnalysisResponse {
  status: 'ok';
  source: 'uploaded_calibrated_dlt';
  job_id: string;
  calibration_id: string;
  calibration?: CalibrationSummary;
  frame_count: number;
  joint_names: string[];
  frames: DLTFrame[];
  raw_frames: DLTFrame[];
  smoothed_frames: DLTFrame[];
  smoothing: SmoothingResponse;
  features: FeatureFrame[];
  biomechanics: BiomechanicsResponse;
  phase_features?: Array<{
    frame_index: number;
    timestamp: number;
    values: Record<string, number>;
  }>;
  phase_segmentation: PhaseSegmentationResponse;
  metrics: {
    mpjpe_mm: number | null;
    mpjpe_available: boolean;
    notes: string;
  };
  rendered_video?: RenderedVideoResponse;
}

export type DLTReconstructionResponse = DemoDltResponse | CalibratedDLTAnalysisResponse;
