export type ActiveTab = 'analyze' | 'history' | 'insights';

export interface SwingSession {
  id: string;
  location: string;
  date: string;
  time: string;
  clubType: string;
  swingScore?: number;
  smashFactor?: number;
  launchAngle?: number;
  rotationSpeed?: number;
  mpjpeMetric?: number;
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

export interface DemoDltJoint {
  x: number;
  y: number;
  z: number;
  observed: boolean;
}

export interface DemoDltFrame {
  frame_index: number;
  timestamp: number;
  joints: Record<string, DemoDltJoint | null>;
}

export interface DemoDltResponse {
  status: 'ok';
  source: 'calibrated_demo' | string;
  calibration: {
    available: boolean;
    type: string;
    metric_scale: boolean;
    notes: string;
  };
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
