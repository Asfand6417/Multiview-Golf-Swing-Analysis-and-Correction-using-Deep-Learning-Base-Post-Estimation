import React from 'react';

interface TennisMetrics {
  kneeFlexion: number; kneeExtVelocity: number; hipRotation: number; shoulderRotation: number;
  xFactor: number; pelvicTilt: number; trunkTilt: number; trunkExtension: number;
  elbowFlexion: number; wristAngle: number; shoulderTilt: number; headStability: number;
  comShift: number; groundReactionProxy: number; contactHeight: number; racquetDropDepth: number;
  swingVelocity: number; angularVelocity: number; followThroughSymmetry: number; repeatabilityScore: number;
}

export default function TennisAnalysis({ metrics }: { metrics: TennisMetrics }) {
  const metricGroups = [
    {
      category: "Kinematic Chain & Power",
      items: [
        { label: "Swing Velocity", value: `${metrics.swingVelocity.toFixed(1)} mph`, optimal: ">115 mph" },
        { label: "X-Factor Stretch", value: `${metrics.xFactor.toFixed(1)}°`, optimal: ">35°" },
        { label: "Ground Force Proxy", value: `${metrics.groundReactionProxy.toFixed(2)} BW`, optimal: "1.5 BW" },
        { label: "Angular Velocity", value: `${metrics.angularVelocity.toFixed(0)} deg/s`, optimal: "High" },
        { label: "Shoulder Rotation", value: `${metrics.shoulderRotation.toFixed(1)}°`, optimal: "110°+" },
        { label: "Hip Rotation", value: `${metrics.hipRotation.toFixed(1)}°`, optimal: "45°+" }
      ]
    },
    {
      category: "Joint Geometry & Posture",
      items: [
        { label: "Knee Flexion (Load)", value: `${metrics.kneeFlexion.toFixed(1)}°`, optimal: "115°" },
        { label: "Knee Ext Velocity", value: `${metrics.kneeExtVelocity.toFixed(0)} deg/s`, optimal: "High" },
        { label: "Elbow Flexion", value: `${metrics.elbowFlexion.toFixed(1)}°`, optimal: "Arm Lag" },
        { label: "Wrist Snap Angle", value: `${metrics.wristAngle.toFixed(1)}°`, optimal: "105°" },
        { label: "Trunk Extension", value: `${metrics.trunkExtension.toFixed(1)}°`, optimal: "Arch >20" },
        { label: "Pelvic Tilt", value: `${metrics.pelvicTilt.toFixed(1)}°`, optimal: "Neutral" }
      ]
    },
    {
      category: "Contact & Consistency Metrics",
      items: [
        { label: "Contact Height", value: `${metrics.contactHeight.toFixed(1)} cm`, optimal: "Apex High" },
        { label: "Racquet Drop Depth", value: `${metrics.racquetDropDepth.toFixed(1)} cm`, optimal: "Deep" },
        { label: "Trunk Tilt", value: `${metrics.trunkTilt.toFixed(1)}°`, optimal: "Stable Axis" },
        { label: "Shoulder Tilt", value: `${metrics.shoulderTilt.toFixed(1)}°`, optimal: "Upon Toss" },
        { label: "Head Stability", value: `${metrics.headStability.toFixed(1)} cm`, optimal: "<3.0 cm" },
        { label: "Com Shift", value: `${metrics.comShift.toFixed(1)} cm`, optimal: "Forward" },
        { label: "Follow-through", value: `${metrics.followThroughSymmetry.toFixed(1)}%`, optimal: ">90%" },
        { label: "Repeatability Score", value: `${metrics.repeatabilityScore.toFixed(1)} / 100`, optimal: "AI Grade" }
      ]
    }
  ];

  return (
    <div className="space-y-8">
      {metricGroups.map((group, gIdx) => (
        <div key={gIdx} className="bg-[#1c1b1b] border border-white/5 rounded-xl p-6">
          <h3 className="text-[12px] font-mono text-[#859585] tracking-widest uppercase mb-4">{group.category}</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {group.items.map((item, iIdx) => (
              <div key={iIdx} className="bg-[#131313] border border-white/5 rounded-lg p-3">
                <span className="text-[9px] text-[#859585] uppercase font-mono block mb-1">{item.label}</span>
                <span className="text-[16px] font-bold text-[#75ff9e] font-sans tracking-tight">{item.value}</span>
                <span className="text-[8px] text-[rgba(255,255,255,0.3)] uppercase font-mono block mt-1">Target: {item.optimal}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
