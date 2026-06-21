"use client";

interface ResultsProps {
    flaws?: string[];
    injury?: string;
    mpjpe?: number | null;
}

export default function AnalysisResult({ flaws = [], injury, mpjpe = null }: ResultsProps) {
    return (
        <div className="bg-kinetic-surface border border-kinetic-border rounded p-4 space-y-4 shadow-lg">
            
            <div className="font-geist text-[12px] uppercase tracking-wider text-kinetic-tertiary mb-2 border-b border-kinetic-border pb-2 flex justify-between">
                <span>Telemetry Readout</span>
                <span className="text-kinetic-primary">NOT_CONNECTED</span>
            </div>

            {/* MPJPE Tracking Score */}
            <div className="bg-[#0a0a0a] border border-kinetic-border rounded p-4 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-kinetic-tertiary opacity-5 blur-[50px] rounded-full"></div>
                <div className="font-geist text-[12px] uppercase text-kinetic-tertiary tracking-wide mb-1">MPJPE Availability</div>
                <div className="text-xl font-bold text-white mb-2">
                    {mpjpe === null ? 'Requires calibrated DLT output' : `${mpjpe.toFixed(2)}mm`}
                </div>
                
                <div className="w-full bg-kinetic-surfaceHigh rounded-sm h-1">
                    <div className="bg-kinetic-primary h-1 rounded-sm shadow-[0_0_10px_rgba(0,228,117,0.8)]" style={{ width: '0%' }}></div>
                </div>
                <div className="font-geist text-[10px] text-kinetic-textMuted mt-2 text-right">No variance computed</div>
            </div>

            {/* Swing Flaws */}
            <div className="bg-[#0a0a0a] border border-kinetic-border rounded p-4">
                <div className="font-geist text-[12px] uppercase text-kinetic-tertiary tracking-wide mb-3">Detected Deviations</div>
                <div className="space-y-2">
                    {(flaws.length > 0 ? flaws : ['No computed deviations']).map((flaw, idx) => (
                        <div key={idx} className="flex items-center gap-3 backdrop-blur-[20px] bg-kinetic-glass border border-white/10 rounded sm px-3 py-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-kinetic-warning shrink-0 shadow-[0_0_5px_rgba(234,179,8,1)]"></div>
                            <span className="text-sm font-medium text-kinetic-textMain">{flaw}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Physiology / Injury */}
            <div className="bg-kinetic-accent/10 border border-kinetic-accent/30 rounded p-4">
                <div className="font-geist text-[12px] uppercase text-kinetic-accent tracking-wide mb-1">Biomechanics Alert</div>
                <p className="text-sm font-medium text-kinetic-textMain">
                    {injury || 'No injury-risk output is available until real analysis runs.'}
                </p>
            </div>
            
        </div>
    );
}
