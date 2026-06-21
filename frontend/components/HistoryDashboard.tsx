"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import {
  TrendingUp,
  Award,
  Zap,
  Activity,
  Calendar,
  ChevronRight,
  MoreVertical,
  Sliders,
  Sparkles,
  MapPin,
  Flame,
  Filter,
  CheckCircle,
  HelpCircle
} from 'lucide-react';
import { SwingSession } from '../types';

interface HistoryDashboardProps {
  onLoadSession: (session: SwingSession) => void;
}

export default function HistoryDashboard({ onLoadSession }: HistoryDashboardProps) {
  const [timeRange, setTimeRange] = useState<'7D' | '30D' | '90D'>('30D');
  const [hoveredPointId, setHoveredPointId] = useState<number | null>(null);

  // Stats data corresponding to screenshot cards
  const summaryStats = [
    {
      id: 'stat-total-analysis',
      title: 'VALIDATED ANALYSIS',
      value: '0',
      badge: 'PENDING',
      badgeColor: 'text-brand-primary bg-brand-primary/10 border-brand-primary/10',
      description: 'Awaiting calibrated DLT output'
    },
    {
      id: 'stat-top-score',
      title: 'TOP WORKSPACE SCORE',
      value: 'N/A',
      badge: 'NO DATA',
      badgeColor: 'text-brand-primary bg-brand-primary/10 border-brand-primary/20',
      description: 'Requires real reconstructed output'
    },
    {
      id: 'stat-avg-smash',
      title: 'AVG SMASH FACTOR',
      value: 'N/A',
      badge: 'NOT MEASURED',
      badgeColor: 'text-brand-blue bg-brand-blue/10 border-brand-blue/20',
      description: 'No launch monitor integration'
    },
    {
      id: 'stat-streak',
      title: 'SESSION STREAK',
      value: '0',
      badge: 'INACTIVE',
      badgeColor: 'text-brand-primary bg-brand-primary/25 border-brand-primary/30',
      description: 'No validated sessions yet'
    }
  ];

  // Placeholder trend data. Do not present it as measured MPJPE until Phase 2 exists.
  const chartData = {
    '7D': [
      { day: 'Mon', error: 10.4, score: 91.2 },
      { day: 'Tue', error: 9.8, score: 92.4 },
      { day: 'Wed', error: 9.5, score: 93.0 },
      { day: 'Thu', error: 8.9, score: 93.8 },
      { day: 'Fri', error: 8.4, score: 94.2 },
      { day: 'Sat', error: 8.2, score: 94.6 },
      { day: 'Sun', error: 8.2, score: 94.8 }
    ],
    '30D': [
      { day: 'Oct 01', error: 14.8, score: 86.4 },
      { day: 'Oct 07', error: 13.5, score: 88.0 },
      { day: 'Oct 14', error: 11.2, score: 90.5 },
      { day: 'Oct 21', error: 9.8, score: 92.8 },
      { day: 'Oct 28', error: 8.5, score: 94.1 },
      { day: 'Today', error: 8.2, score: 94.8 }
    ],
    '90D': [
      { day: 'Aug 01', error: 18.5, score: 81.2 },
      { day: 'Aug 15', error: 16.4, score: 83.5 },
      { day: 'Sep 01', error: 14.2, score: 87.0 },
      { day: 'Sep 15', error: 12.8, score: 89.2 },
      { day: 'Oct 01', error: 11.0, score: 91.0 },
      { day: 'Oct 15', error: 9.4, score: 93.2 },
      { day: 'Today', error: 8.2, score: 94.8 }
    ]
  };

  const points = chartData[timeRange];

  // SVG dimensions for trend graph
  const width = 800;
  const height = 240;
  const paddingX = 60;
  const paddingY = 45;

  // Render SVG Paths
  const maxVal = Math.max(...points.map(p => p.error)) * 1.15;
  const minVal = Math.min(...points.map(p => p.error)) * 0.85;

  const getCoordinates = (p: typeof points[0], index: number) => {
    const x = paddingX + (index / (points.length - 1)) * (width - 2 * paddingX);
    const y = height - paddingY - ((p.error - minVal) / (maxVal - minVal)) * (height - 2 * paddingY);
    return { x, y };
  };

  const pathD = points.map((p, i) => {
    const coords = getCoordinates(p, i);
    return `${i === 0 ? 'M' : 'L'} ${coords.x} ${coords.y}`;
  }).join(' ');

  // SVG Area beneath line for glowing gradient fill
  const areaD = `${pathD} L ${getCoordinates(points[points.length - 1], points.length - 1).x} ${height - paddingY} L ${getCoordinates(points[0], 0).x} ${height - paddingY} Z`;

  // Mock sessions corresponding with high fidelity images and golf analytics parameter readings
  const recentSessions: SwingSession[] = [
    {
      id: 'session-1',
      location: 'Augusta National',
      date: 'Today, 08:30 AM',
      time: '08:30 AM',
      clubType: 'Driver',
      thumbnailUrl: 'https://images.unsplash.com/photo-1592919505780-303950717480?auto=format&fit=crop&q=80&w=480',
      tags: ['Demo record', 'Not validated', '3D Pending']
    },
    {
      id: 'session-2',
      location: 'Precision Laboratory',
      date: 'Oct 26, 04:15 PM',
      time: '04:15 PM',
      clubType: '7-Iron',
      thumbnailUrl: 'https://images.unsplash.com/photo-1535131749006-b7f58c99034b?auto=format&fit=crop&q=80&w=480',
      tags: ['Demo record', 'Not validated', '3D Pending']
    },
    {
      id: 'session-3',
      location: 'Pebble Beach',
      date: 'Oct 24, 10:00 AM',
      time: '14:22 PM',
      clubType: 'Driver',
      thumbnailUrl: 'https://images.unsplash.com/photo-1587174486073-ae5e5cff23aa?auto=format&fit=crop&q=80&w=480',
      tags: ['Demo record', 'Not validated', '3D Pending']
    }
  ];

  return (
    <div id="history-root" className="flex-1 overflow-y-auto p-8 bg-surface-base">
      
      {/* Uppermost Header and Toggles */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <span className="text-xs uppercase font-mono tracking-widest text-[#859585]">HISTORIC DATA INSIGHTS</span>
          <h2 className="text-2xl font-bold text-white tracking-tight mt-1">History & Progress</h2>
        </div>

        {/* Action button tools */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => alert("Excel/CSV export initiated.")}
            className="bg-surface-low border border-white/5 text-brand-secondary hover:text-white px-3.5 py-1.5 rounded text-xs px-4 py-2 font-sans transition flex items-center gap-2 cursor-pointer"
          >
            <span>Export Report</span>
          </button>
        </div>
      </div>

      {/* Summary numerical statistics cards grid */}
      <div id="stat-cards-grid" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        {summaryStats.map((stat) => (
          <div
            key={stat.id}
            className="bg-surface-low border border-white/5 rounded-lg p-5 flex flex-col justify-between hover:border-white/10 transition"
          >
            <div className="flex justify-between items-start mb-2">
              <span className="text-[10px] font-mono text-brand-secondary/40 font-bold uppercase tracking-wider">
                {stat.title}
              </span>
              <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded-sm border font-bold ${stat.badgeColor}`}>
                {stat.badge}
              </span>
            </div>

            <h3 className="text-3xl font-extrabold text-white tracking-tight mt-2 flex items-baseline gap-1 font-sans">
              {stat.value}
              {stat.id === 'stat-streak' && <Flame className="w-5 h-5 text-brand-primary ml-1" />}
            </h3>

            <p className="text-[11px] text-brand-secondary/50 font-sans mt-2">
              {stat.description}
            </p>
          </div>
        ))}
      </div>

      {/* Intermediate Line graph layout panel: PRECISION TRENDS */}
      <div id="precision-trend-container" className="bg-surface-low border border-white/5 rounded-lg p-6 mb-8">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">Validation Trend</h3>
            <p className="text-xs text-brand-secondary/40 mt-0.5">
              No validated DLT or MPJPE history is available until the real reconstruction pipeline is connected.
            </p>
          </div>

          {/* Range selection buttons for graphs */}
          <div className="flex bg-[#121212] p-0.5 rounded border border-white/5 text-[10px] font-mono">
            {(['7D', '30D', '90D'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded transition font-bold font-mono tracking-wider cursor-pointer ${
                  timeRange === range
                    ? 'bg-brand-primary text-black'
                    : 'text-brand-secondary hover:text-white'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>

        {/* Primary Interactive SVG Area */}
        <div className="relative w-full h-[260px] overflow-x-auto">
          <svg className="w-full h-full min-w-[650px]" viewBox={`0 0 ${width} ${height}`}>
            
            {/* Gradients */}
            <defs>
              <linearGradient id="glowGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#75ff9e" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#75ff9e" stopOpacity="0.00" />
              </linearGradient>
            </defs>

            {/* Horizontal grid lines */}
            {Array.from({ length: 5 }).map((_, idx) => {
              const yVal = paddingY + (idx / 4) * (height - 2 * paddingY);
              return (
                <line
                  key={idx}
                  x1={paddingX}
                  y1={yVal}
                  x2={width - paddingX}
                  y2={yVal}
                  stroke="rgba(255,255,255,0.03)"
                  strokeWidth="1"
                />
              );
            })}

            {/* Glowing fill path */}
            <path d={areaD} fill="url(#glowGradient)" />

            {/* Primary line path (Performance Green) */}
            <path
              d={pathD}
              fill="none"
              stroke="#75ff9e"
              strokeWidth="3.5"
              strokeLinecap="round"
              className="drop-shadow-[0_4px_10px_rgba(117,255,158,0.35)]"
            />

            {/* Intersection Nodes circles */}
            {points.map((p, i) => {
              const coords = getCoordinates(p, i);
              const isHovered = hoveredPointId === i;
              return (
                <g key={i}>
                  <circle
                    cx={coords.x}
                    cy={coords.y}
                    r={isHovered ? 8 : 5}
                    fill={isHovered ? "#75ff9e" : "#131313"}
                    stroke="#75ff9e"
                    strokeWidth={isHovered ? 3 : 2}
                    className="cursor-pointer transition-all duration-150"
                  />
                  
                  {/* Invisible larger hover zone */}
                  <circle
                    cx={coords.x}
                    cy={coords.y}
                    r={20}
                    fill="transparent"
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredPointId(i)}
                    onMouseLeave={() => setHoveredPointId(null)}
                  />

                  {/* Date/Label below nodes */}
                  <text
                    x={coords.x}
                    y={height - 15}
                    fill="rgba(229,226,225,0.4)"
                    fontSize="10"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    {p.day}
                  </text>
                </g>
              );
            })}

            {/* Placeholder axis labels until validated history exists */}
            <text x={30} y={paddingY} fill="rgba(229,226,225,0.3)" fontSize="9" fontFamily="monospace">No data</text>
            <text x={30} y={height / 2} fill="rgba(229,226,225,0.3)" fontSize="9" fontFamily="monospace">Pending</text>
            <text x={30} y={height - paddingY} fill="rgba(229,226,225,0.3)" fontSize="9" fontFamily="monospace">Phase 2</text>
          </svg>

          {/* Floating Tooltip of selected joint error values */}
          {hoveredPointId !== null && (
            <div
              className="absolute bg-[#181818] border border-brand-primary/20 p-2.5 rounded text-xs select-none shadow-md"
              style={{
                left: `${paddingX + (hoveredPointId / (points.length - 1)) * (width - 2 * paddingX) - 60}px`,
                top: `${getCoordinates(points[hoveredPointId], hoveredPointId).y - 75}px`
              }}
            >
              <div className="font-mono text-[9px] text-[#859585] uppercase tracking-wider">
                {points[hoveredPointId].day} Analysis
              </div>
              <div className="font-extrabold text-[#75ff9e] text-sm mt-0.5">
                Placeholder: no validated MPJPE
              </div>
              <div className="text-[10px] text-white mt-0.5">
                Phase 2 public-dataset demo pending
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Sessions list cards */}
      <div id="recent-sessions-header" className="flex justify-between items-center mb-5">
        <h3 className="text-lg font-bold text-white tracking-tight">Recent Sessions</h3>
        <button
          onClick={() => alert("Loading all sessions backlog.")}
          className="text-xs text-brand-primary hover:underline font-mono flex items-center gap-1 cursor-pointer"
        >
          <span>View All History</span>
          <ChevronRight className="w-3 h-3" />
        </button>
      </div>

      {/* Grid containing golf course card assets */}
      <div id="recent-sessions-list" className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {recentSessions.map((session) => (
          <div
            key={session.id}
            onClick={() => {
              onLoadSession(session);
              alert(`Successfully loaded ${session.location} session components into the dual synchronizer workspace!`);
            }}
            className="group bg-surface-low border border-white/5 hover:border-brand-primary/30 rounded-lg overflow-hidden flex flex-col justify-between transition duration-200 cursor-pointer text-left active:scale-[0.98]"
          >
            {/* Visual Header course thumbnail representation */}
            <div className="relative aspect-video bg-[#0c0c0c] overflow-hidden">
              <img
                src={session.thumbnailUrl}
                alt={session.location}
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover group-hover:scale-105 transition duration-300 filter grayscale group-hover:grayscale-0 brightness-75"
              />

              {/* Dynamic Overlay tags */}
              <div className="absolute top-3 left-3 flex gap-1.5 flex-wrap">
                {session.tags?.slice(0, 2).map((tag, tIdx) => (
                  <span
                    key={tIdx}
                    className={`text-[8px] font-mono uppercase font-bold px-2 py-0.5 rounded shadow-sm ${
                      tag === '3D Pending'
                        ? 'bg-[#facc15] text-black'
                        : 'bg-black/80 text-white border border-white/10'
                    }`}
                  >
                    {tag}
                  </span>
                ))}
              </div>

              {/* Score label */}
              <div className="absolute top-3 right-3 bg-black/80 border border-brand-primary/20 rounded py-0.5 px-2 text-[10px] font-mono tracking-wider font-extrabold text-brand-primary">
                Demo
              </div>
            </div>

            {/* Lower panel session context metrics */}
            <div className="p-4 flex-1 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start">
                  <h4 className="text-sm font-semibold text-white group-hover:text-brand-primary transition font-sans">
                    {session.location}
                  </h4>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      alert(`Session option configuration: ${session.location}`);
                    }}
                    className="p-1 hover:bg-white/5 rounded text-brand-secondary/60 hover:text-white transition"
                  >
                    <MoreVertical className="w-3.5 h-3.5" />
                  </button>
                </div>
                <span className="text-[10px] font-mono text-brand-secondary/40 block mt-1">
                  Demo record: {session.date} • {session.clubType} club
                </span>
              </div>

              {/* Smash factor parameters line */}
              <div className="grid grid-cols-2 gap-2 border-t border-white/5 mt-4 pt-3 text-[11px] font-mono">
                <div>
                  <span className="text-brand-secondary/40 uppercase text-[9px] block">Smash Factor</span>
                  <span className="text-brand-primary font-bold text-sm mt-0.5 block">Not measured</span>
                </div>
                <div>
                  <span className="text-brand-secondary/40 uppercase text-[9px] block">Launch Angle</span>
                  <span className="text-white font-semibold text-sm mt-0.5 block">Not measured</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
