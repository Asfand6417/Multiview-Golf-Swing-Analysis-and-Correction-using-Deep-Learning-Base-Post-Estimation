"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Activity,
  RefreshCcw,
  Video,
  Layers,
  FileBarChart,
  Plus,
  Compass,
  Cloud,
  LogOut,
  BookOpen
} from 'lucide-react';

interface SidebarProps {
  activeSection: string;
  setActiveSection: (sec: string) => void;
  onNewSession: () => void;
}

export default function Sidebar({
  activeSection,
  setActiveSection,
  onNewSession
}: SidebarProps) {
  const menuItems = [
    {
      id: 'sync',
      label: 'Synchronization',
      icon: RefreshCcw,
      badge: 'Active'
    },
    {
      id: 'biometrics',
      label: 'Biometric 3D Data',
      icon: Layers
    },
    {
      id: 'video',
      label: 'Video Assets',
      icon: Video,
      badge: '12 Files'
    },
    {
      id: 'reports',
      label: 'Model / Reports',
      icon: FileBarChart
    }
  ];

  return (
    <aside
      id="sidebar-container"
      className="w-72 bg-surface-lowest border-r border-white/5 flex flex-col justify-between p-5 select-none h-full"
    >
      {/* Top Section */}
      <div className="flex flex-col gap-6">
        {/* Brand */}
        <div id="brand-header" className="flex items-center gap-3 py-1">
          <div className="w-9 h-9 bg-brand-primary/10 rounded-md border border-brand-primary/30 flex items-center justify-center text-brand-primary shadow-[0_0_15px_rgba(117,255,158,0.15)]">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold tracking-tight text-white flex items-center gap-1.5">
              GOLFSWING <span className="text-brand-primary text-xs bg-brand-primary/10 px-1.5 py-0.5 rounded font-mono decoration-solid">AI</span>
            </h1>
            <span className="text-[10px] font-mono tracking-widest text-brand-secondary/40">KINETIC ELITE V3.4</span>
          </div>
        </div>

        {/* New Session Action */}
        <button
          id="btn-new-session"
          onClick={onNewSession}
          className="w-full bg-brand-primary hover:bg-brand-primary-dim text-black font-semibold py-3 px-4 rounded-md transition-all duration-200 flex items-center justify-center gap-2 text-sm shadow-[0_4px_12px_rgba(117,255,158,0.25)] hover:shadow-[0_4px_18px_rgba(117,255,158,0.35)] cursor-pointer active:scale-[0.98]"
        >
          <Plus className="w-4 h-4 text-black stroke-[3px]" />
          <span>New Session</span>
        </button>

        {/* Navigation Section */}
        <nav id="nav-section" className="flex flex-col gap-1 mt-3">
          <span className="text-[10px] font-mono tracking-widest text-[#859585] uppercase mb-2 px-2">Workspace & Control</span>
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
            return (
              <button
                key={item.id}
                id={`sidebar-link-${item.id}`}
                onClick={() => setActiveSection(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-md transition-all duration-150 group text-left cursor-pointer ${
                  isActive
                    ? 'bg-brand-primary/10 text-brand-primary border-l-2 border-brand-primary font-medium'
                    : 'text-brand-secondary hover:bg-white/5 hover:text-white'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon
                    className={`w-4 h-4 transition-transform duration-200 group-hover:scale-110 ${
                      isActive ? 'text-brand-primary' : 'text-brand-secondary/70 group-hover:text-white'
                    }`}
                  />
                  <span className="text-sm font-sans">{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                      isActive ? 'bg-brand-primary/20 text-brand-primary' : 'bg-white/10 text-brand-secondary/80'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Section Widget */}
      <div className="flex flex-col gap-4 mt-auto">
        {/* Data Acquisition Box */}
        <div
          id="data-acquisition-panel"
          className="bg-surface-low border border-white/5 rounded-md p-3"
        >
          <div className="flex items-center gap-2 mb-2">
            <div className="w-1.5 h-1.5 rounded-full bg-brand-primary animate-pulse" />
            <span className="text-[10px] font-mono tracking-wider font-semibold uppercase text-brand-primary-container">
              Data Acquisition
            </span>
          </div>
          <div className="flex flex-col gap-1 text-[11px]">
            <div className="flex justify-between">
              <span className="text-brand-secondary/60">Multi-View Input</span>
              <span className="text-brand-primary font-medium">Active</span>
            </div>
            <div className="flex justify-between">
              <span className="text-brand-secondary/60">Annotated Videos</span>
              <span className="text-white font-semibold">1,400 Swings</span>
            </div>
            <div className="flex justify-between border-t border-white/5 mt-1.5 pt-1.5">
              <span className="text-brand-secondary/60">Engine Status</span>
              <span className="text-brand-blue font-mono text-[10px]">Landmark active</span>
            </div>
          </div>
        </div>

        {/* Cloud Storage Gauge */}
        <div id="cloud-storage-widget" className="bg-surface-low border border-white/5 rounded-md p-3">
          <div className="flex items-center justify-between text-[10px] font-mono text-brand-secondary/60 mb-1.5">
            <span className="flex items-center gap-1.5 uppercase font-medium">
              <Cloud className="w-3.5 h-3.5 text-brand-blue" />
              Cloud Storage
            </span>
            <span className="text-white">65%</span>
          </div>
          {/* Progress bar container */}
          <div className="w-full bg-[#1e1e1e] rounded-[1px] h-2 overflow-hidden border border-white/5">
            <div className="bg-brand-primary-dim h-full" style={{ width: '65%' }} />
          </div>
          <div className="flex justify-between text-[9px] font-mono text-brand-secondary/40 mt-1">
            <span>6.5 GB active</span>
            <span>10 GB Max Cap</span>
          </div>
        </div>

        {/* App Footer copyright line */}
        <div className="text-[10px] text-brand-secondary/30 font-mono text-center flex items-center justify-center gap-1">
          <span>© 2026 Kinetic Elite</span>
        </div>
      </div>
    </aside>
  );
}
