"use client";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Search,
  Sliders,
  Bell,
  Settings,
  User,
  Activity,
  ChevronRight,
  Sparkles,
  Info
} from 'lucide-react';
import { ActiveTab } from '../types';

interface HeaderProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  activeSection: string;
  setActiveSection: (sec: string) => void;
}

export default function Header({
  activeTab,
  setActiveTab,
  activeSection,
  setActiveSection
}: HeaderProps) {
  // Translate current workspace path based on tabs
  const getSubPath = () => {
    if (activeTab === 'analyze') {
      return activeSection === 'biometrics' ? 'WORKSPACE / 3D SKELETAL' : 'WORKSPACE / SYNC ANALYSIS';
    } else if (activeTab === 'history') {
      return 'DASHBOARD / HISTORIC METRICS';
    } else {
      return 'PIPELINE / NEURAL NETWORK TUNING';
    }
  };

  return (
    <header
      id="main-app-header"
      className="bg-surface-lowest border-b border-white/5 px-6 py-4 flex items-center justify-between select-none h-[72px]"
    >
      {/* Left section: Breadcrumb & Custom Navigation tabs */}
      <div className="flex items-center gap-8">
        <div className="flex flex-col">
          <span className="text-[10px] font-mono tracking-widest text-brand-primary uppercase">
            {getSubPath()}
          </span>
          <div className="flex items-center gap-5 mt-1">
            {/* Primary Action Tabs */}
            <div className="flex bg-[#181818] p-1 rounded-md border border-white/5 text-sm font-sans gap-1">
              {(['analyze', 'history', 'insights'] as ActiveTab[]).map((tab) => (
                <button
                  key={tab}
                  id={`tab-btn-${tab}`}
                  onClick={() => {
                    setActiveTab(tab);
                    if (tab === 'analyze') {
                      setActiveSection('sync'); // Default to synchronization workspace inside analyze
                    } else if (tab === 'history') {
                      setActiveSection('reports'); // Default to reports/charts
                    } else if (tab === 'insights') {
                      setActiveSection('reports');
                    }
                  }}
                  className={`px-4 py-1.5 rounded transition-all duration-150 font-medium capitalize cursor-pointer text-xs ${
                    activeTab === tab
                      ? 'bg-brand-primary text-black font-semibold shadow-[0_2px_8px_rgba(117,255,158,0.2)]'
                      : 'text-brand-secondary hover:text-white hover:bg-white/5'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Pipeline diagnostic flow (Middle visual candy from Image 1) */}
        <div
          id="pipeline-flow-badge"
          className="hidden xl:flex items-center gap-2 bg-surface-low border border-white/5 rounded-full px-3 py-1.5 text-[10px] font-mono text-brand-secondary/70 h-10"
        >
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-secondary/40" />
            <span>POSE DETECTION</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 text-brand-secondary/30" />
          <div className="flex items-center gap-1.5 bg-brand-primary/10 text-brand-primary px-2 py-0.5 rounded-full border border-brand-primary/20">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-primary animate-pulse" />
            <span>LANDMARK DETECTION</span>
            <span className="text-[9px] bg-brand-primary/20 text-brand-primary px-1 rounded">Full</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 text-brand-secondary/30" />
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#dae4ff]/30" />
            <span className="text-brand-tertiary">FINE-TUNED MODEL</span>
          </div>
        </div>
      </div>

      {/* Right Section: Core Utility, Search, Alerts, Setting and Profile */}
      <div id="header-utilities" className="flex items-center gap-4">
        {/* Quick Search */}
        <div className="relative hidden md:block w-48">
          <span className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <Search className="h-3.5 w-3.5 text-brand-secondary/40" />
          </span>
          <input
            type="text"
            placeholder="Search swings..."
            className="w-full pl-9 pr-3 py-1.5 bg-surface-low border border-white/5 rounded text-xs text-white placeholder-brand-secondary/30 bg-opacity-70 focus:outline-none focus:border-brand-primary/40 focus:bg-surface-high font-sans transition-all"
          />
        </div>

        {/* Alert/Notification icon */}
        <button
          id="btn-alerts"
          className="relative p-2 text-brand-secondary hover:text-white hover:bg-white/5 rounded-md transition-colors cursor-pointer"
          title="Notifications"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-brand-primary rounded-full ring-2 ring-surface-lowest" />
        </button>

        {/* Settings button */}
        <button
          id="btn-settings"
          onClick={() => setActiveSection('reports')}
          className="p-2 text-brand-secondary hover:text-white hover:bg-white/5 rounded-md transition-colors cursor-pointer"
          title="System Settings"
        >
          <Settings className="w-4 h-4" />
        </button>

        {/* Divider */}
        <div className="w-px h-6 bg-white/10" />

        {/* User Identity Info Profile (As seen in the mockups) */}
        <div id="user-profile-badge" className="flex items-center gap-2.5">
          <div className="flex flex-col text-right hidden lg:flex">
            <span className="text-xs font-semibold text-white font-sans">Alex Mercer</span>
            <span className="text-[9px] text-brand-primary font-mono tracking-wider uppercase">Pro-Elite Coach</span>
          </div>
          <div className="relative group cursor-pointer">
            <div className="w-9 h-9 rounded-full overflow-hidden border border-brand-primary/40 group-hover:border-brand-primary transition-all duration-300">
              <img
                src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&q=80&w=150"
                alt="Profile"
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover"
              />
            </div>
            {/* Active connection pulse dot */}
            <span className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-brand-primary border-2 border-surface-lowest rounded-full" />
          </div>
        </div>
      </div>
    </header>
  );
}
