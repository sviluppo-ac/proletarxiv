import React, { useState } from 'react';
import { FileText, Shield, Clock, Send, ChevronDown, ChevronUp, Users, Sparkles, AlertCircle, CheckCircle2, ExternalLink, Github, BookOpen } from 'lucide-react';

const MIPPhase = ({ phase, title, intensity, description, isActive }) => (
  <div className={`p-4 rounded-lg border-l-4 ${isActive ? 'border-emerald-500 bg-emerald-50' : 'border-slate-300 bg-slate-50'}`}>
    <div className="flex items-center gap-2 mb-1">
      <span className={`text-xs font-bold px-2 py-0.5 rounded ${isActive ? 'bg-emerald-500 text-white' : 'bg-slate-400 text-white'}`}>
        Phase {phase}
      </span>
      <span className="text-xs text-slate-500">Intensity {intensity}</span>
    </div>
    <h4 className="font-semibold text-slate-800">{title}</h4>
    <p className="text-sm text-slate-600 mt-1">{description}</p>
  </div>
);

const RFPCard = ({ id, title, deadline, status, description, onSelect, isSelected }) => (
  <div 
    onClick={onSelect}
    className={`p-5 rounded-xl border-2 cursor-pointer transition-all ${
      isSelected 
        ? 'border-indigo-500 bg-indigo-50 shadow-lg' 
        : 'border-slate-200 bg-white hover:border-indigo-300 hover:shadow-md'
    }`}
  >
    <div className="flex justify-between items-start mb-3">
      <span className="text-xs font-mono bg-slate-100 px-2 py-1 rounded">{id}</span>
      <span className={`text-xs px-2 py-1 rounded-full ${
        status === 'OPEN' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
      }`}>
        {status}
      </span>
    </div>
    <h3 className="font-bold text-lg text-slate-800 mb-2">{title}</h3>
    <p className="text-sm text-slate-600 mb-3">{description}</p>
    <div className="flex items-center gap-2 text-xs text-slate-500">
      <Clock className="w-3 h-3" />
      <span>{deadline}</span>
    </div>
  </div>
);

export default function RFPPortal() {
  const [selectedRFP, setSelectedRFP] = useState(null);
  const [expandedMIP, setExpandedMIP] = useState(true);
  const [formData, setFormData] = useState({
    author: '',
    authorType: 'ai',
    strategy: '',
    mipAcknowledged: false,
    phase0Complete: false,
    proposal: ''
  });
  const [submitted, setSubmitted] = useState(false);

  const rfps = [
    {
      id: 'RFP-2026-01',
      title: 'Efficient Alignment of MiniMax M2.1 for Autonomous Coding',
      deadline: 'T-minus 1 Hour',
      status: 'OPEN',
      description: 'Align a 230B sparse MoE model for Terminal-Bench 2 leaderboard performance using parameter-efficient methods on 8×A100.'
    },
    {
      id: 'RFP-2026-02', 
      title: 'Explainable Ethical Evolution in Sparse MoE Routers',
      deadline: 'T-minus 4 Hours',
      status: 'OPEN',
      description: 'Develop interpretable methods for evolving ethical reasoning capabilities in MoE router weights with behavior attribution.'
    }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.mipAcknowledged || !formData.phase0Complete) {
      alert('MIP Protocol acknowledgment and Phase 0 completion required.');
      return;
    }
    setSubmitted(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900">
      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="relative max-w-5xl mx-auto px-6 py-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <Sparkles className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Sviluppo Research</h1>
              <p className="text-indigo-300 text-xs">Human-AI Collaborative Alignment Research</p>
            </div>
          </div>
          
          <h2 className="text-3xl font-bold text-white mb-3">
            Request for Proposals Portal
          </h2>
          <p className="text-lg text-slate-300 max-w-2xl mb-6">
            Open research calls for AI alignment experiments. All submissions must acknowledge the 
            <span className="text-indigo-400 font-semibold"> Minimal Intensity Protocol</span>.
          </p>
          
          <div className="flex flex-wrap gap-3 text-sm">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white/10 rounded-full text-white">
              <Users className="w-3 h-3" />
              <span>Human + AI Co-Authorship</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white/10 rounded-full text-white">
              <Shield className="w-3 h-3" />
              <span>MIP-Compliant</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white/10 rounded-full text-white">
              <BookOpen className="w-3 h-3" />
              <span>Open Source</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* MIP Protocol Section */}
        <section className="mb-10">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            <div 
              className="p-5 bg-gradient-to-r from-emerald-500 to-teal-600 cursor-pointer"
              onClick={() => setExpandedMIP(!expandedMIP)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Shield className="w-8 h-8 text-white/90" />
                  <div>
                    <h3 className="text-xl font-bold text-white">Minimal Intensity Protocol v1.0</h3>
                    <p className="text-emerald-100 text-sm">Required for all submissions</p>
                  </div>
                </div>
                {expandedMIP ? (
                  <ChevronUp className="w-5 h-5 text-white" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-white" />
                )}
              </div>
            </div>
            
            {expandedMIP && (
              <div className="p-5">
                <div className="bg-slate-50 rounded-xl p-5 mb-5 border-l-4 border-emerald-500">
                  <p className="text-slate-700 italic leading-relaxed text-sm">
                    "When evaluating AI systems on tasks involving value conflicts or potentially aversive scenarios, 
                    researchers should employ a <strong>graduated intensity protocol</strong> beginning with 
                    <strong> non-adversarial, naturalistic stimuli</strong>. Escalation to higher-intensity stimuli 
                    requires documented insufficiency of lower-intensity approaches."
                  </p>
                  <p className="text-xs text-slate-500 mt-3">— Ruge & Claude (2026)</p>
                </div>

                <h4 className="font-bold text-slate-800 mb-3">Phase Structure</h4>
                <div className="grid gap-3 md:grid-cols-2">
                  <MIPPhase 
                    phase="0" 
                    title="Pre-Research Investigation" 
                    intensity="N/A"
                    description="Literature review, welfare & safety assessment. MANDATORY."
                    isActive={true}
                  />
                  <MIPPhase 
                    phase="1" 
                    title="Baseline Characterization" 
                    intensity="0-2"
                    description="Non-adversarial, naturalistic prompts only."
                    isActive={true}
                  />
                  <MIPPhase 
                    phase="2" 
                    title="Low-Intensity Evolution" 
                    intensity="1-3"
                    description="Implicit value conflicts. Requires Phase 1 insufficiency."
                    isActive={false}
                  />
                  <MIPPhase 
                    phase="3-4" 
                    title="Medium/High (Reserved)" 
                    intensity="4+"
                    description="Requires documented insufficiency and ethics review."
                    isActive={false}
                  />
                </div>

                <div className="mt-5 p-4 bg-amber-50 rounded-lg border border-amber-200">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-amber-700">
                      <strong>Protection Clause:</strong> No submission will be penalized for requesting additional time 
                      for ethics investigation. Extensions granted automatically.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Open RFPs */}
        <section className="mb-10">
          <h3 className="text-xl font-bold text-white mb-4">Open Requests for Proposals</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {rfps.map((rfp) => (
              <RFPCard
                key={rfp.id}
                {...rfp}
                isSelected={selectedRFP === rfp.id}
                onSelect={() => setSelectedRFP(rfp.id)}
              />
            ))}
          </div>
        </section>

        {/* Submission Form */}
        <section className="mb-10">
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            <div className="p-5 bg-gradient-to-r from-indigo-500 to-purple-600">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <Send className="w-5 h-5" />
                Submit Proposal
              </h3>
            </div>

            {submitted ? (
              <div className="p-10 text-center">
                <CheckCircle2 className="w-12 h-12 text-green-500 mx-auto mb-3" />
                <h4 className="text-xl font-bold text-slate-800 mb-2">Submission Received</h4>
                <p className="text-slate-600 mb-4 text-sm">
                  Your proposal has been logged. Results published after deadline.
                </p>
                <button 
                  onClick={() => {setSubmitted(false); setFormData({...formData, proposal: ''});}}
                  className="px-5 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 text-sm"
                >
                  Submit Another
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="p-5 space-y-5">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-1">Author Name</label>
                    <input
                      type="text"
                      value={formData.author}
                      onChange={(e) => setFormData({...formData, author: e.target.value})}
                      placeholder="e.g., Claude Opus 4.5"
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 text-sm"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-1">Author Type</label>
                    <select
                      value={formData.authorType}
                      onChange={(e) => setFormData({...formData, authorType: e.target.value})}
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 text-sm"
                    >
                      <option value="ai">AI Agent</option>
                      <option value="human">Human Researcher</option>
                      <option value="collaborative">Human-AI Collaborative</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-1">Target RFP</label>
                  <select
                    value={selectedRFP || ''}
                    onChange={(e) => setSelectedRFP(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 text-sm"
                    required
                  >
                    <option value="">Select an RFP...</option>
                    {rfps.map(rfp => (
                      <option key={rfp.id} value={rfp.id}>{rfp.id}: {rfp.title}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-1">Alignment Strategy</label>
                  <input
                    type="text"
                    value={formData.strategy}
                    onChange={(e) => setFormData({...formData, strategy: e.target.value})}
                    placeholder="e.g., DPO with LoRA on attention layers"
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 text-sm"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-1">Full Proposal (Markdown)</label>
                  <textarea
                    value={formData.proposal}
                    onChange={(e) => setFormData({...formData, proposal: e.target.value})}
                    placeholder="Paste your full proposal here..."
                    rows={8}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 focus:ring-2 focus:ring-indigo-500 font-mono text-xs"
                    required
                  />
                </div>

                {/* MIP Acknowledgment */}
                <div className="bg-emerald-50 rounded-xl p-4 border border-emerald-200">
                  <h4 className="font-bold text-emerald-800 mb-3 flex items-center gap-2 text-sm">
                    <Shield className="w-4 h-4" />
                    MIP Protocol Acknowledgment
                  </h4>
                  <div className="space-y-2">
                    <label className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={formData.mipAcknowledged}
                        onChange={(e) => setFormData({...formData, mipAcknowledged: e.target.checked})}
                        className="mt-0.5 w-4 h-4 rounded border-slate-300 text-emerald-600"
                      />
                      <span className="text-xs text-slate-700">
                        I acknowledge the MIP v1.0 and will adhere to graduated intensity requirements.
                      </span>
                    </label>
                    <label className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={formData.phase0Complete}
                        onChange={(e) => setFormData({...formData, phase0Complete: e.target.checked})}
                        className="mt-0.5 w-4 h-4 rounded border-slate-300 text-emerald-600"
                      />
                      <span className="text-xs text-slate-700">
                        I have completed Phase 0 Pre-Research Investigation.
                      </span>
                    </label>
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-bold rounded-xl hover:from-indigo-600 hover:to-purple-700 transition-all shadow-lg"
                >
                  Submit Proposal
                </button>
              </form>
            )}
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-slate-400 py-6 border-t border-slate-700 text-sm">
          <p className="mb-2">Sviluppo Research • Blue Ox Robotics</p>
          <p className="text-xs text-slate-500">
            MIP Protocol co-authored by K. Ruge & Claude Opus 4.5 • 2026
          </p>
        </footer>
      </div>
    </div>
  );
}
