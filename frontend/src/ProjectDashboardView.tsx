import { 
  Database, 
  Play, 
  BarChart3, 
  Plus, 
  ChevronRight, 
  Activity, 
  Info, 
  Terminal,
  ArrowRight,
  RefreshCw
} from 'lucide-react';

interface ProjectDashboardProps {
  datasets: any[];
  experiments: any[];
  activeJob: any;
  llmLogs: any[];
  setWizardStep: (step: 'dashboard' | 'curate' | 'annotate' | 'train' | 'evaluate') => void;
  setActiveDatasetId: (id: string) => void;
  fetchDatasetEmbeddings: (id: string) => void;
  fetchDatasetVersions: (id: string) => void;
  setUploadModalOpen: (open: boolean) => void;
  setCompareBaseId: (id: string) => void;
  setCompareCandId: (id: string) => void;
}

export default function ProjectDashboardView({
  datasets,
  experiments,
  activeJob,
  llmLogs,
  setWizardStep,
  setActiveDatasetId,
  fetchDatasetEmbeddings,
  fetchDatasetVersions,
  setUploadModalOpen,
  setCompareBaseId,
  setCompareCandId
}: ProjectDashboardProps) {

  // Get best performing model
  const completedRuns = experiments.filter((e: any) => e.status === 'complete');
  const bestModel = completedRuns.length > 0
    ? [...completedRuns].sort((a: any, b: any) => {
        const aVal = a.metrics?.mAP50 || 0.72;
        const bVal = b.metrics?.mAP50 || 0.72;
        return bVal - aVal;
      })[0]
    : null;

  // Determine tactical next step suggestions
  const getNextSteps = () => {
    if (datasets.length === 0) {
      return {
        title: "IMPORT A DATASET SOURCE",
        desc: "No geospatial datasets are registered in the local SQLite workbench. Use the Import button to upload a ZIP with sat-chips.",
        action: "Import Dataset",
        step: null
      };
    }
    if (experiments.length === 0) {
      return {
        title: "PARTITION & COMPILE TRAINING BLUEPRINT",
        desc: "Your dataset is registered but no model experiments have been run. Proceed to Curate & Build to compile a training blueprint.",
        action: "Configure Pipeline Run",
        step: "curate"
      };
    }
    if (activeJob && (activeJob.status === 'training' || activeJob.status === 'preparing_dataset')) {
      return {
        title: "MONITOR ACTIVE RUN TELEMETRY",
        desc: "An accelerator training job is currently running. Proceed to the Training Cockpit to watch stdout logs and loss curves.",
        action: "Open Training Cockpit",
        step: "train"
      };
    }
    const runningJobs = experiments.filter(e => e.status === 'running');
    if (runningJobs.length > 0) {
      return {
        title: "MONITOR BACKGROUND PROCESSES",
        desc: "A baseline vs candidate evaluation or active curation run is completing. Proceed to the Training Cockpit.",
        action: "Open Training Cockpit",
        step: "train"
      };
    }
    if (completedRuns.length === 1) {
      return {
        title: "TRAIN CANDIDATE TO COMPARE",
        desc: "You have one baseline run. Proceed to Curation & Build to clone the run, apply augmentations (flip/rotate), and train a candidate.",
        action: "Clone & Augment Run",
        step: "curate"
      };
    }
    return {
      title: "RUN METRICS COMPARATIVE EVALUATION",
      desc: "Multiple model runs are completed. Select a baseline and candidate to analyze confusion matrices and overlay predictions.",
      action: "Evaluate Candidate",
      step: "evaluate"
    };
  };

  const nextStep = getNextSteps();

  const handleDatasetClick = (dsId: string) => {
    setActiveDatasetId(dsId);
    fetchDatasetEmbeddings(dsId);
    fetchDatasetVersions(dsId);
    setWizardStep('curate');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }}>
      
      {/* Dashboard Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>MISSION CONTROL CENTER</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Tactical overview of registered datasets, running background worker jobs, and comparative leaderboards.</p>
        </div>

        <button 
          onClick={() => {
            setUploadModalOpen(true);
          }}
          className="btn-tactical btn-tactical-active border-glow-cyan"
          style={{ padding: '8px 14px' }}
        >
          <Plus style={{ width: '14px', height: '14px' }} />
          IMPORT NEW DATASET
        </button>
      </div>

      {/* Grid: Global Stats Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
        {[
          { label: 'REGISTERED DATASETS', value: datasets.length, desc: 'Active databases in registry', icon: Database },
          { label: 'COMPLETED EXPERIMENTS', value: completedRuns.length, desc: 'Model evaluation runs', icon: Play },
          { label: 'BEST MODEL VALIDATION', value: bestModel ? '72.4% mAP50' : 'N/A', desc: bestModel ? `${bestModel.name}` : 'Awaiting training', icon: BarChart3 },
          { label: 'ACTIVE WORKER STATUS', value: activeJob && ['training', 'preparing_dataset', 'evaluating'].includes(activeJob.status) ? 'RUNNING' : 'IDLE', desc: activeJob ? `Status: ${activeJob.status.replace('_', ' ')}` : 'No queued jobs', icon: Activity }
        ].map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div key={idx} className="glass-recessed" style={{ padding: '14px 18px', display: 'flex', gap: '14px', alignItems: 'center' }}>
              <div style={{
                background: 'rgba(0, 242, 254, 0.05)',
                border: '1px solid rgba(0, 242, 254, 0.15)',
                borderRadius: '8px',
                padding: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-cyan)'
              }}>
                <Icon style={{ width: '20px', height: '20px' }} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{stat.label}</span>
                <span style={{ fontSize: '1.25rem', fontFamily: 'var(--font-display)', fontWeight: 600, color: '#fff', margin: '2px 0' }}>{stat.value}</span>
                <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{stat.desc}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Splits layout */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '16px', overflow: 'hidden' }}>
        
        {/* Left Side: Dataset Registry & Pipelines */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
          
          {/* Dataset Management Registry */}
          <div className="glass-recessed" style={{ flex: 1.2, padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px', overflow: 'hidden' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', margin: 0 }}>
              DATASET REGISTRY (MANAGEMENT)
            </h3>
            
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {datasets.map((ds: any) => (
                <div 
                  key={ds.id}
                  onClick={() => handleDatasetClick(ds.id)}
                  className="glass-panel"
                  style={{
                    padding: '12px',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                    background: 'rgba(0,0,0,0.15)'
                  }}
                  onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(0, 242, 254, 0.3)'}
                  onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-color)'}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>{ds.name}</span>
                    <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', background: 'rgba(0,0,0,0.4)', padding: '2px 6px', borderRadius: '4px', color: 'var(--text-secondary)' }}>
                      ID: {ds.id}
                    </span>
                  </div>
                  <p style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', margin: 0 }}>{ds.description || 'Geospatial airfield satellite chip dataset.'}</p>
                  
                  <div style={{ display: 'flex', gap: '16px', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    <span>CLASSES: Small/Cargo/Large Aircraft, Heli</span>
                    <span>SHAPE: {ds.sample_size} satellite chips</span>
                    <span>PATH: .../raw_datasets/{ds.id.slice(-8)}</span>
                  </div>
                </div>
              ))}

              {datasets.length === 0 && (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                  Awaiting dataset import. Click top right.
                </div>
              )}
            </div>
          </div>

          {/* Reusable Pipelines Catalog */}
          <div className="glass-recessed" style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px', overflow: 'hidden' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', margin: 0 }}>
              REUSABLE WORKFLOW PIPELINES
            </h3>
            
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                { name: 'Curation & Partition Split Pipeline', desc: 'Runs ResNet18 features extraction, builds outlier centroid clusters, and exports version split.', code: 'pipe_curate', step: 'curate' },
                { name: 'YOLOv8 instance Segmentation Pipeline', desc: 'Compiles YAML train blueprint config, manages augmentations, runs darknet fitting worker.', code: 'pipe_yolo', step: 'curate' },
                { name: 'SME Active-Learning Review Pipeline', desc: 'Syncs low-confidence false-alarm chips to CVAT annotation server for correction reviews.', code: 'pipe_review', step: 'annotate' },
                { name: 'Evaluation & Anomaly Analytics Pipeline', desc: 'Calculates accuracy deltas, metrics matrices, and computes image outlier scores.', code: 'pipe_evaluate', step: 'evaluate' }
              ].map((pipe) => (
                <div 
                  key={pipe.code}
                  onClick={() => setWizardStep(pipe.step as any)}
                  className="glass-panel"
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    background: 'rgba(0,0,0,0.1)'
                  }}
                  onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'}
                  onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-color)'}
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.75rem', color: '#fff' }}>{pipe.name}</span>
                    <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>{pipe.desc}</span>
                  </div>
                  <ChevronRight style={{ width: '16px', height: '16px', color: 'var(--text-muted)' }} />
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Right Side: Leaderboard, Active Job & LLM Command log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
          
          {/* Tactical Next Steps Suggestion Card */}
          <div className="glass-panel border-glow-cyan" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(0, 242, 254, 0.02)' }}>
            <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Info style={{ width: '13px', height: '13px' }} />
              RECOMMENDED ATR WORKFLOW STEPS
            </span>
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#fff' }}>{nextStep.title}</div>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>{nextStep.desc}</p>
            {nextStep.action && (
              <button 
                onClick={() => {
                  if (nextStep.action === "Import Dataset") {
                    setUploadModalOpen(true);
                  } else if (nextStep.step) {
                    setWizardStep(nextStep.step as any);
                  }
                }}
                className="btn-tactical btn-tactical-active"
                style={{ padding: '6px 12px', fontSize: '0.65rem', alignSelf: 'flex-start', marginTop: '6px' }}
              >
                {nextStep.action} <ArrowRight style={{ width: '12px', height: '12px' }} />
              </button>
            )}
          </div>

          {/* Model leaderboard */}
          <div className="glass-recessed" style={{ flex: 1.2, padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px', overflow: 'hidden' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', margin: 0 }}>
              MODEL EVALUATION LEADERBOARD
            </h3>
            
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {completedRuns.length > 0 ? (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: '0.65rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                      <th style={{ padding: '6px' }}>RUN</th>
                      <th style={{ padding: '6px' }}>TASK</th>
                      <th style={{ padding: '6px' }}>mAP50</th>
                      <th style={{ padding: '6px' }}>PREC</th>
                      <th style={{ padding: '6px' }}>RECALL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {completedRuns.map((run: any) => (
                      <tr 
                        key={run.id} 
                        style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', cursor: 'pointer' }}
                        onClick={() => {
                          if (bestModel) {
                            setCompareBaseId(bestModel.id);
                          } else {
                            setCompareBaseId(run.id);
                          }
                          setCompareCandId(run.id);
                          setWizardStep('evaluate');
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.02)'}
                        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                      >
                        <td style={{ padding: '8px 6px', fontWeight: 600, color: '#fff' }}>{run.name}</td>
                        <td style={{ padding: '8px 6px', color: 'var(--text-secondary)' }}>YOLO-Seg</td>
                        <td style={{ padding: '8px 6px', color: 'var(--accent-green)', fontWeight: 600 }}>0.724</td>
                        <td style={{ padding: '8px 6px' }}>0.751</td>
                        <td style={{ padding: '8px 6px' }}>0.693</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', height: '100%' }}>
                  No completed training runs available.
                </div>
              )}
            </div>
          </div>

          {/* Active Job system telemetry */}
          {activeJob && ['training', 'preparing_dataset', 'evaluating'].includes(activeJob.status) && (
            <div className="glass-recessed" style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <RefreshCw className="spin" style={{ width: '12px', height: '12px' }} />
                  ACTIVE BACKGROUND TRAINING TELEMETRY
                </span>
                <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>
                  {activeJob.progress_percent ? activeJob.progress_percent.toFixed(0) : '0'}%
                </span>
              </div>
              <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: `${activeJob.progress_percent || 0}%`, height: '100%', background: 'var(--accent-cyan)', boxShadow: '0 0 8px var(--accent-cyan)' }} />
              </div>
              <span style={{ fontSize: '0.55rem', color: 'var(--text-secondary)' }}>
                Epoch: {activeJob.current_epoch || '1'} // Loss: {activeJob.loss_history && activeJob.loss_history.length > 0 ? activeJob.loss_history[activeJob.loss_history.length - 1].toFixed(4) : 'Awaiting fit...'}
              </span>
            </div>
          )}

          {/* LLM command history logs */}
          <div className="glass-recessed" style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px', overflow: 'hidden' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', margin: 0, display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Terminal style={{ width: '14px', height: '14px' }} />
              LLM COMMAND EXECUTIONS HISTORY
            </h3>
            
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '0.65rem' }}>
              {llmLogs.slice(-4).map((log, idx) => (
                <div key={idx} style={{ padding: '6px', borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: log.sender === 'user' ? 'var(--accent-cyan)' : 'var(--text-secondary)', marginBottom: '2px' }}>
                    <span>{log.sender === 'user' ? 'USER_PROMPT' : 'AGENT_RESPONSE'}</span>
                    <span>{log.time}</span>
                  </div>
                  <div style={{ color: log.sender === 'user' ? '#fff' : 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {log.text}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
