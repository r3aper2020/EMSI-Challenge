import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  RefreshCw, 
  Database as DbIcon, 
  Terminal, 
  Activity, 
  Layers, 
  BarChart3, 
  ArrowRight, 
  Plus, 
  CheckCircle2, 
  XCircle, 
  Compass, 
  ChevronRight,
  TrendingUp,
  Eye,
  EyeOff,
  Trash2,
  Save,
  Cpu
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000/api';

const getDatasetIdFromVersion = (versionId: string) => {
  if (!versionId) return 'dataset_rareplanes_real';
  if (versionId.startsWith('version_')) {
    const parts = versionId.split('_');
    return parts.slice(1, -1).join('_');
  }
  return 'dataset_rareplanes_real';
};

const getImageUrl = (datasetId: string, imageId: string) => {
  const dsId = datasetId || 'dataset_rareplanes_real';
  return `${API_BASE}/datasets/${dsId}/images/${imageId}`;
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'studio' | 'control'>('dashboard');
  const [llmOpen, setLlmOpen] = useState(true);
  const [llmPrompt, setLlmPrompt] = useState('');
  const [llmLogs, setLlmLogs] = useState<Array<{ sender: 'user' | 'system'; text: string; time: string; payload?: any }>>([
    { sender: 'system', text: 'ATR Assistant terminal online. Ask me to split datasets, configure augmentations, or run training runs.', time: new Date().toLocaleTimeString() }
  ]);
  const [llmLoading, setLlmLoading] = useState(false);
  
  // App Data State
  const [datasets, setDatasets] = useState<any[]>([]);
  const [versions, setVersions] = useState<any[]>([]);
  const [experiments, setExperiments] = useState<any[]>([]);
  
  // Active states
  const [activeDatasetId, setActiveDatasetId] = useState<string>('dataset_rareplanes_real');
  const [embeddings, setEmbeddings] = useState<any[]>([]);
  const [selectedEmbed, setSelectedEmbed] = useState<any>(null);
  const [selectedImageDetails, setSelectedImageDetails] = useState<any>(null);
  
  const [activeJob, setActiveJob] = useState<any>(null);
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(null);
  const [jobLogs, setJobLogs] = useState<string>('');
  const [projectTree, setProjectTree] = useState<any>(null);
  
  const [compareBaseId, setCompareBaseId] = useState<string>('');
  const [compareCandId, setCompareCandId] = useState<string>('');
  const [compareResults, setCompareResults] = useState<any>(null);

  const [compiledVersion, setCompiledVersion] = useState<any>(null);
  const [isGeneratingBlueprint, setIsGeneratingBlueprint] = useState<boolean>(false);

  const experimentsRef = useRef(experiments);
  useEffect(() => {
    experimentsRef.current = experiments;
  }, [experiments]);

  const logsEndRef = useRef<HTMLDivElement>(null);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll helper
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [jobLogs]);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [llmLogs]);

  // Initial Bootstrapping
  useEffect(() => {
    fetchInitialData();
    // Auto-poll if there is a running training job
    const interval = setInterval(() => {
      checkRunningJobs();
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchInitialData = async () => {
    try {
      // Trigger initialization backend scan
      const initRes = await fetch(`${API_BASE}/initialize`, { method: 'POST' });
      const initData = await initRes.json();
      console.log('Backend initialized:', initData);
      
      await fetch(`${API_BASE}/projects`);
      
      const dRes = await fetch(`${API_BASE}/datasets`);
      const dsList = await dRes.json();
      setDatasets(dsList);
      
      if (dsList.length > 0) {
        fetchDatasetEmbeddings(dsList[0].id);
        fetchDatasetVersions(dsList[0].id);
      }
      
      await fetchExperiments();
      await fetchProjectTree();
    } catch (e) {
      console.error('Error fetching initial workbench data:', e);
    }
  };

  const handleDatasetUpload = async () => {
    try {
      const dRes = await fetch(`${API_BASE}/datasets`);
      const dsList = await dRes.json();
      setDatasets(dsList);
    } catch (e) {
      console.error('Error refreshing datasets after upload:', e);
    }
  };

  const fetchDatasetEmbeddings = async (dsId: string) => {
    try {
      const res = await fetch(`${API_BASE}/datasets/${dsId}/embeddings`);
      const data = await res.json();
      setEmbeddings(data);
      if (data.length > 0) {
        setSelectedEmbed(data[0]);
      }
    } catch (e) {
      console.error('Error fetching embeddings:', e);
    }
  };

  const fetchDatasetVersions = async (dsId: string) => {
    try {
      const res = await fetch(`${API_BASE}/dataset-versions?dataset_id=${dsId}`);
      setVersions(await res.json());
    } catch (e) {
      console.error('Error fetching dataset versions:', e);
    }
  };

  const fetchProjectTree = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/proj_default/tree`);
      if (res.ok) {
        const data = await res.json();
        setProjectTree(data);
      }
    } catch (e) {
      console.error('Error fetching project tree:', e);
    }
  };

  const fetchExperiments = async () => {
    try {
      const res = await fetch(`${API_BASE}/experiments`);
      const data = await res.json();
      setExperiments(data);
      if (data.length >= 2) {
        setCompareBaseId(data[1].id);
        setCompareCandId(data[0].id);
      } else if (data.length === 1) {
        setCompareBaseId(data[0].id);
      }
      
      // Auto-refresh project-pipeline-run tree as well
      await fetchProjectTree();
    } catch (e) {
      console.error('Error fetching experiments:', e);
    }
  };

  const checkRunningJobs = async () => {
    let activeExpId = activeExperimentId;
    if (!activeExpId) {
      const activeRunningExp = experimentsRef.current.find(e => e.status === 'training' || e.status === 'queued' || e.status === 'preparing_dataset');
      if (activeRunningExp) {
        activeExpId = activeRunningExp.id;
      }
    }
    
    if (activeExpId) {
      try {
        const res = await fetch(`${API_BASE}/experiments/${activeExpId}/job`);
        if (res.ok) {
          const job = await res.json();
          setActiveJob(job);
          setJobLogs(job.logs || '');
          fetchExperiments(); // Keep historical runs list status in sync in real time
          
          if (job.status === 'complete' || job.status === 'failed' || job.status === 'cancelled') {
            setActiveExperimentId(null);
          }
        }
      } catch (e) {
        console.error('Error polling job status:', e);
      }
    } else {
      setActiveJob((prev: any) => {
        if (prev && (prev.status === 'complete' || prev.status === 'failed' || prev.status === 'cancelled')) {
          return prev;
        }
        return null;
      });
    }
  };

  const [viewTab, setViewTab] = useState<'evaluator' | 'cockpit'>('evaluator');
  const [expandedPipelines, setExpandedPipelines] = useState<Record<string, boolean>>({
    pipe_curation: true,
    pipe_labeling_review: false,
    pipe_yolo: true,
    pipe_maskrcnn: false,
    pipe_eval_review: false,
  });

  const togglePipeline = (pipeId: string) => {
    setExpandedPipelines(prev => ({
      ...prev,
      [pipeId]: !prev[pipeId]
    }));
  };

  const selectRun = (exp: any) => {
    if (exp.pipeline_id === 'pipe_yolo' || exp.pipeline_id === 'pipe_maskrcnn') {
      setActiveExperimentId(exp.id);
      setViewTab('cockpit');
      setActiveTab('control');
    } else {
      let mockLogs = '';
      if (exp.pipeline_id === 'pipe_curation') {
        mockLogs = `[System] Dataset Curation Run: ${exp.name}
[System] Status: ${exp.status.toUpperCase()}
[System] Scanned dataset: dataset_rareplanes_real
[System] Detected 2 potential duplicate aircraft chips.
[System] Flagged 3 low-resolution / shadowed airfield samples.
[System] Curation blueprint checks passed successfully.`;
      } else if (exp.pipeline_id === 'pipe_labeling_review') {
        mockLogs = `[System] Labeling Review Run: ${exp.name}
[System] Status: ${exp.status.toUpperCase()}
[System] Loaded active learning proposal queue...
[System] Detected 15 auto-labeled aircraft suggestions.
[System] Verified 12 suggestions with high-confidence thresholds (>85%).
[System] SME feedback logged.`;
      } else if (exp.pipeline_id === 'pipe_eval_review') {
        mockLogs = `[System] Evaluation Review Run: ${exp.name}
[System] Status: ${exp.status.toUpperCase()}
[System] Scanned ground truth labels vs model predictions...
[System] Calculated confidence distributions: min=0.25, max=0.98.
[System] Visual inspection complete. Top errors queued for active learning rerun.`;
      } else {
        mockLogs = `[System] Pipeline Run: ${exp.name}
[System] Status: ${exp.status.toUpperCase()}
[System] Execution complete.`;
      }

      setActiveJob({
        id: exp.id,
        status: exp.status,
        logs: mockLogs,
        loss_history: [],
        map50_history: [],
        current_epoch: 0,
        total_epochs: 1
      });
      setJobLogs(mockLogs);
      setViewTab('cockpit');
      
      if (exp.pipeline_id === 'pipe_curation' || exp.pipeline_id === 'pipe_labeling_review') {
        setActiveTab('studio');
      } else {
        setActiveTab('control');
      }
    }
  };

  const handleLaunchPipeline = async (pipe: any) => {
    if (pipe.type === 'segmentation_training') {
      setActiveTab('studio');
      return;
    }
    
    try {
      const res = await fetch(`${API_BASE}/pipelines/${pipe.id}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: `Interactive ${pipe.name}` })
      });
      if (res.ok) {
        const expData = await res.json();
        await fetchProjectTree();
        
        const initialLogs = `[System] Launched Pipeline Run: ${expData.name}
[System] Status: RUNNING
[System] Initializing pipeline workflow...`;
        
        setActiveJob({
          id: expData.id,
          status: 'running',
          logs: initialLogs,
          loss_history: [],
          map50_history: [],
          current_epoch: 0,
          total_epochs: 1
        });
        setJobLogs(initialLogs);
        setViewTab('cockpit');
        
        if (pipe.id === 'pipe_curation' || pipe.id === 'pipe_labeling_review') {
          setActiveTab('studio');
        } else {
          setActiveTab('control');
        }
        
        let pollCount = 0;
        const interval = setInterval(async () => {
          pollCount++;
          const treeRes = await fetch(`${API_BASE}/projects/proj_default/tree`);
          if (treeRes.ok) {
            const treeData = await treeRes.json();
            const updatedPipe = treeData.pipelines?.find((p: any) => p.id === pipe.id);
            const updatedExp = updatedPipe?.experiments?.find((e: any) => e.id === expData.id);
            if (updatedExp) {
              if (updatedExp.status === 'complete' || updatedExp.status === 'failed' || pollCount > 10) {
                clearInterval(interval);
                await fetchProjectTree();
                selectRun(updatedExp);
              } else {
                const tickLogs = `[System] Launched Pipeline Run: ${expData.name}
[System] Status: RUNNING
[System] Process running... (tick ${pollCount}/3)
[System] Scanning dataset airfield samples...
[System] Active proposals processing...`;
                setActiveJob({
                  id: expData.id,
                  status: 'running',
                  logs: tickLogs,
                  loss_history: [],
                  map50_history: [],
                  current_epoch: 0,
                  total_epochs: 1
                });
                setJobLogs(tickLogs);
              }
            }
          }
        }, 1000);
      }
    } catch (err) {
      console.error("Failed to launch pipeline run:", err);
    }
  };

  const triggerQuickPrompt = (promptText: string) => {
    setLlmPrompt(promptText);
    setLlmOpen(true);
  };

  // Submit LLM Commands
  const handleLlmSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!llmPrompt.trim()) return;
    
    const userMsg = llmPrompt;
    setLlmPrompt('');
    setLlmLogs(prev => [...prev, { sender: 'user', text: userMsg, time: new Date().toLocaleTimeString() }]);
    setLlmLoading(true);
    
    try {
      const res = await fetch(`${API_BASE}/llm/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userMsg })
      });
      
      const data = await res.json();
      setLlmLoading(false);
      
      if (data.status === 'success') {
        setLlmLogs(prev => [...prev, { 
          sender: 'system', 
          text: data.message, 
          time: new Date().toLocaleTimeString(),
          payload: data.workflow_result 
        }]);
        
        // Refresh background data
        fetchExperiments();
        if (data.payload && data.payload.dataset_id) {
          fetchDatasetVersions(data.payload.dataset_id);
        }
        
        // Auto navigate to control view if job started
        if (data.action === 'create_experiment' || data.action === 'clone_experiment') {
          setActiveTab('control');
          if (data.workflow_result && data.workflow_result.experiment_id) {
            setActiveExperimentId(data.workflow_result.experiment_id);
          }
        }
      } else {
        setLlmLogs(prev => [...prev, { sender: 'system', text: `Failed: ${data.message}`, time: new Date().toLocaleTimeString() }]);
      }
    } catch (err: any) {
      setLlmLoading(false);
      setLlmLogs(prev => [...prev, { sender: 'system', text: `Network error: ${err.message}`, time: new Date().toLocaleTimeString() }]);
    }
  };

  // Step 1 of Pipeline Studio: Split dataset & compile configuration (Pipeline Generation)
  const handleGeneratePipelineBlueprint = async (formData: any) => {
    setIsGeneratingBlueprint(true);
    setJobLogs(`[System] Initiating Pipeline Blueprint Generation...
[System] Compiling Stage 1: Ingestion & Data Partitioning...
[System] Running split partition on '${formData.dataset_id}' (Seed: ${formData.split_seed})...
`);
    try {
      const val_split = parseFloat((1 - formData.train_split).toFixed(2));
      const splitRes = await fetch(`${API_BASE}/datasets/split`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: formData.dataset_id,
          version_tag: formData.version_tag,
          train_split: formData.train_split,
          val_split: val_split,
          split_seed: formData.split_seed
        })
      });
      
      if (!splitRes.ok) {
        const errData = await splitRes.json();
        throw new Error(errData.detail || "Split partition failed");
      }
      
      const version = await splitRes.json();
      setCompiledVersion(version);
      
      setJobLogs(prev => prev + `[System] Split partition complete! Created dataset version: ${version.id}
[System] Ingested Satellite Chips: ${version.manifest?.train?.length + version.manifest?.val?.length} chips
[System] Output location: ${version.export_path}/dataset.yaml
[System] Blueprint generated successfully. Ready for execution.
`);
      return version;
    } catch (e: any) {
      console.error('Error generating pipeline split:', e);
      setJobLogs(prev => prev + `[Error] Blueprint generation failed: ${e.message}\n`);
      throw e;
    } finally {
      setIsGeneratingBlueprint(false);
    }
  };

  // Step 2 of Pipeline Studio: Execute the model training job on the compiled split (Pipeline Execution)
  const handleExecutePipelineRun = async (formData: any, versionId: string) => {
    setActiveJob({
      id: `pipeline_${Date.now()}`,
      status: "preparing_dataset",
      progress_percent: 2.0,
      loss_history: [],
      map50_history: [],
      current_epoch: 0,
      total_epochs: formData.epochs,
      logs: ""
    });
    setJobLogs(prev => prev + `[System] Launching Pipeline Execution on Accelerator...
[System] Stage 2: Applying Augmentations (fliplr=${formData.fliplr ? 'ON' : 'OFF'}, flipud=${formData.flipud ? 'ON' : 'OFF'}, degrees=${formData.degrees})...
[System] Stage 3: Initializing YOLOv8-seg neural net model training...
\n`);

    try {
      const expRes = await fetch(`${API_BASE}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: 'proj_default',
          dataset_version_id: versionId,
          name: formData.name,
          task_type: formData.task_type,
          model_type: formData.model_type,
          config: {
            epochs: formData.epochs,
            batch: formData.batch,
            imgsz: formData.imgsz,
            mock: formData.mock,
            augmentations: {
              fliplr: formData.fliplr,
              flipud: formData.flipud,
              degrees: formData.degrees
            }
          }
        })
      });
      
      if (!expRes.ok) {
        const errData = await expRes.json();
        throw new Error(errData.detail || "Training execution failed");
      }
      
      const expData = await expRes.json();
      if (expData.experiment && expData.experiment.id) {
        setActiveExperimentId(expData.experiment.id);
      }
      
      setActiveTab('control');
      fetchExperiments();
      fetchDatasetVersions(formData.dataset_id);
    } catch (e: any) {
      console.error('Error starting pipeline execution:', e);
      setJobLogs(prev => prev + `[Error] Pipeline execution failed: ${e.message}\n`);
      setActiveJob({
        id: "pipeline_failed",
        status: "failed",
        progress_percent: 0.0,
        loss_history: [],
        map50_history: [],
        current_epoch: 0,
        total_epochs: formData.epochs
      });
    }
  };

  // Run experiment metrics comparison
  useEffect(() => {
    if (compareBaseId && compareCandId) {
      loadComparison();
    }
  }, [compareBaseId, compareCandId, experiments]);

  const loadComparison = async () => {
    try {
      const baseExp = experiments.find(e => e.id === compareBaseId);
      const candExp = experiments.find(e => e.id === compareCandId);
      
      if (!baseExp || !candExp) return;
      
      // Fetch evaluations for both
      const baseEvalRes = await fetch(`${API_BASE}/experiments/${compareBaseId}/evaluation`);
      const candEvalRes = await fetch(`${API_BASE}/experiments/${compareCandId}/evaluation`);
      
      const baseEval = baseEvalRes.ok ? await baseEvalRes.json() : null;
      const candEval = candEvalRes.ok ? await candEvalRes.json() : null;
      
      setCompareResults({
        base: { exp: baseExp, eval: baseEval },
        candidate: { exp: candExp, eval: candEval }
      });
    } catch (e) {
      console.error('Error loading comparison metrics:', e);
    }
  };


  return (
    <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Background Elements */}
      <div className="mesh-background">
        <div className="mesh-blob-1"></div>
        <div className="mesh-blob-2"></div>
        <div className="mesh-blob-3"></div>
      </div>
      <div className="tactical-grid"></div>

      {/* 1. Technical Navigation Header */}
      <header className="glass-panel" style={{ 
        margin: '12px 16px 6px 16px', 
        padding: '12px 24px', 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)' 
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Compass className="gradient-text-cyan" style={{ width: '28px', height: '28px' }} />
          <div>
            <h1 className="gradient-text-cyan" style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>
              ATR WORKBENCH
            </h1>
            <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
              Tactical Model Production Center // V1.0.0
            </span>
          </div>
        </div>
        
        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '8px' }}>
          {[
            { id: 'dashboard', label: 'DASHBOARD', icon: Compass },
            { id: 'studio', label: '1. PIPELINE STUDIO', icon: DbIcon },
            { id: 'control', label: '2. MISSION CONTROL', icon: Play },
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button 
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`btn-tactical ${active ? 'btn-tactical-active' : ''}`}
                style={{ padding: '8px 12px', fontSize: '0.7rem' }}
              >
                <Icon style={{ width: '13px', height: '13px' }} />
                {tab.label}
              </button>
            );
          })}
        </nav>

        {/* Global Connection Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div className="glass-recessed" style={{ padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
            <span className="led-indicator green"></span>
            LOCAL INSTANCE ACTIVE
          </div>
          <button 
            onClick={() => setLlmOpen(!llmOpen)}
            className={`btn-tactical ${llmOpen ? 'btn-tactical-active' : ''}`}
            style={{ padding: '8px 14px' }}
          >
            <Terminal style={{ width: '15px', height: '15px' }} />
            ASSISTANT
          </button>
        </div>
      </header>


      {/* Main Workspace Frame */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', padding: '0 16px 12px 16px', gap: '12px' }}>
        
        {/* Left Column: Pipelines & Runs nested Accordion tree */}
        <div className="glass-panel" style={{ width: '320px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto', flexShrink: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', marginBottom: '4px' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', letterSpacing: '0.05em', margin: 0, color: 'var(--accent-cyan)' }}>
              PROJECT WORKFLOWS
            </h3>
            <button
              onClick={() => {
                setActiveTab('dashboard');
                setActiveExperimentId(null);
              }}
              className={`btn-tactical ${activeTab === 'dashboard' ? 'btn-tactical-active border-glow-cyan' : ''}`}
              style={{ padding: '2px 6px', fontSize: '0.55rem' }}
            >
              OVERVIEW
            </button>
          </div>

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px', overflowY: 'auto' }}>
            {projectTree && projectTree.pipelines ? (
              projectTree.pipelines.map((pipe: any) => {
                const isExpanded = !!expandedPipelines[pipe.id];
                return (
                  <div key={pipe.id} style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '8px' }}>
                    {/* Pipeline Accordion Header */}
                    <div 
                      style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center', 
                        padding: '6px 10px', 
                        background: 'rgba(255, 255, 255, 0.03)',
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }} 
                      onClick={() => togglePipeline(pipe.id)}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                        <span style={{ 
                          fontSize: '0.6rem', 
                          transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', 
                          transition: 'transform 0.15s ease', 
                          display: 'inline-block', 
                          color: 'var(--text-secondary)' 
                        }}>▶</span>
                        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                          <span style={{ fontWeight: 600, fontSize: '0.65rem', color: '#fff', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{pipe.name}</span>
                          <span style={{ fontSize: '0.55rem', color: 'var(--text-muted)' }}>{pipe.experiments?.length || 0} runs</span>
                        </div>
                      </div>
                      
                      {/* Launch Run Button */}
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleLaunchPipeline(pipe);
                        }}
                        className="btn-tactical" 
                        style={{ 
                          padding: '2px 6px', 
                          fontSize: '0.55rem', 
                          borderColor: 'var(--accent-cyan)',
                          color: 'var(--accent-cyan)',
                          background: 'rgba(0, 242, 254, 0.03)',
                          flexShrink: 0
                        }}
                      >
                        LAUNCH
                      </button>
                    </div>

                    {/* Pipeline Runs list */}
                    {isExpanded && (
                      <div style={{ 
                        paddingLeft: '12px', 
                        display: 'flex', 
                        flexDirection: 'column', 
                        gap: '4px', 
                        borderLeft: '1px dashed rgba(255,255,255,0.08)', 
                        marginLeft: '8px', 
                        marginTop: '4px' 
                      }}>
                        {pipe.experiments && pipe.experiments.map((exp: any) => {
                          const isSelected = activeJob?.id === exp.id || activeJob?.experiment_id === exp.id || activeExperimentId === exp.id;
                          return (
                            <div 
                              key={exp.id}
                              onClick={() => selectRun(exp)}
                              className="glass-panel"
                              style={{
                                padding: '6px 10px',
                                fontSize: '0.65rem',
                                fontFamily: 'var(--font-mono)',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                cursor: 'pointer',
                                borderColor: isSelected ? 'var(--accent-cyan)' : 'var(--border-color)',
                                background: isSelected ? 'rgba(0, 242, 254, 0.05)' : 'rgba(0,0,0,0.15)',
                                transition: 'all 0.15s ease'
                              }}
                            >
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', overflow: 'hidden' }}>
                                <span style={{ fontWeight: 600, color: isSelected ? 'var(--accent-cyan)' : '#fff', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{exp.name}</span>
                                <span style={{ color: 'var(--text-muted)', fontSize: '0.5rem' }}>{exp.id}</span>
                              </div>

                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
                                {exp.status === 'complete' ? (
                                  <span style={{ color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '2px', fontSize: '0.55rem' }}>
                                    <CheckCircle2 style={{ width: '9px', height: '9px' }} /> DONE
                                  </span>
                                ) : exp.status === 'failed' ? (
                                  <span style={{ color: 'var(--accent-red)', display: 'flex', alignItems: 'center', gap: '2px', fontSize: '0.55rem' }}>
                                    <XCircle style={{ width: '9px', height: '9px' }} /> FAIL
                                  </span>
                                ) : (
                                  <span style={{ color: 'var(--accent-orange)', display: 'flex', alignItems: 'center', gap: '2px', fontSize: '0.55rem' }}>
                                    <RefreshCw className="spin" style={{ width: '9px', height: '9px' }} /> RUN
                                  </span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                        {(!pipe.experiments || pipe.experiments.length === 0) && (
                          <div style={{ padding: '6px 10px', color: 'var(--text-muted)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}>
                            No runs completed yet.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.7rem' }}>
                Loading projects, pipelines and runs...
              </div>
            )}
          </div>

          {/* AI command prompt pre-fills */}
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Terminal style={{ width: '10px', height: '10px' }} />
              COCKPIT QUICK TRIGGERS
            </span>
            <div style={{ display: 'flex', gap: '4px', flexDirection: 'column' }}>
              <button 
                onClick={() => triggerQuickPrompt('export the rareplanes dataset with 80/20 split to train an instance segmentation model')}
                className="btn-tactical" 
                style={{ padding: '4px 8px', fontSize: '0.55rem', width: '100%', justifyContent: 'center' }}
              >
                Split 80/20 & Train Model
              </button>
              <button 
                onClick={() => triggerQuickPrompt('Redo the previous experiment but apply horizontal flip and random rotation of 30 degrees')}
                className="btn-tactical" 
                style={{ padding: '4px 8px', fontSize: '0.55rem', width: '100%', justifyContent: 'center' }}
              >
                Clone Run + Rotate/Flip Aug
              </button>
            </div>
          </div>
        </div>

        {/* Active Screen Container */}
        <main className="glass-panel" style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', padding: '20px' }}>
          {activeTab === 'dashboard' && (
            <DashboardView 
              datasets={datasets}
              experiments={experiments}
              versions={versions}
              projectTree={projectTree}
              setActiveTab={setActiveTab}
            />
          )}
          {activeTab === 'studio' && (
            <PipelineStudioView 
              datasets={datasets} 
              embeddings={embeddings} 
              selectedEmbed={selectedEmbed}
              setSelectedEmbed={setSelectedEmbed}
              selectedImageDetails={selectedImageDetails}
              setSelectedImageDetails={setSelectedImageDetails}
              activeDatasetId={activeDatasetId}
              setActiveDatasetId={setActiveDatasetId}
              fetchDatasetEmbeddings={fetchDatasetEmbeddings}
              fetchDatasetVersions={fetchDatasetVersions}
              onDatasetUpload={handleDatasetUpload}
              versions={versions}
              handleGeneratePipelineBlueprint={handleGeneratePipelineBlueprint}
              handleExecutePipelineRun={handleExecutePipelineRun}
              isGeneratingBlueprint={isGeneratingBlueprint}
              compiledVersion={compiledVersion}
              setCompiledVersion={setCompiledVersion}
            />
          )}
          {activeTab === 'control' && (
            <MissionControlView 
              activeJob={activeJob} 
              jobLogs={jobLogs}
              logsEndRef={logsEndRef}
              experiments={experiments}
              datasets={datasets}
              compareBaseId={compareBaseId}
              compareCandId={compareCandId}
              setCompareBaseId={setCompareBaseId}
              setCompareCandId={setCompareCandId}
              compareResults={compareResults}
              viewTab={viewTab}
              setViewTab={setViewTab}
            />
          )}
        </main>

        {/* Sliding LLM Orchestrator Side Panel */}
        {llmOpen && (
          <aside className="glass-panel border-glow-cyan" style={{ 
            width: '380px', 
            display: 'flex', 
            flexDirection: 'column', 
            overflow: 'hidden',
            background: 'rgba(5, 8, 20, 0.85)'
          }}>
            <div style={{ 
              padding: '16px', 
              borderBottom: '1px solid var(--border-color)', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              background: 'rgba(0,0,0,0.2)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Terminal style={{ width: '18px', height: '18px', color: 'var(--accent-cyan)' }} />
                <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '0.9rem', letterSpacing: '0.05em' }}>LLM COCKPIT ORCHESTRATOR</h2>
              </div>
              <span className="led-indicator cyan"></span>
            </div>

            {/* Chat History Terminal */}
            <div style={{ flex: 1, padding: '16px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              {llmLogs.map((log, idx) => (
                <div key={idx} style={{ 
                  alignSelf: log.sender === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '90%',
                  background: log.sender === 'user' ? 'rgba(0, 242, 254, 0.06)' : 'rgba(255, 255, 255, 0.02)',
                  border: log.sender === 'user' ? '1px solid rgba(0, 242, 254, 0.2)' : '1px solid rgba(255,255,255,0.05)',
                  padding: '10px 12px',
                  borderRadius: '8px'
                }}>
                  <div style={{ fontSize: '0.65rem', color: log.sender === 'user' ? 'var(--accent-cyan)' : 'var(--text-secondary)', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                    <span>{log.sender === 'user' ? 'USER@WORKBENCH' : 'LLM_PARSER_AGENT'}</span>
                    <span>{log.time}</span>
                  </div>
                  <div style={{ color: log.sender === 'user' ? '#fff' : 'var(--text-primary)', whiteSpace: 'pre-wrap', lineHeight: '1.4' }}>
                    {log.text}
                  </div>
                </div>
              ))}
              {llmLoading && (
                <div style={{ display: 'flex', gap: '8px', color: 'var(--accent-cyan)', alignItems: 'center' }}>
                  <RefreshCw className="spin" style={{ width: '14px', height: '14px' }} />
                  <span>Analyzing intent and orchestrating pipeline actions...</span>
                </div>
              )}
              <div ref={terminalEndRef} />
            </div>

            {/* Chat Input Console */}
            <form onSubmit={handleLlmSubmit} style={{ padding: '12px', borderTop: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.1)' }}>
              <div style={{ display: 'flex', gap: '8px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '6px 10px', alignItems: 'center' }}>
                <ChevronRight style={{ width: '16px', height: '16px', color: 'var(--accent-cyan)' }} />
                <input 
                  type="text" 
                  value={llmPrompt}
                  onChange={e => setLlmPrompt(e.target.value)}
                  placeholder="Enter orchestration command..." 
                  style={{ 
                    flex: 1, 
                    background: 'none', 
                    border: 'none', 
                    color: '#fff', 
                    outline: 'none', 
                    fontFamily: 'var(--font-mono)', 
                    fontSize: '0.8rem' 
                  }}
                />
                <button type="submit" style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer' }}>
                  <ArrowRight style={{ width: '16px', height: '16px' }} />
                </button>
              </div>
            </form>
          </aside>
        )}

      </div>
    </div>
  );
}// ============================================
// CONSTANTS & HELPERS FOR ANNOTATION/EXPLORE
// ==============================================
const CLASS_COLORS: Record<number, string> = {
  0: '#00f2fe', // Cyan - Small Aircraft
  1: '#00ff87', // Green - Cargo Plane
  2: '#ff9900', // Orange - Large Aircraft
  3: '#ff0055'  // Red - Helicopter
};

const CLASS_NAMES: Record<number, string> = {
  0: 'Small Aircraft',
  1: 'Cargo Plane',
  2: 'Large Aircraft',
  3: 'Helicopter'
};

// ============================================
// VIEW COMPONENT: 1. EXPLORE & CURATE (FiftyOne-Style)
// ===========================================// ============================================
// VIEW COMPONENT: 1. PIPELINE STUDIO (Curation & Build)
// ============================================
function PipelineStudioView({
  datasets,
  embeddings,
  selectedEmbed,
  setSelectedEmbed,
  selectedImageDetails,
  setSelectedImageDetails,
  activeDatasetId,
  setActiveDatasetId,
  fetchDatasetEmbeddings,
  fetchDatasetVersions,
  onDatasetUpload,
  versions: _versions,
  handleGeneratePipelineBlueprint,
  handleExecutePipelineRun,
  isGeneratingBlueprint,
  compiledVersion,
  setCompiledVersion
}: any) {
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadName, setUploadName] = useState('');
  const [uploadDesc, setUploadDesc] = useState('');
  const [uploadTask, setUploadTask] = useState('instance_segmentation');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadingStatus, setUploadingStatus] = useState('');
  const [uploadError, setUploadError] = useState('');

  // Control side-panel tab: curate data filters vs build pipeline configuration
  const [controlTab, setControlTab] = useState<'curate' | 'build'>('curate');

  // FiftyOne Curation Filters State
  const [searchId, setSearchId] = useState('');
  const minGsd = 0.1;
  const [maxGsd, setMaxGsd] = useState(1.0);
  const [minTargets, setMinTargets] = useState(0);
  const [showClusterMap, setShowClusterMap] = useState(false);
  const [hoveredCardIdx, setHoveredCardIdx] = useState<string | null>(null);

  // Space-Based Radar (SAR) Curation Parameters
  const [sensorType, setSensorType] = useState('all');
  const [sarPolarization, setSarPolarization] = useState('all');
  const [minIncidence, setMinIncidence] = useState(15.0);
  const [maxIncidence, setMaxIncidence] = useState(45.0);
  const [simulateBackscatter, setSimulateBackscatter] = useState(false);

  // Pipeline Parameters Form State (Combined)
  const [formData, setFormData] = useState({
    name: 'YOLOv8-seg airfield run',
    dataset_id: activeDatasetId || 'dataset_rareplanes_real',
    version_tag: 'run_v1',
    train_split: 0.8,
    val_split: 0.2,
    split_seed: 42,
    task_type: 'instance_segmentation',
    model_type: 'yolo_seg',
    epochs: 3,
    batch: 2,
    imgsz: 512,
    fliplr: false,
    flipud: false,
    degrees: 0.0,
    mock: true
  });

  const [compiledBlueprint, setCompiledBlueprint] = useState<string | null>(null);

  // Synchronize activeDatasetId changes with pipeline configuration formData
  useEffect(() => {
    if (activeDatasetId) {
      setFormData(prev => ({ ...prev, dataset_id: activeDatasetId }));
    }
  }, [activeDatasetId]);

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setCompiledBlueprint(null); // Clear blueprint when parameters change to enforce rebuild
    setCompiledVersion(null);
  };

  const handleGenerateBlueprintClick = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const version = await handleGeneratePipelineBlueprint(formData);
      const val_split = parseFloat((1 - formData.train_split).toFixed(2));
      const trainCount = Math.round((version.manifest?.train?.length) || (embeddings.length * formData.train_split));
      const valCount = Math.round((version.manifest?.val?.length) || (embeddings.length - trainCount));
      
      const yamlStr = `pipeline:
  name: "${formData.name}"
  version_tag: "${formData.version_tag}"
  dataset_id: "${formData.dataset_id}"
  generated_version_id: "${version.id}"
  stages:
    - data_split:
        train_ratio: ${formData.train_split} (${trainCount} chips)
        val_ratio: ${val_split} (${valCount} chips)
        seed: ${formData.split_seed}
        output_yaml: "${version.export_path}/dataset.yaml"
    - augmentations:
        horizontal_flip: ${formData.fliplr}
        vertical_flip: ${formData.flipud}
        random_rotation: ${formData.degrees}°
    - model_training:
        architecture: "YOLOv8-seg"
        epochs: ${formData.epochs}
        batch_size: ${formData.batch}
        image_size: ${formData.imgsz}px
        mock_execution: ${formData.mock}`;
      setCompiledBlueprint(yamlStr);
    } catch (err) {
      console.error("Failed to generate blueprint:", err);
    }
  };

  const onExecutePipeline = () => {
    if (!compiledVersion) return;
    handleExecutePipelineRun(formData, compiledVersion.id);
  };

  // FiftyOne Curation Filtering Logic
  const filteredEmbeddings = embeddings.filter((emb: any) => {
    if (searchId && !emb.image_id.toLowerCase().includes(searchId.toLowerCase())) return false;
    const gsd = emb.metadata?.gsd || 0.3;
    if (gsd < minGsd || gsd > maxGsd) return false;
    const targets = emb.labels?.length || 0;
    if (targets < minTargets) return false;
    
    // Geospatial Sensor Type Filter
    const type = emb.metadata?.sensor_type || 'Optical (WorldView-3)';
    if (sensorType === 'optical' && !type.includes('Optical')) return false;
    if (sensorType === 'sar' && !type.includes('SAR')) return false;
    
    // Space-Based SAR parameter filters
    if (sensorType === 'sar' || type.includes('SAR')) {
      const pol = emb.metadata?.sar_polarization || 'N/A';
      if (sarPolarization !== 'all' && pol !== sarPolarization) return false;
      
      const inc = emb.metadata?.incidence_angle || 0;
      if (inc < minIncidence || inc > maxIncidence) return false;
    }
    
    return true;
  });

  // Calculate statistics
  const totalAnnotatedTargets = embeddings.reduce((acc: number, cur: any) => acc + (cur.labels?.length || 0), 0);
  const avgGsd = embeddings.length > 0 ? (embeddings.reduce((acc: number, cur: any) => acc + (cur.metadata?.gsd || 0.3), 0) / embeddings.length).toFixed(3) : 'N/A';
  const utmZones = Array.from(new Set(embeddings.map((e: any) => e.metadata?.utm_zone || '18N'))).join(', ');

  const totalChips = embeddings.length || 60;
  const trainCount = Math.round(totalChips * formData.train_split);
  const valCount = totalChips - trainCount;

  const handleCardClick = (emb: any) => {
    setSelectedEmbed(emb);
    setSelectedImageDetails(emb);
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setUploadingStatus('Ingesting ZIP and extracting geospatial imagery...');
    setUploadError('');

    const formDataUpload = new FormData();
    formDataUpload.append('file', uploadFile);
    formDataUpload.append('name', uploadName);
    formDataUpload.append('description', uploadDesc);
    formDataUpload.append('task_type', uploadTask);

    try {
      const response = await fetch(`${API_BASE}/datasets/upload`, {
        method: 'POST',
        body: formDataUpload
      });

      const result = await response.json();
      if (response.ok && result.status === 'success') {
        setUploadingStatus('Running ResNet18 feature extraction & PCA clustering...');
        
        await new Promise(resolve => setTimeout(resolve, 1500));
        await onDatasetUpload();
        
        const newDsId = result.dataset.id;
        setActiveDatasetId(newDsId);
        await fetchDatasetEmbeddings(newDsId);
        await fetchDatasetVersions(newDsId);
        
        setUploadingStatus('');
        setUploadModalOpen(false);
        setUploadName('');
        setUploadDesc('');
        setUploadFile(null);
      } else {
        setUploadError(result.detail || 'Failed to parse ZIP archive.');
        setUploadingStatus('');
      }
    } catch (err: any) {
      setUploadError(err.message || 'Network connection failed.');
      setUploadingStatus('');
    }
  };

  const renderScatterPlot = () => {
    if (embeddings.length === 0) return null;
    return embeddings.map((pt: any) => {
      const svgX = ((pt.x + 10) / 20) * 360 + 20;
      const svgY = ((pt.y + 10) / 20) * 360 + 20;
      
      const isSelected = selectedEmbed && selectedEmbed.image_id === pt.image_id;
      const isFilteredOut = !filteredEmbeddings.find((f: any) => f.image_id === pt.image_id);
      
      let dotColor = 'rgba(0, 114, 255, 0.7)'; // Default blue
      const sceneType = pt.metadata?.scene_type;
      if (sceneType === 'runway_intersection') dotColor = 'rgba(0, 242, 254, 0.8)';
      if (sceneType === 'taxiway') dotColor = 'rgba(0, 255, 137, 0.8)';
      if (sceneType === 'cargo_ramp') dotColor = 'rgba(255, 153, 0, 0.8)';
      
      return (
        <circle 
          key={pt.image_id}
          cx={svgX}
          cy={svgY}
          r={isSelected ? 7 : 4}
          fill={isSelected ? '#fff' : dotColor}
          stroke={isSelected ? 'var(--accent-cyan)' : 'none'}
          strokeWidth={2}
          opacity={isFilteredOut ? 0.15 : 1}
          style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
          onClick={() => {
            setSelectedEmbed(pt);
          }}
        />
      );
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }}>
      
      {/* Step Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        opacity: selectedImageDetails ? 0.4 : 1,
        pointerEvents: selectedImageDetails ? 'none' : 'auto',
        transition: 'all 0.3s ease'
      }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>STEP 1: PIPELINE STUDIO</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Curate dataset chips, annotate target aircraft, configure pipeline splits & augmentations, and compile the YAML model training blueprint.</p>
        </div>
        
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label>ACTIVE DATASET SOURCE</label>
            <select 
              value={activeDatasetId} 
              onChange={e => {
                const dsId = e.target.value;
                setActiveDatasetId(dsId);
                fetchDatasetEmbeddings(dsId);
                fetchDatasetVersions(dsId);
              }}
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '6px 10px', borderRadius: '4px', outline: 'none', width: '220px' }}
            >
              {datasets && datasets.map((ds: any) => (
                <option key={ds.id} value={ds.id}>{ds.name} ({ds.sample_size} chips)</option>
              ))}
            </select>
          </div>
          
          <button 
            onClick={() => setUploadModalOpen(true)}
            className="btn-tactical btn-tactical-active"
            style={{ padding: '8px 14px', marginTop: '16px' }}
          >
            <Plus style={{ width: '14px', height: '14px' }} />
            IMPORT DATASET
          </button>
        </div>
      </div>

      {/* Grid: Stats & Telemetry */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(4, 1fr)', 
        gap: '10px',
        opacity: selectedImageDetails ? 0.4 : 1,
        pointerEvents: selectedImageDetails ? 'none' : 'auto',
        transition: 'all 0.3s ease'
      }}>
        {[
          { label: 'TOTAL SATELLITE CHIPS', value: embeddings.length, desc: 'Ingested worldview-3 frames' },
          { label: 'ANNOTATED TARGETS', value: totalAnnotatedTargets, desc: 'Aircraft polygons cataloged' },
          { label: 'AVERAGE CHIP RESOLUTION', value: `${avgGsd}m`, desc: 'Ground Sample Distance (GSD)' },
          { label: 'COORDINATE PROJECTIONS', value: utmZones || '18N', desc: 'Active UTM zones detected' },
        ].map((stat, idx) => (
          <div key={idx} className="glass-recessed" style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{stat.label}</span>
            <div style={{ fontSize: '1.2rem', fontFamily: 'var(--font-display)', fontWeight: 600, color: 'var(--accent-cyan)', margin: '2px 0' }}>{stat.value}</div>
            <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{stat.desc}</span>
          </div>
        ))}
      </div>
      
      {/* Sub-navigation Switcher */}
      <div style={{ 
        display: 'flex', 
        gap: '8px', 
        borderBottom: '1px solid var(--border-color)', 
        paddingBottom: '10px',
        opacity: selectedImageDetails ? 0.4 : 1,
        pointerEvents: selectedImageDetails ? 'none' : 'auto',
        transition: 'all 0.3s ease'
      }}>
        <button 
          onClick={() => setControlTab('curate')} 
          className={`btn-tactical ${controlTab === 'curate' ? 'btn-tactical-active border-glow-cyan' : ''}`}
          style={{ fontSize: '0.7rem', padding: '6px 12px' }}
        >
          1. CURATE & LABEL
        </button>
        <button 
          onClick={() => setControlTab('build')} 
          className={`btn-tactical ${controlTab === 'build' ? 'btn-tactical-active border-glow-cyan' : ''}`}
          style={{ fontSize: '0.7rem', padding: '6px 12px' }}
        >
          2. BUILD PIPELINE BLUE-PRINT
        </button>
      </div>

      {controlTab === 'curate' ? (
        // CURATE VIEW (Full Width, Curation Filters integrated horizontally)
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
          
          {/* Horizontal Curation Filter Strip */}
          <div className="glass-panel" style={{ 
            padding: '12px 16px', 
            display: 'flex', 
            flexWrap: 'wrap', 
            gap: '16px', 
            alignItems: 'center', 
            background: 'rgba(0,0,0,0.25)',
            opacity: selectedImageDetails ? 0.4 : 1,
            pointerEvents: selectedImageDetails ? 'none' : 'auto',
            transition: 'all 0.3s ease'
          }}>
            
            {/* Filter Group: Search & Sensor */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                <label>SEARCH BY ID</label>
                <input 
                  type="text" 
                  value={searchId}
                  onChange={e => setSearchId(e.target.value)}
                  placeholder="e.g. chip_01"
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '4px 8px', borderRadius: '4px', outline: 'none', fontSize: '0.7rem', width: '110px' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                <label>SENSOR TYPE</label>
                <select 
                  value={sensorType}
                  onChange={e => setSensorType(e.target.value)}
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '4px 8px', borderRadius: '4px', outline: 'none', fontSize: '0.7rem' }}
                >
                  <option value="all">All Sensors</option>
                  <option value="optical">Optical (WorldView)</option>
                  <option value="sar">SAR (Space Radar)</option>
                </select>
              </div>
            </div>

            {/* SAR specific options (inline and compact if selected) */}
            {(sensorType === 'sar' || sensorType === 'all') && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderLeft: '1px solid rgba(255,255,255,0.08)', paddingLeft: '12px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                  <label>SAR POLARIZATION</label>
                  <select 
                    value={sarPolarization}
                    onChange={e => setSarPolarization(e.target.value)}
                    style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '4px 8px', borderRadius: '4px', outline: 'none', fontSize: '0.7rem' }}
                  >
                    <option value="all">All Polarizations</option>
                    <option value="VV">VV (Co-polarization)</option>
                    <option value="VH">VH (Cross-polarization)</option>
                    <option value="VV+VH">VV+VH (Dual-pol)</option>
                  </select>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', width: '120px' }}>
                    <label>INCIDENCE RANGE</label>
                    <span>{minIncidence.toFixed(0)}°-{maxIncidence.toFixed(0)}°</span>
                  </div>
                  <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                    <input 
                      type="range"
                      min="15.0"
                      max="45.0"
                      step="1.0"
                      value={minIncidence}
                      onChange={e => setMinIncidence(parseFloat(e.target.value))}
                      style={{ accentColor: 'var(--accent-cyan)', width: '60px', height: '6px' }}
                    />
                    <input 
                      type="range"
                      min="15.0"
                      max="45.0"
                      step="1.0"
                      value={maxIncidence}
                      onChange={e => setMaxIncidence(parseFloat(e.target.value))}
                      style={{ accentColor: 'var(--accent-cyan)', width: '60px', height: '6px' }}
                    />
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                  <input 
                    type="checkbox" 
                    id="sar-backscatter-simulate-inline"
                    checked={simulateBackscatter} 
                    onChange={e => setSimulateBackscatter(e.target.checked)}
                    style={{ cursor: 'pointer', accentColor: 'var(--accent-cyan)' }}
                  />
                  <label htmlFor="sar-backscatter-simulate-inline" style={{ cursor: 'pointer', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                    SIMULATE SAR
                  </label>
                </div>
              </div>
            )}

            {/* General Filter options */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderLeft: '1px solid rgba(255,255,255,0.08)', paddingLeft: '12px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '110px' }}>
                  <label>MAX RESOLUTION</label>
                  <span>{maxGsd.toFixed(1)}m</span>
                </div>
                <input 
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={maxGsd}
                  onChange={e => setMaxGsd(parseFloat(e.target.value))}
                  style={{ accentColor: 'var(--accent-cyan)', width: '110px', height: '6px', marginTop: '6px' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100px' }}>
                  <label>MIN TARGETS</label>
                  <span>{minTargets}</span>
                </div>
                <input 
                  type="range"
                  min="0"
                  max="8"
                  step="1"
                  value={minTargets}
                  onChange={e => setMinTargets(parseInt(e.target.value) || 0)}
                  style={{ accentColor: 'var(--accent-cyan)', width: '100px', height: '6px', marginTop: '6px' }}
                />
              </div>
            </div>

            {/* View & action triggers */}
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
              <button 
                onClick={() => setShowClusterMap(!showClusterMap)}
                className="btn-tactical border-glow-cyan"
                style={{ padding: '6px 12px', fontSize: '0.65rem' }}
              >
                {showClusterMap ? 'Show Image Grid' : 'Show PCA Embeddings'}
              </button>
              {selectedEmbed && (
                <button
                  onClick={() => setSelectedImageDetails(selectedEmbed)}
                  className="btn-tactical btn-tactical-active"
                  style={{ padding: '6px 12px', fontSize: '0.65rem' }}
                >
                  Label Chip <ArrowRight style={{ width: '12px', height: '12px' }} />
                </button>
              )}
            </div>

          </div>

          {/* Main workspace for Grid/Map (100% width) */}
          <div className="glass-recessed" style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {selectedImageDetails ? (
              <AnnotateTargetsView 
                selectedEmbed={selectedEmbed}
                setSelectedEmbed={setSelectedEmbed}
                selectedImageDetails={selectedImageDetails}
                setSelectedImageDetails={setSelectedImageDetails}
                activeDatasetId={activeDatasetId}
                fetchDatasetEmbeddings={fetchDatasetEmbeddings}
              />
            ) : showClusterMap ? (
              // Embeddings map
              <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '16px', overflow: 'hidden' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', overflow: 'hidden' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                    <span>PCA CLUSTERING DEVIATION (RESNET-18 EXTRACTED)</span>
                    <span>Filtered: {filteredEmbeddings.length} / {embeddings.length}</span>
                  </div>
                  
                  <div style={{ flex: 1, border: '1px solid var(--border-color)', borderRadius: '8px', background: 'rgba(0,0,0,0.4)', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <svg width="400" height="400" style={{ maxWidth: '100%', maxHeight: '100%' }}>
                      <line x1="200" y1="10" x2="200" y2="390" stroke="rgba(255,255,255,0.05)" />
                      <line x1="10" y1="200" x2="390" y2="200" stroke="rgba(255,255,255,0.05)" />
                      {renderScatterPlot()}
                    </svg>
                    
                    <div style={{ position: 'absolute', bottom: '10px', left: '10px', display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '0.55rem', fontFamily: 'var(--font-mono)', background: 'rgba(0,0,0,0.6)', padding: '6px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <span style={{ color: 'rgba(255, 153, 0, 0.8)' }}>● Cargo Ramp Airfields</span>
                      <span style={{ color: 'rgba(0, 242, 254, 0.8)' }}>● Runway Intersections</span>
                      <span style={{ color: 'rgba(0, 255, 137, 0.8)' }}>● Taxiways / Aprons</span>
                      <span style={{ color: 'rgba(0, 114, 255, 0.7)' }}>● Other Features</span>
                    </div>
                  </div>
                </div>

                {/* Selected node card detail panel */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', overflowY: 'auto' }}>
                  <h4 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '6px' }}>
                    CHIP METADATA DETAILS
                  </h4>
                  {selectedEmbed ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div style={{ height: '180px', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-color)', position: 'relative' }}>
                        <img 
                          src={getImageUrl(activeDatasetId, selectedEmbed.image_id)} 
                          alt="Selected chip preview"
                          style={{ 
                            width: '100%', 
                            height: '100%', 
                            objectFit: 'cover',
                            filter: simulateBackscatter && selectedEmbed.metadata?.sensor_type === 'SAR'
                              ? 'grayscale(1) contrast(2.2) brightness(1.2) sepia(0.5) hue-rotate(140deg)'
                              : 'none',
                            transition: 'filter 0.3s ease'
                          }}
                        />
                        <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }} viewBox="0 0 512 512">
                          {selectedEmbed.labels && selectedEmbed.labels.map((lbl: any, idx: number) => {
                            const bbox = lbl.bbox || [0,0,0,0];
                            const color = CLASS_COLORS[lbl.class_id] || '#00f2fe';
                            return (
                              <rect 
                                key={idx}
                                x={bbox[0] * 512}
                                y={bbox[1] * 512}
                                width={Math.max(10, (bbox[2] - bbox[0]) * 512)}
                                height={Math.max(10, (bbox[3] - bbox[1]) * 512)}
                                fill="none"
                                stroke={color}
                                strokeWidth={3}
                              />
                            );
                          })}
                        </svg>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                        <div className="glass-panel" style={{ padding: '8px', background: 'none' }}>
                          <div style={{ color: 'var(--text-secondary)' }}>IMAGE ID</div>
                          <div style={{ color: '#fff', fontWeight: 600 }}>{selectedEmbed.image_id}</div>
                        </div>
                        <div className="glass-panel" style={{ padding: '8px', background: 'none' }}>
                          <div style={{ color: 'var(--text-secondary)' }}>SITE CODE</div>
                          <div style={{ color: '#fff', fontWeight: 600 }}>{selectedEmbed.metadata?.airport_code || 'KIAD'}</div>
                        </div>
                        <div className="glass-panel" style={{ padding: '8px', background: 'none' }}>
                          <div style={{ color: 'var(--text-secondary)' }}>RESOLUTION</div>
                          <div style={{ color: '#fff', fontWeight: 600 }}>{selectedEmbed.metadata?.gsd || 0.3}m GSD</div>
                        </div>
                        <div className="glass-panel" style={{ padding: '8px', background: 'none' }}>
                          <div style={{ color: 'var(--text-secondary)' }}>TARGETS</div>
                          <div style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>{selectedEmbed.labels?.length || 0} Aircraft</div>
                        </div>
                        <div className="glass-panel" style={{ padding: '8px', background: 'none' }}>
                          <div style={{ color: 'var(--text-secondary)' }}>SENSOR TYPE</div>
                          <div style={{ color: selectedEmbed.metadata?.sensor_type === 'SAR' ? 'var(--accent-cyan)' : '#fff', fontWeight: 600 }}>
                            {selectedEmbed.metadata?.sensor_type || 'Optical (WorldView)'}
                          </div>
                        </div>
                        <div className="glass-panel" style={{ padding: '8px', background: 'none' }}>
                          <div style={{ color: 'var(--text-secondary)' }}>POL / LOOK ANGLE</div>
                          <div style={{ color: '#fff', fontWeight: 600 }}>
                            {selectedEmbed.metadata?.sensor_type === 'SAR'
                              ? `${selectedEmbed.metadata?.sar_polarization} / ${selectedEmbed.metadata?.incidence_angle}°`
                              : 'N/A'}
                          </div>
                        </div>
                      </div>

                      <div style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '6px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <div style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>BOUNDS FOOTPRINT (E / N)</div>
                        <div>Min: E {selectedEmbed.metadata?.bounds?.min_easting?.toFixed(1) || 296000}m / N {selectedEmbed.metadata?.bounds?.min_northing?.toFixed(1) || 4312000}m</div>
                        <div>Max: E {selectedEmbed.metadata?.bounds?.max_easting?.toFixed(1) || 296256}m / N {selectedEmbed.metadata?.bounds?.max_northing?.toFixed(1) || 4312256}m</div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textAlign: 'center', marginTop: '40px' }}>
                      Click a node on the cluster map to analyze coordinates.
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', height: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  <span>FIFTYONE IMAGE GRIDS (WorldView Imagery)</span>
                  <span>Filtered: {filteredEmbeddings.length} / {embeddings.length} chips</span>
                </div>
                
                <div style={{ flex: 1, overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '12px' }}>
                  {filteredEmbeddings.map((emb: any) => {
                    const numTargets = emb.labels?.length || 0;
                    const isSelected = selectedEmbed && selectedEmbed.image_id === emb.image_id;
                    const isHovered = hoveredCardIdx === emb.image_id;
                    return (
                      <div 
                        key={emb.image_id} 
                        onClick={() => handleCardClick(emb)}
                        onMouseEnter={() => setHoveredCardIdx(emb.image_id)}
                        onMouseLeave={() => setHoveredCardIdx(null)}
                        className="glass-panel" 
                        style={{ 
                          padding: '6px', 
                          cursor: 'pointer', 
                          display: 'flex', 
                          flexDirection: 'column', 
                          gap: '6px',
                          borderColor: isSelected ? 'var(--accent-cyan)' : isHovered ? 'rgba(255,255,255,0.2)' : 'var(--border-color)',
                          boxShadow: isSelected ? '0 0 12px rgba(0, 242, 254, 0.15)' : 'none',
                          transition: 'all 0.15s ease'
                        }}
                      >
                        <div style={{ height: '90px', background: '#000', borderRadius: '6px', overflow: 'hidden', position: 'relative' }}>
                          <img 
                            src={getImageUrl(activeDatasetId, emb.image_id)} 
                            alt={emb.image_id} 
                            style={{ 
                              width: '100%', 
                              height: '100%', 
                              objectFit: 'cover',
                              filter: simulateBackscatter && emb.metadata?.sensor_type === 'SAR'
                                ? 'grayscale(1) contrast(2.2) brightness(1.2) sepia(0.5) hue-rotate(140deg)'
                                : 'none',
                              transition: 'filter 0.3s ease'
                            }}
                          />
                          
                          {(isHovered || isSelected) && emb.labels && (
                            <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }} viewBox="0 0 512 512">
                              {emb.labels.map((lbl: any, idx: number) => {
                                const bbox = lbl.bbox || [0,0,0,0];
                                const color = CLASS_COLORS[lbl.class_id] || '#00f2fe';
                                return (
                                  <rect 
                                    key={idx}
                                    x={bbox[0] * 512}
                                    y={bbox[1] * 512}
                                    width={Math.max(12, (bbox[2] - bbox[0]) * 512)}
                                    height={Math.max(12, (bbox[3] - bbox[1]) * 512)}
                                    fill="none"
                                    stroke={color}
                                    strokeWidth={4}
                                  />
                                );
                              })}
                            </svg>
                          )}
                          
                          <div style={{ position: 'absolute', top: '4px', right: '4px', background: 'rgba(0,0,0,0.6)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: numTargets > 0 ? 'var(--accent-orange)' : 'var(--text-secondary)' }}>
                            {numTargets} aircraft
                          </div>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)', padding: '0 2px' }}>
                          <span style={{ color: '#fff', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{emb.image_id}</span>
                          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.55rem' }}>
                            <span>GSD: {emb.metadata?.gsd || 0.3}m</span>
                            <span style={{ color: emb.metadata?.sensor_type === 'SAR' ? 'var(--accent-cyan)' : 'var(--text-secondary)', fontWeight: 600 }}>
                              {emb.metadata?.sensor_type === 'SAR' ? `SAR (${emb.metadata?.sar_polarization})` : 'OPTICAL'}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {filteredEmbeddings.length === 0 && (
                    <div style={{ gridColumn: '1/-1', padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
                      No sat-chips match the curation filters selected. Try relaxing filters.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        // BUILD VIEW (centered card layout, full width)
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'flex-start', overflowY: 'auto', padding: '10px 0' }}>
          <div className="glass-panel" style={{ width: '100%', maxWidth: '960px', display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px', padding: '24px', background: 'rgba(5, 8, 20, 0.45)' }}>
            
            {/* Left side: parameters form */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.9rem', color: 'var(--accent-cyan)', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', margin: 0, letterSpacing: '0.05em' }}>
                PIPELINE CONFIGURATION SCHEMATIC
              </h3>
              
              <form onSubmit={handleGenerateBlueprintClick} style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <label>EXPERIMENT/RUN ID</label>
                  <input 
                    type="text" 
                    value={formData.name}
                    onChange={e => handleInputChange('name', e.target.value)}
                    style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label>PIPELINE TAG</label>
                    <input 
                      type="text" 
                      value={formData.version_tag}
                      onChange={e => handleInputChange('version_tag', e.target.value)}
                      style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                    />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label>EPOCHS</label>
                    <input 
                      type="number" 
                      value={formData.epochs}
                      onChange={e => handleInputChange('epochs', parseInt(e.target.value) || 3)}
                      style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                    />
                  </div>
                </div>

                {/* Split Settings */}
                <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.05)', padding: '12px', borderRadius: '6px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ color: 'var(--accent-cyan)', fontWeight: 600, fontSize: '0.65rem' }}>STAGE 2: SPLITTING</div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <label>TRAIN SPLIT</label>
                      <span>{Math.round(formData.train_split * 100)}%</span>
                    </div>
                    <input 
                      type="range"
                      min="0.5"
                      max="0.9"
                      step="0.05"
                      value={formData.train_split}
                      onChange={e => handleInputChange('train_split', parseFloat(e.target.value))}
                      style={{ accentColor: 'var(--accent-cyan)', width: '100%' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                      <span>Train Count: {trainCount} chips</span>
                      <span>Val Count: {valCount} chips</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label>PARTITION SEED</label>
                    <input 
                      type="number" 
                      value={formData.split_seed}
                      onChange={e => handleInputChange('split_seed', parseInt(e.target.value) || 42)}
                      style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                    />
                  </div>
                </div>

                {/* Custom Augmentations */}
                <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.05)', padding: '12px', borderRadius: '6px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ color: 'var(--accent-cyan)', fontWeight: 600, fontSize: '0.65rem' }}>STAGE 3: DATA AUGMENTATIONS</div>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <label>HORIZONTAL FLIP (fliplr)</label>
                    <input 
                      type="checkbox"
                      checked={formData.fliplr}
                      onChange={e => handleInputChange('fliplr', e.target.checked)}
                      style={{ accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
                    />
                  </div>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <label>VERTICAL FLIP (flipud)</label>
                    <input 
                      type="checkbox"
                      checked={formData.flipud}
                      onChange={e => handleInputChange('flipud', e.target.checked)}
                      style={{ accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
                    />
                  </div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <label>ROTATION</label>
                      <span>{formData.degrees}°</span>
                    </div>
                    <input 
                      type="range"
                      min="0"
                      max="180"
                      value={formData.degrees}
                      onChange={e => handleInputChange('degrees', parseFloat(e.target.value))}
                      style={{ accentColor: 'var(--accent-cyan)', width: '100%' }}
                    />
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label>BATCH SIZE</label>
                    <select 
                      value={formData.batch}
                      onChange={e => handleInputChange('batch', parseInt(e.target.value) || 2)}
                      style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                    >
                      <option value={2}>2</option>
                      <option value={4}>4</option>
                      <option value={8}>8</option>
                    </select>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <label>IMAGE SIZE</label>
                    <select 
                      value={formData.imgsz}
                      onChange={e => handleInputChange('imgsz', parseInt(e.target.value) || 512)}
                      style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                    >
                      <option value={256}>256px</option>
                      <option value={512}>512px</option>
                      <option value={640}>640px</option>
                    </select>
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderTop: '1px dashed var(--border-color)' }}>
                  <label htmlFor="pipeline-run-mock" style={{ cursor: 'pointer' }}>MOCK PIPELINE RUN</label>
                  <input 
                    type="checkbox"
                    id="pipeline-run-mock"
                    checked={formData.mock}
                    onChange={e => handleInputChange('mock', e.target.checked)}
                    style={{ accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
                  />
                </div>

                {isGeneratingBlueprint ? (
                  <button 
                    type="button" 
                    disabled={true}
                    className="btn-tactical btn-tactical-active" 
                    style={{ width: '100%', justifyContent: 'center', padding: '12px', marginTop: '10px' }}
                  >
                    <RefreshCw className="spin" style={{ width: '14px', height: '14px' }} /> GENERATING BLUEPRINT...
                  </button>
                ) : !compiledBlueprint ? (
                  <button 
                    type="submit" 
                    className="btn-tactical btn-tactical-active border-glow-cyan" 
                    style={{ width: '100%', justifyContent: 'center', padding: '12px', marginTop: '10px' }}
                  >
                    <Cpu style={{ width: '14px', height: '14px' }} /> GENERATE BLUEPRINT SCHEMATIC
                  </button>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '10px' }}>
                    <button 
                      type="button"
                      onClick={onExecutePipeline}
                      className="btn-tactical btn-tactical-active border-glow-green" 
                      style={{ width: '100%', justifyContent: 'center', padding: '12px', color: 'var(--accent-green)', borderColor: 'var(--accent-green)' }}
                    >
                      <Play style={{ width: '14px', height: '14px' }} /> LAUNCH PIPELINE RUN
                    </button>
                    
                    <button 
                      type="button"
                      onClick={() => { setCompiledBlueprint(null); setCompiledVersion(null); }}
                      className="btn-tactical" 
                      style={{ width: '100%', justifyContent: 'center', padding: '8px', fontSize: '0.75rem' }}
                    >
                      RE-CONFIG PIPELINE
                    </button>
                  </div>
                )}
              </form>
            </div>

            {/* Right side: Blueprint YAML preview & launch info */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', borderLeft: '1px solid var(--border-color)', paddingLeft: '24px' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.9rem', color: 'var(--accent-cyan)', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', margin: 0, letterSpacing: '0.05em' }}>
                YAML BLUEPRINT PREVIEW
              </h3>

              {compiledBlueprint ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: '100%' }}>
                  <div style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span className="led-indicator green"></span>
                    <span>BLUEPRINT SCHEMATIC COMPILED READY</span>
                  </div>
                  <pre style={{ 
                    flex: 1,
                    background: 'rgba(0,0,0,0.9)', 
                    border: '1px solid var(--border-color)', 
                    borderRadius: '8px', 
                    padding: '16px', 
                    fontSize: '0.75rem', 
                    fontFamily: 'var(--font-mono)', 
                    color: 'var(--accent-green)',
                    overflow: 'auto',
                    lineHeight: '1.4',
                    maxHeight: '400px'
                  }}>
                    {compiledBlueprint}
                  </pre>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', lineHeight: '1.4', margin: 0 }}>
                    Confirm the parameters are correct and launch the run. This will register a version in our dataset manager and spawn an accelerator worker job to execute training.
                  </p>
                </div>
              ) : (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', border: '1px dashed var(--border-color)', borderRadius: '8px', padding: '40px', textAlign: 'center' }}>
                  <Cpu style={{ width: '40px', height: '40px', color: 'var(--text-muted)', marginBottom: '16px' }} />
                  <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>No compiled blueprint available.</span>
                  <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '8px', maxWidth: '300px' }}>
                    Adjust parameters in the left pane and click "Generate Blueprint" to preview the YAML file structure.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ZIP Ingestion Modal */}
      {uploadModalOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(4, 6, 15, 0.85)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div className="glass-panel border-glow-cyan" style={{ width: '450px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', color: '#fff' }}>
              INGEST NEW GEOSPATIAL DATASET
            </h3>
            
            <form onSubmit={handleUploadSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label>DATASET NAME</label>
                <input 
                  type="text" 
                  value={uploadName}
                  onChange={e => setUploadName(e.target.value)}
                  placeholder="e.g. RarePlanes Cargo Ramp"
                  required
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label>DESCRIPTION</label>
                <textarea 
                  value={uploadDesc}
                  onChange={e => setUploadDesc(e.target.value)}
                  placeholder="Describe dataset location, GSD, target categories..."
                  rows={3}
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none', resize: 'none' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label>TARGET OBJECT TASK</label>
                <select 
                  value={uploadTask}
                  onChange={e => setUploadTask(e.target.value)}
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                >
                  <option value="instance_segmentation">Instance Segmentation (Masks)</option>
                  <option value="object_detection">Object Detection (Boxes)</option>
                </select>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label>GEOSPATIAL CHIPS ARCHIVE (.ZIP)</label>
                <input 
                  type="file" 
                  accept=".zip"
                  onChange={e => {
                    if (e.target.files && e.target.files.length > 0) {
                      setUploadFile(e.target.files[0]);
                    }
                  }}
                  required
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '8px 12px', borderRadius: '4px', outline: 'none' }}
                />
                <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>ZIP must contain images/ folder with PNG/JPG/JPEG files.</span>
              </div>

              {uploadError && (
                <div style={{ color: 'var(--accent-red)', fontSize: '0.7rem' }}>
                  Error: {uploadError}
                </div>
              )}

              {uploadingStatus && (
                <div style={{ color: 'var(--accent-cyan)', fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <RefreshCw className="spin" style={{ width: '12px', height: '12px' }} />
                  {uploadingStatus}
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '8px' }}>
                <button 
                  type="button" 
                  onClick={() => {
                    setUploadModalOpen(false);
                    setUploadError('');
                    setUploadingStatus('');
                  }}
                  className="btn-tactical"
                >
                  CANCEL
                </button>
                <button 
                  type="submit" 
                  disabled={!!uploadingStatus}
                  className="btn-tactical btn-tactical-active"
                >
                  INGEST ARCHIVE
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// VIEW COMPONENT: 2. ANNOTATE TARGETS (CVAT-Style)
// ============================================
function AnnotateTargetsView({
  selectedEmbed: _selectedEmbed,
  setSelectedEmbed: _setSelectedEmbed,
  selectedImageDetails,
  setSelectedImageDetails,
  activeDatasetId,
  fetchDatasetEmbeddings
}: any) {
  const [drawingMode, setDrawingMode] = useState<'polygon' | 'bbox'>('polygon');
  const [activeClassId, setActiveClassId] = useState<number>(0);
  const [editLabels, setEditLabels] = useState<any[]>([]);
  const [tempPoints, setTempPoints] = useState<any[]>([]);
  const [selectedLabelIdx, setSelectedLabelIdx] = useState<number | null>(null);
  const [hoveredLabelIdx, setHoveredLabelIdx] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isDrawingBbox, setIsDrawingBbox] = useState(false);
  const [bboxStart, setBboxStart] = useState<[number, number] | null>(null);
  const [hiddenLayers, setHiddenLayers] = useState<Record<number, boolean>>({});
  const [saveStatus, setSaveStatus] = useState<string>('');

  // AI Auto-Labeling Suggestions Queue
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [loadingAutoLabel, setLoadingAutoLabel] = useState(false);

  useEffect(() => {
    if (selectedImageDetails) {
      setEditLabels(selectedImageDetails.labels || []);
      setTempPoints([]);
      setSelectedLabelIdx(null);
      setSaveStatus('');
      setSuggestions([]); // Clear suggestions when moving to another image
    }
  }, [selectedImageDetails]);

  const handleAutoLabel = async () => {
    if (!selectedImageDetails) return;
    setLoadingAutoLabel(true);
    setSaveStatus('Running pipeline target inference...');
    try {
      const res = await fetch(`${API_BASE}/datasets/${activeDatasetId}/images/${selectedImageDetails.image_id}/auto-label`, {
        method: 'POST'
      });
      const data = await res.json();
      setLoadingAutoLabel(false);
      if (res.ok && data.status === 'success') {
        setSuggestions(data.suggestions || []);
        setSaveStatus(`Model inference generated ${data.suggestions?.length || 0} suggestions.`);
      } else {
        setSaveStatus(`Auto-label failed: ${data.detail || 'Model pipeline error'}`);
      }
    } catch (e: any) {
      setLoadingAutoLabel(false);
      setSaveStatus(`Auto-label network error: ${e.message}`);
    }
  };

  const approveSuggestion = (suggId: string) => {
    const sugg = suggestions.find(s => s.id === suggId);
    if (!sugg) return;
    setEditLabels(prev => [...prev, {
      class_id: sugg.class_id,
      polygon: sugg.polygon,
      bbox: sugg.bbox
    }]);
    setSuggestions(prev => prev.filter(s => s.id !== suggId));
  };

  const rejectSuggestion = (suggId: string) => {
    setSuggestions(prev => prev.filter(s => s.id !== suggId));
  };

  const approveAllSuggestions = () => {
    const newLabels = suggestions.map(s => ({
      class_id: s.class_id,
      polygon: s.polygon,
      bbox: s.bbox
    }));
    setEditLabels(prev => [...prev, ...newLabels]);
    setSuggestions([]);
    setSaveStatus(`Approved all suggested detections.`);
  };

  // Keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setTempPoints([]);
        setIsDrawingBbox(false);
        setBboxStart(null);
      } else if (e.key === 'Backspace' && tempPoints.length > 0) {
        setTempPoints(prev => prev.slice(0, -1));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tempPoints]);

  const handleCanvasMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setMousePos({ x, y });
  };

  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = parseFloat(((e.clientX - rect.left) / rect.width).toFixed(5));
    const y = parseFloat(((e.clientY - rect.top) / rect.height).toFixed(5));

    if (drawingMode === 'polygon') {
      // If clicked near the first vertex and we have >= 3 points, close it
      if (tempPoints.length >= 3) {
        const d = Math.hypot(tempPoints[0][0] - x, tempPoints[0][1] - y);
        if (d < 0.03) {
          completePolygon();
          return;
        }
      }
      setTempPoints([...tempPoints, [x, y]]);
    } else if (drawingMode === 'bbox') {
      if (!isDrawingBbox) {
        setIsDrawingBbox(true);
        setBboxStart([x, y]);
      } else if (bboxStart) {
        setIsDrawingBbox(false);
        const x1 = Math.min(bboxStart[0], x);
        const y1 = Math.min(bboxStart[1], y);
        const x2 = Math.max(bboxStart[0], x);
        const y2 = Math.max(bboxStart[1], y);
        const polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
        const bbox = [x1, y1, x2, y2];
        setEditLabels([...editLabels, { class_id: activeClassId, polygon, bbox }]);
        setBboxStart(null);
      }
    }
  };

  const completePolygon = () => {
    if (tempPoints.length < 3) return;
    const xs = tempPoints.map(p => p[0]);
    const ys = tempPoints.map(p => p[1]);
    const bbox = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    setEditLabels([...editLabels, { class_id: activeClassId, polygon: tempPoints, bbox }]);
    setTempPoints([]);
  };

  const deleteLayer = (idx: number) => {
    const updated = editLabels.filter((_, i) => i !== idx);
    setEditLabels(updated);
    setSelectedLabelIdx(null);
  };

  const toggleLayerVisibility = (idx: number) => {
    setHiddenLayers({
      ...hiddenLayers,
      [idx]: !hiddenLayers[idx]
    });
  };

  const handleSaveLabels = async () => {
    setSaveStatus('Saving label configuration...');
    try {
      const res = await fetch(`${API_BASE}/datasets/${activeDatasetId}/images/${selectedImageDetails.image_id}/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels: editLabels })
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setSaveStatus('Configuration saved successfully.');
        await fetchDatasetEmbeddings(activeDatasetId);
      } else {
        setSaveStatus(`Error: ${data.detail || 'Could not write label changes'}`);
      }
    } catch (e: any) {
      setSaveStatus(`Network error: ${e.message}`);
    }
  };

  if (!selectedImageDetails) {
    return (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        <Compass style={{ width: '40px', height: '40px' }} />
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', marginTop: '12px' }}>NO IMAGE CHIP SELECT</div>
        <button onClick={() => setSelectedImageDetails(null)} className="btn-tactical btn-tactical-active" style={{ marginTop: '12px' }}>
          Select Frame from Grid
        </button>
      </div>
    );
  }

  const isSaveDisabled = saveStatus === 'Saving label configuration...';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }}>
      
      {/* breadcrumbs */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>STEP 2: ANNOTATE TARGETS</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Use CVAT-style polygon mapping or box drawing to label tactical aviation targets.</p>
        </div>
        
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          Active: <span style={{ color: '#fff', fontWeight: 600 }}>{selectedImageDetails.image_id}</span> // Site: {selectedImageDetails.metadata?.airport_code || 'KIAD'}
        </div>
      </div>

      {/* Workspace */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 340px', gap: '16px', overflow: 'hidden' }}>
        
        {/* Canvas Pane */}
        <div className="glass-recessed" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px', overflow: 'hidden', position: 'relative' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', padding: '2px 6px' }}>
            <span>CVAT INTERACTIVE DRAWING CANVAS (512x512)</span>
            <span>Cursor: X {(mousePos.x*512).toFixed(0)}px / Y {(mousePos.y*512).toFixed(0)}px</span>
          </div>

          <div style={{ flex: 1, background: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <div style={{ width: '480px', height: '480px', position: 'relative', border: '1px solid rgba(255,255,255,0.05)', boxShadow: '0 10px 30px rgba(0,0,0,0.8)' }}>
              
              <svg 
                viewBox="0 0 512 512" 
                onClick={handleCanvasClick}
                onMouseMove={handleCanvasMouseMove}
                style={{ 
                  width: '100%', 
                  height: '100%', 
                  cursor: 'crosshair',
                  background: `url(${getImageUrl(activeDatasetId, selectedImageDetails.image_id)}) no-repeat center center`,
                  backgroundSize: 'cover'
                }}
              >
                {/* Existing layers */}
                {editLabels.map((lbl, idx) => {
                  if (hiddenLayers[idx]) return null;
                  const isSelected = selectedLabelIdx === idx;
                  const isHovered = hoveredLabelIdx === idx;
                  const color = CLASS_COLORS[lbl.class_id] || '#00f2fe';
                  const pointsStr = lbl.polygon.map((p: any) => `${p[0] * 512},${p[1] * 512}`).join(' ');

                  return (
                    <polygon 
                      key={idx}
                      points={pointsStr}
                      fill={color}
                      fillOpacity={isSelected ? 0.3 : 0.1}
                      stroke={isSelected || isHovered ? '#fff' : color}
                      strokeWidth={isSelected || isHovered ? 4 : 2}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedLabelIdx(idx);
                      }}
                      onMouseEnter={() => setHoveredLabelIdx(idx)}
                      onMouseLeave={() => setHoveredLabelIdx(null)}
                      style={{ transition: 'all 0.1s ease', cursor: 'pointer' }}
                    />
                  );
                })}

                {/* AI Suggestions Overlays */}
                {suggestions.map((sugg, idx) => {
                  const pointsStr = sugg.polygon.map((p: any) => `${p[0] * 512},${p[1] * 512}`).join(' ');
                  return (
                    <polygon 
                      key={`sugg-${idx}`}
                      points={pointsStr}
                      fill="#d946ef"
                      fillOpacity={0.08}
                      stroke="#d946ef"
                      strokeWidth={2}
                      strokeDasharray="4,4"
                      style={{ cursor: 'pointer' }}
                    >
                      <title>{`AI Suggestion: Class ${CLASS_NAMES[sugg.class_id]} (Conf: ${Math.round(sugg.confidence * 100)}%)`}</title>
                    </polygon>
                  );
                })}

                {/* Drawing polygon path */}
                {drawingMode === 'polygon' && tempPoints.length > 0 && (
                  <>
                    <polyline 
                      points={tempPoints.map(p => `${p[0]*512},${p[1]*512}`).join(' ')}
                      fill="none"
                      stroke="var(--accent-cyan)"
                      strokeWidth={2}
                    />
                    <line 
                      x1={tempPoints[tempPoints.length - 1][0]*512}
                      y1={tempPoints[tempPoints.length - 1][1]*512}
                      x2={mousePos.x*512}
                      y2={mousePos.y*512}
                      stroke="var(--accent-cyan)"
                      strokeWidth={1.5}
                      strokeDasharray="3,3"
                    />
                    {tempPoints.map((p, idx) => (
                      <circle 
                        key={idx}
                        cx={p[0]*512}
                        cy={p[1]*512}
                        r={idx === 0 ? 5 : 3}
                        fill={idx === 0 ? 'var(--accent-orange)' : 'var(--accent-cyan)'}
                        stroke="#fff"
                        onClick={(e) => {
                          if (idx === 0) {
                            e.stopPropagation();
                            completePolygon();
                          }
                        }}
                      />
                    ))}
                  </>
                )}

                {/* Drawing bbox rect */}
                {drawingMode === 'bbox' && isDrawingBbox && bboxStart && (
                  <rect 
                    x={Math.min(bboxStart[0], mousePos.x)*512}
                    y={Math.min(bboxStart[1], mousePos.y)*512}
                    width={Math.abs(mousePos.x - bboxStart[0])*512}
                    height={Math.abs(mousePos.y - bboxStart[1])*512}
                    fill="none"
                    stroke="var(--accent-cyan)"
                    strokeWidth={2}
                    strokeDasharray="4,4"
                  />
                )}
              </svg>

            </div>
          </div>
        </div>

        {/* Sidebar Controls */}
        <div className="glass-recessed" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px', overflowY: 'auto' }}>
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
            LABELS & LAYERS TREE
          </h3>

          {/* Mode Selector */}
          <div style={{ display: 'flex', gap: '6px' }}>
            <button 
              onClick={() => { setDrawingMode('polygon'); setTempPoints([]); }}
              className={`btn-tactical ${drawingMode === 'polygon' ? 'btn-tactical-active' : ''}`}
              style={{ flex: 1, padding: '6px', fontSize: '0.65rem' }}
            >
              Polygon Tool
            </button>
            <button 
              onClick={() => { setDrawingMode('bbox'); setTempPoints([]); }}
              className={`btn-tactical ${drawingMode === 'bbox' ? 'btn-tactical-active' : ''}`}
              style={{ flex: 1, padding: '6px', fontSize: '0.65rem' }}
            >
              Box Tool
            </button>
          </div>

          {/* Class Selector */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
            <label>ACTIVE CLASS LABEL</label>
            <select 
              value={activeClassId}
              onChange={e => setActiveClassId(parseInt(e.target.value))}
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '6px 8px', borderRadius: '4px', outline: 'none' }}
            >
              {Object.keys(CLASS_NAMES).map(cid => (
                <option key={cid} value={cid}>{CLASS_NAMES[parseInt(cid)]}</option>
              ))}
            </select>
          </div>

          {/* AI Auto-Label Action Trigger */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <button 
              onClick={handleAutoLabel}
              disabled={loadingAutoLabel}
              className="btn-tactical border-glow-cyan"
              style={{ width: '100%', justifyContent: 'center', padding: '8px 10px', fontSize: '0.75rem', display: 'flex', gap: '8px', alignItems: 'center' }}
            >
              <Cpu style={{ width: '14px', height: '14px', animation: loadingAutoLabel ? 'spin 1.5s linear infinite' : 'none' }} />
              {loadingAutoLabel ? 'RUNNING MODEL INFERENCE...' : 'AUTO-LABEL WITH AI'}
            </button>
            
            {suggestions.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', background: 'rgba(217, 70, 239, 0.05)', border: '1px dashed rgba(217, 70, 239, 0.3)', padding: '10px', borderRadius: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                  <span style={{ color: '#d946ef', fontWeight: 600 }}>AI SUGGESTIONS ({suggestions.length})</span>
                  <button 
                    onClick={approveAllSuggestions}
                    style={{ background: 'none', border: 'none', color: '#d946ef', cursor: 'pointer', fontSize: '0.6rem', textDecoration: 'underline' }}
                  >
                    Approve All
                  </button>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxHeight: '130px', overflowY: 'auto' }}>
                  {suggestions.map((sugg) => (
                    <div 
                      key={sugg.id} 
                      style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.03)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ color: '#fff' }}>{CLASS_NAMES[sugg.class_id]}</span>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.55rem' }}>Conf: {Math.round(sugg.confidence * 100)}%</span>
                      </div>
                      <div style={{ display: 'flex', gap: '4px' }}>
                        <button 
                          onClick={() => approveSuggestion(sugg.id)}
                          style={{ background: 'none', border: 'none', color: 'var(--accent-green)', cursor: 'pointer', padding: '2px', fontWeight: 'bold' }}
                          title="Approve Target"
                        >
                          ✓
                        </button>
                        <button 
                          onClick={() => rejectSuggestion(sugg.id)}
                          style={{ background: 'none', border: 'none', color: 'var(--accent-red)', cursor: 'pointer', padding: '2px', fontWeight: 'bold' }}
                          title="Reject Target"
                        >
                          ✗
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Layers list */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px', overflowY: 'auto', minHeight: '160px' }}>
            <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', borderBottom: '1px dashed rgba(255,255,255,0.05)', paddingBottom: '4px' }}>
              LAYERS ({editLabels.length})
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {editLabels.map((lbl, idx) => {
                const isSelected = selectedLabelIdx === idx;
                const isHovered = hoveredLabelIdx === idx;
                const isHidden = hiddenLayers[idx];
                const color = CLASS_COLORS[lbl.class_id] || '#00f2fe';
                return (
                  <div 
                    key={idx}
                    onMouseEnter={() => setHoveredLabelIdx(idx)}
                    onMouseLeave={() => setHoveredLabelIdx(null)}
                    onClick={() => setSelectedLabelIdx(idx)}
                    className="glass-panel"
                    style={{ 
                      padding: '6px 10px', 
                      fontSize: '0.65rem', 
                      fontFamily: 'var(--font-mono)', 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      background: isSelected ? 'rgba(255,255,255,0.03)' : 'none',
                      borderColor: isSelected || isHovered ? color : 'rgba(255,255,255,0.05)',
                      cursor: 'pointer'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', backgroundColor: color }} />
                      <span>{CLASS_NAMES[lbl.class_id]}</span>
                    </div>

                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <button 
                        onClick={(e) => { e.stopPropagation(); toggleLayerVisibility(idx); }}
                        style={{ background: 'none', border: 'none', color: isHidden ? 'var(--text-muted)' : 'var(--text-secondary)', cursor: 'pointer' }}
                        title="Toggle visibility"
                      >
                        {isHidden ? <EyeOff style={{ width: '12px', height: '12px' }} /> : <Eye style={{ width: '12px', height: '12px' }} />}
                      </button>
                      <button 
                        onClick={(e) => { e.stopPropagation(); deleteLayer(idx); }}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-red)', cursor: 'pointer' }}
                        title="Delete layer"
                      >
                        <Trash2 style={{ width: '12px', height: '12px' }} />
                      </button>
                    </div>
                  </div>
                );
              })}
              {editLabels.length === 0 && (
                <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                  No target vector paths drawn yet. Click canvas to drop vertices.
                </div>
              )}
            </div>
          </div>

          {/* Quick Cheatsheet */}
          <div style={{ background: 'rgba(0,0,0,0.15)', border: '1px solid rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 600, marginBottom: '2px' }}>KEYBOARD SHORTCUTS</span>
            <span>[Esc] - Reset / Cancel active shape</span>
            <span>[Backspace] - Undo last polygon node</span>
            <span>Double Click canvas - Complete polygon</span>
          </div>

          {/* Alert status / save */}
          {saveStatus && (
            <div style={{ 
              fontSize: '0.7rem', 
              fontFamily: 'var(--font-mono)', 
              color: saveStatus.includes('success') ? 'var(--accent-green)' : saveStatus.includes('Error') ? 'var(--accent-red)' : 'var(--accent-cyan)'
            }}>
              {saveStatus}
            </div>
          )}

          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={handleSaveLabels}
              disabled={isSaveDisabled}
              className="btn-tactical btn-tactical-active"
              style={{ flex: 1, justifyContent: 'center', padding: '10px', fontSize: '0.75rem' }}
            >
              <Save style={{ width: '14px', height: '14px' }} />
              SAVE CHANGES
            </button>
            <button 
              onClick={() => setSelectedImageDetails(null)}
              className="btn-tactical"
              style={{ padding: '10px', fontSize: '0.75rem' }}
            >
              ← BACK TO CURATION
            </button>
          </div>
        </div>

      </div>

    </div>
  );
}



// ============================================
// VIEW COMPONENT: 4. MISSION CONTROL (Telemetry & Eval)
// ============================================
function MissionControlView({
  activeJob,
  jobLogs,
  logsEndRef,
  experiments,
  datasets,
  compareBaseId,
  compareCandId,
  setCompareBaseId,
  setCompareCandId,
  compareResults,
  viewTab,
  setViewTab
}: any) {

  // Auto switch to cockpit if a job starts executing
  useEffect(() => {
    if (activeJob && ['preparing_dataset', 'queued', 'training', 'evaluating'].includes(activeJob.status)) {
      if (setViewTab) setViewTab('cockpit');
    }
  }, [activeJob]);

  const renderSVGChart = (history: number[], strokeColor: string) => {
    if (!history || history.length === 0) {
      return (
        <div style={{ height: '110px', background: 'rgba(0,0,0,0.25)', border: '1px solid var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          Awaiting telemetry...
        </div>
      );
    }
    const width = 300;
    const height = 110;
    const padding = 15;
    const maxVal = Math.max(...history, 1.0);
    
    let points = "";
    history.forEach((val, index) => {
      const x = padding + (index / (history.length - 1 || 1)) * (width - 2 * padding);
      const y = height - padding - (val / maxVal) * (height - 2 * padding);
      points += `${x},${y} `;
    });

    return (
      <svg width="100%" height="110" style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '6px' }}>
        <polyline
          fill="none"
          stroke={strokeColor}
          strokeWidth="2"
          points={points.trim()}
        />
        {history.map((val, index) => {
          const x = padding + (index / (history.length - 1 || 1)) * (width - 2 * padding);
          const y = height - padding - (val / maxVal) * (height - 2 * padding);
          return (
            <circle
              key={index}
              cx={x}
              cy={y}
              r="3"
              fill={strokeColor}
            />
          );
        })}
      </svg>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }}>
      
      {/* Step Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>MISSION CONTROL</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Monitor local MPS/CPU hardware executions, capture validation metrics, and perform comparative evaluations on trained ATR models.</p>
        </div>

        {/* Global Connection / Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div className="glass-recessed" style={{ padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
            <span className={`led-indicator ${activeJob && !['complete', 'failed', 'cancelled'].includes(activeJob.status) ? 'orange pulse' : 'green'}`}></span>
            ACCELERATOR STATUS: {activeJob && !['complete', 'failed', 'cancelled'].includes(activeJob.status) ? activeJob.status.toUpperCase() : 'IDLE'}
          </div>
        </div>
      </div>

      {/* Grid: Stats & Telemetry */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {[
          { label: 'DATASETS REGISTERED', value: datasets ? datasets.length : 0, sub: '1 Active Source', icon: DbIcon },
          { label: 'EXPERIMENTS RUN', value: experiments ? experiments.length : 0, sub: 'Total training attempts', icon: Layers },
          { label: 'LATEST ACCURACY (mAP50)', value: experiments && experiments.length > 0 && experiments[0].status === 'complete' ? '0.72' : 'N/A', sub: experiments && experiments.length > 0 ? experiments[0].name : 'No runs completed yet', icon: TrendingUp },
          { label: 'ACTIVE GPU WORKERS', value: '1', sub: 'Local MPS Execution enabled', icon: Activity },
        ].map((item, idx) => {
          const Icon = item.icon;
          return (
            <div key={idx} className="glass-recessed" style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '80px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{item.label}</span>
                <Icon style={{ width: '14px', height: '14px', color: 'var(--accent-cyan)' }} />
              </div>
              <div>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem', fontWeight: 700, margin: '2px 0' }}>{item.value}</div>
                <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{item.sub}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Workspace (occupies full space) */}
      <div className="glass-recessed" style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        
        {/* View Tab Selector */}
        <div style={{ display: 'flex', justifyItems: 'flex-start', gap: '8px', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px', marginBottom: '12px' }}>
          <button 
            onClick={() => setViewTab('evaluator')}
            className={`btn-tactical ${viewTab === 'evaluator' ? 'btn-tactical-active border-glow-cyan' : ''}`}
            style={{ fontSize: '0.7rem', padding: '6px 12px' }}
          >
            <BarChart3 style={{ width: '12px', height: '12px' }} />
            MODEL COMPARATIVE EVALUATOR
          </button>
          <button 
            onClick={() => setViewTab('cockpit')}
            className={`btn-tactical ${viewTab === 'cockpit' ? 'btn-tactical-active border-glow-orange' : ''}`}
            style={{ fontSize: '0.7rem', padding: '6px 12px' }}
          >
            <Activity style={{ width: '12px', height: '12px' }} />
            REAL-TIME MONITORING COCKPIT {activeJob && !['complete', 'failed', 'cancelled'].includes(activeJob.status) && '●'}
          </button>
        </div>

          {viewTab === 'cockpit' ? (
            // REAL-TIME MONITORING COCKPIT
            activeJob ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', letterSpacing: '0.05em', margin: 0 }}>
                    MONITORING COCKPIT
                  </h3>
                  <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>
                    ACTIVE PIPELINE EXECUTION ({activeJob.status.toUpperCase()})
                  </span>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>LOSS HISTORY</div>
                    {renderSVGChart(activeJob.loss_history, '#ff9900')}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>VALIDATION mAP50</div>
                    {renderSVGChart(activeJob.map50_history, '#00ff87')}
                  </div>
                </div>

                {/* Console Logs Terminal */}
                <div style={{ 
                  flex: 1, 
                  background: 'rgba(0,0,0,0.95)', 
                  border: '1px solid var(--border-color)', 
                  borderRadius: '8px', 
                  padding: '10px', 
                  overflowY: 'auto',
                  fontFamily: 'var(--font-mono)', 
                  fontSize: '0.7rem', 
                  color: 'rgba(0, 242, 254, 0.95)',
                  boxShadow: 'inset 0 4px 10px rgba(0,0,0,0.95)',
                  lineHeight: '1.4'
                }}>
                  <div style={{ color: 'var(--text-secondary)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px', marginBottom: '6px', fontSize: '0.6rem', display: 'flex', justifyContent: 'space-between' }}>
                    <span>PROCESS STREAMING CONSOLE LOGS</span>
                    <span>PROCESS_ID: {activeJob.id}</span>
                  </div>
                  <div style={{ whiteSpace: 'pre-wrap' }}>
                    {jobLogs}
                  </div>
                  <div ref={logsEndRef} />
                </div>
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                <Activity style={{ width: '40px', height: '40px' }} />
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', marginTop: '12px' }}>ACCELERATOR IDLE</div>
                <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', maxWidth: '300px', textAlign: 'center', marginTop: '4px' }}>
                  No active training process is currently executing. Build a pipeline blueprint and launch execution to stream metrics.
                </p>
              </div>
            )
          ) : (
            // MODEL COMPARATIVE EVALUATOR
            <div style={{ flex: 1, overflowY: 'auto' }}>
              <ModelEvaluatorView 
                experiments={experiments}
                compareBaseId={compareBaseId}
                compareCandId={compareCandId}
                setCompareBaseId={setCompareBaseId}
                setCompareCandId={setCompareCandId}
                compareResults={compareResults}
              />
            </div>
          )}

      </div>
    </div>
  );
}

// ============================================
// VIEW COMPONENT: 5. MODEL EVALUATOR & COMPARER
// ============================================
function ModelEvaluatorView({
  experiments,
  compareBaseId,
  compareCandId,
  setCompareBaseId,
  setCompareCandId,
  compareResults
}: any) {
  
  const renderEvaluatedChips = () => {
    if (!compareResults || !compareResults.candidate || !compareResults.candidate.eval) return null;
    const evals = compareResults.candidate.eval.predictions || [];
    
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
        {evals.slice(0, 4).map((sample: any) => {
          return (
            <div key={sample.image_id} className="glass-panel" style={{ padding: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ height: '110px', background: '#000', borderRadius: '4px', overflow: 'hidden', position: 'relative', display: 'flex', justifyContent: 'center' }}>
                <div style={{ position: 'relative', width: '110px', height: '110px' }}>
                  <img 
                    src={getImageUrl(getDatasetIdFromVersion(compareResults.candidate.eval.dataset_version_id), sample.image_id)} 
                    alt={sample.image_id} 
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                  
                  {/* SVG overlay for drawing predictions bounding boxes */}
                  <svg 
                    style={{ 
                      position: 'absolute', 
                      top: 0, 
                      left: 0, 
                      width: '100%', 
                      height: '100%', 
                      pointerEvents: 'none' 
                    }}
                    viewBox="0 0 512 512"
                  >
                    {sample.predictions && sample.predictions.map((p: any, idx: number) => {
                      const color = p.type === 'TP' ? '#00ff87' : p.type === 'FP' ? '#ff0055' : '#ff9900';
                      return (
                        <g key={idx}>
                          <rect 
                            x={p.bbox[0]} 
                            y={p.bbox[1]} 
                            width={Math.max(15, p.bbox[2] - p.bbox[0])} 
                            height={Math.max(15, p.bbox[3] - p.bbox[1])} 
                            fill="none" 
                            stroke={color} 
                            strokeWidth="10" 
                            strokeDasharray={p.type === 'FN' ? '16,16' : 'none'}
                          />
                          <text 
                            x={p.bbox[0] + 5} 
                            y={p.bbox[1] > 40 ? p.bbox[1] - 12 : p.bbox[1] + 35} 
                            fill={color} 
                            fontSize="28" 
                            fontFamily="var(--font-mono)"
                            fontWeight="bold"
                            style={{ paintOrder: 'stroke', stroke: '#000', strokeWidth: '6px', strokeLinejoin: 'round' }}
                          >
                            {p.type}:{p.confidence > 0 ? (p.confidence*100).toFixed(0)+'%' : 'FN'}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              </div>
              <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', textAlign: 'center' }}>
                {sample.image_id}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }}>
      
      {/* View Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>STEP 5: EVALUATE & ITERATE</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Select two experimental runs to review metrics differential and false alarm clusters.</p>
        </div>
        
        {/* Selector Dropdowns */}
        <div style={{ display: 'flex', gap: '12px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label>BASELINE EXPERIMENT</label>
            <select 
              value={compareBaseId} 
              onChange={e => setCompareBaseId(e.target.value)}
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '6px 10px', borderRadius: '4px', outline: 'none' }}
            >
              <option value="">Select Baseline</option>
              {experiments.map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
            </select>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label>CANDIDATE RUN</label>
            <select 
              value={compareCandId} 
              onChange={e => setCompareCandId(e.target.value)}
              style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', color: '#fff', padding: '6px 10px', borderRadius: '4px', outline: 'none' }}
            >
              <option value="">Select Candidate</option>
              {experiments.map((e: any) => <option key={e.id} value={e.id}>{e.name}</option>)}
            </select>
          </div>
        </div>
      </div>

      {compareResults && compareResults.base && compareResults.candidate ? (
        <div style={{ flex: 1, display: 'grid', gridTemplateRows: '1fr auto', gap: '16px', overflow: 'hidden' }}>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '16px', overflow: 'hidden' }}>
            
            {/* Left: Comparison Metrics Table */}
            <div className="glass-recessed" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'auto' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
                ACCURACY STATS DELTA (SEGMENTATION MASK)
              </h3>
              
              <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '10px 8px' }}>METRIC</th>
                    <th style={{ padding: '10px 8px' }}>{compareResults.base.exp.name.slice(0, 15)}...</th>
                    <th style={{ padding: '10px 8px' }}>{compareResults.candidate.exp.name.slice(0, 15)}...</th>
                    <th style={{ padding: '10px 8px' }}>DELTA</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { key: 'mAP50', label: 'mAP50 (IoU=0.5)' },
                    { key: 'mAP50-95', label: 'mAP50-95 (avg)' },
                    { key: 'precision', label: 'Precision' },
                    { key: 'recall', label: 'Recall' },
                    { key: 'f1', label: 'F1 Score' },
                  ].map((row) => {
                    const baseVal = compareResults.base.eval ? compareResults.base.eval[row.key] || 0.0 : 0.0;
                    const candVal = compareResults.candidate.eval ? compareResults.candidate.eval[row.key] || 0.0 : 0.0;
                    const delta = candVal - baseVal;
                    const isPositive = delta >= 0;
                    
                    return (
                      <tr key={row.key} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                        <td style={{ padding: '12px 8px', fontWeight: 600 }}>{row.label}</td>
                        <td style={{ padding: '12px 8px' }}>{baseVal.toFixed(3)}</td>
                        <td style={{ padding: '12px 8px', color: '#fff' }}>{candVal.toFixed(3)}</td>
                        <td style={{ padding: '12px 8px', color: delta === 0 ? 'var(--text-secondary)' : isPositive ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>
                          {delta === 0 ? '0.000' : (isPositive ? '+' : '') + delta.toFixed(3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Right: Configurations Diff */}
            <div className="glass-recessed" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'auto' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
                HYPERPARAMETER DIFF
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                {[
                  { label: 'Export Version', base: compareResults.base.exp.dataset_version_id, cand: compareResults.candidate.exp.dataset_version_id },
                  { label: 'Epochs', base: compareResults.base.exp.config.epochs, cand: compareResults.candidate.exp.config.epochs },
                  { label: 'Aug: Horiz Flip', base: compareResults.base.exp.config.augmentations?.fliplr ? 'TRUE' : 'FALSE', cand: compareResults.candidate.exp.config.augmentations?.fliplr ? 'TRUE' : 'FALSE' },
                  { label: 'Aug: Rotation', base: `${compareResults.base.exp.config.augmentations?.degrees || 0.0}°`, cand: `${compareResults.candidate.exp.config.augmentations?.degrees || 0.0}°` },
                ].map((item, idx) => {
                  const hasDiff = item.base !== item.cand;
                  return (
                    <div key={idx} style={{ 
                      padding: '8px 10px', 
                      borderRadius: '4px',
                      background: hasDiff ? 'rgba(255, 153, 0, 0.03)' : 'rgba(0,0,0,0.1)',
                      border: hasDiff ? '1px solid rgba(255, 153, 0, 0.15)' : '1px solid rgba(255,255,255,0.03)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px'
                    }}>
                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.65rem' }}>{item.label.toUpperCase()}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: '#fff', gap: '8px' }}>
                        <span 
                          title={String(item.base)} 
                          style={{ 
                            textOverflow: 'ellipsis', 
                            overflow: 'hidden', 
                            whiteSpace: 'nowrap', 
                            maxWidth: '120px', 
                            flex: 1 
                          }}
                        >
                          Base: {item.base}
                        </span>
                        <ArrowRight style={{ width: '12px', height: '12px', color: hasDiff ? 'var(--accent-orange)' : 'var(--text-muted)', flexShrink: 0 }} />
                        <span 
                          title={String(item.cand)} 
                          style={{ 
                            textOverflow: 'ellipsis', 
                            overflow: 'hidden', 
                            whiteSpace: 'nowrap', 
                            maxWidth: '120px', 
                            color: hasDiff ? 'var(--accent-cyan)' : '#fff',
                            textAlign: 'right',
                            flex: 1 
                          }}
                        >
                          Cand: {item.cand}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>

          {/* Bottom Side: Predictions Inspection */}
          <div className="glass-recessed" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em' }}>
                VISUAL VALIDATION CHIP OVERLAYS (CANDIDATE RUN)
              </h3>
              <div style={{ display: 'flex', gap: '16px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                <span style={{ color: 'var(--accent-green)' }}>● TP = TRUE POSITIVE</span>
                <span style={{ color: 'var(--accent-red)' }}>● FP = FALSE POSITIVE</span>
                <span style={{ color: 'var(--accent-orange)' }}>● FN = FALSE NEGATIVE</span>
              </div>
            </div>
            {renderEvaluatedChips()}
          </div>

        </div>
      ) : (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
          <BarChart3 style={{ width: '40px', height: '40px' }} />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', marginTop: '12px' }}>INSUFFICIENT DATA TO COMPARE</div>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', maxWidth: '300px', textAlign: 'center', marginTop: '4px' }}>
            Please trigger at least two experimental runs using the assistant or lab builder to compare metrics deltas.
          </p>
        </div>
      )}

    </div>
  );
}


// ============================================
// VIEW COMPONENT: DASHBOARD OVERVIEW & ROADMAP
// ============================================
function DashboardView({
  datasets,
  experiments,
  versions,
  projectTree,
  setActiveTab
}: any) {
  const totalDatasets = datasets.length;
  const totalExperiments = experiments.length;
  
  // Extract completed runs with evaluations
  const completedRuns: any[] = [];
  const recentRuns: any[] = [];
  
  if (projectTree && projectTree.pipelines) {
    projectTree.pipelines.forEach((pipe: any) => {
      if (pipe.experiments) {
        pipe.experiments.forEach((exp: any) => {
          // Add to recent runs
          recentRuns.push({
            id: exp.id,
            name: exp.name,
            pipelineName: pipe.name,
            pipelineId: pipe.id,
            status: exp.status,
            createdAt: exp.created_at
          });
          
          // If completed and has evaluation, add to leaderboard
          if (exp.status === 'complete' && exp.evaluation) {
            completedRuns.push({
              id: exp.id,
              name: exp.name,
              pipelineName: pipe.name,
              pipelineId: pipe.id,
              version: exp.dataset_version_id,
              modelType: exp.model_type,
              epochs: exp.config?.epochs || 'N/A',
              mAP50: exp.evaluation.map50 || 0.0,
              precision: exp.evaluation.precision || 0.0,
              recall: exp.evaluation.recall || 0.0,
              f1: exp.evaluation.f1 || 0.0
            });
          }
        });
      }
    });
  }

  // Sort leaderboard by mAP50 descending
  completedRuns.sort((a, b) => b.mAP50 - a.mAP50);
  
  // Sort recent activity by createdAt descending
  recentRuns.sort((a, b) => b.createdAt - a.createdAt);
  const displayRecentRuns = recentRuns.slice(0, 5);

  const topScore = completedRuns.length > 0 ? completedRuns[0].mAP50.toFixed(3) : 'N/A';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }}>
      {/* Dashboard Header */}
      <div>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>WORKBENCH DASHBOARD</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>An end-to-end overview of your geospatial ATR target training pipeline, versions, and validation comparative leaderboard.</p>
      </div>

      {/* Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        {[
          { label: 'INGESTED DATASETS', value: totalDatasets, desc: `${datasets.reduce((acc: number, cur: any) => acc + (cur.sample_size || 0), 0)} airfield chips` },
          { label: 'EXPERIMENTS INITIATED', value: totalExperiments, desc: `${versions.length} split configurations compiled` },
          { label: 'TOP MODEL ACCURACY (mAP50)', value: topScore, desc: 'Leaderboard champion score' },
          { label: 'GPU ACCELERATOR', value: 'MPS ACTIVE', desc: 'Hardware acceleration enabled' },
        ].map((stat, idx) => (
          <div key={idx} className="glass-recessed" style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{stat.label}</span>
            <div style={{ fontSize: '1.2rem', fontFamily: 'var(--font-display)', fontWeight: 600, color: 'var(--accent-cyan)', margin: '2px 0' }}>{stat.value}</div>
            <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{stat.desc}</span>
          </div>
        ))}
      </div>

      {/* Main Content Panels */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '16px', overflow: 'hidden' }}>
        
        {/* Left: Performance Leaderboard */}
        <div className="glass-recessed" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px', overflow: 'auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', margin: 0 }}>
              ATR ACCURACY LEADERBOARD
            </h3>
            <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>
              VAL ACCURACY METRICS (SEGMENTATION MASK)
            </span>
          </div>

          {completedRuns.length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '8px' }}>RANK</th>
                  <th style={{ padding: '8px' }}>RUN NAME</th>
                  <th style={{ padding: '8px' }}>PIPELINE</th>
                  <th style={{ padding: '8px' }}>EPOCHS</th>
                  <th style={{ padding: '8px' }}>mAP50</th>
                  <th style={{ padding: '8px' }}>PRECISION</th>
                  <th style={{ padding: '8px' }}>RECALL</th>
                  <th style={{ padding: '8px' }}>F1 SCORE</th>
                </tr>
              </thead>
              <tbody>
                {completedRuns.map((run, index) => (
                  <tr key={run.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', background: index === 0 ? 'rgba(0,255,135,0.02)' : 'none' }}>
                    <td style={{ padding: '10px 8px', fontWeight: 'bold', color: index === 0 ? 'var(--accent-green)' : 'inherit' }}>
                      #{index + 1}
                    </td>
                    <td style={{ padding: '10px 8px', color: '#fff', fontWeight: 600 }}>{run.name}</td>
                    <td style={{ padding: '10px 8px' }}>
                      <span className="glass-panel" style={{ padding: '2px 6px', fontSize: '0.55rem', border: '1px solid rgba(0, 242, 254, 0.3)', color: 'var(--accent-cyan)', background: 'none', borderRadius: '3px' }}>
                        {run.pipelineName.replace('Pipeline', '')}
                      </span>
                    </td>
                    <td style={{ padding: '10px 8px' }}>{run.epochs}</td>
                    <td style={{ padding: '10px 8px', color: 'var(--accent-cyan)', fontWeight: 'bold' }}>{run.mAP50.toFixed(3)}</td>
                    <td style={{ padding: '10px 8px' }}>{run.precision.toFixed(3)}</td>
                    <td style={{ padding: '10px 8px' }}>{run.recall.toFixed(3)}</td>
                    <td style={{ padding: '10px 8px' }}>{run.f1.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              <Activity style={{ width: '30px', height: '30px', marginBottom: '8px' }} />
              <div style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>NO TRAINING RUNS EVALUATED</div>
              <p style={{ fontSize: '0.65rem', textAlign: 'center', marginTop: '4px', maxWidth: '280px' }}>
                Deploy blueprints and run model training to rank pipeline configurations.
              </p>
            </div>
          )}
        </div>

        {/* Right: Recent Runs & Pipeline Shortcuts */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
          
          {/* Recent Runs */}
          <div className="glass-recessed" style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'auto' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', margin: 0 }}>
              RECENT SYSTEM ACTIVITY
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {displayRecentRuns.length > 0 ? (
                displayRecentRuns.map((run: any) => {
                  const isRunning = ['queued', 'training', 'preparing_dataset', 'evaluating'].includes(run.status);
                  const isSuccess = run.status === 'complete';
                  const isFailed = run.status === 'failed';
                  const ledClass = isRunning ? 'orange pulse' : isSuccess ? 'green' : isFailed ? 'red' : 'red';
                  
                  return (
                    <div key={run.id} className="glass-panel" style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px', background: 'rgba(0,0,0,0.1)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.7rem', color: '#fff', fontWeight: 600, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '180px' }}>
                          {run.name}
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}>
                          <span className={`led-indicator ${ledClass}`}></span>
                          {run.status.toUpperCase()}
                        </div>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.6rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                        <span>Pipeline: {run.pipelineName.replace('Pipeline', '')}</span>
                        <span>{new Date(run.createdAt * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', textAlign: 'center', marginTop: '20px', fontFamily: 'var(--font-mono)' }}>
                  Awaiting system executions...
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions Shortcuts */}
          <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px', background: 'rgba(5, 8, 20, 0.4)' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', letterSpacing: '0.05em', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px', margin: 0 }}>
              QUICK LAUNCH SHORTCUTS
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <button 
                onClick={() => setActiveTab('studio')} 
                className="btn-tactical" 
                style={{ fontSize: '0.65rem', padding: '8px', justifyContent: 'center' }}
              >
                Data Prep Studio →
              </button>
              <button 
                onClick={() => setActiveTab('control')} 
                className="btn-tactical" 
                style={{ fontSize: '0.65rem', padding: '8px', justifyContent: 'center' }}
              >
                Mission Control →
              </button>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
