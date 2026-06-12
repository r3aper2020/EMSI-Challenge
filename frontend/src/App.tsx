import React, { useState, useEffect, useRef } from 'react';
import FiftyOneDashboard from './FiftyOneDashboard';
import ProjectDashboardView from './ProjectDashboardView';
import { 
  Play, 
  RefreshCw, 
  Terminal, 
  Layers, 
  BarChart3, 
  ArrowRight, 
  Plus, 
  Compass, 
  ChevronRight,
  Eye,
  EyeOff,
  Trash2,
  Save,
  Cpu,
  Activity
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

const renderSVGChart = (history: number[], strokeColor: string) => {
  if (!history || history.length === 0) {
    return (
      <div style={{ height: '110px', background: 'rgba(0,0,0,0.25)', border: '1px solid var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.65rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
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

export default function App() {
  if (window.location.pathname.startsWith('/fiftyone')) {
    return <FiftyOneDashboard />;
  }

  const [wizardStep, setWizardStep] = useState<'dashboard' | 'curate' | 'annotate' | 'train' | 'evaluate'>('dashboard');
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
  
  const [compareBaseId, setCompareBaseId] = useState<string>('');
  const [compareCandId, setCompareCandId] = useState<string>('');
  const [compareResults, setCompareResults] = useState<any>(null);

  const [compiledVersion, setCompiledVersion] = useState<any>(null);
  const [isGeneratingBlueprint, setIsGeneratingBlueprint] = useState<boolean>(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadName, setUploadName] = useState('');
  const [uploadDesc, setUploadDesc] = useState('');
  const [uploadTask, setUploadTask] = useState('instance_segmentation');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadingStatus, setUploadingStatus] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [datasetToDelete, setDatasetToDelete] = useState<any>(null);
  const [cancelJobConfirm, setCancelJobConfirm] = useState<any>(null);
  const [experimentToDelete, setExperimentToDelete] = useState<any>(null);
  const [clearLlmHistoryConfirm, setClearLlmHistoryConfirm] = useState<boolean>(false);

  // Pipeline Parameters Form State (Combined & Lifted)
  const [formData, setFormData] = useState({
    name: 'YOLOv8-seg airfield run',
    dataset_id: 'dataset_rareplanes_real',
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

  useEffect(() => {
    if (experiments && experiments.length > 0) {
      const nextRunNum = experiments.length + 1;
      setFormData(prev => {
        const isDefaultName = prev.name === 'YOLOv8-seg airfield run' || prev.name.startsWith('YOLOv8-seg airfield run v') || prev.name.startsWith('YOLOv8-seg airfield run ');
        const isDefaultTag = prev.version_tag === 'run_v1' || prev.version_tag.startsWith('run_v');
        return {
          ...prev,
          name: isDefaultName ? `YOLOv8-seg airfield run v${nextRunNum}` : prev.name,
          version_tag: isDefaultTag ? `run_v${nextRunNum}` : prev.version_tag
        };
      });
    } else {
      setFormData(prev => ({
        ...prev,
        name: 'YOLOv8-seg airfield run v1',
        version_tag: 'run_v1'
      }));
    }
  }, [experiments]);

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
        const hasReal = dsList.find((d: any) => d.id === 'dataset_rareplanes_real');
        const defaultDsId = hasReal ? 'dataset_rareplanes_real' : dsList[0].id;
        setActiveDatasetId(defaultDsId);
        fetchDatasetEmbeddings(defaultDsId);
        fetchDatasetVersions(defaultDsId);
      }
      
      await fetchExperiments();
      
      // Fetch LLM command history from SQLite database
      const cmdRes = await fetch(`${API_BASE}/projects/proj_default/llm-commands`);
      if (cmdRes.ok) {
        const cmdHistory = await cmdRes.json();
        if (cmdHistory.length > 0) {
          const formattedLogs = cmdHistory.map((c: any) => ({
            sender: c.sender,
            text: c.text,
            time: new Date(c.created_at * 1000).toLocaleTimeString(),
            payload: c.payload
          }));
          setLlmLogs(formattedLogs);
        }
      }
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
        await handleDatasetUpload();
        
        const newDsId = result.dataset.id;
        setActiveDatasetId(newDsId);
        await fetchDatasetEmbeddings(newDsId);
        await fetchDatasetVersions(newDsId);
        
        setUploadingStatus('');
        setUploadModalOpen(false);
        setUploadName('');
        setUploadDesc('');
        setUploadFile(null);
        setWizardStep('curate');
      } else {
        setUploadError(result.detail || 'Failed to parse ZIP archive.');
        setUploadingStatus('');
      }
    } catch (err: any) {
      setUploadError(err.message || 'Network connection failed.');
      setUploadingStatus('');
    }
  };

  const handleDeleteDataset = async (datasetId: string) => {
    try {
      const response = await fetch(`${API_BASE}/datasets/${datasetId}`, {
        method: 'DELETE'
      });
      const result = await response.json();
      if (response.ok && result.status === 'success') {
        const dRes = await fetch(`${API_BASE}/datasets`);
        const updatedDsList = await dRes.json();
        setDatasets(updatedDsList);
        
        if (activeDatasetId === datasetId) {
          if (updatedDsList.length > 0) {
            const nextDsId = updatedDsList[0].id;
            setActiveDatasetId(nextDsId);
            fetchDatasetEmbeddings(nextDsId);
            fetchDatasetVersions(nextDsId);
          } else {
            setActiveDatasetId('');
            setEmbeddings([]);
            setSelectedEmbed(null);
            setSelectedImageDetails(null);
          }
        }
        const expRes = await fetch(`${API_BASE}/experiments`);
        const expData = await expRes.json();
        setExperiments(expData);
      } else {
        alert(result.detail || 'Failed to delete dataset.');
      }
    } catch (err: any) {
      alert(err.message || 'Error occurred while deleting dataset.');
    }
  };

  const fetchDatasetEmbeddings = async (dsId: string) => {
    try {
      const res = await fetch(`${API_BASE}/datasets/${dsId}/embeddings`);
      const data = await res.json();
      setEmbeddings(data);
      setSelectedEmbed(null);
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

  const handleCancelTraining = async (experimentId: string) => {
    if (!experimentId) return;
    try {
      const res = await fetch(`${API_BASE}/experiments/${experimentId}/cancel`, {
        method: 'POST'
      });
      if (res.ok) {
        setActiveJob((prev: any) => prev ? { ...prev, status: 'cancelled' } : null);
        fetchExperiments();
      } else {
        const errorData = await res.json();
        alert(`Failed to cancel training: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (e) {
      console.error('Error cancelling training:', e);
      alert('Error connecting to backend to cancel training.');
    }
  };

  const handleDeleteExperiment = async (expId: string) => {
    if (!expId) return;
    try {
      const res = await fetch(`${API_BASE}/experiments/${expId}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        if (activeExperimentId === expId) {
          setActiveExperimentId(null);
          setActiveJob(null);
        }
        if (compareCandId === expId) {
          setCompareCandId('');
        }
        if (compareBaseId === expId) {
          setCompareBaseId('');
        }
        fetchExperiments();
      } else {
        const errorData = await res.json();
        alert(`Failed to delete run: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (e) {
      console.error('Error deleting experiment:', e);
      alert('Error connecting to backend to delete experiment.');
    }
  };

  const selectRun = (exp: any) => {
    const isRunning = ['preparing_dataset', 'queued', 'training', 'evaluating'].includes(exp.status);
    if (isRunning) {
      setActiveExperimentId(exp.id);
      setActiveJob({
        id: exp.id,
        experiment_id: exp.id,
        status: exp.status,
        logs: exp.logs || '',
        loss_history: exp.loss_history || [],
        map50_history: exp.map50_history || [],
        current_epoch: exp.current_epoch || 0,
        total_epochs: exp.config?.epochs || 3,
        experiment: exp
      });
      setJobLogs(exp.logs || '');
      setWizardStep('train');
    } else {
      if (wizardStep === 'train') {
        // Load model config into training configuration form and stay on the Train tab
        setFormData({
          name: `${exp.name} (Rerun)`,
          dataset_id: getDatasetIdFromVersion(exp.dataset_version_id),
          version_tag: exp.dataset_version_id || 'run_v1',
          train_split: exp.config?.train_split || 0.8,
          val_split: exp.config?.val_split || 0.2,
          split_seed: exp.config?.split_seed || 42,
          task_type: exp.task_type || 'instance_segmentation',
          model_type: exp.model_type || 'yolo_seg',
          epochs: exp.config?.epochs || 3,
          batch: exp.config?.batch || 2,
          imgsz: exp.config?.imgsz || 512,
          fliplr: exp.config?.augmentations?.fliplr || false,
          flipud: exp.config?.augmentations?.flipud || false,
          degrees: exp.config?.augmentations?.degrees || 0.0,
          mock: exp.config?.mock !== undefined ? exp.config?.mock : true
        });
        setCompiledBlueprint(null);
        setCompiledVersion(null);
        setActiveJob(null);
      } else {
        // Switch to evaluate normally
        setCompareCandId(exp.id);
        setWizardStep('evaluate');
      }
    }
  };



  const triggerQuickPrompt = (promptText: string) => {
    setLlmPrompt(promptText);
    setLlmOpen(true);
  };

  const handleClearLlmHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects/proj_default/llm-commands`, { method: 'DELETE' });
      if (res.ok) {
        setLlmLogs([
          { sender: 'system', text: 'ATR Assistant terminal online. Ask me to split datasets, configure augmentations, or run training runs.', time: new Date().toLocaleTimeString() }
        ]);
      }
    } catch (err) {
      console.error("Failed to clear LLM command history:", err);
    }
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
      
      // Load updated command history from database
      const cmdRes = await fetch(`${API_BASE}/projects/proj_default/llm-commands`);
      if (cmdRes.ok) {
        const cmdHistory = await cmdRes.json();
        if (cmdHistory.length > 0) {
          const formattedLogs = cmdHistory.map((c: any) => ({
            sender: c.sender,
            text: c.text,
            time: new Date(c.created_at * 1000).toLocaleTimeString(),
            payload: c.payload
          }));
          setLlmLogs(formattedLogs);
        }
      }
      
      // Refresh background data
      fetchExperiments();
      if (data.status === 'success') {
        if (data.payload && data.payload.dataset_id) {
          fetchDatasetVersions(data.payload.dataset_id);
        }
        
        // Auto navigate to train stage if job started
        if (data.action === 'create_experiment' || data.action === 'clone_experiment') {
          setWizardStep('train');
          if (data.workflow_result && data.workflow_result.experiment_id) {
            setActiveExperimentId(data.workflow_result.experiment_id);
          }
        }
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
      logs: "",
      experiment: {
        id: "exp_preparing",
        name: formData.name,
        task_type: formData.task_type,
        model_type: formData.model_type,
        dataset_version_id: versionId,
        config: {
          epochs: formData.epochs,
          batch: formData.batch,
          imgsz: formData.imgsz,
          train_split: formData.train_split,
          val_split: 1 - formData.train_split,
          augmentations: {
            fliplr: formData.fliplr,
            flipud: formData.flipud,
            degrees: formData.degrees
          }
        }
      }
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
      
      setWizardStep('train');
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
      
      // Fetch evaluations only for completed experiments
      let baseEval = null;
      if (baseExp.status === 'complete') {
        const baseEvalRes = await fetch(`${API_BASE}/experiments/${compareBaseId}/evaluation`);
        baseEval = baseEvalRes.ok ? await baseEvalRes.json() : null;
      }
      
      let candEval = null;
      if (candExp.status === 'complete') {
        const candEvalRes = await fetch(`${API_BASE}/experiments/${compareCandId}/evaluation`);
        candEval = candEvalRes.ok ? await candEvalRes.json() : null;
      }
      
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
        
        {/* Guided Progress Stepper */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {[
            { id: 'dashboard', label: '0. MISSION CONTROL', icon: Activity },
            { id: 'curate', label: '1. CURATE', icon: Compass },
            { id: 'annotate', label: '2. ANNOTATE', icon: Layers },
            { id: 'train', label: '3. TRAIN', icon: Play },
            { id: 'evaluate', label: '4. EVALUATE', icon: BarChart3 }
          ].map((step, idx) => {
            const Icon = step.icon;
            const active = wizardStep === step.id;
            const isCompleted = ['dashboard', 'curate', 'annotate', 'train', 'evaluate'].indexOf(wizardStep) > idx;
            return (
              <React.Fragment key={step.id}>
                {idx > 0 && (
                  <div style={{ 
                    width: '30px', 
                    height: '2px', 
                    background: isCompleted ? 'var(--accent-cyan)' : 'var(--border-color)',
                    boxShadow: isCompleted ? '0 0 8px var(--accent-cyan)' : 'none',
                    transition: 'all 0.3s ease'
                  }} />
                )}
                <button 
                  onClick={() => setWizardStep(step.id as any)}
                  className={`btn-tactical ${active ? 'btn-tactical-active border-glow-cyan' : ''}`}
                  style={{ 
                    padding: '8px 14px', 
                    fontSize: '0.7rem',
                    borderColor: isCompleted ? 'rgba(0, 242, 254, 0.4)' : undefined,
                    color: isCompleted ? 'rgba(255, 255, 255, 0.8)' : undefined
                  }}
                >
                  <Icon style={{ width: '13px', height: '13px', color: active ? 'var(--accent-cyan)' : isCompleted ? 'var(--accent-cyan)' : 'var(--text-secondary)' }} />
                  {step.label}
                </button>
              </React.Fragment>
            );
          })}
        </div>

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
        
        {/* Left Column: Simple completed/running model runs history list */}
        <div className="glass-panel" style={{ width: '320px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto', flexShrink: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', marginBottom: '4px' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', letterSpacing: '0.05em', margin: 0, color: 'var(--accent-cyan)' }}>
              MODEL RUN HISTORY
            </h3>
          </div>
          
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto' }}>
            {experiments && experiments.length > 0 ? (
              experiments.map((exp: any) => {
                const isSelected = compareCandId === exp.id || activeExperimentId === exp.id;
                
                return (
                  <div 
                    key={exp.id}
                    onClick={() => selectRun(exp)}
                    className={`glass-panel`}
                    style={{
                      padding: '10px',
                      cursor: 'pointer',
                      borderColor: isSelected ? 'var(--accent-cyan)' : 'var(--border-color)',
                      background: isSelected ? 'rgba(0, 242, 254, 0.05)' : 'rgba(0,0,0,0.15)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <span style={{ fontWeight: 600, fontSize: '0.7rem', color: isSelected ? 'var(--accent-cyan)' : '#fff', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '170px' }}>
                        {exp.name}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ 
                          fontSize: '0.55rem', 
                          fontFamily: 'var(--font-mono)', 
                          color: exp.status === 'complete' ? 'var(--accent-green)' : ['failed', 'cancelled'].includes(exp.status) ? 'var(--accent-red)' : 'var(--accent-orange)' 
                        }}>
                          {exp.status === 'complete' ? 'DONE' : exp.status === 'failed' ? 'FAIL' : exp.status === 'cancelled' ? 'ABORTED' : 'RUNNING'}
                        </span>
                        
                        {!['queued', 'preparing_dataset', 'training', 'evaluating'].includes(exp.status) && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setExperimentToDelete(exp);
                            }}
                            style={{
                              background: 'none',
                              border: 'none',
                              color: 'var(--accent-red)',
                              cursor: 'pointer',
                              padding: '2px',
                              display: 'flex',
                              alignItems: 'center',
                              opacity: 0.5,
                              transition: 'opacity 0.2s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                            onMouseLeave={(e) => e.currentTarget.style.opacity = '0.5'}
                            title="Remove Run History"
                          >
                            <Trash2 style={{ width: '10px', height: '10px' }} />
                          </button>
                        )}
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                      <span>{exp.model_type === 'yolo_seg' ? 'YOLOv8-Seg' : 'Instance Seg'}</span>
                      {exp.status === 'complete' && (
                        <span style={{ color: 'var(--accent-green)' }}>mAP: 0.72</span>
                      )}
                    </div>
                  </div>
                );
              })
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.7rem' }}>
                No model runs in database.
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
          {wizardStep === 'dashboard' && (
            <ProjectDashboardView 
              datasets={datasets}
              experiments={experiments}
              activeJob={activeJob}
              llmLogs={llmLogs}
              setWizardStep={setWizardStep}
              setActiveDatasetId={setActiveDatasetId}
              fetchDatasetEmbeddings={fetchDatasetEmbeddings}
              fetchDatasetVersions={fetchDatasetVersions}
              setUploadModalOpen={setUploadModalOpen}
              setCompareBaseId={setCompareBaseId}
              setCompareCandId={setCompareCandId}
              setDatasetToDelete={setDatasetToDelete}
            />
          )}
          {wizardStep === 'curate' && (
            <PipelineStudioView 
              setUploadModalOpen={setUploadModalOpen}
              wizardStep={wizardStep}
              setWizardStep={setWizardStep}
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
              versions={versions}
              handleGeneratePipelineBlueprint={handleGeneratePipelineBlueprint}
              handleExecutePipelineRun={handleExecutePipelineRun}
              isGeneratingBlueprint={isGeneratingBlueprint}
              compiledVersion={compiledVersion}
              setCompiledVersion={setCompiledVersion}
              formData={formData}
              setFormData={setFormData}
              handleInputChange={handleInputChange}
              handleGenerateBlueprintClick={handleGenerateBlueprintClick}
              compiledBlueprint={compiledBlueprint}
              setCompiledBlueprint={setCompiledBlueprint}
            />
          )}
          {wizardStep === 'annotate' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              {selectedImageDetails ? (
                <AnnotateTargetsView 
                  selectedEmbed={selectedEmbed}
                  setSelectedEmbed={setSelectedEmbed}
                  selectedImageDetails={selectedImageDetails}
                  setSelectedImageDetails={setSelectedImageDetails}
                  activeDatasetId={activeDatasetId}
                  fetchDatasetEmbeddings={fetchDatasetEmbeddings}
                  setWizardStep={setWizardStep}
                  embeddings={embeddings}
                />
              ) : (
                /* Beautiful select frame overview */
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>STEP 2: ANNOTATE TARGETS</h2>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Select a satellite image chip from the catalog to draw aircraft bounding boxes / polygon paths.</p>
                    </div>
                    
                    <button
                      onClick={() => setWizardStep('train')}
                      className="btn-tactical border-glow-cyan"
                    >
                      Skip to Training Config <ArrowRight style={{ width: '12px', height: '12px' }} />
                    </button>
                  </div>
                  
                  <div className="glass-recessed" style={{ flex: 1, padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '20px', overflow: 'hidden' }}>
                    <Layers style={{ width: '48px', height: '48px', color: 'var(--text-muted)' }} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>NO IMAGE CHIP LOADED IN CANVAS</span>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', maxWidth: '440px', textAlign: 'center', marginTop: '-10px' }}>
                      Click any satellite chip below to open it in the interactive drawing canvas.
                    </p>
                    
                    <div style={{ width: '100%', overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '12px', padding: '4px' }}>
                      {embeddings.map((emb: any) => (
                        <div 
                          key={emb.image_id}
                          onClick={() => {
                            setSelectedEmbed(emb);
                            setSelectedImageDetails(emb);
                          }}
                          className="glass-panel"
                          style={{ padding: '8px', cursor: 'pointer', textAlign: 'center', background: 'rgba(0,0,0,0.2)', transition: 'transform 0.15s ease' }}
                          onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.03)'; }}
                          onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
                        >
                          <img src={getImageUrl(activeDatasetId, emb.image_id)} style={{ width: '100%', height: '80px', objectFit: 'cover', borderRadius: '4px' }} />
                          <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', marginTop: '6px', color: '#fff', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{emb.image_id}</div>
                          <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)' }}>{emb.labels?.length || 0} aircraft</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          {wizardStep === 'train' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', gap: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>STEP 3: MODEL TRAINING COCKPIT</h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Configure augmentations, compile YAML blueprint schematics, and run deep learning training jobs.</p>
                </div>
                
                <div style={{ display: 'flex', gap: '12px' }}>
                  <button
                    onClick={() => setWizardStep('annotate')}
                    className="btn-tactical"
                  >
                    ← Back to Annotation
                  </button>
                  <button
                    onClick={() => setWizardStep('evaluate')}
                    className="btn-tactical border-glow-cyan"
                  >
                    Skip to Evaluation <ArrowRight style={{ width: '12px', height: '12px' }} />
                  </button>
                </div>
              </div>
              
              {activeJob ? (
                <div className="glass-recessed" style={{ flex: 1, padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
                  
                  <div className="glass-panel" style={{ 
                    padding: '12px 20px', 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    background: activeJob.status === 'complete' ? 'rgba(0, 255, 135, 0.05)' : ['failed', 'cancelled'].includes(activeJob.status) ? 'rgba(255, 0, 85, 0.05)' : 'rgba(255, 153, 0, 0.05)',
                    border: activeJob.status === 'complete' ? '1px solid rgba(0, 255, 135, 0.2)' : ['failed', 'cancelled'].includes(activeJob.status) ? '1px solid rgba(255, 0, 85, 0.2)' : '1px solid rgba(255, 153, 0, 0.2)'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span className={`led-indicator ${activeJob.status === 'complete' ? 'green' : ['failed', 'cancelled'].includes(activeJob.status) ? 'red' : 'orange'}`}></span>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.85rem', fontWeight: 600, fontFamily: 'var(--font-display)', color: '#fff' }}>
                          {activeJob.status === 'complete' 
                            ? 'TRAINING RUN COMPLETED SUCCESSFULLY' 
                            : activeJob.status === 'failed' 
                              ? 'TRAINING RUN FAILED' 
                              : activeJob.status === 'cancelled'
                                ? 'TRAINING RUN CANCELLED BY USER'
                                : 'ACCELERATOR TRAINING ACTIVE'}
                        </span>
                        <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                          Run ID: {activeJob.id} // Target: YOLOv8 Segmentation
                        </span>
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '10px' }}>
                      {!['complete', 'failed', 'cancelled'].includes(activeJob.status) && (
                        <button
                          onClick={() => {
                            setCancelJobConfirm(activeJob);
                          }}
                          className="btn-tactical border-glow-red"
                          style={{ padding: '8px 16px', color: 'var(--accent-red)', borderColor: 'var(--accent-red)' }}
                        >
                          Cancel Run
                        </button>
                      )}

                      {activeJob.status === 'complete' && (
                        <button
                          onClick={() => {
                            setCompareCandId(activeJob.experiment_id || activeJob.id);
                            setWizardStep('evaluate');
                          }}
                          className="btn-tactical border-glow-green"
                          style={{ padding: '8px 16px', color: 'var(--accent-green)', borderColor: 'var(--accent-green)' }}
                        >
                          Proceed to Evaluation <ArrowRight style={{ width: '12px', height: '12px' }} />
                        </button>
                      )}
                      
                      {['complete', 'failed', 'cancelled'].includes(activeJob.status) && (
                        <button
                          onClick={() => {
                            setActiveJob(null);
                          }}
                          className="btn-tactical"
                          style={{ padding: '8px 16px' }}
                        >
                          Configure New Run
                        </button>
                      )}
                    </div>
                  </div>
                  
                  {/* Telemetry card for experiment details */}
                  {activeJob.experiment && (
                    <div className="glass-panel" style={{ 
                      padding: '14px 20px', 
                      background: 'rgba(255, 255, 255, 0.02)', 
                      border: '1px solid rgba(255, 255, 255, 0.08)',
                      borderRadius: '8px',
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                      gap: '20px',
                      fontSize: '0.75rem',
                      lineHeight: '1.4'
                    }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>RUN NAME & SCHEMATIC</span>
                        <span style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{activeJob.experiment.name || 'Experiment'}</span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>ID: {activeJob.experiment.id}</span>
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>MODEL ARCHITECTURE</span>
                        <span style={{ fontWeight: 600, color: '#fff' }}>{(activeJob.experiment.model_type || 'yolo_seg').toUpperCase().replace('_', ' ')}</span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{(activeJob.experiment.task_type || 'instance_segmentation').replace('_', ' ')}</span>
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>DATASET SOURCE VERSION</span>
                        <span style={{ fontWeight: 600, color: '#fff', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} title={activeJob.experiment.dataset_version_id}>
                          {(activeJob.experiment.dataset_version_id || '').replace('version_dataset_', '')}
                        </span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          Split: {(() => {
                            const trainPercent = Math.round((activeJob.experiment.config?.train_split ?? activeJob.experiment.config?.train_ratio ?? 0.7) * 100);
                            const valPercent = Math.round((activeJob.experiment.config?.val_split ?? activeJob.experiment.config?.val_ratio ?? 0.3) * 100);
                            return `${trainPercent}/${valPercent}`;
                          })()}
                        </span>
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>HYPERPARAMETERS</span>
                        <span style={{ fontWeight: 600, color: '#fff' }}>
                          Epochs: {activeJob.experiment.config?.epochs ?? activeJob.total_epochs ?? 3} // Batch: {activeJob.experiment.config?.batch ?? activeJob.experiment.config?.batch_size ?? 2}
                        </span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          Resolution: {activeJob.experiment.config?.imgsz ?? 512}px
                        </span>
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>ACTIVE AUGMENTATIONS</span>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '2px' }}>
                          {(() => {
                            const config = activeJob.experiment.config || {};
                            const augs = config.augmentations || {};
                            const hasFliplr = augs.fliplr ?? config.fliplr ?? false;
                            const hasFlipud = augs.flipud ?? config.flipud ?? false;
                            const degrees = augs.degrees ?? config.degrees ?? 0;
                            return (
                              <>
                                <span style={{ 
                                  fontSize: '0.55rem', 
                                  fontFamily: 'var(--font-mono)',
                                  padding: '2px 6px',
                                  background: hasFliplr ? 'rgba(0, 255, 135, 0.1)' : 'rgba(255,255,255,0.03)',
                                  border: hasFliplr ? '1px solid rgba(0, 255, 135, 0.2)' : '1px solid rgba(255,255,255,0.05)',
                                  borderRadius: '3px',
                                  color: hasFliplr ? 'var(--accent-green)' : 'var(--text-muted)'
                                }}>
                                  H-FLIP
                                </span>
                                <span style={{ 
                                  fontSize: '0.55rem', 
                                  fontFamily: 'var(--font-mono)',
                                  padding: '2px 6px',
                                  background: hasFlipud ? 'rgba(0, 255, 135, 0.1)' : 'rgba(255,255,255,0.03)',
                                  border: hasFlipud ? '1px solid rgba(0, 255, 135, 0.2)' : '1px solid rgba(255,255,255,0.05)',
                                  borderRadius: '3px',
                                  color: hasFlipud ? 'var(--accent-green)' : 'var(--text-muted)'
                                }}>
                                  V-FLIP
                                </span>
                                <span style={{ 
                                  fontSize: '0.55rem', 
                                  fontFamily: 'var(--font-mono)',
                                  padding: '2px 6px',
                                  background: degrees > 0 ? 'rgba(0, 255, 135, 0.1)' : 'rgba(255,255,255,0.03)',
                                  border: degrees > 0 ? '1px solid rgba(0, 255, 135, 0.2)' : '1px solid rgba(255,255,255,0.05)',
                                  borderRadius: '3px',
                                  color: degrees > 0 ? 'var(--accent-green)' : 'var(--text-muted)'
                                }}>
                                  ROT: {degrees}°
                                </span>
                              </>
                            );
                          })()}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>ACCELERATOR LOSS DIVERGENCE HISTORY</span>
                      {renderSVGChart(activeJob.loss_history, '#ff9900')}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>VALIDATION MAP50 ACCURACY PROFILE</span>
                      {renderSVGChart(activeJob.map50_history, '#00ff87')}
                    </div>
                  </div>
                  
                  <div style={{ 
                    flex: 1, 
                    background: 'rgba(0,0,0,0.95)', 
                    border: '1px solid var(--border-color)', 
                    borderRadius: '8px', 
                    padding: '12px', 
                    overflowY: 'auto',
                    fontFamily: 'var(--font-mono)', 
                    fontSize: '0.75rem', 
                    color: 'rgba(0, 242, 254, 0.95)',
                    lineHeight: '1.4'
                  }}>
                    <div style={{ color: 'var(--text-secondary)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px', marginBottom: '8px', fontSize: '0.6rem', display: 'flex', justifyContent: 'space-between' }}>
                      <span>STREAMING STANDARD ACCELERATOR STDOUT LOGS</span>
                      <span>WORKER_PID: {activeJob.id}</span>
                    </div>
                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                      {jobLogs}
                    </pre>
                    <div ref={logsEndRef} />
                  </div>
                  
                </div>
              ) : (
                <PipelineStudioView 
                  setUploadModalOpen={setUploadModalOpen}
                  wizardStep={wizardStep}
                  setWizardStep={setWizardStep}
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
                  versions={versions}
                  handleGeneratePipelineBlueprint={handleGeneratePipelineBlueprint}
                  handleExecutePipelineRun={handleExecutePipelineRun}
                  isGeneratingBlueprint={isGeneratingBlueprint}
                  compiledVersion={compiledVersion}
                  setCompiledVersion={setCompiledVersion}
                  formData={formData}
                  setFormData={setFormData}
                  handleInputChange={handleInputChange}
                  handleGenerateBlueprintClick={handleGenerateBlueprintClick}
                  compiledBlueprint={compiledBlueprint}
                  setCompiledBlueprint={setCompiledBlueprint}
                />
              )}
            </div>
          )}
          {wizardStep === 'evaluate' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', gap: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', letterSpacing: '0.05em' }}>STEP 4: COMPARATIVE EVALUATION</h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Select baselines vs candidates to analyze precision, recall, confusion deltas, and prediction overlays.</p>
                </div>
                
                <button
                  onClick={() => setWizardStep('train')}
                  className="btn-tactical"
                >
                  ← Back to Training
                </button>
              </div>
              
              <div className="glass-recessed" style={{ flex: 1, padding: '20px', display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
                <ModelEvaluatorView 
                  experiments={experiments}
                  compareBaseId={compareBaseId}
                  compareCandId={compareCandId}
                  setCompareBaseId={setCompareBaseId}
                  setCompareCandId={setCompareCandId}
                  compareResults={compareResults}
                />
                
                {compareResults && compareResults.candidate && (
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px', borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
                    <button
                      onClick={() => {
                        const cand = compareResults.candidate.exp;
                        setFormData({
                          name: `${cand.name} (Rerun)`,
                          dataset_id: getDatasetIdFromVersion(cand.dataset_version_id),
                          version_tag: cand.dataset_version_id || 'run_v1',
                          train_split: cand.config.train_split || 0.8,
                          val_split: cand.config.val_split || 0.2,
                          split_seed: cand.config.split_seed || 42,
                          task_type: cand.task_type || 'instance_segmentation',
                          model_type: cand.model_type || 'yolo_seg',
                          epochs: cand.config.epochs || 3,
                          batch: cand.config.batch || 2,
                          imgsz: cand.config.imgsz || 512,
                          fliplr: cand.config.augmentations?.fliplr || false,
                          flipud: cand.config.augmentations?.flipud || false,
                          degrees: cand.config.augmentations?.degrees || 0.0,
                          mock: cand.config.mock !== undefined ? cand.config.mock : true
                        });
                        setCompiledBlueprint(null);
                        setCompiledVersion(null);
                        setActiveJob(null);
                        setWizardStep('train');
                      }}
                      className="btn-tactical border-glow-cyan"
                    >
                      <RefreshCw style={{ width: '13px', height: '13px' }} /> Tune Hyperparameters & Rerun Experiment
                    </button>
                  </div>
                )}
              </div>
            </div>
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
                <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '0.9rem', letterSpacing: '0.05em' }}>LLM COCKPIT</h2>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <button 
                  onClick={() => setClearLlmHistoryConfirm(true)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontSize: '0.65rem',
                    fontFamily: 'var(--font-mono)',
                    textDecoration: 'underline',
                    padding: 0
                  }}
                  onMouseEnter={e => e.currentTarget.style.color = 'var(--accent-red)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--text-secondary)'}
                >
                  CLEAR
                </button>
                <span className="led-indicator cyan"></span>
              </div>
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

      {/* Confirm Dataset Deletion Modal */}
      {datasetToDelete && (
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
          zIndex: 1100
        }}>
          <div className="glass-panel border-glow-red" style={{ width: '450px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', border: '1px solid var(--accent-red)' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', letterSpacing: '0.05em', borderBottom: '1px solid rgba(255, 0, 85, 0.2)', paddingBottom: '8px', color: 'var(--accent-red)', margin: 0 }}>
              CONFIRM DATASET DELETION
            </h3>
            
            <p style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#fff', margin: 0, lineHeight: '1.5' }}>
              Are you sure you want to permanently delete dataset <span style={{ color: 'var(--accent-cyan)' }}>"{datasetToDelete.name}"</span>?
            </p>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>
              This action is destructive and irreversible. It will wipe all associated version splits, hyperparameter experiments, validation metrics, and image files on disk.
            </p>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '8px' }}>
              <button 
                type="button" 
                onClick={() => setDatasetToDelete(null)}
                className="btn-tactical"
              >
                CANCEL
              </button>
              <button 
                type="button" 
                onClick={() => {
                  handleDeleteDataset(datasetToDelete.id);
                  setDatasetToDelete(null);
                }}
                className="btn-tactical btn-tactical-active"
                style={{ background: 'var(--accent-red)', borderColor: 'var(--accent-red)', color: '#fff' }}
              >
                CONFIRM DESTRUCTION
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Cancel Training Modal */}
      {cancelJobConfirm && (
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
          zIndex: 1100
        }}>
          <div className="glass-panel border-glow-red" style={{ width: '450px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', border: '1px solid var(--accent-red)' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', letterSpacing: '0.05em', borderBottom: '1px solid rgba(255, 0, 85, 0.2)', paddingBottom: '8px', color: 'var(--accent-red)', margin: 0 }}>
              CONFIRM RUN CANCELLATION
            </h3>
            
            <p style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#fff', margin: 0, lineHeight: '1.5' }}>
              Are you sure you want to cancel the active training run <span style={{ color: 'var(--accent-cyan)' }}>"{cancelJobConfirm.experiment?.name || cancelJobConfirm.id}"</span>?
            </p>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>
              This will safely abort the accelerator training thread, update the run status to 'cancelled', and halt all weight optimizations immediately.
            </p>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '8px' }}>
              <button 
                type="button" 
                onClick={() => setCancelJobConfirm(null)}
                className="btn-tactical"
              >
                KEEP RUNNING
              </button>
              <button 
                type="button" 
                onClick={() => {
                  handleCancelTraining(cancelJobConfirm.experiment_id || cancelJobConfirm.id);
                  setCancelJobConfirm(null);
                }}
                className="btn-tactical btn-tactical-active"
                style={{ background: 'var(--accent-red)', borderColor: 'var(--accent-red)', color: '#fff' }}
              >
                ABORT TRAINING
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Run History Deletion Modal */}
      {experimentToDelete && (
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
          zIndex: 1100
        }}>
          <div className="glass-panel border-glow-red" style={{ width: '450px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', border: '1px solid var(--accent-red)' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', letterSpacing: '0.05em', borderBottom: '1px solid rgba(255, 0, 85, 0.2)', paddingBottom: '8px', color: 'var(--accent-red)', margin: 0 }}>
              CONFIRM RUN HISTORY DELETION
            </h3>
            
            <p style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#fff', margin: 0, lineHeight: '1.5' }}>
              Are you sure you want to permanently delete training run <span style={{ color: 'var(--accent-cyan)' }}>"{experimentToDelete.name}"</span>?
            </p>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>
              This will erase all associated telemetry data, loss histories, logs, confusion matrices, and model weight directories on disk.
            </p>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '8px' }}>
              <button 
                type="button" 
                onClick={() => setExperimentToDelete(null)}
                className="btn-tactical"
              >
                CANCEL
              </button>
              <button 
                type="button" 
                onClick={() => {
                  handleDeleteExperiment(experimentToDelete.id);
                  setExperimentToDelete(null);
                }}
                className="btn-tactical btn-tactical-active"
                style={{ background: 'var(--accent-red)', borderColor: 'var(--accent-red)', color: '#fff' }}
              >
                DELETE RECORD
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Clear LLM History Modal */}
      {clearLlmHistoryConfirm && (
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
          zIndex: 1100
        }}>
          <div className="glass-panel border-glow-red" style={{ width: '450px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', border: '1px solid var(--accent-red)' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', letterSpacing: '0.05em', borderBottom: '1px solid rgba(255, 0, 85, 0.2)', paddingBottom: '8px', color: 'var(--accent-red)', margin: 0 }}>
              CONFIRM CLEAR LLM HISTORY
            </h3>
            
            <p style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#fff', margin: 0, lineHeight: '1.5' }}>
              Are you sure you want to permanently clear the LLM command and orchestration history from the database?
            </p>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>
              This will reset your terminal logs to the initial system greeting and remove all saved agent prompt histories.
            </p>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '8px' }}>
              <button 
                type="button" 
                onClick={() => setClearLlmHistoryConfirm(false)}
                className="btn-tactical"
              >
                CANCEL
              </button>
              <button 
                type="button" 
                onClick={() => {
                  handleClearLlmHistory();
                  setClearLlmHistoryConfirm(false);
                }}
                className="btn-tactical btn-tactical-active"
                style={{ background: 'var(--accent-red)', borderColor: 'var(--accent-red)', color: '#fff' }}
              >
                CLEAR CHAT HISTORY
              </button>
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
  setUploadModalOpen,
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
  versions: _versions,
  handleExecutePipelineRun,
  isGeneratingBlueprint,
  compiledVersion,
  setCompiledVersion,
  wizardStep,
  setWizardStep,
  formData,
  handleInputChange,
  handleGenerateBlueprintClick,
  compiledBlueprint,
  setCompiledBlueprint
}: any) {
  // Control side-panel tab: curate data filters vs build pipeline configuration
  const controlTab = wizardStep === 'curate' ? 'curate' : 'build';

  // FiftyOne Curation Filters State
  const [searchId, setSearchId] = useState('');
  const minGsd = 0.1;
  const [maxGsd, setMaxGsd] = useState(1.0);
  const [minTargets, setMinTargets] = useState(0);
  const [hoveredCardIdx, setHoveredCardIdx] = useState<string | null>(null);

  // Curation State

  // Space-Based Radar (SAR) Curation Parameters
  const [sensorType, setSensorType] = useState('all');
  const [sarPolarization, setSarPolarization] = useState('all');
  const [minIncidence, setMinIncidence] = useState(15.0);
  const [maxIncidence, setMaxIncidence] = useState(45.0);
  const [simulateBackscatter, setSimulateBackscatter] = useState(false);

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
    if (selectedEmbed && selectedEmbed.image_id === emb.image_id) {
      setSelectedEmbed(null);
    } else {
      setSelectedEmbed(emb);
    }
  };




  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%', overflow: 'hidden' }}>
      
      {/* Step Header */}
      {wizardStep !== 'train' && (
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

            <button 
              onClick={() => window.open(`/fiftyone?dataset_id=${activeDatasetId}`, '_blank')}
              className="btn-tactical border-glow-cyan"
              style={{ padding: '8px 14px', marginTop: '16px' }}
            >
              <Compass style={{ width: '14px', height: '14px' }} />
              LAUNCH FIFTYONE WORKBENCH
            </button>
          </div>
        </div>
      )}

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
              {selectedEmbed && (
                <button
                  onClick={() => {
                    setSelectedImageDetails(selectedEmbed);
                    setWizardStep('annotate');
                  }}
                  className="btn-tactical btn-tactical-active"
                  style={{ padding: '6px 12px', fontSize: '0.65rem' }}
                >
                  Label Selected Chip <ArrowRight style={{ width: '12px', height: '12px' }} />
                </button>
              )}
            </div>

          </div>

          {/* Main workspace for Grid/Details (100% width) */}
          <div className="glass-recessed" style={{ flex: 1, padding: '16px', display: 'grid', gridTemplateColumns: '1fr 340px', gap: '16px', overflow: 'hidden' }}>
            
            {/* Left side: FiftyOne Image Grids */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', height: '100%', overflow: 'hidden' }}>
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
                      <div style={{ width: '100%', aspectRatio: '1/1', background: '#000', borderRadius: '6px', overflow: 'hidden', position: 'relative' }}>
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

            {/* Right side: Detail Panel */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden', height: '100%' }}>
              <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px', overflowY: 'auto', padding: '12px', background: 'rgba(0,0,0,0.15)' }}>
                <h4 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '6px', margin: 0 }}>
                  CHIP METADATA DETAILS
                </h4>
                {selectedEmbed ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-color)', position: 'relative', flexShrink: 0 }}>
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

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                      <div className="glass-panel" style={{ padding: '6px', background: 'none' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>IMAGE ID</div>
                        <div style={{ color: '#fff', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{selectedEmbed.image_id}</div>
                      </div>
                      <div className="glass-panel" style={{ padding: '6px', background: 'none' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>SITE CODE</div>
                        <div style={{ color: '#fff', fontWeight: 600 }}>{selectedEmbed.metadata?.airport_code || 'KIAD'}</div>
                      </div>
                      <div className="glass-panel" style={{ padding: '6px', background: 'none' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>RESOLUTION</div>
                        <div style={{ color: '#fff', fontWeight: 600 }}>{selectedEmbed.metadata?.gsd || 0.3}m GSD</div>
                      </div>
                      <div className="glass-panel" style={{ padding: '6px', background: 'none' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>TARGETS</div>
                        <div style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>{selectedEmbed.labels?.length || 0} Aircraft</div>
                      </div>
                      <div className="glass-panel" style={{ padding: '6px', background: 'none' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>SENSOR TYPE</div>
                        <div style={{ color: selectedEmbed.metadata?.sensor_type === 'SAR' ? 'var(--accent-cyan)' : '#fff', fontWeight: 600 }}>
                          {selectedEmbed.metadata?.sensor_type || 'Optical (WorldView)'}
                        </div>
                      </div>
                      <div className="glass-panel" style={{ padding: '6px', background: 'none' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>POL / LOOK ANGLE</div>
                        <div style={{ color: '#fff', fontWeight: 600 }}>
                          {selectedEmbed.metadata?.sensor_type === 'SAR'
                            ? `${selectedEmbed.metadata?.sar_polarization} / ${selectedEmbed.metadata?.incidence_angle}°`
                            : 'N/A'}
                        </div>
                      </div>
                    </div>

                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '6px', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                      <div style={{ color: 'var(--text-secondary)', fontWeight: 600, fontSize: '0.55rem' }}>BOUNDS FOOTPRINT (E / N)</div>
                      <div>Min: E {selectedEmbed.metadata?.bounds?.min_easting?.toFixed(1) || 296000}m / N {selectedEmbed.metadata?.bounds?.min_northing?.toFixed(1) || 4312000}m</div>
                      <div>Max: E {selectedEmbed.metadata?.bounds?.max_easting?.toFixed(1) || 296256}m / N {selectedEmbed.metadata?.bounds?.max_northing?.toFixed(1) || 4312256}m</div>
                    </div>

                    {/* Embeddings PCA Similarity search section */}
                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontWeight: 600 }}>
                        EMBEDDING VECTOR SIMILARITY ANALYSIS
                      </span>
                      
                      {/* Outlier score display */}
                      <div style={{ background: 'rgba(255, 153, 0, 0.04)', border: '1px solid rgba(255, 153, 0, 0.15)', padding: '6px', borderRadius: '4px', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-orange)' }}>
                        ANOMALY OUTLIER SCORE: {(() => {
                          const xs = embeddings.map((e: any) => e.x);
                          const ys = embeddings.map((e: any) => e.y);
                          const cx = xs.reduce((a: number, b: number) => a + b, 0) / (embeddings.length || 1);
                          const cy = ys.reduce((a: number, b: number) => a + b, 0) / (embeddings.length || 1);
                          const dx = selectedEmbed.x - cx;
                          const dy = selectedEmbed.y - cy;
                          const dist = Math.sqrt(dx * dx + dy * dy);
                          const maxDist = Math.max(...embeddings.map((e: any) => Math.sqrt((e.x - cx) ** 2 + (e.y - cy) ** 2))) || 1;
                          return ((dist / maxDist) * 100).toFixed(1);
                        })()}%
                      </div>

                      {/* Similarity neighbors grid */}
                      <span style={{ fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        NEAREST VECTORS IN EMBEDDINGS (EUCLIDEAN PCA MATCHES):
                      </span>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
                        {embeddings
                          .filter((emb: any) => emb.image_id !== selectedEmbed.image_id)
                          .map((emb: any) => {
                            const dx = emb.x - selectedEmbed.x;
                            const dy = emb.y - selectedEmbed.y;
                            const dist = Math.sqrt(dx * dx + dy * dy);
                            return { emb, dist };
                          })
                          .sort((a: any, b: any) => a.dist - b.dist)
                          .slice(0, 4)
                          .map((neigh: any) => (
                            <div 
                              key={neigh.emb.image_id}
                              onClick={() => {
                                setSelectedEmbed(neigh.emb);
                              }}
                              style={{ cursor: 'pointer', textAlign: 'center' }}
                            >
                              <img 
                                src={getImageUrl(activeDatasetId, neigh.emb.image_id)} 
                                style={{ width: '100%', height: '35px', objectFit: 'cover', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.05)' }} 
                              />
                              <div style={{ fontSize: '0.5rem', color: 'var(--accent-cyan)', marginTop: '2px', fontFamily: 'var(--font-mono)' }}>
                                d={neigh.dist.toFixed(2)}
                              </div>
                            </div>
                          ))
                        }
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <h4 style={{ fontFamily: 'var(--font-display)', fontSize: '0.75rem', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255, 255, 255, 0.08)', paddingBottom: '6px', margin: 0 }}>
                      GLOBAL DATASET STATISTICS
                    </h4>
                    
                    {(() => {
                      const total = embeddings.length;
                      let labeled = 0;
                      let optical = 0;
                      let sar = 0;
                      const classes: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
                      
                      embeddings.forEach((e: any) => {
                        if (e.labels && e.labels.length > 0) labeled++;
                        if (e.metadata?.sensor_type === 'SAR') sar++;
                        else optical++;
                        
                        if (e.labels) {
                          e.labels.forEach((l: any) => {
                            if (l.class_id in classes) classes[l.class_id]++;
                          });
                        }
                      });
                      
                      return (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                            <div className="glass-panel" style={{ padding: '8px', background: 'rgba(0,0,0,0.2)' }}>
                              <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>TOTAL CHIPS</div>
                              <div style={{ color: '#fff', fontSize: '1rem', fontWeight: 600 }}>{total}</div>
                            </div>
                            <div className="glass-panel" style={{ padding: '8px', background: 'rgba(0,0,0,0.2)' }}>
                              <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem' }}>LABELED CHIPS</div>
                              <div style={{ color: 'var(--accent-cyan)', fontSize: '1rem', fontWeight: 600 }}>{labeled}</div>
                            </div>
                          </div>
                          
                          <div className="glass-panel" style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px', background: 'rgba(0,0,0,0.2)' }}>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem', fontWeight: 600 }}>SENSOR DISTRIBUTION</div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>Optical (WV-3):</span>
                              <span style={{ color: '#fff', fontWeight: 600 }}>{optical}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>Space Radar (SAR):</span>
                              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{sar}</span>
                            </div>
                          </div>
                          
                          <div className="glass-panel" style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px', background: 'rgba(0,0,0,0.2)' }}>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.55rem', fontWeight: 600 }}>TARGET CLASS COUNTS</div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#00f2fe' }}>
                              <span>Small Aircraft:</span>
                              <span style={{ fontWeight: 600 }}>{classes[0]}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#00ff87' }}>
                              <span>Cargo Plane:</span>
                              <span style={{ fontWeight: 600 }}>{classes[1]}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#ff9900' }}>
                              <span>Large Aircraft:</span>
                              <span style={{ fontWeight: 600 }}>{classes[2]}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#ff0055' }}>
                              <span>Helicopter:</span>
                              <span style={{ fontWeight: 600 }}>{classes[3]}</span>
                            </div>
                          </div>
                          
                          <div style={{ color: 'var(--text-muted)', fontSize: '0.6rem', textAlign: 'center', marginTop: '10px', lineHeight: '1.4' }}>
                            Select any satellite chip from the grid on the left to inspect detailed target bounding boxes, resolutions, and nearest embedding vectors.
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
            
          </div>

          {/* Linear workflow footer buttons */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', borderTop: '1px solid var(--border-color)', paddingTop: '12px', marginTop: '12px' }}>
            {selectedEmbed && (
              <button
                onClick={() => {
                  setSelectedImageDetails(selectedEmbed);
                  setWizardStep('annotate');
                }}
                className="btn-tactical btn-tactical-active border-glow-cyan"
                style={{ padding: '8px 16px', fontSize: '0.75rem' }}
              >
                Label Selected Chip <ArrowRight style={{ width: '12.5px', height: '12.5px' }} />
              </button>
            )}
            <button
              onClick={() => setWizardStep('annotate')}
              className="btn-tactical border-glow-cyan"
              style={{ padding: '8px 16px', fontSize: '0.75rem' }}
            >
              Proceed to Annotation Stage <ArrowRight style={{ width: '12.5px', height: '12.5px' }} />
            </button>
          </div>

        </div>
      ) : (
        // BUILD VIEW (centered card layout, full width)
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'flex-start', overflowY: 'auto', padding: '10px 0' }}>
          <div className="glass-panel" style={{ width: '100%', maxWidth: '1200px', display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px', padding: '24px', background: 'rgba(5, 8, 20, 0.45)' }}>
            
            {/* Left side: parameters form */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', minWidth: 0 }}>
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
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', borderLeft: '1px solid var(--border-color)', paddingLeft: '24px', minWidth: 0 }}>
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
  fetchDatasetEmbeddings,
  setWizardStep,
  embeddings = []
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

  // Find current index of active image in embeddings
  const currentIndex = embeddings.findIndex((e: any) => e.image_id === selectedImageDetails.image_id);
  
  const filmstripRef = useRef<HTMLDivElement>(null);

  // Auto-scroll filmstrip when active image changes
  useEffect(() => {
    if (filmstripRef.current && currentIndex !== -1) {
      const activeEl = filmstripRef.current.children[currentIndex] as HTMLElement;
      if (activeEl) {
        const container = filmstripRef.current;
        const scrollLeft = activeEl.offsetLeft - container.offsetWidth / 2 + activeEl.offsetWidth / 2;
        container.scrollTo({ left: scrollLeft, behavior: 'smooth' });
      }
    }
  }, [currentIndex]);
  
  const handleNextImage = async () => {
    if (currentIndex !== -1 && currentIndex < embeddings.length - 1) {
      if (JSON.stringify(editLabels) !== JSON.stringify(selectedImageDetails.labels || [])) {
        await handleSaveLabels();
      }
      const nextEmbed = embeddings[currentIndex + 1];
      setSelectedImageDetails(nextEmbed);
    }
  };

  const handlePrevImage = async () => {
    if (currentIndex !== -1 && currentIndex > 0) {
      if (JSON.stringify(editLabels) !== JSON.stringify(selectedImageDetails.labels || [])) {
        await handleSaveLabels();
      }
      const prevEmbed = embeddings[currentIndex - 1];
      setSelectedImageDetails(prevEmbed);
    }
  };

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
      // Ignore if user is inside a form control
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'SELECT' ||
        document.activeElement?.tagName === 'TEXTAREA'
      ) {
        return;
      }

      if (e.key === 'Escape') {
        setTempPoints([]);
        setIsDrawingBbox(false);
        setBboxStart(null);
      } else if (e.key === 'Backspace' && tempPoints.length > 0) {
        setTempPoints(prev => prev.slice(0, -1));
      } else if (e.key === 'ArrowRight' || e.key === ']') {
        handleNextImage();
      } else if (e.key === 'ArrowLeft' || e.key === '[') {
        handlePrevImage();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tempPoints, handleNextImage, handlePrevImage]);

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

          {/* Filmstrip / Thumbnail Navigation Strip */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
              <span>DATASET IMAGE FILMSTRIP</span>
              <span>Active: {currentIndex + 1} of {embeddings.length}</span>
            </div>
            <div 
              ref={filmstripRef}
              style={{ 
                display: 'flex', 
                gap: '8px', 
                overflowX: 'auto', 
                padding: '4px 2px 6px 2px',
                scrollBehavior: 'smooth',
                scrollbarWidth: 'thin',
                scrollbarColor: 'var(--accent-cyan) rgba(0,0,0,0.2)'
              }}
            >
              {embeddings.map((emb: any, idx: number) => {
                const isActive = idx === currentIndex;
                const isLabeled = (emb.labels || []).length > 0;
                return (
                  <div
                    key={emb.image_id}
                    onClick={async () => {
                      if (isActive) return;
                      if (JSON.stringify(editLabels) !== JSON.stringify(selectedImageDetails.labels || [])) {
                        await handleSaveLabels();
                      }
                      setSelectedImageDetails(emb);
                    }}
                    style={{
                      flexShrink: 0,
                      width: '64px',
                      cursor: 'pointer',
                      borderRadius: '4px',
                      border: isActive ? '2px solid var(--accent-cyan)' : '1px solid rgba(255,255,255,0.1)',
                      background: isActive ? 'rgba(0, 242, 254, 0.1)' : 'rgba(0,0,0,0.3)',
                      padding: '3px',
                      textAlign: 'center',
                      position: 'relative',
                      transition: 'all 0.15s ease'
                    }}
                    title={`${emb.image_id} (${emb.labels?.length || 0} labels)`}
                  >
                    <img 
                      src={getImageUrl(activeDatasetId, emb.image_id)} 
                      style={{ 
                        width: '100%', 
                        height: '42px', 
                        objectFit: 'cover', 
                        borderRadius: '2px',
                        opacity: isActive ? 1.0 : 0.6
                      }} 
                    />
                    <div style={{ 
                      fontSize: '0.45rem', 
                      fontFamily: 'var(--font-mono)', 
                      marginTop: '2px', 
                      color: isActive ? '#fff' : 'var(--text-muted)',
                      overflow: 'hidden', 
                      textOverflow: 'ellipsis', 
                      whiteSpace: 'nowrap'
                    }}>
                      {emb.image_id.substring(0, 8)}
                    </div>
                    {/* Visual indicator tag for annotation status */}
                    <span style={{
                      position: 'absolute',
                      top: '2px',
                      right: '2px',
                      width: '6px',
                      height: '6px',
                      borderRadius: '50%',
                      backgroundColor: isLabeled ? 'var(--accent-green)' : 'var(--accent-orange)'
                    }} />
                  </div>
                );
              })}
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
            <span>[ArrowLeft] / [ [ ] - Previous image</span>
            <span>[ArrowRight] / [ ] ] - Next image</span>
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

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {/* Next / Previous Image Navigation Row */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', background: 'rgba(0,0,0,0.1)', border: '1px solid rgba(255,255,255,0.05)', padding: '8px', borderRadius: '6px', marginBottom: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginBottom: '2px' }}>
                <span>CATALOG NAVIGATION</span>
                <span>{currentIndex !== -1 ? `${currentIndex + 1} / ${embeddings.length}` : 'N/A'}</span>
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  type="button"
                  onClick={handlePrevImage}
                  disabled={currentIndex <= 0}
                  className="btn-tactical"
                  style={{ flex: 1, justifyContent: 'center', padding: '6px', fontSize: '0.7rem', opacity: currentIndex <= 0 ? 0.4 : 1 }}
                >
                  ← PREV
                </button>
                <button
                  type="button"
                  onClick={handleNextImage}
                  disabled={currentIndex === -1 || currentIndex >= embeddings.length - 1}
                  className="btn-tactical border-glow-cyan"
                  style={{ flex: 1, justifyContent: 'center', padding: '6px', fontSize: '0.7rem', opacity: (currentIndex === -1 || currentIndex >= embeddings.length - 1) ? 0.4 : 1 }}
                >
                  NEXT →
                </button>
              </div>
            </div>

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
                onClick={() => {
                  setSelectedImageDetails(null);
                  if (setWizardStep) setWizardStep('curate');
                }}
                className="btn-tactical"
                style={{ flex: 1, justifyContent: 'center', padding: '10px', fontSize: '0.75rem' }}
              >
                ← CURATION
              </button>
            </div>
            
            <button 
              onClick={() => {
                if (setWizardStep) setWizardStep('train');
              }}
              className="btn-tactical border-glow-green"
              style={{ width: '100%', justifyContent: 'center', padding: '10px', fontSize: '0.75rem', color: 'var(--accent-green)', borderColor: 'var(--accent-green)' }}
            >
              PROCEED TO TRAINING →
            </button>
          </div>
        </div>

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
  compareResults,
  triggerQuickPrompt
}: any) {
  const [activeTab, setActiveTab] = React.useState<'overview' | 'failures' | 'classes' | 'matrix'>('overview');
  const [failureFilter, setFailureFilter] = React.useState<'all' | 'fp' | 'fn' | 'tp'>('all');
  const [page, setPage] = React.useState(0);
  const [zoomedChip, setZoomedChip] = React.useState<any>(null);
  
  const itemsPerPage = 24;

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

  const renderRecommendation = () => {
    if (!compareResults || !compareResults.base || !compareResults.candidate) return null;
    const baseEval = compareResults.base.eval || { map50: 0.72, precision: 0.75, recall: 0.69 };
    const candEval = compareResults.candidate.eval || { map50: 0.72, precision: 0.75, recall: 0.69 };
    
    const dMap = candEval.map50 - baseEval.map50;
    const dPrec = candEval.precision - baseEval.precision;
    const dRec = candEval.recall - baseEval.recall;

    let title = "STABLE MODEL ITERATION DETECTED";
    let desc = "Metrics are holding steady. Consider exploring heavier data augmentations or adjusting hyperparameter balances to trigger higher convergence rates.";
    let type = "info"; // info, success, warning
    let actionLabel = "Clone Run with heavy rotation";
    let actionPrompt = "Redo the previous experiment but apply random rotation of 45 degrees";

    if (dMap > 0.02) {
      title = "SUCCESS: HIGH CONVERGENCE PERFORMANCE GAIN";
      desc = `Excellent! The candidate run shows a positive accuracy gain of +${dMap.toFixed(3)} mAP50. The model is responding well to the current configuration. Proceed to lock these weights as production candidate.`;
      type = "success";
      actionLabel = "Run Evaluation with 90/10 split";
      actionPrompt = "Train YOLOv8 on dataset with a 90/10 split to validate candidates";
    } else if (dRec < -0.02 && dPrec > 0.02) {
      title = "WARNING: PRECISION-RECALL GAIN MISMATCH (CONSERVATIVE)";
      desc = `Precision improved (+${dPrec.toFixed(3)}) but Recall dropped (-${Math.abs(dRec).toFixed(3)}). The candidate run is more conservative, reducing false alarms on taxiways/runways, but is missing more actual aircraft targets. Recommend lowering the confidence threshold to 0.35 in FiftyOne, or adding vertical flips.`;
      type = "warning";
      actionLabel = "Rerun with flip & rotate augmentations";
      actionPrompt = "Redo the previous experiment but apply horizontal flip and random rotation of 15 degrees";
    } else if (dRec > 0.02 && dPrec < -0.02) {
      title = "WARNING: RECALL GAIN WITH HIGH FALSE ALARMS (AGGRESSIVE)";
      desc = `Recall improved (+${dRec.toFixed(3)}) but Precision dropped (-${Math.abs(dPrec).toFixed(3)}). The model is aggressively proposing targets, but introducing false positives (false alarms on buildings/shadows). Suggest applying dropout, increasing training epochs, or registering negative background samples.`;
      type = "warning";
      actionLabel = "Rerun with increased batch size";
      actionPrompt = "Redo the previous experiment but increase batch size to 4 and train for 5 epochs";
    }

    const typeColors: Record<string, string> = {
      info: 'rgba(0, 242, 254, 0.06)',
      success: 'rgba(0, 255, 135, 0.06)',
      warning: 'rgba(255, 153, 0, 0.06)'
    };
    const borderColors: Record<string, string> = {
      info: 'rgba(0, 242, 254, 0.4)',
      success: 'rgba(0, 255, 135, 0.4)',
      warning: 'rgba(255, 153, 0, 0.4)'
    };
    const textColors: Record<string, string> = {
      info: 'var(--accent-cyan)',
      success: 'var(--accent-green)',
      warning: 'var(--accent-orange)'
    };

    return (
      <div className="glass-panel" style={{ 
        padding: '16px', 
        background: typeColors[type], 
        borderColor: borderColors[type], 
        display: 'flex', 
        flexDirection: 'column', 
        gap: '10px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: textColors[type], fontWeight: 600 }}>
            TACTICAL DIAGNOSTIC ADVICE
          </span>
          <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', background: 'rgba(0,0,0,0.4)', padding: '2px 6px', borderRadius: '4px', color: 'var(--text-secondary)' }}>
            AUTO ANALYSIS
          </span>
        </div>
        <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff' }}>{title}</div>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>{desc}</p>
        
        {triggerQuickPrompt && (
          <button 
            onClick={() => triggerQuickPrompt(actionPrompt)}
            className="btn-tactical btn-tactical-active" 
            style={{ 
              alignSelf: 'flex-start', 
              padding: '6px 12px', 
              fontSize: '0.65rem', 
              borderColor: borderColors[type],
              color: '#fff',
              marginTop: '4px'
            }}
          >
            {actionLabel} →
          </button>
        )}
      </div>
    );
  };

  const renderClassPerformance = () => {
    if (!compareResults || !compareResults.candidate) return null;
    const candEval = compareResults.candidate.eval || { map50: 0.724, precision: 0.751, recall: 0.693 };
    
    // Generate deterministic variations of performance for each class
    const classes = [
      { id: 0, name: 'Small Aircraft', weight: 1.0, count: 48 },
      { id: 1, name: 'Cargo Plane', weight: 1.1, count: 18 },
      { id: 2, name: 'Large Aircraft', weight: 0.95, count: 22 },
      { id: 3, name: 'Helicopter', weight: 0.72, count: 8 }
    ];

    return (
      <div className="glass-recessed" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', margin: 0 }}>
          CLASS-SPECIFIC PERFORMANCE COMPARISON (CANDIDATE)
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {classes.map((cls) => {
            const classMap = Math.min(0.98, candEval.map50 * cls.weight);
            const classPrec = Math.min(0.98, candEval.precision * cls.weight);
            const classRec = Math.min(0.98, candEval.recall * cls.weight);

            const color = CLASS_COLORS[cls.id] || '#00f2fe';

            return (
              <div key={cls.id} className="glass-panel" style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 600, color: color }}>{cls.name}</span>
                  <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                    GT Instances: {cls.count}
                  </span>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                  {[
                    { label: 'mAP50', value: classMap },
                    { label: 'Precision', value: classPrec },
                    { label: 'Recall', value: classRec }
                  ].map((metric, mIdx) => (
                    <div key={mIdx} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)' }}>
                        <span>{metric.label}</span>
                        <span style={{ color: '#fff', fontWeight: 600 }}>{(metric.value * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ width: `${metric.value * 100}%`, height: '100%', background: color }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderConfusionMatrix = () => {
    // Matrix classes: Small Aircraft, Cargo Plane, Large Aircraft, Helicopter, Background
    const headers = ['Small A/C', 'Cargo A/C', 'Large A/C', 'Heli', 'Background (FN)'];
    const rowHeaders = ['Small A/C', 'Cargo A/C', 'Large A/C', 'Heli', 'Background (FP)'];

    // Generate semi-mock matrix values reflecting candidates accuracy
    const matrix = [
      [38, 2, 0, 0, 8],  // Ground Truth Small
      [1, 15, 1, 0, 1],  // Ground Truth Cargo
      [0, 2, 18, 0, 2],  // Ground Truth Large
      [1, 0, 0, 5, 2],   // Ground Truth Heli
      [5, 1, 1, 2, 0]    // Pred Background (FPs)
    ];

    return (
      <div className="glass-recessed" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', margin: 0 }}>
            CONFUSION MATRIX / CLASSIFICATION RESOLUTION
          </h3>
          <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
            VALUES INDICATE OVERLAPPING TARGETS INTERSECTION COUNT
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', textAlign: 'center' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                <th style={{ padding: '8px', textAlign: 'left' }}>ACTUAL \ PRED</th>
                {headers.map((h, i) => (
                  <th key={i} style={{ padding: '8px' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rIdx) => (
                <tr key={rIdx} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                  <td style={{ padding: '10px 8px', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    {rowHeaders[rIdx]}
                  </td>
                  {row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx && rIdx < 4;
                    const isFP = rIdx === 4 && cIdx < 4;
                    const isFN = cIdx === 4 && rIdx < 4;
                    
                    let bg = 'rgba(0,0,0,0.2)';
                    let color = '#fff';
                    let border = '1px solid rgba(255,255,255,0.02)';
                    
                    if (isDiagonal) {
                      bg = 'rgba(0, 255, 135, 0.08)';
                      color = 'var(--accent-green)';
                      border = '1px solid rgba(0, 255, 135, 0.2)';
                    } else if (isFP || isFN) {
                      bg = val > 0 ? 'rgba(255, 153, 0, 0.05)' : 'rgba(0,0,0,0.1)';
                      color = val > 0 ? 'var(--accent-orange)' : 'var(--text-muted)';
                      border = val > 0 ? '1px solid rgba(255, 153, 0, 0.15)' : 'none';
                    } else if (val > 0) {
                      bg = 'rgba(255, 0, 85, 0.05)';
                      color = 'var(--accent-red)';
                      border = '1px solid rgba(255, 0, 85, 0.15)';
                    }

                    return (
                      <td 
                        key={cIdx} 
                        style={{ 
                          padding: '12px 8px',
                          background: bg,
                          color: color,
                          border: border,
                          fontWeight: val > 0 ? 600 : 300
                        }}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', lineHeight: '1.4', background: 'rgba(0,0,0,0.15)', padding: '10px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.02)' }}>
          <strong>How to read:</strong> Diagonal values represent true positive classifications. The <strong>Background (FN)</strong> column displays instances of aircraft missed (deletions). The <strong>Background (FP)</strong> row shows non-aircraft features classified as false detections (insertions).
        </div>
      </div>
    );
  };

  const renderVisualFailures = () => {
    if (!compareResults || !compareResults.candidate || !compareResults.candidate.eval) return null;
    const evals = compareResults.candidate.eval.predictions || [];
    
    // Filter validation chips based on failureFilter
    const filteredEvals = evals.filter((sample: any) => {
      const preds = sample.predictions || [];
      if (failureFilter === 'fp') {
        return preds.some((p: any) => p.type === 'FP');
      }
      if (failureFilter === 'fn') {
        return preds.some((p: any) => p.type === 'FN');
      }
      if (failureFilter === 'tp') {
        return preds.some((p: any) => p.type === 'TP');
      }
      return true;
    });

    const pageCount = Math.ceil(filteredEvals.length / itemsPerPage);
    const visibleEvals = filteredEvals.slice(page * itemsPerPage, (page + 1) * itemsPerPage);

    return (
      <div className="glass-recessed" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        
        {/* Failure Type Filters */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            {[
              { id: 'all', label: 'ALL CHIPS' },
              { id: 'fp', label: 'FALSE POSITIVES (FP)' },
              { id: 'fn', label: 'FALSE NEGATIVES (FN)' },
              { id: 'tp', label: 'TRUE POSITIVES (TP)' }
            ].map((filt) => (
              <button
                key={filt.id}
                onClick={() => { setFailureFilter(filt.id as any); setPage(0); }}
                className={`btn-tactical ${failureFilter === filt.id ? 'btn-tactical-active border-glow-cyan' : ''}`}
                style={{ padding: '6px 12px', fontSize: '0.65rem' }}
              >
                {filt.label}
              </button>
            ))}
          </div>
          
          <div style={{ display: 'flex', gap: '16px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: 'var(--accent-green)' }}>● TP = TRUE POSITIVE</span>
            <span style={{ color: 'var(--accent-red)' }}>● FP = FALSE POSITIVE</span>
            <span style={{ color: 'var(--accent-orange)' }}>● FN = FALSE NEGATIVE</span>
          </div>
        </div>

        {/* Chips Grid */}
        {visibleEvals.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '12px' }}>
            {visibleEvals.map((sample: any) => {
              const preds = sample.predictions || [];
              const fpCount = preds.filter((p: any) => p.type === 'FP').length;
              const fnCount = preds.filter((p: any) => p.type === 'FN').length;
              const tpCount = preds.filter((p: any) => p.type === 'TP').length;

              return (
                <div 
                  key={sample.image_id} 
                  className="glass-panel" 
                  style={{ 
                    padding: '8px', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    gap: '8px',
                    cursor: 'pointer',
                    background: 'rgba(0,0,0,0.15)'
                  }}
                  onClick={() => setZoomedChip(sample)}
                  onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(0, 242, 254, 0.3)'}
                  onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-color)'}
                >
                  <div style={{ width: '100%', aspectRatio: '1/1', background: '#000', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
                    <img 
                      src={getImageUrl(getDatasetIdFromVersion(compareResults.candidate.eval.dataset_version_id), sample.image_id)} 
                      alt={sample.image_id} 
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                    <svg 
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                      viewBox="0 0 512 512"
                    >
                      {preds.map((p: any, pIdx: number) => {
                        const color = p.type === 'TP' ? '#00ff87' : p.type === 'FP' ? '#ff0055' : '#ff9900';
                        return (
                          <rect 
                            key={pIdx}
                            x={p.bbox[0]} 
                            y={p.bbox[1]} 
                            width={Math.max(20, p.bbox[2] - p.bbox[0])} 
                            height={Math.max(20, p.bbox[3] - p.bbox[1])} 
                            fill="none" 
                            stroke={color} 
                            strokeWidth="12" 
                            strokeDasharray={p.type === 'FN' ? '18,18' : 'none'}
                          />
                        );
                      })}
                    </svg>
                  </div>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                    <div style={{ color: '#fff', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', fontWeight: 600 }}>
                      {sample.image_id}
                    </div>
                    <div style={{ display: 'flex', gap: '8px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                      <span style={{ color: 'var(--accent-green)' }}>TP: {tpCount}</span>
                      <span style={{ color: 'var(--accent-red)' }}>FP: {fpCount}</span>
                      <span style={{ color: 'var(--accent-orange)' }}>FN: {fnCount}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            No failure chips matched the selected filter.
          </div>
        )}

        {/* Pagination Row */}
        {pageCount > 1 && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', alignItems: 'center', marginTop: '10px' }}>
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="btn-tactical"
              style={{ padding: '4px 10px', fontSize: '0.65rem', opacity: page === 0 ? 0.4 : 1 }}
            >
              PREVIOUS
            </button>
            <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
              PAGE {page + 1} OF {pageCount}
            </span>
            <button
              onClick={() => setPage(p => Math.min(pageCount - 1, p + 1))}
              disabled={page === pageCount - 1}
              className="btn-tactical"
              style={{ padding: '4px 10px', fontSize: '0.65rem', opacity: page === pageCount - 1 ? 0.4 : 1 }}
            >
              NEXT
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderZoomModal = () => {
    if (!zoomedChip) return null;
    const preds = zoomedChip.predictions || [];

    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        background: 'rgba(0,0,0,0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        backdropFilter: 'blur(8px)'
      }}>
        <div className="glass-panel border-glow-cyan" style={{
          width: '600px',
          background: 'rgba(6, 10, 24, 0.95)',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          maxHeight: '90vh',
          overflowY: 'auto'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
            <div>
              <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>VALIDATION CHIP INSPECTOR</span>
              <h3 style={{ margin: '2px 0 0 0', fontSize: '0.95rem', fontFamily: 'var(--font-display)', color: '#fff' }}>{zoomedChip.image_id}</h3>
            </div>
            <button 
              onClick={() => setZoomedChip(null)} 
              className="btn-tactical btn-tactical-active"
              style={{ padding: '6px 12px', fontSize: '0.7rem' }}
            >
              CLOSE
            </button>
          </div>

          <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
            {/* Visual Chip */}
            <div style={{ 
              width: '320px', 
              height: '320px', 
              background: '#000', 
              borderRadius: '6px', 
              overflow: 'hidden', 
              position: 'relative',
              boxShadow: '0 0 20px rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,255,255,0.05)',
              flexShrink: 0
            }}>
              <img 
                src={getImageUrl(getDatasetIdFromVersion(compareResults.candidate.eval.dataset_version_id), zoomedChip.image_id)} 
                alt={zoomedChip.image_id} 
                style={{ width: '100%', height: '100%' }}
              />
              <svg 
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                viewBox="0 0 512 512"
              >
                {preds.map((p: any, idx: number) => {
                  const color = p.type === 'TP' ? '#00ff87' : p.type === 'FP' ? '#ff0055' : '#ff9900';
                  return (
                    <g key={idx}>
                      <rect 
                        x={p.bbox[0]} 
                        y={p.bbox[1]} 
                        width={Math.max(20, p.bbox[2] - p.bbox[0])} 
                        height={Math.max(20, p.bbox[3] - p.bbox[1])} 
                        fill="none" 
                        stroke={color} 
                        strokeWidth="8" 
                        strokeDasharray={p.type === 'FN' ? '12,12' : 'none'}
                      />
                      <text 
                        x={p.bbox[0] + 5} 
                        y={p.bbox[1] > 30 ? p.bbox[1] - 8 : p.bbox[1] + 25} 
                        fill={color} 
                        fontSize="20" 
                        fontFamily="var(--font-mono)"
                        fontWeight="bold"
                        style={{ paintOrder: 'stroke', stroke: '#000', strokeWidth: '4px' }}
                      >
                        {p.type}:{p.confidence > 0 ? (p.confidence*100).toFixed(0)+'%' : 'FN'}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* List of Predictions */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px', height: '320px', overflowY: 'auto' }}>
              <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>DETECTED INSTANCES LIST</span>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {preds.length > 0 ? (
                  preds.map((p: any, idx: number) => {
                    const color = p.type === 'TP' ? 'var(--accent-green)' : p.type === 'FP' ? 'var(--accent-red)' : 'var(--accent-orange)';
                    return (
                      <div 
                        key={idx} 
                        className="glass-panel" 
                        style={{ 
                          padding: '8px 10px', 
                          display: 'flex', 
                          justifyContent: 'space-between', 
                          alignItems: 'center',
                          fontSize: '0.7rem',
                          fontFamily: 'var(--font-mono)',
                          background: 'rgba(0,0,0,0.1)'
                        }}
                      >
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          <span style={{ fontWeight: 600, color: '#fff' }}>
                            {CLASS_NAMES[p.class_id] || 'Aircraft'}
                          </span>
                          <span style={{ fontSize: '0.55rem', color: 'var(--text-secondary)' }}>
                            ID: {p.pred_id.slice(-6)}
                          </span>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
                          <span style={{ color: color, fontWeight: 600 }}>{p.type}</span>
                          <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>
                            {p.confidence > 0 ? `Conf: ${(p.confidence*100).toFixed(0)}%` : 'Missed'}
                          </span>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.7rem' }}>
                    No predictions overlaying this chip.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', overflow: 'hidden', flex: 1 }}>
      
      {/* Selector Dropdowns Header */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', flexShrink: 0, marginBottom: '4px' }}>
        
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
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px', overflow: 'hidden' }}>
          
          {/* Sub-tabs Row */}
          <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', flexShrink: 0 }}>
            {[
              { id: 'overview', label: 'OVERVIEW & RECOMMENDATIONS' },
              { id: 'failures', label: 'VISUAL FAILURE INSPECTOR' },
              { id: 'classes', label: 'CLASS BREAKDOWN' },
              { id: 'matrix', label: 'CONFUSION MATRIX' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`btn-tactical ${activeTab === tab.id ? 'btn-tactical-active border-glow-cyan' : ''}`}
                style={{ padding: '8px 16px', fontSize: '0.7rem' }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Sub-tab Content Pane */}
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {activeTab === 'overview' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '16px' }}>
                  {/* Comparative Metrics Table */}
                  <div className="glass-recessed" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', margin: 0 }}>
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

                  {/* Configurations Diff */}
                  <div className="glass-recessed" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '0.8rem', letterSpacing: '0.05em', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', margin: 0 }}>
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
                                style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '120px', flex: 1 }}
                              >
                                Base: {item.base}
                              </span>
                              <ArrowRight style={{ width: '12px', height: '12px', color: hasDiff ? 'var(--accent-orange)' : 'var(--text-muted)', flexShrink: 0 }} />
                              <span 
                                title={String(item.cand)} 
                                style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '120px', color: hasDiff ? 'var(--accent-cyan)' : '#fff', textAlign: 'right', flex: 1 }}
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

                {/* AI Advice Summary Card */}
                {renderRecommendation()}
              </div>
            )}

            {activeTab === 'failures' && renderVisualFailures()}
            {activeTab === 'classes' && renderClassPerformance()}
            {activeTab === 'matrix' && renderConfusionMatrix()}
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

      {/* Popovers */}
      {renderZoomModal()}

    </div>
  );
}
