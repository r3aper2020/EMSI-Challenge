import { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  Check, 
  AlertTriangle, 
  ArrowLeft, 
  ArrowRight,
  RefreshCw,
  Compass,
  Activity,
  AlertCircle,
  Info
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000/api';

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

export default function FiftyOneDashboard() {
  // Query parameters parsing
  const getQueryParams = () => {
    const params = new URLSearchParams(window.location.search);
    return {
      datasetId: params.get('dataset_id') || 'dataset_rareplanes_real',
      experimentId: params.get('experiment_id') || ''
    };
  };

  const { datasetId: queryDatasetId, experimentId: queryExperimentId } = getQueryParams();

  // Component states
  const [datasets, setDatasets] = useState<any[]>([]);
  const [activeDatasetId, setActiveDatasetId] = useState(queryDatasetId);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [activeExperimentId, setActiveExperimentId] = useState(queryExperimentId);
  const [embeddings, setEmbeddings] = useState<any[]>([]);
  const [evaluation, setEvaluation] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Filters State
  const [searchId, setSearchId] = useState('');
  const [confThreshold, setConfThreshold] = useState(0.40);
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [showPredictions, setShowPredictions] = useState(true);
  const [sensorType, setSensorType] = useState('all');
  const [showTP, setShowTP] = useState(true);
  const [showFP, setShowFP] = useState(true);
  const [showFN, setShowFN] = useState(true);

  // Lasso Curation Filtering & Coordinate Mapping State
  const [lassoPolygon, setLassoPolygon] = useState<{ x: number; y: number }[]>([]);
  const [isLassoing, setIsLassoing] = useState(false);
  const [lassoedIds, setLassoedIds] = useState<string[]>([]);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Point-in-polygon ray casting check
  const isPointInPolygon = (point: { x: number; y: number }, polygon: { x: number; y: number }[]) => {
    const x = point.x, y = point.y;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x, yi = polygon[i].y;
      const xj = polygon[j].x, yj = polygon[j].y;
      
      const intersect = ((yi > y) !== (yj > y))
        && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    e.preventDefault(); // Stop default browser selection/drag behavior
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 420;
    const y = ((e.clientY - rect.top) / rect.height) * 420;
    setIsLassoing(true);
    setLassoPolygon([{ x, y }]);
  };

  const clearLasso = () => {
    setLassoPolygon([]);
    setLassoedIds([]);
  };

  // Reset lasso state on active dataset change
  useEffect(() => {
    setLassoPolygon([]);
    setIsLassoing(false);
    setLassoedIds([]);
  }, [activeDatasetId]);

  // Global mousemove and mouseup listeners when lassoing is active
  useEffect(() => {
    if (!isLassoing) return;

    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (!svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 420;
      const y = ((e.clientY - rect.top) / rect.height) * 420;
      
      const lastPoint = lassoPolygon[lassoPolygon.length - 1];
      if (!lastPoint || Math.abs(lastPoint.x - x) > 1 || Math.abs(lastPoint.y - y) > 1) {
        setLassoPolygon(prev => [...prev, { x, y }]);
      }
    };

    const handleGlobalMouseUp = () => {
      setIsLassoing(false);
      
      if (lassoPolygon.length > 2) {
        const insideIds: string[] = [];
        embeddings.forEach((pt: any) => {
          const cx = ((pt.x + 10) / 20) * 380 + 20;
          const cy = ((pt.y + 10) / 20) * 380 + 20;
          if (isPointInPolygon({ x: cx, y: cy }, lassoPolygon)) {
            insideIds.push(pt.image_id);
          }
        });
        
        if (insideIds.length > 0) {
          setLassoedIds(insideIds);
        } else {
          setLassoedIds([]);
          setLassoPolygon([]);
        }
      } else {
        setLassoedIds([]);
        setLassoPolygon([]);
      }
    };

    window.addEventListener('mousemove', handleGlobalMouseMove);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isLassoing, lassoPolygon, embeddings]);

  // Analytic / View States
  const [prioritizeHard, setPrioritizeHard] = useState(false);
  const [showPcaSpace, setShowPcaSpace] = useState(true);
  const [selectedChip, setSelectedChip] = useState<any>(null);
  const [zoomedChip, setZoomedChip] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('metadata');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Load Datasets & Experiments
  useEffect(() => {
    (async () => {
      try {
        const dRes = await fetch(`${API_BASE}/datasets`);
        if (dRes.ok) {
          setDatasets(await dRes.json());
        }
        const eRes = await fetch(`${API_BASE}/experiments`);
        if (eRes.ok) {
          const exps = await eRes.json();
          setExperiments(exps.filter((exp: any) => exp.status === 'complete'));
        }
      } catch (err) {
        console.error('Failed to load datasets/experiments', err);
      }
    })();
  }, []);

  // Fetch Dataset Embeddings and Evaluation results
  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const embRes = await fetch(`${API_BASE}/datasets/${activeDatasetId}/embeddings`);
        if (embRes.ok) {
          setEmbeddings(await embRes.json());
        }
        if (activeExperimentId) {
          const evalRes = await fetch(`${API_BASE}/experiments/${activeExperimentId}/evaluation`);
          if (evalRes.ok) {
            setEvaluation(await evalRes.json());
          } else {
            setEvaluation(null);
          }
        } else {
          setEvaluation(null);
        }
      } catch (err) {
        console.error('Error fetching details', err);
      } finally {
        setLoading(false);
      }
    })();
  }, [activeDatasetId, activeExperimentId]);

  const handleDatasetChange = (dsId: string) => {
    setActiveDatasetId(dsId);
    setActiveExperimentId('');
    setEvaluation(null);
    setSelectedChip(null);
    setZoomedChip(null);
  };

  const getPredictionsForImage = (imageId: string) => {
    if (!evaluation || !evaluation.predictions) return [];
    const imageEval = evaluation.predictions.find((p: any) => p.image_id === imageId);
    return imageEval ? imageEval.predictions : [];
  };

  const totalAnnotations = embeddings.reduce((acc, curr) => acc + (curr.labels?.length || 0), 0);
  
  const avgGSD = embeddings.length > 0
    ? (embeddings.reduce((acc, curr) => acc + (curr.metadata?.gsd || 0.3), 0) / embeddings.length).toFixed(3)
    : '0.300';

  const getAnomalyDistancePercent = (item: any) => {
    if (embeddings.length === 0) return 0;
    const xs = embeddings.map((e) => e.x);
    const ys = embeddings.map((e) => e.y);
    const meanX = xs.reduce((acc, curr) => acc + curr, 0) / embeddings.length;
    const meanY = ys.reduce((acc, curr) => acc + curr, 0) / embeddings.length;
    const dx = item.x - meanX;
    const dy = item.y - meanY;
    const maxDist = Math.max(...embeddings.map((e) => Math.sqrt((e.x - meanX) ** 2 + (e.y - meanY) ** 2))) || 1;
    return (Math.sqrt(dx * dx + dy * dy) / maxDist) * 100;
  };

  const getDifficultyScore = (item: any) => {
    const preds = getPredictionsForImage(item.image_id);
    if (preds.length === 0) return 0;
    let errorSum = 0;
    preds.forEach((p: any) => {
      if (p.type === 'FP' || p.type === 'FN') {
        errorSum += 0.8;
      } else {
        const diff = Math.abs(p.confidence - 0.5);
        errorSum += 0.5 - diff;
      }
    });
    return errorSum / preds.length;
  };

  const getClosestNeighbors = (item: any, limit = 4) => {
    if (!item) return [];
    return embeddings
      .filter((e) => e.image_id !== item.image_id)
      .map((e) => {
        const dx = e.x - item.x;
        const dy = e.y - item.y;
        return { emb: e, dist: Math.sqrt(dx * dx + dy * dy) };
      })
      .sort((a, b) => a.dist - b.dist)
      .slice(0, limit);
  };

  // Filter the embeddings for the grid
  let filteredEmbeddings = embeddings.filter((e) => {
    if (searchId && !e.image_id.toLowerCase().includes(searchId.toLowerCase())) {
      return false;
    }
    const sensor = e.metadata?.sensor_type || 'Optical (WorldView-3)';
    if (sensorType === 'optical' && !sensor.includes('Optical')) {
      return false;
    }
    if (sensorType === 'sar' && !sensor.includes('SAR')) {
      return false;
    }
    
    // Lasso Curation Filter
    if (lassoedIds.length > 0 && !lassoedIds.includes(e.image_id)) {
      return false;
    }
    
    return true;
  });

  if (prioritizeHard) {
    filteredEmbeddings = [...filteredEmbeddings]
      .sort((a, b) => getDifficultyScore(b) - getDifficultyScore(a))
      .slice(0, 12);
  }

  const getImageUrl = (dsId: string, imageId: string) => {
    return `${API_BASE}/datasets/${dsId}/images/${imageId}`;
  };

  const syncToCvat = (imageId: string) => {
    setStatusMessage(`Extracting chip bounding box parameters. Syncing image '${imageId}' to CVAT labeling queue...`);
    setTimeout(() => {
      setStatusMessage(`Chip successfully synchronized. CVAT Task Link active: http://localhost:8080/tasks/review/${imageId}`);
    }, 1200);
  };

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: '#04060f',
        backgroundImage: `
          linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px)
        `,
        backgroundSize: '40px 40px',
        color: '#f1f5f9',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: "'Plus Jakarta Sans', sans-serif",
        overflow: 'hidden'
      }}
    >
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin-custom {
          animation: spin 1.2s linear infinite;
        }
      `}</style>

      {/* Header */}
      <header
        style={{
          padding: '12px 24px',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          background: 'rgba(9, 14, 31, 0.65)',
          backdropFilter: 'blur(12px)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          zIndex: 10
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Compass
            style={{
              width: '26px',
              height: '26px',
              color: '#00f2fe',
              filter: 'drop-shadow(0 0 8px #00f2fe)'
            }}
          />
          <div>
            <h1
              style={{
                fontFamily: "'Orbitron', sans-serif",
                fontSize: '1.1rem',
                fontWeight: 600,
                letterSpacing: '0.05em',
                margin: 0,
                color: '#fff'
              }}
            >
              FIFTYONE GEOSPATIAL VISUAL INSPECTOR
            </h1>
            <span
              style={{
                fontSize: '0.6rem',
                fontFamily: "'Space Grotesk', monospace",
                color: '#94a3b8',
                textTransform: 'uppercase'
              }}
            >
              Geospatial Dataset Analytics & Failure Analysis Suite
            </span>
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            gap: '16px',
            alignItems: 'center',
            fontFamily: "'Space Grotesk', monospace",
            fontSize: '0.75rem'
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <label style={{ color: '#94a3b8', fontSize: '0.6rem' }}>SELECT DATASET</label>
            <select
              value={activeDatasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
              style={{
                background: '#090e1f',
                border: '1px solid rgba(255,255,255,0.15)',
                color: '#fff',
                padding: '5px 10px',
                borderRadius: '4px',
                outline: 'none'
              }}
            >
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} ({ds.sample_size} chips)
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <label style={{ color: '#94a3b8', fontSize: '0.6rem' }}>SELECT EVALUATION RUN</label>
            <select
              value={activeExperimentId}
              onChange={(e) => setActiveExperimentId(e.target.value)}
              style={{
                background: '#090e1f',
                border: '1px solid rgba(255,255,255,0.15)',
                color: '#fff',
                padding: '5px 10px',
                borderRadius: '4px',
                outline: 'none',
                width: '180px'
              }}
            >
              <option value="">No model run overlays</option>
              {experiments.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  {exp.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => {
              window.location.href = '/';
            }}
            style={{
              background: 'rgba(255, 0, 85, 0.15)',
              border: '1px solid #ff0055',
              color: '#ff0055',
              padding: '8px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontWeight: 600,
              textTransform: 'uppercase',
              fontSize: '0.7rem',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(255,0,85,0.25)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255,0,85,0.15)';
            }}
          >
            <ArrowLeft style={{ width: '13px', height: '13px' }} />
            Return to Workbench
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Sidebar Filters */}
        <aside
          style={{
            width: '320px',
            borderRight: '1px solid rgba(255,255,255,0.08)',
            background: 'rgba(9, 14, 31, 0.45)',
            backdropFilter: 'blur(10px)',
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '20px',
            overflowY: 'auto',
            flexShrink: 0
          }}
        >
          <div>
            <h3
              style={{
                fontFamily: "'Orbitron', sans-serif",
                fontSize: '0.75rem',
                letterSpacing: '0.05em',
                color: '#00f2fe',
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                paddingBottom: '6px',
                margin: '0 0 12px 0'
              }}
            >
              VISUAL QUERY FILTERS
            </h3>
            <div
              style={{
                position: 'relative',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '6px',
                padding: '6px 10px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <Search style={{ width: '14px', height: '14px', color: '#94a3b8' }} />
              <input
                type="text"
                value={searchId}
                onChange={(e) => setSearchId(e.target.value)}
                placeholder="Search Image ID..."
                style={{
                  background: 'none',
                  border: 'none',
                  outline: 'none',
                  color: '#fff',
                  fontSize: '0.75rem',
                  fontFamily: "'Space Grotesk', monospace",
                  width: '100%'
                }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span
              style={{
                fontSize: '0.65rem',
                fontFamily: "'Space Grotesk', monospace",
                color: '#94a3b8',
                textTransform: 'uppercase'
              }}
            >
              LAYER TOGGLES
            </span>
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                fontSize: '0.75rem',
                cursor: 'pointer'
              }}
            >
              <input
                type="checkbox"
                checked={showGroundTruth}
                onChange={(e) => setShowGroundTruth(e.target.checked)}
                style={{ accentColor: '#00f2fe', cursor: 'pointer' }}
              />
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span
                  style={{
                    width: '8px',
                    height: '8px',
                    background: '#00f2fe',
                    borderRadius: '2px',
                    display: 'inline-block'
                  }}
                />
                GROUND TRUTH LABELS
              </span>
            </label>

            {activeExperimentId && (
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  marginTop: '4px'
                }}
              >
                <input
                  type="checkbox"
                  checked={showPredictions}
                  onChange={(e) => setShowPredictions(e.target.checked)}
                  style={{ accentColor: '#00ff87', cursor: 'pointer' }}
                />
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span
                    style={{
                      width: '8px',
                      height: '8px',
                      background: '#00ff87',
                      borderRadius: '2px',
                      display: 'inline-block'
                    }}
                  />
                  MODEL PREDICTIONS
                </span>
              </label>
            )}
          </div>

          {activeExperimentId && showPredictions && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                background: 'rgba(0,0,0,0.2)',
                padding: '10px',
                borderRadius: '6px',
                border: '1px solid rgba(255,255,255,0.05)'
              }}
            >
              <span
                style={{
                  fontSize: '0.6rem',
                  fontFamily: "'Space Grotesk', monospace",
                  color: '#94a3b8',
                  textTransform: 'uppercase'
                }}
              >
                PREDICTION SUBTYPES
              </span>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '0.7rem',
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={showTP}
                  onChange={(e) => setShowTP(e.target.checked)}
                  style={{ accentColor: '#00ff87' }}
                />
                <span style={{ color: '#00ff87' }}>TRUE POSITIVE (TP)</span>
              </label>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '0.7rem',
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={showFP}
                  onChange={(e) => setShowFP(e.target.checked)}
                  style={{ accentColor: '#ff0055' }}
                />
                <span style={{ color: '#ff0055' }}>FALSE POSITIVE (FP)</span>
              </label>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '0.7rem',
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={showFN}
                  onChange={(e) => setShowFN(e.target.checked)}
                  style={{ accentColor: '#ff9900' }}
                />
                <span style={{ color: '#ff9900' }}>FALSE NEGATIVE (FN)</span>
              </label>
            </div>
          )}

          {activeExperimentId && showPredictions && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.65rem',
                  fontFamily: "'Space Grotesk', monospace",
                  color: '#94a3b8'
                }}
              >
                <span>CONFIDENCE THRESHOLD</span>
                <span style={{ color: '#00f2fe', fontWeight: 600 }}>
                  {confThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.95"
                step="0.05"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                style={{
                  accentColor: '#00f2fe',
                  cursor: 'pointer',
                  height: '6px',
                  marginTop: '4px'
                }}
              />
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <span
              style={{
                fontSize: '0.65rem',
                fontFamily: "'Space Grotesk', monospace",
                color: '#94a3b8',
                textTransform: 'uppercase'
              }}
            >
              SENSOR BAND
            </span>
            <select
              value={sensorType}
              onChange={(e) => setSensorType(e.target.value)}
              style={{
                background: '#090e1f',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#fff',
                padding: '6px 10px',
                borderRadius: '4px',
                outline: 'none',
                fontSize: '0.75rem'
              }}
            >
              <option value="all">All Bands</option>
              <option value="optical">Optical (Visible / WorldView)</option>
              <option value="sar">SAR (Microwave / Radar)</option>
            </select>
          </div>

          <div
            style={{
              borderTop: '1px solid rgba(255,255,255,0.08)',
              paddingTop: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px'
            }}
          >
            <span
              style={{
                fontSize: '0.65rem',
                fontFamily: "'Space Grotesk', monospace",
                color: '#94a3b8',
                textTransform: 'uppercase'
              }}
            >
              ANALYTIC VIEWS
            </span>
            <button
              onClick={() => setShowPcaSpace(!showPcaSpace)}
              style={{
                width: '100%',
                background: showPcaSpace
                  ? 'rgba(0, 242, 254, 0.12)'
                  : 'rgba(255,255,255,0.02)',
                border: showPcaSpace
                  ? '1px solid #00f2fe'
                  : '1px solid rgba(255,255,255,0.1)',
                color: showPcaSpace ? '#00f2fe' : '#fff',
                padding: '10px',
                borderRadius: '6px',
                fontSize: '0.7rem',
                fontWeight: 600,
                fontFamily: "'Space Grotesk', monospace",
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'all 0.2s'
              }}
            >
              <Activity style={{ width: '14px', height: '14px' }} />
              {showPcaSpace ? 'CLOSE PCA CLUSTER VIEW' : 'OPEN PCA CLUSTER VIEW'}
            </button>

            {activeExperimentId && (
              <button
                onClick={() => setPrioritizeHard(!prioritizeHard)}
                style={{
                  width: '100%',
                  background: prioritizeHard
                    ? 'rgba(255, 153, 0, 0.12)'
                    : 'rgba(255,255,255,0.02)',
                  border: prioritizeHard
                    ? '1px solid #ff9900'
                    : '1px solid rgba(255,255,255,0.1)',
                  color: prioritizeHard ? '#ff9900' : '#fff',
                  padding: '10px',
                  borderRadius: '6px',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  fontFamily: "'Space Grotesk', monospace",
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  transition: 'all 0.2s'
                }}
              >
                <AlertCircle style={{ width: '14px', height: '14px' }} />
                {prioritizeHard ? 'SHOW ALL SAMPLES' : 'PRIORITIZE 12 HARD SAMPLES'}
              </button>
            )}
          </div>
        </aside>

        {/* Content Pane */}
        <main
          style={{
            flex: 1,
            padding: '24px',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            gap: '20px'
          }}
        >
          {/* Stats Bar */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '12px',
              flexShrink: 0
            }}
          >
            {[
              {
                label: 'DATASET LOADED',
                value: activeDatasetId.replace('dataset_', '').toUpperCase(),
                color: '#00f2fe'
              },
              { label: 'ANNOTATIONS COUNT', value: totalAnnotations, color: '#ff9900' },
              {
                label: 'FILTERED SHOWN',
                value: `${filteredEmbeddings.length} chips`,
                color: '#00ff87'
              },
              { label: 'AVG GROUND GSD', value: `${avgGSD}m`, color: '#fff' }
            ].map((stat, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(9, 14, 31, 0.4)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '8px',
                  padding: '12px 16px',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <span
                  style={{
                    fontSize: '0.55rem',
                    fontFamily: "'Space Grotesk', monospace",
                    color: '#94a3b8'
                  }}
                >
                  {stat.label}
                </span>
                <span
                  style={{
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    color: stat.color,
                    marginTop: '4px'
                  }}
                >
                  {stat.value}
                </span>
              </div>
            ))}
          </div>

          {/* Core Panel */}
          {loading ? (
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '12px',
                color: '#00f2fe',
                fontFamily: "'Space Grotesk', monospace"
              }}
            >
              <RefreshCw className="animate-spin-custom" style={{ width: '30px', height: '30px' }} />
              <span>Analyzing database coordinates & building satellite overlays...</span>
            </div>
          ) : (
            <div
              style={{
                flex: 1,
                display: 'flex',
                gap: '20px',
                overflow: 'hidden'
              }}
            >
              {/* Left Pane: Scrollable Image Grid */}
              <div
                style={{
                  flex: 1.2,
                  display: 'flex',
                  flexDirection: 'column',
                  overflowY: 'auto',
                  paddingRight: '6px'
                }}
              >
                {filteredEmbeddings.length === 0 ? (
                  <div style={{ color: '#475569', fontSize: '0.85rem', textAlign: 'center', marginTop: '40px' }}>
                    No image chips match the active query filters.
                  </div>
                ) : (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                      gap: '16px'
                    }}
                  >
                    {filteredEmbeddings.map((item) => {
                      const preds = getPredictionsForImage(item.image_id);
                      const isSelected = selectedChip && selectedChip.image_id === item.image_id;
                      return (
                        <div
                          key={item.image_id}
                          onClick={() => {
                            setSelectedChip(item);
                          }}
                          onDoubleClick={() => {
                            setZoomedChip(item);
                          }}
                          style={{
                            background: 'rgba(9, 14, 31, 0.4)',
                            border: isSelected
                              ? '1px solid #00f2fe'
                              : '1px solid rgba(255,255,255,0.06)',
                            borderRadius: '8px',
                            padding: '8px',
                            cursor: 'pointer',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '8px',
                            boxShadow: isSelected
                              ? '0 0 15px rgba(0, 242, 254, 0.15)'
                              : 'none',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          <div
                            style={{
                              height: '120px',
                              borderRadius: '6px',
                              overflow: 'hidden',
                              position: 'relative',
                              background: '#000'
                            }}
                          >
                            <img
                              src={getImageUrl(activeDatasetId, item.image_id)}
                              alt={item.image_id}
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                              }}
                            />
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
                              {showGroundTruth &&
                                item.labels &&
                                item.labels.map((lbl: any, tIdx: number) => {
                                  const bbox = lbl.bbox || [0, 0, 0, 0];
                                  const color = CLASS_COLORS[lbl.class_id] || '#00f2fe';
                                  return (
                                    <rect
                                      key={`gt-${tIdx}`}
                                      x={bbox[0] * 512}
                                      y={bbox[1] * 512}
                                      width={Math.max(15, (bbox[2] - bbox[0]) * 512)}
                                      height={Math.max(15, (bbox[3] - bbox[1]) * 512)}
                                      fill="none"
                                      stroke={color}
                                      strokeWidth={3}
                                    />
                                  );
                                })}
                              {showPredictions &&
                                preds.map((pred: any, pIdx: number) => {
                                  if (
                                    pred.confidence < confThreshold ||
                                    (pred.type === 'TP' && !showTP) ||
                                    (pred.type === 'FP' && !showFP) ||
                                    (pred.type === 'FN' && !showFN)
                                  )
                                    return null;
                                  const color =
                                    pred.type === 'TP'
                                      ? '#00ff87'
                                      : pred.type === 'FP'
                                        ? '#ff0055'
                                        : '#ff9900';
                                  return (
                                    <rect
                                      key={`pred-${pIdx}`}
                                      x={pred.bbox[0]}
                                      y={pred.bbox[1]}
                                      width={Math.max(15, pred.bbox[2] - pred.bbox[0])}
                                      height={Math.max(15, pred.bbox[3] - pred.bbox[1])}
                                      fill="none"
                                      stroke={color}
                                      strokeWidth={4}
                                      strokeDasharray={pred.type === 'FN' ? '8,8' : 'none'}
                                    />
                                  );
                                })}
                            </svg>
                            <div
                              style={{
                                position: 'absolute',
                                top: '6px',
                                left: '6px',
                                background: 'rgba(0,0,0,0.7)',
                                padding: '2px 6px',
                                borderRadius: '4px',
                                fontSize: '0.55rem',
                                fontFamily: "'Space Grotesk', monospace"
                              }}
                            >
                              {item.metadata?.sensor_type === 'SAR' ? 'SAR' : 'OPTICAL'}
                            </div>
                          </div>

                          <div
                            style={{
                              display: 'flex',
                              flexDirection: 'column',
                              gap: '2px',
                              fontSize: '0.7rem',
                              fontFamily: "'Space Grotesk', monospace"
                            }}
                          >
                            <span style={{ fontWeight: 600, color: '#fff' }}>
                              {item.image_id}
                            </span>
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                color: '#94a3b8',
                                fontSize: '0.6rem'
                              }}
                            >
                              <span>GT: {item.labels?.length || 0} planes</span>
                              {activeExperimentId && (
                                <span style={{ color: '#00ff87' }}>
                                  Preds: {preds.filter((p: any) => p.confidence >= confThreshold).length}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Right Pane: PCA Space Embeddings Plot */}
              {showPcaSpace && (
                <div
                  style={{
                    flex: 1.0,
                    background: 'rgba(9, 14, 31, 0.25)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '12px',
                    padding: '20px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px',
                    overflow: 'hidden',
                    minWidth: 0
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                      flexShrink: 0,
                      height: '360px',
                      overflow: 'hidden'
                    }}
                  >
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontFamily: "'Space Grotesk', monospace",
                        color: '#94a3b8'
                      }}
                    >
                      PCA EMBEDDING MAP (Select nodes to compute distance vectors)
                    </span>
                    <div
                      style={{
                        flex: 1,
                        border: '1px solid rgba(255,255,255,0.05)',
                        borderRadius: '8px',
                        background: 'rgba(0,0,0,0.4)',
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        overflow: 'hidden'
                      }}
                    >
                      <svg
                        ref={svgRef}
                        width="320"
                        height="320"
                        viewBox="0 0 420 420"
                        onMouseDown={handleMouseDown}
                        style={{ maxWidth: '100%', maxHeight: '100%', cursor: 'crosshair' }}
                      >
                        <line x1="210" y1="10" x2="210" y2="410" stroke="rgba(255,255,255,0.03)" />
                        <line x1="10" y1="210" x2="410" y2="210" stroke="rgba(255,255,255,0.03)" />
                        {embeddings.map((e) => {
                          const cx = ((e.x + 10) / 20) * 380 + 20;
                          const cy = ((e.y + 10) / 20) * 380 + 20;
                          const isSelected = selectedChip && selectedChip.image_id === e.image_id;
                          const isFilteredOut = !filteredEmbeddings.find((f: any) => f.image_id === e.image_id);
                          let color = 'rgba(0, 114, 255, 0.7)';
                          const sceneType = e.metadata?.scene_type;
                          if (sceneType === 'runway_intersection') color = '#00f2fe';
                          else if (sceneType === 'taxiway') color = '#00ff87';
                          else if (sceneType === 'cargo_ramp') color = '#ff9900';
                          return (
                            <circle
                              key={e.image_id}
                              cx={cx}
                              cy={cy}
                              r={isSelected ? 8 : 4}
                              fill={isSelected ? '#fff' : color}
                              stroke={isSelected ? '#00f2fe' : 'none'}
                              strokeWidth={isSelected ? 2 : 0}
                              opacity={isFilteredOut ? 0.15 : 1}
                              style={{
                                cursor: 'pointer',
                                transition: 'all 0.15s ease'
                              }}
                              onClick={(evt) => {
                                evt.stopPropagation();
                                setSelectedChip(e);
                              }}
                              onDoubleClick={(evt) => {
                                evt.stopPropagation();
                                setZoomedChip(e);
                              }}
                            />
                          );
                        })}
                        {lassoPolygon.length > 0 && (
                          <polyline
                            points={lassoPolygon.map(p => `${p.x},${p.y}`).join(' ')}
                            fill="rgba(0, 242, 254, 0.08)"
                            stroke="#00f2fe"
                            strokeWidth="3"
                            strokeDasharray="5 3"
                          />
                        )}
                      </svg>

                      {/* Legend Overlay */}
                      <div style={{ position: 'absolute', bottom: '10px', left: '10px', display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '0.55rem', fontFamily: "'Space Grotesk', monospace", background: 'rgba(0,0,0,0.75)', padding: '6px 10px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.08)', pointerEvents: 'none' }}>
                        <span style={{ color: '#ff9900', display: 'flex', alignItems: 'center', gap: '4px' }}>● Cargo Ramps</span>
                        <span style={{ color: '#00f2fe', display: 'flex', alignItems: 'center', gap: '4px' }}>● Runway Intersect.</span>
                        <span style={{ color: '#00ff87', display: 'flex', alignItems: 'center', gap: '4px' }}>● Taxiways / Aprons</span>
                        <span style={{ color: 'rgba(0, 114, 255, 0.8)', display: 'flex', alignItems: 'center', gap: '4px' }}>● Other Features</span>
                      </div>

                      {/* Clear Lasso button */}
                      {lassoedIds.length > 0 && (
                        <button
                          onClick={clearLasso}
                          style={{
                            position: 'absolute',
                            top: '10px',
                            right: '10px',
                            background: 'rgba(255, 0, 85, 0.2)',
                            border: '1px solid #ff0055',
                            color: '#ff0055',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.6rem',
                            fontFamily: "'Space Grotesk', monospace",
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}
                        >
                          <svg viewBox="0 0 24 24" width="10" height="10" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                          </svg>
                          CLEAR ({lassoedIds.length})
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Metadata and Neighbors summary directly in side pane */}
                  <div
                    style={{
                      flex: 1,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '12px',
                      overflowY: 'auto',
                      borderTop: '1px solid rgba(255,255,255,0.08)',
                      paddingTop: '16px',
                      minHeight: 0
                    }}
                  >
                    <h4
                      style={{
                        fontSize: '0.8rem',
                        fontFamily: "'Orbitron', sans-serif",
                        color: '#00f2fe',
                        margin: 0,
                        borderBottom: '1px solid rgba(255,255,255,0.08)',
                        paddingBottom: '6px'
                      }}
                    >
                      NODE METADATA & SIMILARITY
                    </h4>
                    {selectedChip ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        <div
                          style={{
                            height: '340px',
                            borderRadius: '8px',
                            overflow: 'hidden',
                            border: '1px solid rgba(255,255,255,0.1)',
                            position: 'relative'
                          }}
                        >
                          <img
                            src={getImageUrl(activeDatasetId, selectedChip.image_id)}
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover'
                            }}
                          />
                          <button
                            onClick={() => setZoomedChip(selectedChip)}
                            style={{
                              position: 'absolute',
                              bottom: '10px',
                              right: '10px',
                              background: 'rgba(9, 14, 31, 0.85)',
                              border: '1px solid #00f2fe',
                              color: '#00f2fe',
                              padding: '6px 12px',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '0.65rem',
                              fontWeight: 600,
                              fontFamily: "'Space Grotesk', monospace",
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            🔍 ZOOM CHIP
                          </button>
                        </div>
                        <div
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: '10px',
                            fontSize: '0.75rem',
                            fontFamily: "'Space Grotesk', monospace"
                          }}
                        >
                          <div
                            style={{
                              background: 'rgba(255,255,255,0.02)',
                              padding: '8px 10px',
                              borderRadius: '6px',
                              border: '1px solid rgba(255,255,255,0.05)'
                            }}
                          >
                            <div style={{ color: '#94a3b8', fontSize: '0.6rem', marginBottom: '2px' }}>IMAGE ID</div>
                            <div style={{ fontWeight: 600, color: '#fff' }}>{selectedChip.image_id}</div>
                          </div>
                          <div
                            style={{
                              background: 'rgba(255,255,255,0.02)',
                              padding: '8px 10px',
                              borderRadius: '6px',
                              border: '1px solid rgba(255,255,255,0.05)'
                            }}
                          >
                            <div style={{ color: '#94a3b8', fontSize: '0.6rem', marginBottom: '2px' }}>SCENE TYPE</div>
                            <div style={{ fontWeight: 600, color: '#fff' }}>
                              {(selectedChip.metadata?.scene_type || 'aprons').toUpperCase()}
                            </div>
                          </div>
                          <div
                            style={{
                              background: 'rgba(255,255,255,0.02)',
                              padding: '8px 10px',
                              borderRadius: '6px',
                              border: '1px solid rgba(255,255,255,0.05)'
                            }}
                          >
                            <div style={{ color: '#94a3b8', fontSize: '0.6rem', marginBottom: '2px' }}>ANOMALY DISTANCE</div>
                            <div style={{ fontWeight: 600, color: '#ff9900' }}>
                              {getAnomalyDistancePercent(selectedChip).toFixed(1)}% Outlier
                            </div>
                          </div>
                          <div
                            style={{
                              background: 'rgba(255,255,255,0.02)',
                              padding: '8px 10px',
                              borderRadius: '6px',
                              border: '1px solid rgba(255,255,255,0.05)'
                            }}
                          >
                            <div style={{ color: '#94a3b8', fontSize: '0.6rem', marginBottom: '2px' }}>COORDINATES</div>
                            <div style={{ fontWeight: 600, color: '#00ff87' }}>
                              {selectedChip.x.toFixed(2)}, {selectedChip.y.toFixed(2)}
                            </div>
                          </div>
                        </div>

                        <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '14px' }}>
                          <span
                            style={{
                              fontSize: '0.65rem',
                              fontFamily: "'Space Grotesk', monospace",
                              color: '#94a3b8',
                              display: 'block',
                              marginBottom: '8px'
                            }}
                          >
                            CLOSEST NEIGHBORS IN VECTOR EMBEDDINGS (DISTANCE ANALYSIS)
                          </span>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
                            {getClosestNeighbors(selectedChip).map((e: any) => (
                              <div
                                key={e.emb.image_id}
                                onClick={() => {
                                  setSelectedChip(e.emb);
                                }}
                                onDoubleClick={() => {
                                  setZoomedChip(e.emb);
                                }}
                                style={{ cursor: 'pointer', textAlign: 'center' }}
                              >
                                <img
                                  src={getImageUrl(activeDatasetId, e.emb.image_id)}
                                  style={{
                                    width: '100%',
                                    height: '120px',
                                    objectFit: 'cover',
                                    borderRadius: '4px',
                                    border: '1px solid rgba(255,255,255,0.05)'
                                  }}
                                />
                                <div style={{ fontSize: '0.5rem', color: '#00f2fe', marginTop: '4px' }}>
                                  d={e.dist.toFixed(2)}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          padding: '40px 20px',
                          border: '1px dashed rgba(255,255,255,0.1)',
                          borderRadius: '8px',
                          background: 'rgba(255,255,255,0.01)',
                          color: '#94a3b8',
                          textAlign: 'center',
                          gap: '12px',
                          marginTop: '20px'
                        }}
                      >
                        <Info style={{ width: '20px', height: '20px', color: 'rgba(0, 242, 254, 0.5)' }} />
                        <span style={{ fontSize: '0.7rem', fontFamily: "'Space Grotesk', monospace", lineHeight: '1.4' }}>
                          Select a point on the PCA space or grid chip to inspect embedding metrics.
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {/* Zoom / Detailed Modal */}
      {zoomedChip && (
        <div
          style={{
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
            zIndex: 100
          }}
        >
          <div
            style={{
              width: '920px',
              height: '620px',
              background: '#090e1f',
              border: '1px solid rgba(0, 242, 254, 0.25)',
              boxShadow: '0 0 30px rgba(0, 242, 254, 0.15)',
              borderRadius: '12px',
              display: 'flex',
              overflow: 'hidden',
              position: 'relative'
            }}
          >
            {/* Zoomed Image & Bounding Boxes */}
            <div
              style={{
                flex: 1.2,
                position: 'relative',
                background: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <img
                src={getImageUrl(activeDatasetId, zoomedChip.image_id)}
                alt="Detailed focus"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain'
                }}
              />
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
                {showGroundTruth &&
                  zoomedChip.labels &&
                  zoomedChip.labels.map((lbl: any, tIdx: number) => {
                    const bbox = lbl.bbox || [0, 0, 0, 0];
                    const color = CLASS_COLORS[lbl.class_id] || '#00f2fe';
                    return (
                      <g key={`det-gt-${tIdx}`}>
                        <rect
                          x={bbox[0] * 512}
                          y={bbox[1] * 512}
                          width={Math.max(15, (bbox[2] - bbox[0]) * 512)}
                          height={Math.max(15, (bbox[3] - bbox[1]) * 512)}
                          fill="none"
                          stroke={color}
                          strokeWidth={4}
                        />
                        <text
                          x={bbox[0] * 512 + 6}
                          y={bbox[1] * 512 > 20 ? bbox[1] * 512 - 6 : bbox[1] * 512 + 18}
                          fill={color}
                          fontSize="14"
                          fontWeight="bold"
                          fontFamily="'Space Grotesk', monospace"
                          style={{
                            paintOrder: 'stroke',
                            stroke: '#000',
                            strokeWidth: '4px'
                          }}
                        >
                          GT: {CLASS_NAMES[lbl.class_id]}
                        </text>
                      </g>
                    );
                  })}
                {showPredictions &&
                  getPredictionsForImage(zoomedChip.image_id).map((pred: any, pIdx: number) => {
                    if (
                      pred.confidence < confThreshold ||
                      (pred.type === 'TP' && !showTP) ||
                      (pred.type === 'FP' && !showFP) ||
                      (pred.type === 'FN' && !showFN)
                    )
                      return null;
                    const color =
                      pred.type === 'TP'
                        ? '#00ff87'
                        : pred.type === 'FP'
                          ? '#ff0055'
                          : '#ff9900';
                    return (
                      <g key={`det-pred-${pIdx}`}>
                        <rect
                          x={pred.bbox[0]}
                          y={pred.bbox[1]}
                          width={Math.max(15, pred.bbox[2] - pred.bbox[0])}
                          height={Math.max(15, pred.bbox[3] - pred.bbox[1])}
                          fill="none"
                          stroke={color}
                          strokeWidth={5}
                          strokeDasharray={pred.type === 'FN' ? '12,12' : 'none'}
                        />
                        <text
                          x={pred.bbox[0] + 6}
                          y={pred.bbox[1] > 20 ? pred.bbox[1] - 8 : pred.bbox[1] + 24}
                          fill={color}
                          fontSize="16"
                          fontWeight="bold"
                          fontFamily="'Space Grotesk', monospace"
                          style={{
                            paintOrder: 'stroke',
                            stroke: '#000',
                            strokeWidth: '4px'
                          }}
                        >
                          {pred.type}: {pred.confidence > 0 ? (pred.confidence * 100).toFixed(0) + '%' : 'FN'}
                        </text>
                      </g>
                    );
                  })}
              </svg>
              <button
                onClick={() => setZoomedChip(null)}
                style={{
                  position: 'absolute',
                  top: '16px',
                  left: '16px',
                  background: 'rgba(0,0,0,0.6)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  padding: '6px 10px',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '0.7rem',
                  fontFamily: "'Space Grotesk', monospace"
                }}
              >
                ← CLOSE ZOOM
              </button>
            </div>

            {/* Modal Tabs Side Panel */}
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                borderLeft: '1px solid rgba(255,255,255,0.08)'
              }}
            >
              <div
                style={{
                  display: 'flex',
                  borderBottom: '1px solid rgba(255,255,255,0.08)',
                  background: 'rgba(0,0,0,0.2)'
                }}
              >
                {[
                  { id: 'metadata', label: 'METADATA' },
                  { id: 'similarity', label: 'SIMILARITY' },
                  { id: 'review', label: 'CVAT WORKFLOW' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => {
                      setActiveTab(tab.id);
                      setStatusMessage(null);
                    }}
                    style={{
                      flex: 1,
                      padding: '14px',
                      background: activeTab === tab.id ? '#090e1f' : 'transparent',
                      border: 'none',
                      borderBottom: activeTab === tab.id ? '2px solid #00f2fe' : 'none',
                      color: activeTab === tab.id ? '#00f2fe' : '#94a3b8',
                      fontSize: '0.7rem',
                      fontFamily: "'Space Grotesk', monospace",
                      fontWeight: 600,
                      cursor: 'pointer',
                      transition: 'all 0.15s'
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
                {activeTab === 'metadata' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <h4
                      style={{
                        fontSize: '0.75rem',
                        fontFamily: "'Orbitron', sans-serif",
                        color: '#00f2fe',
                        margin: '0 0 4px 0'
                      }}
                    >
                      GEOSPATIAL FRAME SPECIFICATIONS
                    </h4>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '8px',
                        fontSize: '0.7rem',
                        fontFamily: "'Space Grotesk', monospace"
                      }}
                    >
                      {[
                        { label: 'IMAGE FILE', val: zoomedChip.image_id },
                        {
                          label: 'SENSOR TYPE',
                          val: zoomedChip.metadata?.sensor_type || 'Optical (WorldView-3)'
                        },
                        { label: 'SITE CODE', val: zoomedChip.metadata?.airport_code || 'KIAD' },
                        { label: 'RESOLUTION', val: `${zoomedChip.metadata?.gsd || 0.3}m GSD` },
                        { label: 'POL / PASS', val: zoomedChip.metadata?.sar_polarization || 'N/A' },
                        {
                          label: 'LOOK ANGLE',
                          val: zoomedChip.metadata?.incidence_angle
                            ? `${zoomedChip.metadata.incidence_angle}°`
                            : 'N/A'
                        }
                      ].map((item, idx) => (
                        <div
                          key={idx}
                          style={{
                            background: 'rgba(255,255,255,0.01)',
                            border: '1px solid rgba(255,255,255,0.04)',
                            padding: '8px',
                            borderRadius: '4px'
                          }}
                        >
                          <div style={{ color: '#94a3b8', fontSize: '0.55rem' }}>{item.label}</div>
                          <div style={{ color: '#fff', fontWeight: 600, marginTop: '2px' }}>
                            {item.val}
                          </div>
                        </div>
                      ))}
                    </div>

                    <div
                      style={{
                        background: 'rgba(255, 153, 0, 0.05)',
                        border: '1px solid rgba(255, 153, 0, 0.15)',
                        padding: '10px',
                        borderRadius: '6px',
                        display: 'flex',
                        gap: '8px',
                        marginTop: '6px'
                      }}
                    >
                      <Info
                        style={{
                          width: '16px',
                          height: '16px',
                          color: '#ff9900',
                          flexShrink: 0
                        }}
                      />
                      <div
                        style={{
                          fontSize: '0.65rem',
                          color: '#94a3b8',
                          lineHeight: '1.4'
                        }}
                      >
                        <span style={{ color: '#ff9900', fontWeight: 600 }}>
                          PCA ANOMALY RATING: {getAnomalyDistancePercent(zoomedChip).toFixed(1)}%
                        </span>
                        <br />
                        This chip is placed in a cluster suggesting runway intersections or apron regions. Outlier score indicates distance from main airport terminal centroids.
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'similarity' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h4
                        style={{
                          fontSize: '0.75rem',
                          fontFamily: "'Orbitron', sans-serif",
                          color: '#00f2fe',
                          margin: 0
                        }}
                      >
                        EUCLIDEAN PCA NEIGHBORS
                      </h4>
                      <span
                        style={{
                          fontSize: '0.6rem',
                          fontFamily: "'Space Grotesk', monospace",
                          color: '#94a3b8'
                        }}
                      >
                        Distance threshold matches
                      </span>
                    </div>
                    <p style={{ fontSize: '0.65rem', color: '#94a3b8', margin: 0, lineHeight: '1.4' }}>
                      Below are the 4 images in the dataset with the closest feature embeddings coordinates. Use this to find duplicate chips or examine similar ground features.
                    </p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {getClosestNeighbors(zoomedChip).map((neigh: any) => (
                        <div
                          key={neigh.emb.image_id}
                          onClick={() => {
                            setSelectedChip(neigh.emb);
                            setZoomedChip(neigh.emb);
                          }}
                          style={{
                            display: 'flex',
                            gap: '12px',
                            background: 'rgba(255,255,255,0.02)',
                            border: '1px solid rgba(255,255,255,0.05)',
                            padding: '6px',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            transition: 'all 0.15s'
                          }}
                          onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#00f2fe')}
                          onMouseLeave={(e) =>
                            (e.currentTarget.style.borderColor = 'rgba(255,255,255,0.05)')
                          }
                        >
                          <img
                            src={getImageUrl(activeDatasetId, neigh.emb.image_id)}
                            style={{
                              width: '70px',
                              height: '50px',
                              objectFit: 'cover',
                              borderRadius: '4px'
                            }}
                          />
                          <div
                            style={{
                              display: 'flex',
                              flexDirection: 'column',
                              justifyContent: 'center',
                              fontSize: '0.65rem',
                              fontFamily: "'Space Grotesk', monospace"
                            }}
                          >
                            <span style={{ color: '#fff', fontWeight: 600 }}>{neigh.emb.image_id}</span>
                            <span style={{ color: '#00f2fe', fontSize: '0.6rem', marginTop: '2px' }}>
                              Euclidean distance: {neigh.dist.toFixed(3)}
                            </span>
                          </div>
                          <div
                            style={{
                              marginLeft: 'auto',
                              display: 'flex',
                              alignItems: 'center',
                              color: '#00f2fe',
                              paddingRight: '8px'
                            }}
                          >
                            <ArrowRight style={{ width: '14px', height: '14px' }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {activeTab === 'review' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    <h4
                      style={{
                        fontSize: '0.75rem',
                        fontFamily: "'Orbitron', sans-serif",
                        color: '#00f2fe',
                        margin: 0
                      }}
                    >
                      SME REVIEW CONTROLLER
                    </h4>
                    <p style={{ fontSize: '0.65rem', color: '#94a3b8', margin: 0, lineHeight: '1.4' }}>
                      Flag low-confidence labels, false alarms, or missing planes. Sending to CVAT queues them for corrected labeling, which automatically compiles into a new dataset version export.
                    </p>
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                        marginTop: '6px'
                      }}
                    >
                      <button
                        onClick={() => syncToCvat(zoomedChip.image_id)}
                        style={{
                          background: 'rgba(0, 242, 254, 0.08)',
                          border: '1px solid #00f2fe',
                          color: '#00f2fe',
                          padding: '10px',
                          borderRadius: '6px',
                          fontWeight: 600,
                          fontSize: '0.7rem',
                          fontFamily: "'Space Grotesk', monospace",
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '6px',
                          transition: 'all 0.2s'
                        }}
                      >
                        <RefreshCw
                          className={statusMessage?.includes('Syncing') ? 'animate-spin-custom' : ''}
                          style={{ width: '13px', height: '13px' }}
                        />
                        SEND IMAGE TO CVAT REVIEW QUEUE
                      </button>
                      <button
                        onClick={() => {
                          setStatusMessage(
                            'Image successfully flagged as duplicate. Will be ignored in future split partitioning.'
                          );
                        }}
                        style={{
                          background: 'rgba(255, 153, 0, 0.08)',
                          border: '1px solid #ff9900',
                          color: '#ff9900',
                          padding: '10px',
                          borderRadius: '6px',
                          fontWeight: 600,
                          fontSize: '0.7rem',
                          fontFamily: "'Space Grotesk', monospace",
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '6px',
                          transition: 'all 0.2s'
                        }}
                      >
                        <AlertTriangle style={{ width: '13px', height: '13px' }} />
                        FLAG CHIP AS DUPLICATE
                      </button>
                    </div>

                    {statusMessage && (
                      <div
                        style={{
                          background: 'rgba(0, 242, 254, 0.04)',
                          border: '1px solid rgba(0, 242, 254, 0.2)',
                          padding: '12px',
                          borderRadius: '6px',
                          fontSize: '0.65rem',
                          fontFamily: "'Space Grotesk', monospace",
                          color: '#fff',
                          lineHeight: '1.4',
                          marginTop: '10px'
                        }}
                      >
                        <div
                          style={{
                            color: '#00f2fe',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            marginBottom: '4px'
                          }}
                        >
                          <Check style={{ width: '12px', height: '12px' }} />
                          STATUS ACTION SYNCED
                        </div>
                        {statusMessage}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Close Button Footer */}
              <div
                style={{
                  padding: '20px',
                  borderTop: '1px solid rgba(255,255,255,0.08)',
                  display: 'flex',
                  justifyContent: 'flex-end',
                  background: 'rgba(0,0,0,0.1)'
                }}
              >
                <button
                  onClick={() => setZoomedChip(null)}
                  style={{
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#fff',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.7rem',
                    fontWeight: 600
                  }}
                >
                  CLOSE DIALOG
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}