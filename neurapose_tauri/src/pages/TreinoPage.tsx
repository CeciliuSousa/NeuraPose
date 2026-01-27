import { useState, useEffect, useRef } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    Dumbbell,
    Settings2,
    Zap,
    RotateCcw
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { Terminal } from '../components/ui/Terminal';
import { StatusMessage } from '../components/ui/StatusMessage';
import { PathSelector } from '../components/ui/PathSelector';
import { ConfigCard } from '../components/ui/ConfigCard';
import { DeviceSelector } from '../components/ui/DeviceSelector';
import { ProcessControls } from '../components/ui/ProcessControls';

// Modelos disponíveis com nomes extensos
const MODEL_OPTIONS = [
    { value: 'tft', label: 'Temporal Fusion Transformer', description: 'Melhor para séries temporais complexas' },
    { value: 'lstm', label: 'LSTM', description: 'Redes neurais recorrentes padrão' },
    { value: 'robust', label: 'RobustLSTM', description: 'LSTM com normalização robusta' },
    { value: 'pooled', label: 'PooledLSTM', description: 'LSTM com pooling temporal' },
    { value: 'bilstm', label: 'BiLSTM', description: 'Processa em ambas direções' },
    { value: 'attention', label: 'AttentionLSTM', description: 'LSTM com mecanismo de atenção' },
    { value: 'tcn', label: 'Temporal Convolutional Network', description: 'Convoluções temporais' },
    { value: 'transformer', label: 'Transformer', description: 'Arquitetura de atenção pura' },
    { value: 'wavenet', label: 'WaveNet', description: 'Convoluções causais dilatadas' },
];

// Valores padrão dos parâmetros (do config_master.py)
const DEFAULT_PARAMS = {
    epochs: 5000,       // Usuário pediu 5000 como padrão
    batch_size: 32,
    lr: 0.0003,
    dropout: 0.3,
    hidden_size: 128,
    num_layers: 2,
    num_heads: 8,
    kernel_size: 5,
};

type TrainMode = 'treinar' | 'retreinar';

export default function TreinoPage() {
    const [mode, setMode] = useState<TrainMode>('treinar');
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'processing' | 'info' } | null>(null);

    // Dataset e Modelo selecionados
    const [datasetPath, setDatasetPath] = useState('');
    const [pretrainedPath, setPretrainedPath] = useState('');

    // Parâmetros de treinamento
    const [params, setParams] = useState(DEFAULT_PARAMS);
    const [modelType, setModelType] = useState('tft');
    const [device, setDevice] = useState<'cuda' | 'cpu'>('cuda');

    // Estados do Explorer
    const [explorerOpen, setExplorerOpen] = useState<'dataset' | 'pretrained' | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});

    // Nomes derivados
    const datasetName = datasetPath ? datasetPath.replace(/\\/g, '/').split('/').pop() || '' : '';

    // Carregar caminhos do backend e restaurar estado
    useEffect(() => {
        // Restaurar estado se houver treino em andamento
        APIService.healthCheck().then(res => {
            // Só mostra mensagem se for ESTE processo (train)
            if (res.data.processing && res.data.current_process === 'train') {
                setLoading(true);
                setMessage({ text: '⏳ Treinamento em andamento...', type: 'processing' });
            }
        }).catch(() => { });

        APIService.getConfig().then(res => {
            const data = res.data as any;
            if (data.status === 'success') {
                setRoots(data.paths);
            }
        });

        // Restaurar logs do localStorage
        const savedLogs = localStorage.getItem('np_train_logs');
        if (savedLogs) setLogs(JSON.parse(savedLogs));



    }, []);



    // Refs para controle de buffer e logs
    const bufferRef = useRef<string[]>([]);

    // Logs via WebSocket
    useEffect(() => {
        if (loading) {
            let flushInterval: ReturnType<typeof setInterval>;

            import('../services/websocket').then(mod => {
                const ws = mod.default;
                ws.connectLogs('train');
                ws.connectStatus();

                const handleLogs = (data: any) => {
                    // 1. Tratamento robusto do payload (igual aos outros arquivos funcionais)
                    const newLogs = Array.isArray(data) ? data : (data.logs || []);
                    // const total = data.total || 0;

                    // Adiciona ao buffer persistente
                    if (newLogs.length > 0) {
                        bufferRef.current.push(...newLogs);
                    }
                };

                // Flush do buffer para o estado a cada 100ms
                flushInterval = setInterval(() => {
                    if (bufferRef.current.length > 0) {
                        const logsToFlush = [...bufferRef.current];
                        bufferRef.current = []; // Limpa buffer

                        setLogs((prev) => {
                            // Limpa logs visualmente feios (opcional) e junta
                            const processedLogs = logsToFlush.map(l => l.replace(/\r/g, '')).filter(l => l.trim());

                            const updated = [...prev, ...processedLogs];

                            // Mantém apenas as últimas 1000 linhas para performance/evitar crash
                            if (updated.length > 1000) {
                                return updated.slice(-1000);
                            }
                            return updated;
                        });
                    }
                }, 100);

                const handleStatus = (status: any) => {
                    if (!status.is_running && loading) {
                        setLoading(false);
                        setMessage({ text: '✅ Treinamento concluído!', type: 'success' });
                        ws.disconnectLogs();
                    }
                };

                ws.events.on('logs', handleLogs);
                ws.events.on('status', handleStatus);

                return () => {
                    ws.events.off('logs', handleLogs);
                    ws.events.off('status', handleStatus);
                };
            });

            return () => {
                if (flushInterval) clearInterval(flushInterval);
            };
        }
    }, [loading]);

    const handleTrain = async () => {
        if (!datasetPath) {
            setMessage({ text: 'Por favor, selecione um dataset para treinamento.', type: 'error' });
            return;
        }
        if (mode === 'retreinar' && !pretrainedPath) {
            setMessage({ text: 'Para retreinar, selecione um modelo pré-treinado.', type: 'error' });
            return;
        }

        setLoading(true);
        setMessage({ text: `⏳ Iniciando ${mode === 'treinar' ? 'treinamento' : 'retreinamento'}...`, type: 'processing' });
        // setLogs(prev => [...prev, `[INFO] Modo: ${mode.toUpperCase()}`]); // Removido para limpeza
        // setLogs(prev => [...prev, `[INFO] Dataset: ${datasetName}`]);     // Removido para limpeza
        // setLogs(prev => [...prev, `[INFO] Modelo: ${MODEL_OPTIONS.find(m => m.value === modelType)?.label}`]); // Removido para limpeza
        // setLogs(prev => [...prev, `[INFO] Hardware: ${device === 'cuda' ? 'GPU (CUDA)' : 'CPU'}`]); // Removido para limpeza

        try {
            const data = {
                dataset_path: datasetPath,
                model_type: modelType,
                device: device,
                pretrained_path: mode === 'retreinar' ? pretrainedPath : undefined,
                ...params
            };

            if (mode === 'treinar') {
                await APIService.startTraining(data);
            } else {
                await APIService.retrainTraining(data);
            }
        } catch (error: any) {
            setLoading(false);
            setMessage({ text: `❌ Erro: ${error.message}`, type: 'error' });
            setLogs(prev => [...prev, `[ERRO] ${error.message}`]);
        }
    };

    const handleStop = async () => {
        try {
            await APIService.stopTraining();
            setMessage({ text: '🛑 Interrupção de treinamento solicitada...', type: 'info' });
        } catch (error: any) {
            setMessage({ text: `❌ Erro ao parar: ${error.message}`, type: 'error' });
        }
    };

    const resetParams = () => {
        setParams(DEFAULT_PARAMS);
        setMessage({ text: 'Parâmetros restaurados para valores padrão.', type: 'info' });
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Treinamento de Modelo"
                description="Página para treinamento de modelos temporais baseados em LSTM"
                icon={Dumbbell}
            />

            {/* Seletor de Modo */}
            <div className="flex gap-2 p-1 bg-muted/50 rounded-xl w-fit">
                <button
                    onClick={() => setMode('treinar')}
                    className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${mode === 'treinar'
                        ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/20'
                        : 'hover:bg-muted text-muted-foreground'
                        }`}
                >
                    <Zap className="w-4 h-4" />
                    Treinar Novo
                </button>
                <button
                    onClick={() => setMode('retreinar')}
                    className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${mode === 'retreinar'
                        ? 'bg-orange-500 text-white shadow-lg shadow-orange-500/20'
                        : 'hover:bg-muted text-muted-foreground'
                        }`}
                >
                    <RotateCcw className="w-4 h-4" />
                    Retreinar
                </button>
            </div>

            {/* Status Message */}
            {message && (
                <StatusMessage
                    message={message.text}
                    type={message.type}
                    onClose={() => setMessage(null)}
                    autoCloseDelay={message.type === 'success' ? 5000 : undefined}
                />
            )}

            <div className="grid gap-6 lg:grid-cols-2">
                {/* Config Panel */}
                <div className="space-y-6">
                    <ConfigCard title="Configuração Avançada" icon={Settings2}>
                        <div className="space-y-6">
                            {/* Dataset Selection */}
                            <div className="space-y-4">
                                <h3 className="font-semibold text-sm text-muted-foreground flex items-center gap-2">
                                    <Dumbbell className="w-4 h-4" /> Dados e Modelo
                                </h3>

                                <div className="space-y-2">
                                    <PathSelector
                                        label="Dataset (.pt)"
                                        value={datasetPath}
                                        onSelect={() => setExplorerOpen('dataset')}
                                        placeholder="Selecione o diretório para treinar..."
                                        icon={Dumbbell}
                                    />
                                    {datasetName && (
                                        <p className="text-xs text-muted-foreground mt-1">
                                            Dataset: <span className="font-mono text-primary">{datasetName}</span>
                                        </p>
                                    )}
                                </div>

                                {mode === 'retreinar' && (
                                    <div className="space-y-2 p-4 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                                        <PathSelector
                                            label="Modelo Pré-treinado"
                                            value={pretrainedPath}
                                            onSelect={() => setExplorerOpen('pretrained')}
                                            placeholder="Selecione um modelo para retreino..."
                                        />
                                    </div>
                                )}
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-muted-foreground">Arquitetura</label>
                                <div className="grid grid-cols-3 gap-2">
                                    {MODEL_OPTIONS.map(opt => (
                                        <button
                                            key={opt.value}
                                            onClick={() => setModelType(opt.value)}
                                            className={`p-3 rounded-lg text-left border-2 transition-all ${modelType === opt.value
                                                ? 'bg-primary/10 border-primary text-primary'
                                                : 'bg-secondary/30 border-border text-muted-foreground hover:border-primary/50'
                                                }`}
                                            title={opt.description}
                                        >
                                            <span className="text-xs font-bold block truncate">{opt.label}</span>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <DeviceSelector
                                value={device}
                                onChange={setDevice}
                            />

                            <div className="p-4 bg-muted/20 rounded-xl border border-border">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="font-semibold text-sm text-muted-foreground">
                                        Hiperparâmetros
                                    </h3>
                                    <button
                                        onClick={resetParams}
                                        className="text-xs text-muted-foreground hover:text-primary flex items-center gap-1"
                                    >
                                        <RotateCcw className="w-3 h-3" />
                                        Restaurar
                                    </button>
                                </div>

                                <div className="grid grid-cols-4 gap-4 mb-4 text-xs">
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">Épocas</label>
                                        <input type="number" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.epochs} onChange={e => setParams({ ...params, epochs: parseInt(e.target.value) || 0 })} />
                                    </div>
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">Batch</label>
                                        <input type="number" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.batch_size} onChange={e => setParams({ ...params, batch_size: parseInt(e.target.value) || 0 })} />
                                    </div>
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">LR</label>
                                        <input type="number" step="0.0001" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.lr} onChange={e => setParams({ ...params, lr: parseFloat(e.target.value) || 0 })} />
                                    </div>
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">Drop</label>
                                        <input type="number" step="0.05" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.dropout} onChange={e => setParams({ ...params, dropout: parseFloat(e.target.value) || 0 })} />
                                    </div>
                                </div>

                                <div className="grid grid-cols-4 gap-4 text-xs">
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">Hidden Size</label>
                                        <input type="number" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.hidden_size} onChange={e => setParams({ ...params, hidden_size: parseInt(e.target.value) || 0 })} />
                                    </div>
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">Num Layers</label>
                                        <input type="number" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.num_layers} onChange={e => setParams({ ...params, num_layers: parseInt(e.target.value) || 0 })} />
                                    </div>
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">Num Heads</label>
                                        <input type="number" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.num_heads} onChange={e => setParams({ ...params, num_heads: parseInt(e.target.value) || 0 })} />
                                    </div>
                                    <div className="space-y-1">
                                        <label className="text-muted-foreground">Kernel Size</label>
                                        <input type="number" className="w-full px-2 py-1.5 rounded border border-border bg-background" value={params.kernel_size} onChange={e => setParams({ ...params, kernel_size: parseInt(e.target.value) || 0 })} />
                                    </div>
                                </div>
                            </div>

                            <ProcessControls
                                isProcessing={loading}
                                canStart={!!datasetPath && (mode !== 'retreinar' || !!pretrainedPath)}
                                onStart={handleTrain}
                                onStop={handleStop}
                                labels={{
                                    start: mode === 'treinar' ? 'Iniciar Treinamento' : 'Iniciar Retreinamento',
                                    stop: 'Parar Treinamento'
                                }}
                            />
                        </div>
                    </ConfigCard>
                </div>

                {/* Terminal Panel */}
                <div>
                    <Terminal
                        logs={logs}
                        title="Console de Treinamento"
                        isLoading={loading}
                        onClear={async () => {
                            setLogs([]);
                            localStorage.removeItem('np_train_logs');
                            try { await APIService.clearLogs('train'); } catch (e) { console.error(e); }
                        }}
                    />
                </div>
            </div>

            {/* File Explorer Modal - Dataset */}
            <FileExplorerModal
                isOpen={explorerOpen === 'dataset'}
                onClose={() => setExplorerOpen(null)}
                onSelect={(path) => {
                    setDatasetPath(path);
                    setExplorerOpen(null);
                }}
                initialPath={datasetPath || roots.datasets}
                rootPath={roots.datasets}
                title="Selecionar Dataset para Treinamento"
            />

            {/* File Explorer Modal - Pretrained Model */}
            <FileExplorerModal
                isOpen={explorerOpen === 'pretrained'}
                onClose={() => setExplorerOpen(null)}
                onSelect={(path) => {
                    setPretrainedPath(path);
                    setExplorerOpen(null);
                }}
                initialPath={pretrainedPath || roots.modelos_treinados}
                rootPath={roots.modelos_treinados}
                title="Selecionar Modelo Pré-treinado"
            />
        </div>
    );
}
