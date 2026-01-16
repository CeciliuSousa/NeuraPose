import { useState, useEffect } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    Dumbbell,
    Play,
    RefreshCcw,
    FolderInput,
    Settings2,
    Zap,
    RotateCcw
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { shortenPath } from '../lib/utils';
import { Terminal } from '../components/ui/Terminal';
import { StatusMessage } from '../components/ui/StatusMessage';

// Modelos disponíveis com nomes extensos
const MODEL_OPTIONS = [
    { value: 'tft', label: 'Temporal Fusion Transformer', description: 'Melhor para séries temporais complexas' },
    { value: 'lstm', label: 'LSTM Clássico', description: 'Redes neurais recorrentes padrão' },
    { value: 'robust', label: 'RobustLSTM', description: 'LSTM com normalização robusta' },
    { value: 'pooled', label: 'PooledLSTM', description: 'LSTM com pooling temporal' },
    { value: 'bilstm', label: 'BiLSTM (Bidirecional)', description: 'Processa em ambas direções' },
    { value: 'attention', label: 'AttentionLSTM', description: 'LSTM com mecanismo de atenção' },
    { value: 'tcn', label: 'TCN (Temporal Conv. Network)', description: 'Convoluções temporais' },
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

    // Carregar caminhos do backend
    useEffect(() => {
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
            }
        });
    }, []);

    // Polling de logs
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (loading) {
            interval = setInterval(async () => {
                try {
                    const res = await APIService.getLogs();
                    setLogs(res.data.logs);

                    const health = await APIService.healthCheck();
                    if (!health.data.processing) {
                        setLoading(false);
                        setMessage({ text: '✅ Treinamento concluído!', type: 'success' });
                    }
                } catch (e) { console.error(e); }
            }, 1000);
        }
        return () => { if (interval) clearInterval(interval); };
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
        setLogs(prev => [...prev, `[INFO] Modo: ${mode.toUpperCase()}`]);
        setLogs(prev => [...prev, `[INFO] Dataset: ${datasetName}`]);
        setLogs(prev => [...prev, `[INFO] Modelo: ${MODEL_OPTIONS.find(m => m.value === modelType)?.label}`]);
        setLogs(prev => [...prev, `[INFO] Hardware: ${device === 'cuda' ? 'GPU (CUDA)' : 'CPU'}`]);

        try {
            const endpoint = mode === 'treinar' ? '/train/start' : '/train/retrain';
            await fetch(`http://localhost:8000${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_path: datasetPath,
                    model_type: modelType,
                    device: device,
                    pretrained_path: mode === 'retreinar' ? pretrainedPath : undefined,
                    ...params
                })
            });
        } catch (error: any) {
            setLoading(false);
            setMessage({ text: `❌ Erro: ${error.message}`, type: 'error' });
            setLogs(prev => [...prev, `[ERRO] ${error.message}`]);
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
                description="Treine ou retreine modelos LSTM/Transformer para detecção de anomalias comportamentais."
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
                    {/* Dataset Selection */}
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <Dumbbell className="w-5 h-5 text-primary" />
                            Dados de Entrada
                        </h3>

                        <div className="space-y-4">
                            {/* Dataset Path */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Dataset (.pt)</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={shortenPath(datasetPath)}
                                        title={datasetPath}
                                        readOnly
                                        className="flex-1 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm cursor-pointer"
                                        placeholder="Selecione uma pasta para treinamento..."
                                        onClick={() => setExplorerOpen('dataset')}
                                    />
                                    <button
                                        onClick={() => setExplorerOpen('dataset')}
                                        className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                                    >
                                        <FolderInput className="w-4 h-4" />
                                    </button>
                                </div>
                                {datasetName && (
                                    <p className="text-xs text-muted-foreground">
                                        Dataset: <span className="font-mono text-primary">{datasetName}</span>
                                    </p>
                                )}
                            </div>

                            {/* Pretrained Model (only for retreinar) */}
                            {mode === 'retreinar' && (
                                <div className="space-y-2 p-4 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                                    <label className="text-sm font-medium text-orange-400">Modelo Pré-treinado</label>
                                    <div className="flex gap-2">
                                        <input
                                            type="text"
                                            value={shortenPath(pretrainedPath)}
                                            title={pretrainedPath}
                                            readOnly
                                            className="flex-1 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm cursor-pointer"
                                            placeholder="Selecione um modelo para retreino..."
                                            onClick={() => setExplorerOpen('pretrained')}
                                        />
                                        <button
                                            onClick={() => setExplorerOpen('pretrained')}
                                            className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                                        >
                                            <FolderInput className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Model Selection */}
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <Settings2 className="w-5 h-5 text-primary" />
                            Arquitetura do Modelo
                        </h3>

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

                        {/* Hardware Selection */}
                        <div className="mt-6 space-y-2">
                            <label className="text-sm font-medium text-muted-foreground italic">Hardware para Treinamento</label>
                            <div className="grid grid-cols-2 gap-2 p-1 bg-muted rounded-xl">
                                <button
                                    onClick={() => setDevice('cuda')}
                                    className={`py-2 text-xs font-bold rounded-lg transition-all ${device === 'cuda' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-background/50 text-muted-foreground'}`}
                                >
                                    GPU (CUDA)
                                </button>
                                <button
                                    onClick={() => setDevice('cpu')}
                                    className={`py-2 text-xs font-bold rounded-lg transition-all ${device === 'cpu' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-background/50 text-muted-foreground'}`}
                                >
                                    CPU
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Hyperparameters */}
                    <div className="rounded-xl border border-border bg-card p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="font-semibold text-lg flex items-center gap-2">
                                Hiperparâmetros
                            </h3>
                            <button
                                onClick={resetParams}
                                className="text-xs text-muted-foreground hover:text-primary flex items-center gap-1"
                            >
                                <RotateCcw className="w-3 h-3" />
                                Restaurar Padrão
                            </button>
                        </div>

                        <div className="grid grid-cols-4 gap-4 mb-4">
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Épocas</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    value={params.epochs}
                                    onChange={e => setParams({ ...params, epochs: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Batch Size</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    value={params.batch_size}
                                    onChange={e => setParams({ ...params, batch_size: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Learning Rate</label>
                                <input
                                    type="number"
                                    step="0.0001"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm font-mono"
                                    value={params.lr}
                                    onChange={e => setParams({ ...params, lr: parseFloat(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Dropout</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    value={params.dropout}
                                    onChange={e => setParams({ ...params, dropout: parseFloat(e.target.value) || 0 })}
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-4 gap-4">
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Hidden Size</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    value={params.hidden_size}
                                    onChange={e => setParams({ ...params, hidden_size: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Num Layers</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    value={params.num_layers}
                                    onChange={e => setParams({ ...params, num_layers: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Num Heads</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    value={params.num_heads}
                                    onChange={e => setParams({ ...params, num_heads: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Kernel Size</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    value={params.kernel_size}
                                    onChange={e => setParams({ ...params, kernel_size: parseInt(e.target.value) || 0 })}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Train Button */}
                    <button
                        onClick={handleTrain}
                        disabled={loading || !datasetPath}
                        className={`w-full py-4 rounded-xl font-bold flex justify-center items-center gap-3 text-lg transition-all shadow-xl ${loading || !datasetPath
                            ? 'bg-muted cursor-not-allowed text-muted-foreground'
                            : mode === 'treinar'
                                ? 'bg-primary text-primary-foreground hover:brightness-110 shadow-primary/20'
                                : 'bg-orange-500 text-white hover:brightness-110 shadow-orange-500/20'
                            }`}
                    >
                        {loading ? (
                            <>
                                <RefreshCcw className="w-6 h-6 animate-spin" />
                                {mode === 'treinar' ? 'Treinando...' : 'Retreinando...'}
                            </>
                        ) : (
                            <>
                                <Play className="w-6 h-6 fill-current" />
                                {mode === 'treinar' ? 'Iniciar Treinamento' : 'Iniciar Retreinamento'}
                            </>
                        )}
                    </button>
                </div>

                {/* Terminal Panel */}
                <div>
                    <Terminal
                        logs={logs}
                        title="Console de Treinamento"
                        height="700px"
                        isLoading={loading}
                        onClear={async () => {
                            setLogs([]);
                            try { await APIService.clearLogs(); } catch (e) { console.error(e); }
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
