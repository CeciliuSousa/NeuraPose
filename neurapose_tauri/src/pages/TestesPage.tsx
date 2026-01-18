import { useState, useEffect } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    PlayCircle,
    FolderInput,
    FileCode,
    Cpu,
    Zap,
    RefreshCcw,
    Database,
    Binary,
    StopCircle
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { Terminal } from '../components/ui/Terminal';
import { StatusMessage } from '../components/ui/StatusMessage';
import { VideoPreviewPanel } from '../components/ui/VideoPreviewPanel';
import { PreviewToggle } from '../components/ui/PreviewToggle';
import { useProcessingStatus } from '../hooks/useProcessingStatus';

export default function TestesPage() {
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'processing' } | null>(null);

    const [config, setConfig] = useState({
        modelPath: '',
        datasetPath: '',
        device: 'cuda',
        showPreview: false
    });

    const [explorerTarget, setExplorerTarget] = useState<'model' | 'dataset' | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});
    const { setPageStatus } = useProcessingStatus();

    // Load config and restore state
    useEffect(() => {
        // Restaurar estado se houver teste em andamento
        APIService.healthCheck().then(res => {
            // Só mostra mensagem se for ESTE processo (test)
            if (res.data.processing && res.data.current_process === 'test') {
                setLoading(true);
                setMessage({ text: '⏳ Teste em andamento...', type: 'processing' });
                setPageStatus('test', 'processing');
            }
        }).catch(() => { });

        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
            }
        });

        // Restaurar logs do localStorage
        const savedLogs = localStorage.getItem('np_test_logs');
        if (savedLogs) setLogs(JSON.parse(savedLogs));

        // Restore Config
        const savedConfig = localStorage.getItem('np_test_config');
        if (savedConfig) {
            try {
                setConfig(JSON.parse(savedConfig));
            } catch (e) {
                console.error("Failed to parse saved config", e);
            }
        }
    }, []);

    // Save Config on Change
    useEffect(() => {
        localStorage.setItem('np_test_config', JSON.stringify(config));
    }, [config]);


    // Polling de Logs
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (loading) {
            interval = setInterval(async () => {
                try {
                    const res = await APIService.getLogs('test');
                    setLogs(res.data.logs);
                    localStorage.setItem('np_test_logs', JSON.stringify(res.data.logs));

                    const health = await APIService.healthCheck();
                    if (!health.data.processing) {
                        setLoading(false);
                        setMessage({ text: '✅ Teste concluído! Verifique os resultados.', type: 'success' });
                        setPageStatus('test', 'success');
                    }
                } catch (e) { console.error(e); }
            }, 1000);
        }
        return () => { if (interval) clearInterval(interval); };
    }, [loading]);

    const handleStop = async () => {
        try {
            await APIService.stopTesting();
            setLogs(prev => [...prev, '[INFO] Solicitação de parada enviada...']);
            setMessage({ text: '⚠️ Parando teste...', type: 'processing' });
        } catch (e) { console.error(e); }
    };

    const handleRunTest = async () => {
        if (!config.modelPath || !config.datasetPath) {
            setMessage({ text: 'Por favor, selecione o modelo e o dataset de teste.', type: 'error' });
            return;
        }
        setLoading(true);
        setMessage({ text: '⏳ Executando testes de validação...', type: 'processing' });
        setLogs(prev => [...prev, `[INFO] Iniciando validação de modelo...`]);
        setPageStatus('test', 'processing');
        try {
            await APIService.startTesting({
                model_path: config.modelPath,
                dataset_path: config.datasetPath,
                device: config.device,
                show_preview: config.showPreview
            });
        } catch (error: any) {
            setLoading(false);
            setMessage({ text: `❌ Erro: ${error.response?.data?.detail || error.message}`, type: 'error' });
            setLogs(prev => [...prev, `[ERRO] ${error.response?.data?.detail || error.message}`]);
        }
    };

    return (
        <div className="space-y-6 max-w-6xl mx-auto">
            <PageHeader
                title="Validação e Performance"
                description="Execute testes de precisão em modelos treinados e valide o desempenho real."
            />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Configuration Panel */}
                <div className="lg:col-span-12 xl:col-span-6 space-y-6">
                    <div className="bg-card border border-border rounded-2xl p-8 shadow-md">
                        <h3 className="font-bold text-2xl mb-8 flex items-center gap-3">
                            <Binary className="w-6 h-6 text-primary" />
                            Setup de Teste
                        </h3>

                        <div className="space-y-6 mb-10">
                            {/* Model Selection */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Modelo Treinado (.pth / .pt)</label>
                                <div className="flex gap-2">
                                    <div className="flex-1 relative">
                                        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground">
                                            <FileCode className="w-4 h-4" />
                                        </div>
                                        <input
                                            type="text"
                                            className="w-full pl-11 pr-4 py-3 rounded-xl bg-background border border-border text-xs font-mono outline-none focus:ring-2 focus:ring-primary/40 truncate cursor-pointer"
                                            value={config.modelPath ? config.modelPath.replace(/\\/g, '/').split('/').pop() || '' : ''}
                                            title={config.modelPath}
                                            readOnly
                                            placeholder="Selecione o modelo para teste..."
                                            onClick={() => setExplorerTarget('model')}
                                        />
                                    </div>
                                    <button
                                        onClick={() => setExplorerTarget('model')}
                                        className="px-4 py-3 bg-secondary rounded-xl border border-border hover:bg-primary/10 hover:text-primary transition-all shrink-0"
                                    >
                                        <FolderInput className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>

                            {/* Dataset Selection */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Dataset de Teste</label>
                                <div className="flex gap-2">
                                    <div className="flex-1 relative">
                                        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground">
                                            <Database className="w-4 h-4" />
                                        </div>
                                        <input
                                            type="text"
                                            className="w-full pl-11 pr-4 py-3 rounded-xl bg-background border border-border text-xs font-mono outline-none focus:ring-2 focus:ring-primary/40 truncate cursor-pointer"
                                            value={config.datasetPath ? config.datasetPath.replace(/\\/g, '/').split('/').pop() || '' : ''}
                                            title={config.datasetPath}
                                            readOnly
                                            placeholder="Selecione o dataset para teste..."
                                            onClick={() => setExplorerTarget('dataset')}
                                        />
                                    </div>
                                    <button
                                        onClick={() => setExplorerTarget('dataset')}
                                        className="px-4 py-3 bg-secondary rounded-xl border border-border hover:bg-primary/10 hover:text-primary transition-all shrink-0"
                                    >
                                        <FolderInput className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>

                            {/* Device Toggle */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider italic">Motor de Inferência</label>
                                <div className="grid grid-cols-2 gap-3 p-1.5 bg-muted/50 rounded-2xl">
                                    <button
                                        onClick={() => setConfig({ ...config, device: 'cuda' })}
                                        className={`flex items-center justify-center gap-2 py-3 text-sm font-bold rounded-xl transition-all ${config.device === 'cuda' ? 'bg-primary text-primary-foreground shadow-lg' : 'hover:bg-background/80 text-muted-foreground'}`}
                                    >
                                        <Zap className="w-4 h-4" />
                                        GPU (CUDA)
                                    </button>
                                    <button
                                        onClick={() => setConfig({ ...config, device: 'cpu' })}
                                        className={`flex items-center justify-center gap-2 py-3 text-sm font-bold rounded-xl transition-all ${config.device === 'cpu' ? 'bg-primary text-primary-foreground shadow-lg' : 'hover:bg-background/80 text-muted-foreground'}`}
                                    >
                                        <Cpu className="w-4 h-4" />
                                        CPU
                                    </button>
                                </div>
                            </div>

                            {/* Preview Toggle */}
                            <PreviewToggle
                                checked={config.showPreview}
                                onChange={(value) => setConfig({ ...config, showPreview: value })}
                                isProcessing={loading}
                            />
                        </div>

                        {!loading ? (
                            <button
                                onClick={handleRunTest}
                                className="w-full py-4 rounded-xl font-bold text-primary-foreground flex justify-center items-center gap-3 text-lg transition-all shadow-xl bg-primary hover:brightness-110 hover:scale-[1.01] active:scale-95 shadow-primary/20"
                            >
                                <PlayCircle className="w-6 h-6 fill-current" />
                                Iniciar Bateria de Testes
                            </button>
                        ) : (
                            <div className="flex gap-3">
                                <button
                                    disabled
                                    className="flex-1 py-4 rounded-xl font-bold bg-muted text-muted-foreground flex justify-center items-center gap-3"
                                >
                                    <RefreshCcw className="w-6 h-6 animate-spin" />
                                    Validando...
                                </button>
                                <button
                                    onClick={handleStop}
                                    className="px-6 py-4 rounded-xl font-bold bg-red-500 text-white flex justify-center items-center gap-2 hover:bg-red-600 transition-all"
                                >
                                    <StopCircle className="w-6 h-6" />
                                    Parar
                                </button>
                            </div>
                        )}
                    </div>

                    {/* <div className="bg-primary/5 border border-primary/10 p-6 rounded-2xl">
                        <h4 className="font-bold text-primary mb-2 flex items-center gap-2">
                            <TestTube2 className="w-5 h-5" />
                            O que será avaliado?
                        </h4>
                        <ul className="text-sm space-y-2 text-muted-foreground italic">
                            <li className="flex items-center gap-2 line-through opacity-50"><ChevronRight className="w-3 h-3" /> Relatórios de Acurácia e F1-Score</li>
                            <li className="flex items-center gap-2 line-through opacity-50"><ChevronRight className="w-3 h-3" /> Matriz de Confusão por Classe</li>
                            <li className="flex items-center gap-2"><ChevronRight className="w-3 h-3 text-primary" /> Logs detalhados de predição em tempo real</li>
                            <li className="flex items-center gap-2"><ChevronRight className="w-3 h-3 text-primary" /> Tempo de inferência por frame</li>
                        </ul>
                    </div> */}
                </div>

                {/* Real-time Logs */}
                <div className="lg:col-span-12 xl:col-span-6 space-y-6">
                    {/* Status Message */}
                    {message && (
                        <div className="mb-4">
                            <StatusMessage
                                message={message.text}
                                type={message.type}
                                onClose={() => setMessage(null)}
                                autoCloseDelay={message.type === 'success' ? 5000 : undefined}
                            />
                        </div>
                    )}

                    {/* Terminal Component */}
                    <Terminal
                        logs={logs}
                        title="Console de Validação"
                        height="500px"
                        isLoading={loading}
                        onClear={async () => {
                            setLogs([]);
                            try { await APIService.clearLogs('test'); } catch (e) { console.error(e); }
                        }}
                    />

                    {/* Video Preview - Below Terminal */}
                    <VideoPreviewPanel
                        isVisible={loading && config.showPreview}
                        title="Preview de Inferência"
                    />
                </div>
            </div>

            <FileExplorerModal
                isOpen={explorerTarget !== null}
                onClose={() => setExplorerTarget(null)}
                onSelect={(path) => {
                    if (explorerTarget === 'model') {
                        // Ao selecionar um Modelo, busca o model_best.pt
                        let finalPath = path;
                        // Simulação de busca no front (o backend já tratará, mas informamos o usuário)
                        // Note: finalPath remains the folder path, but backend will look for model_best.pt
                        setConfig({ ...config, modelPath: finalPath });
                    }
                    if (explorerTarget === 'dataset') {
                        // Ao selecionar um Dataset, busca teste/videos
                        // O backend já faz o ajuste se enviarmos a pasta raiz do dataset
                        setConfig({ ...config, datasetPath: path });
                    }
                    setExplorerTarget(null);
                }}
                initialPath={explorerTarget === 'model' ? roots.modelos_treinados : roots.datasets}
                rootPath={explorerTarget === 'model' ? roots.modelos_treinados : roots.datasets}
                title={explorerTarget === 'model' ? "Selecionar Modelo (.pt)" : "Selecionar Diretório do Dataset"}
            />
        </div>
    );
}
