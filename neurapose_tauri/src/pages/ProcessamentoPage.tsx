import { useState, useEffect } from 'react';
import {
    Video,
    Play,
    Pause,
    Square,
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { Terminal } from '../components/ui/Terminal';
import { VideoPreviewPanel } from '../components/ui/VideoPreviewPanel';
import { PreviewToggle } from '../components/ui/PreviewToggle';
import { PathSelector } from '../components/ui/PathSelector';

export default function ProcessamentoPage() {
    // Form State
    const [config, setConfig] = useState({
        inputPath: '',
        datasetName: '',
        device: 'cuda',
        showPreview: false,
    });

    // Processing State
    const [loading, setLoading] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [explorerTarget, setExplorerTarget] = useState<'input' | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});
    const [progress, setProgress] = useState(0);

    // Carregamento inicial de caminhos e logs - apenas UMA VEZ na montagem
    useEffect(() => {
        const savedLogs = localStorage.getItem('np_process_logs');
        if (savedLogs) setLogs(JSON.parse(savedLogs));

        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
                setConfig(prev => ({
                    ...prev,
                    inputPath: ''
                }));
            }
        }).catch(err => console.error("Erro ao carregar caminhos do backend:", err));
    }, []);

    // Logs & Health Polling - apenas quando loading muda
    useEffect(() => {
        const savedConfig = localStorage.getItem('np_process_config');
        if (savedConfig && !config.inputPath) {
            const parsed = JSON.parse(savedConfig);
            setConfig((prev: any) => ({ ...prev, ...parsed, inputPath: prev.inputPath }));
        }

        const savedLoading = localStorage.getItem('np_process_loading');
        if (savedLoading === 'true' && !loading) setLoading(true);

        let interval: any;

        if (loading) {
            localStorage.setItem('np_process_loading', 'true');
            interval = setInterval(async () => {
                try {
                    const res = await APIService.getLogs('process');
                    const newLogs = res.data.logs;
                    setLogs(newLogs);
                    localStorage.setItem('np_process_logs', JSON.stringify(newLogs));

                    const progressLine = [...newLogs].reverse().find(l => l.includes('[PROGRESSO]'));
                    if (progressLine) {
                        const match = progressLine.match(/(\d+)%/);
                        if (match) setProgress(parseInt(match[1]));
                    }

                    const health = await APIService.healthCheck();
                    setIsPaused(health.data.paused);

                    if (!health.data.processing && loading) {
                        setLoading(false);
                        localStorage.setItem('np_process_loading', 'false');
                    }
                } catch (e) {
                    console.error("Erro ao buscar status:", e);
                }
            }, 5000);
        } else {
            localStorage.setItem('np_process_loading', 'false');
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [loading]);

    useEffect(() => {
        localStorage.setItem('np_process_config', JSON.stringify(config));
    }, [config]);

    const handleProcess = async () => {
        if (!config.inputPath) {
            alert("Por favor, selecione o diretório de entrada.");
            return;
        }

        setLoading(true);
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Iniciando comunicação com o servidor...`]);

        try {
            await APIService.startProcessing({
                input_path: config.inputPath,
                dataset_name: config.datasetName,
                device: config.device,
                show_preview: config.showPreview
            });

        } catch (error: any) {
            console.error("Erro ao iniciar:", error);
            const errMsg = error.response?.data?.detail || error.message;
            setLogs(prev => [...prev, `[ERRO] ${errMsg}`]);
            setLoading(false);
        }
    };

    const handleStop = async () => {
        try {
            await APIService.stopProcess();
            setLogs(prev => [...prev, `[INFO] Solicitação de parada enviada...`]);
        } catch (e) { console.error(e); }
    };

    const togglePause = async () => {
        try {
            if (isPaused) await APIService.resumeProcess();
            else await APIService.pauseProcess();
            setIsPaused(!isPaused);
        } catch (e) { console.error(e); }
    };

    const openExplorer = () => setExplorerTarget('input');
    const closeExplorer = () => setExplorerTarget(null);

    const handleSelectPath = (path: string) => {
        setConfig({ ...config, inputPath: path });
        closeExplorer();
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center gap-3 border-b border-border pb-4">
                <div className="p-2 bg-primary/10 rounded-md">
                    <Video className="w-6 h-6 text-primary" />
                </div>
                <h1 className="text-2xl font-bold">Processamento de Vídeo</h1>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left: Configuration */}
                <div className="space-y-6">
                    <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
                        <h2 className="text-lg font-semibold mb-6">Configuração de Diretórios</h2>

                        <div className="space-y-4">
                            <div className="space-y-4">
                                <PathSelector
                                    label="Diretório de Entrada (Vídeos)"
                                    value={config.inputPath}
                                    onSelect={() => openExplorer()}
                                    placeholder="Selecione o diretório para processar..."
                                />
                            </div>

                            {/* <div className="space-y-2">
                                <label className="text-sm font-medium text-muted-foreground">Nome do Dataset de Saída</label>
                                <input
                                    type="text"
                                    value={config.datasetName}
                                    onChange={(e) => setConfig({ ...config, datasetName: e.target.value })}
                                    placeholder="Deixe vazio para usar o nome do diretório de entrada"
                                    className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                />
                                <p className="text-xs text-muted-foreground italic">
                                    Saída: resultados-processamentos/{config.datasetName || '[nome-da-pasta]'}-processado/
                                </p>
                            </div> */}

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-muted-foreground italic">Hardware para Inferência</label>
                                <div className="grid grid-cols-2 gap-2 p-1 bg-muted rounded-xl">
                                    <button
                                        onClick={() => setConfig({ ...config, device: 'cuda' })}
                                        className={`py-2 text-xs font-bold rounded-lg transition-all ${config.device === 'cuda' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-background/50 text-muted-foreground'}`}
                                    >
                                        GPU (CUDA)
                                    </button>
                                    <button
                                        onClick={() => setConfig({ ...config, device: 'cpu' })}
                                        className={`py-2 text-xs font-bold rounded-lg transition-all ${config.device === 'cpu' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-background/50 text-muted-foreground'}`}
                                    >
                                        CPU
                                    </button>
                                </div>
                            </div>


                            <PreviewToggle
                                checked={config.showPreview}
                                onChange={(value) => setConfig({ ...config, showPreview: value })}
                                isProcessing={loading}
                            />

                            <div className="pt-4 space-y-3">
                                {!loading ? (
                                    <button
                                        onClick={handleProcess}
                                        className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:brightness-110 transition-all flex items-center justify-center gap-2 shadow-lg shadow-primary/20"
                                    >
                                        <Play className="w-5 h-5 fill-current" />
                                        Iniciar Processamento
                                    </button>
                                ) : (
                                    <div className="flex gap-2">
                                        <button
                                            onClick={togglePause}
                                            className="flex-1 py-3 bg-orange-500 text-white rounded-lg font-semibold hover:bg-orange-600 transition-all flex items-center justify-center gap-2"
                                        >
                                            {isPaused ? <Play className="w-5 h-5 fill-current" /> : <Pause className="w-5 h-5 fill-current" />}
                                            {isPaused ? 'Continuar' : 'Pausar'}
                                        </button>
                                        <button
                                            onClick={handleStop}
                                            className="flex-1 py-3 bg-red-500 text-white rounded-lg font-semibold hover:bg-red-600 transition-all flex items-center justify-center gap-2"
                                        >
                                            <Square className="w-5 h-5 fill-current" />
                                            Parar
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Preview Video */}
                    <VideoPreviewPanel
                        isVisible={loading && config.showPreview}
                        title="Processamento em tempo real"
                    />
                </div>

                {/* Right: Terminal Output */}
                <Terminal
                    logs={logs.filter(log => !log.includes('[PROGRESSO]'))}
                    title="Console de processamentos"
                    height="550px"
                    progress={loading ? progress : undefined}
                    isLoading={loading}
                    isPaused={isPaused}
                    onClear={async () => {
                        setLogs([]);
                        localStorage.removeItem('np_process_logs');
                        try { await APIService.clearLogs('process'); } catch (e) { console.error(e); }
                    }}
                />
            </div>

            <FileExplorerModal
                isOpen={explorerTarget !== null}
                onClose={closeExplorer}
                onSelect={handleSelectPath}
                initialPath={roots.videos}
                rootPath={roots.videos}
                title="Selecionar Diretório de Entrada (Vídeos)"
            />
        </div>
    );
}
