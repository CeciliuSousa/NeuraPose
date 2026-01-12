'use client';

import { useState, useEffect, useRef } from 'react';
import {
    Video,
    Play,
    Pause,
    Square,
    FolderInput,
    Loader2
} from 'lucide-react';
import { APIService } from '@/services/api';
import { FileExplorerModal } from '@/components/file-explorer-modal';

export default function ProcessamentoPage() {
    // Form State
    const [config, setConfig] = useState({
        inputPath: '',
        datasetName: '', // Nome do dataset de saída (sufixo -processado adicionado automaticamente)
        device: 'cuda', // 'cuda' ou 'cpu'
        showPreview: false,
    });

    // Processing State
    const [loading, setLoading] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [explorerTarget, setExplorerTarget] = useState<'input' | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});

    const terminalRef = useRef<HTMLDivElement>(null);
    const [autoScroll, setAutoScroll] = useState(true);

    // Logs & Health Polling
    useEffect(() => {
        // 1. Tentar carregar estado do localStorage
        const savedConfig = localStorage.getItem('np_process_config');
        if (savedConfig) setConfig(JSON.parse(savedConfig));

        const savedLogs = localStorage.getItem('np_process_logs');
        if (savedLogs) setLogs(JSON.parse(savedLogs));

        const savedLoading = localStorage.getItem('np_process_loading');
        if (savedLoading === 'true') setLoading(true);

        // 2. Inicializa paths padrões do backend
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                const { videos } = res.data.paths;
                setRoots(res.data.paths);
                // Só atualiza se o campo estiver vazio
                setConfig(prev => ({
                    ...prev,
                    inputPath: prev.inputPath || videos
                }));
            }
        }).catch(err => console.error("Erro ao carregar caminhos do backend:", err));

        let interval: NodeJS.Timeout;

        if (loading) {
            localStorage.setItem('np_process_loading', 'true');
            interval = setInterval(async () => {
                try {
                    const res = await APIService.getLogs();
                    const newLogs = res.data.logs;
                    setLogs(newLogs);
                    localStorage.setItem('np_process_logs', JSON.stringify(newLogs));

                    // Checa também o status (pause/execução)
                    const health = await APIService.healthCheck();
                    setIsPaused(health.data.paused);

                    // Se o servidor diz que não está mais processando, paramos o polling
                    if (!health.data.processing && loading) {
                        setLoading(false);
                        localStorage.setItem('np_process_loading', 'false');
                    }
                } catch (e) {
                    console.error("Erro ao buscar status:", e);
                }
            }, 1000);
        } else {
            localStorage.setItem('np_process_loading', 'false');
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [loading]);

    // Persistir config ao mudar
    useEffect(() => {
        localStorage.setItem('np_process_config', JSON.stringify(config));
    }, [config]);

    // Auto-scroll terminal (apenas se autoScroll estiver ativo)
    useEffect(() => {
        if (terminalRef.current && autoScroll) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [logs, autoScroll]);

    // Detecta se o usuário scrollou manualmente
    const handleTerminalScroll = () => {
        if (terminalRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = terminalRef.current;
            // Se estiver perto do final (10px de tolerância), ativa auto-scroll
            const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
            setAutoScroll(isAtBottom);
        }
    };

    const handleProcess = async () => {
        if (!config.inputPath) {
            alert("Por favor, selecione a pasta de entrada.");
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
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-muted-foreground">Pasta de Entrada (Vídeos)</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={config.inputPath}
                                        onChange={(e) => setConfig({ ...config, inputPath: e.target.value })}
                                        placeholder="Ex: C:\Videos\Entrada"
                                        className="flex-1 bg-background border border-border rounded-md px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary/50 transition-all font-mono"
                                    />
                                    <button
                                        onClick={() => openExplorer()}
                                        className="px-3 py-2 bg-secondary rounded-md border border-border hover:bg-secondary/80 transition-colors"
                                    >
                                        <FolderInput className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-muted-foreground">Nome do Dataset de Saída</label>
                                <input
                                    type="text"
                                    value={config.datasetName}
                                    onChange={(e) => setConfig({ ...config, datasetName: e.target.value })}
                                    placeholder="Deixe vazio para usar o nome da pasta de entrada"
                                    className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                />
                                <p className="text-xs text-muted-foreground italic">
                                    Saída: resultados-processamentos/{config.datasetName || '[nome-da-pasta]'}-processado/
                                </p>
                            </div>

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


                            <div className="pt-2">
                                <label className="flex items-center gap-3 cursor-pointer group">
                                    <div className="relative">
                                        <input
                                            type="checkbox"
                                            checked={config.showPreview}
                                            onChange={(e) => setConfig({ ...config, showPreview: e.target.checked })}
                                            className="sr-only peer"
                                        />
                                        <div className="w-10 h-5 bg-muted rounded-full peer peer-checked:bg-primary transition-colors"></div>
                                        <div className="absolute left-1 top-1 w-3 h-3 bg-white rounded-full transition-transform peer-checked:translate-x-5"></div>
                                    </div>
                                    <span className="text-sm font-medium group-hover:text-primary transition-colors">Mostrar Preview em tempo real</span>
                                </label>
                            </div>

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
                    {loading && config.showPreview && (
                        <div className="bg-card border border-border rounded-xl overflow-hidden shadow-lg animate-in fade-in zoom-in duration-300">
                            <div className="px-4 py-2 bg-muted/50 border-b border-border flex items-center justify-between">
                                <span className="text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                    Live Preview
                                </span>
                            </div>
                            <div className="aspect-video bg-black flex items-center justify-center relative">
                                <img
                                    src="http://localhost:8000/video_feed"
                                    alt="Video Stream"
                                    className="w-full h-full object-contain"
                                    onError={(e: any) => {
                                        e.target.style.display = 'none';
                                    }}
                                    onLoad={(e: any) => {
                                        e.target.style.display = 'block';
                                    }}
                                />
                                <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
                                    <Video className="w-12 h-12" />
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right: Terminal Output */}
                <div className="flex flex-col h-full bg-slate-950 rounded-xl border border-border shadow-2xl overflow-hidden min-h-[500px] lg:min-h-0">
                    <div className="flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-white/5">
                        <div className="flex items-center gap-2">
                            <div className="flex gap-1.5">
                                <div className="w-3 h-3 rounded-full bg-red-500/50" />
                                <div className="w-3 h-3 rounded-full bg-orange-500/50" />
                                <div className="w-3 h-3 rounded-full bg-green-500/50" />
                            </div>
                            <span className="text-xs font-mono text-slate-400 ml-2">Terminal Output</span>
                        </div>
                        <button
                            onClick={() => setLogs([])}
                            className="text-[10px] uppercase font-bold text-slate-500 hover:text-white transition-colors"
                        >
                            Limpar
                        </button>
                    </div>
                    <div
                        ref={terminalRef}
                        onScroll={handleTerminalScroll}
                        className="flex-1 p-4 font-mono text-sm overflow-y-auto space-y-1 max-h-[400px] scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent"
                    >
                        {logs.length === 0 && (
                            <div className="text-slate-700 italic flex items-center justify-center h-full">
                                Aguardando início do processo...
                            </div>
                        )}
                        {logs.map((log, i) => {
                            const isError = log.includes('[ERRO]');
                            const isInfo = log.includes('[INFO]');
                            const isTqdm = log.includes('|') && log.includes('%');

                            return (
                                <div key={i} className={`
                                    whitespace-pre-wrap break-all border-l-2 pl-3 py-0.5
                                    ${isError ? 'text-red-400 border-red-500 bg-red-500/5' :
                                        isInfo ? 'text-blue-400 border-blue-500 bg-blue-500/5' :
                                            isTqdm ? 'text-emerald-400 border-emerald-500/30' :
                                                'text-slate-300 border-transparent'}
                                `}>
                                    {log}
                                </div>
                            );
                        })}
                    </div>
                    {loading && (
                        <div className="bg-slate-900 px-4 py-2 border-t border-white/5 flex items-center justify-between">
                            <span className="text-[10px] text-slate-500 font-mono animate-pulse">
                                {isPaused ? 'PROCESSO PAUSADO' : 'EXECUTANDO NO SERVIDOR...'}
                            </span>
                            <div className="flex gap-1">
                                <div className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <FileExplorerModal
                isOpen={explorerTarget !== null}
                onClose={closeExplorer}
                onSelect={handleSelectPath}
                initialPath={config.inputPath}
                rootPath={roots.videos}
                title="Selecionar Pasta de Entrada (Vídeos)"
            />
        </div>
    );
}
