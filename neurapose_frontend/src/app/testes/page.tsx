'use client';

import { useState, useEffect, useRef } from 'react';
import { PageHeader } from '@/components/ui/page-header';
import {
    TestTube2,
    PlayCircle,
    Terminal as TerminalIcon,
    ChevronRight,
    Search,
    FileCode,
    Cpu,
    Zap,
    RefreshCcw,
    Database,
    Binary
} from 'lucide-react';
import { APIService } from '@/services/api';
import { FileExplorerModal } from '@/components/file-explorer-modal';

export default function TestsPage() {
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const terminalRef = useRef<HTMLDivElement>(null);

    const [config, setConfig] = useState({
        modelPath: '',
        datasetPath: '',
        device: 'cuda'
    });

    const [explorerTarget, setExplorerTarget] = useState<'model' | 'dataset' | null>(null);

    // Polling de Logs
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (loading) {
            interval = setInterval(async () => {
                try {
                    const res = await APIService.getLogs();
                    setLogs(res.data.logs);

                    const health = await APIService.healthCheck();
                    if (!health.data.processing) {
                        setLoading(false);
                    }
                } catch (e) { console.error(e); }
            }, 1000);
        }
        return () => { if (interval) clearInterval(interval); };
    }, [loading]);

    // Auto-scroll terminal
    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [logs]);

    const handleRunTest = async () => {
        if (!config.modelPath || !config.datasetPath) {
            alert("Selecione o modelo e o dataset de teste.");
            return;
        }
        setLoading(true);
        setLogs(prev => [...prev, `[INFO] Iniciando validação de modelo...`]);
        try {
            await APIService.startTesting({
                model_path: config.modelPath,
                dataset_path: config.datasetPath,
                device: config.device
            });
        } catch (error: any) {
            setLoading(false);
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
                                            className="w-full pl-11 pr-4 py-3 rounded-xl bg-background border border-border text-xs font-mono outline-none focus:ring-2 focus:ring-primary/40 truncate"
                                            value={config.modelPath}
                                            readOnly
                                            placeholder="Selecione o arquivo do modelo..."
                                        />
                                    </div>
                                    <button
                                        onClick={() => setExplorerTarget('model')}
                                        className="px-4 py-3 bg-secondary rounded-xl border border-border hover:bg-primary/10 hover:text-primary transition-all shrink-0"
                                    >
                                        <Search className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>

                            {/* Dataset Selection */}
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Dataset de Validação</label>
                                <div className="flex gap-2">
                                    <div className="flex-1 relative">
                                        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground">
                                            <Database className="w-4 h-4" />
                                        </div>
                                        <input
                                            type="text"
                                            className="w-full pl-11 pr-4 py-3 rounded-xl bg-background border border-border text-xs font-mono outline-none focus:ring-2 focus:ring-primary/40 truncate"
                                            value={config.datasetPath}
                                            readOnly
                                            placeholder="Selecione a pasta do dataset..."
                                        />
                                    </div>
                                    <button
                                        onClick={() => setExplorerTarget('dataset')}
                                        className="px-4 py-3 bg-secondary rounded-xl border border-border hover:bg-primary/10 hover:text-primary transition-all shrink-0"
                                    >
                                        <Search className="w-5 h-5" />
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
                        </div>

                        <button
                            onClick={handleRunTest}
                            disabled={loading}
                            className={`w-full py-4 rounded-xl font-bold text-primary-foreground flex justify-center items-center gap-3 text-lg transition-all shadow-xl
                                ${loading ? 'bg-muted cursor-not-allowed text-muted-foreground' : 'bg-primary hover:brightness-110 hover:scale-[1.01] active:scale-95 shadow-primary/20'}
                            `}
                        >
                            {loading ? <RefreshCcw className="w-6 h-6 animate-spin" /> : <PlayCircle className="w-6 h-6 fill-current" />}
                            {loading ? 'Validando...' : 'Iniciar Bateria de Testes'}
                        </button>
                    </div>

                    <div className="bg-primary/5 border border-primary/10 p-6 rounded-2xl">
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
                    </div>
                </div>

                {/* Real-time Logs */}
                <div className="lg:col-span-12 xl:col-span-6 flex flex-col h-full bg-slate-950 rounded-2xl border border-border shadow-2xl overflow-hidden min-h-[500px]">
                    <div className="flex items-center justify-between px-6 py-4 bg-slate-900/50 border-b border-white/10">
                        <div className="flex items-center gap-3">
                            <TerminalIcon className="w-5 h-5 text-primary" />
                            <span className="text-xs font-mono font-bold text-slate-300">Resumo de Validação</span>
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
                        className="flex-1 p-6 font-mono text-xs overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-white/10"
                    >
                        {logs.length === 0 && (
                            <div className="text-slate-700 italic flex flex-col items-center justify-center h-full gap-4">
                                <Search className="w-12 h-12 opacity-20" />
                                <span>Aguardando início dos testes...</span>
                            </div>
                        )}
                        {logs.map((log, i) => {
                            const isError = log.includes('[ERRO]');
                            const isInfo = log.includes('[INFO]');
                            const isMetric = log.includes('Acc:') || log.includes('F1:');

                            return (
                                <div key={i} className={`
                                    whitespace-pre-wrap break-all border-l-2 pl-4 py-1.5 transition-all
                                    ${isError ? 'text-red-400 border-red-500 bg-red-500/5' :
                                        isInfo ? 'text-blue-400 border-blue-500 bg-blue-500/5' :
                                            isMetric ? 'text-amber-400 border-amber-500/30 font-bold' :
                                                'text-slate-300 border-transparent'}
                                `}>
                                    {log}
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            <FileExplorerModal
                isOpen={explorerTarget !== null}
                onClose={() => setExplorerTarget(null)}
                onSelect={(path) => {
                    if (explorerTarget === 'model') setConfig({ ...config, modelPath: path });
                    if (explorerTarget === 'dataset') setConfig({ ...config, datasetPath: path });
                    setExplorerTarget(null);
                }}
                initialPath={explorerTarget === 'model' ? config.modelPath : config.datasetPath}
                title={explorerTarget === 'model' ? "Selecionar Modelo (.pt)" : "Selecionar Pasta do Dataset"}
            />
        </div>
    );
}
