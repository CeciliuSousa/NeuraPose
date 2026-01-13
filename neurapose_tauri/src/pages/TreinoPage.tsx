import { useState, useEffect, useRef } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    Dumbbell,
    Database,
    Play,
    Split,
    RefreshCcw,
    Terminal as TerminalIcon,
    FolderInput,
    Layers
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';

export default function TreinoPage() {
    const [activeTab, setActiveTab] = useState<'train' | 'split'>('train');
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const terminalRef = useRef<HTMLDivElement>(null);

    // Train Config State
    const [trainConfig, setTrainConfig] = useState({
        epochs: 100,
        batchSize: 32,
        lr: 0.0003,
        modelName: 'meu-modelo-neurapose',
        datasetName: 'dataset-final',
        temporalModel: 'tft'
    });

    // Split Config State
    const [splitConfig, setSplitConfig] = useState({
        inputDir: '',
        datasetName: 'meu-dataset',
        trainSplit: 'treino',
        testSplit: 'teste'
    });

    const [explorerOpen, setExplorerOpen] = useState(false);
    const [roots, setRoots] = useState<Record<string, string>>({});

    useEffect(() => {
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
                setSplitConfig(prev => ({
                    ...prev,
                    inputDir: res.data.paths.reidentificacoes || ''
                }));
            }
        });
    }, []);


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

    const handleTrain = async () => {
        setLoading(true);
        setLogs(prev => [...prev, `[INFO] Iniciando treinamento...`]);
        try {
            await APIService.startTraining({
                epochs: trainConfig.epochs,
                batch_size: trainConfig.batchSize,
                learning_rate: trainConfig.lr,
                model_name: trainConfig.modelName,
                dataset_name: trainConfig.datasetName,
                temporal_model: trainConfig.temporalModel
            });
        } catch (error: any) {
            setLoading(false);
            setLogs(prev => [...prev, `[ERRO] ${error.response?.data?.detail || error.message}`]);
        }
    };

    const handleSplit = async () => {
        if (!splitConfig.inputDir) {
            alert("Selecione a pasta de entrada para o split.");
            return;
        }
        setLoading(true);
        setLogs(prev => [...prev, `[INFO] Iniciando divisão do dataset...`]);
        try {
            await APIService.splitDataset({
                input_dir_process: splitConfig.inputDir,
                dataset_name: splitConfig.datasetName,
                train_split: splitConfig.trainSplit,
                test_split: splitConfig.testSplit
            });
        } catch (error: any) {
            setLoading(false);
            setLogs(prev => [...prev, `[ERRO] ${error.response?.data?.detail || error.message}`]);
        }
    };

    return (
        <div className="space-y-6 max-w-6xl mx-auto">
            <PageHeader
                title="Pipeline de Inteligência"
                description="Prepare seus dados e treine modelos temporais para detecção de anomalias."
            />

            <div className="flex gap-4 p-1 bg-muted/50 rounded-2xl w-fit mb-6">
                <button
                    onClick={() => setActiveTab('train')}
                    className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'train' ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/20' : 'hover:bg-muted text-muted-foreground'}`}
                >
                    <Dumbbell className="w-4 h-4" />
                    Treinamento
                </button>
                <button
                    onClick={() => setActiveTab('split')}
                    className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'split' ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/20' : 'hover:bg-muted text-muted-foreground'}`}
                >
                    <Split className="w-4 h-4" />
                    Gerar Dataset (Split)
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Configuration Column */}
                <div className="lg:col-span-12 xl:col-span-7 space-y-6">

                    {activeTab === 'train' ? (
                        <div className="bg-card border border-border rounded-2xl p-8 shadow-md">
                            <h3 className="font-bold text-2xl mb-8 flex items-center gap-3">
                                <Layers className="w-6 h-6 text-primary" />
                                Hiperparâmetros de Treino
                            </h3>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Identificador do Modelo</label>
                                    <input
                                        type="text"
                                        className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm outline-none focus:ring-2 focus:ring-primary/40 font-medium"
                                        value={trainConfig.modelName}
                                        onChange={e => setTrainConfig({ ...trainConfig, modelName: e.target.value })}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Dataset para Treino</label>
                                    <input
                                        type="text"
                                        className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm outline-none focus:ring-2 focus:ring-primary/40 font-medium"
                                        value={trainConfig.datasetName}
                                        onChange={e => setTrainConfig({ ...trainConfig, datasetName: e.target.value })}
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Épocas</label>
                                    <input
                                        type="number"
                                        className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm outline-none focus:ring-2 focus:ring-primary/40"
                                        value={trainConfig.epochs}
                                        onChange={e => setTrainConfig({ ...trainConfig, epochs: parseInt(e.target.value) })}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Batch Size</label>
                                    <input
                                        type="number"
                                        className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm outline-none focus:ring-2 focus:ring-primary/40"
                                        value={trainConfig.batchSize}
                                        onChange={e => setTrainConfig({ ...trainConfig, batchSize: parseInt(e.target.value) })}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">L. Rate</label>
                                    <input
                                        type="number"
                                        step="0.0001"
                                        className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm outline-none focus:ring-2 focus:ring-primary/40 font-mono"
                                        value={trainConfig.lr}
                                        onChange={e => setTrainConfig({ ...trainConfig, lr: parseFloat(e.target.value) })}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Arquitetura</label>
                                    <select
                                        className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm outline-none focus:ring-2 focus:ring-primary/40 font-bold"
                                        value={trainConfig.temporalModel}
                                        onChange={e => setTrainConfig({ ...trainConfig, temporalModel: e.target.value })}
                                    >
                                        <option value="tft">TFT</option>
                                        <option value="lstm">LSTM</option>
                                    </select>
                                </div>
                            </div>

                            <button
                                onClick={handleTrain}
                                disabled={loading}
                                className={`w-full py-4 rounded-xl font-bold text-primary-foreground flex justify-center items-center gap-3 text-lg transition-all shadow-xl
                                    ${loading ? 'bg-muted cursor-not-allowed text-muted-foreground' : 'bg-primary hover:brightness-110 hover:scale-[1.01] active:scale-95 shadow-primary/20'}
                                `}
                            >
                                {loading ? <RefreshCcw className="w-6 h-6 animate-spin" /> : <Play className="w-6 h-6 fill-current" />}
                                {loading ? 'Task em execução...' : 'Iniciar Ciclo de Treino'}
                            </button>
                        </div>
                    ) : (
                        <div className="bg-card border border-border rounded-2xl p-8 shadow-md">
                            <h3 className="font-bold text-2xl mb-8 flex items-center gap-3">
                                <Database className="w-6 h-6 text-primary" />
                                Configuração do Split
                            </h3>

                            <div className="space-y-6 mb-8">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Diretório de Resultados (Fonte)</label>
                                    <div className="flex gap-2">
                                        <input
                                            type="text"
                                            className="flex-1 px-4 py-3 rounded-xl bg-background border border-border text-xs font-mono outline-none focus:ring-2 focus:ring-primary/40"
                                            value={splitConfig.inputDir}
                                            onChange={e => setSplitConfig({ ...splitConfig, inputDir: e.target.value })}
                                            placeholder="Caminho para pasta resultados-reidentificacoes"
                                        />
                                        <button
                                            onClick={() => setExplorerOpen(true)}
                                            className="px-4 py-3 bg-secondary rounded-xl border border-border hover:bg-primary/10 hover:text-primary transition-all"
                                        >
                                            <FolderInput className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Nome de Saída do Dataset</label>
                                    <input
                                        type="text"
                                        className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm outline-none focus:ring-2 focus:ring-primary/40 font-medium"
                                        value={splitConfig.datasetName}
                                        onChange={e => setSplitConfig({ ...splitConfig, datasetName: e.target.value })}
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Subpasta Treino</label>
                                        <input
                                            type="text"
                                            className="w-full px-4 py-3 rounded-xl bg-background border border-border text-xs outline-none focus:ring-2 focus:ring-primary/40"
                                            value={splitConfig.trainSplit}
                                            onChange={e => setSplitConfig({ ...splitConfig, trainSplit: e.target.value })}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Subpasta Teste</label>
                                        <input
                                            type="text"
                                            className="w-full px-4 py-3 rounded-xl bg-background border border-border text-xs outline-none focus:ring-2 focus:ring-primary/40"
                                            value={splitConfig.testSplit}
                                            onChange={e => setSplitConfig({ ...splitConfig, testSplit: e.target.value })}
                                        />
                                    </div>
                                </div>
                            </div>

                            <button
                                onClick={handleSplit}
                                disabled={loading}
                                className={`w-full py-4 rounded-xl font-bold text-primary-foreground flex justify-center items-center gap-3 text-lg transition-all shadow-xl
                                    ${loading ? 'bg-muted cursor-not-allowed text-muted-foreground' : 'bg-primary hover:brightness-110 hover:scale-[1.01] shadow-primary/20'}
                                `}
                            >
                                {loading ? <RefreshCcw className="w-6 h-6 animate-spin" /> : <Split className="w-6 h-6" />}
                                {loading ? 'Dividindo...' : 'Gerar Novo Dataset'}
                            </button>
                        </div>
                    )}
                </div>

                {/* Terminal Column */}
                <div className="lg:col-span-12 xl:col-span-5 flex flex-col h-full bg-slate-950 rounded-2xl border border-border shadow-2xl overflow-hidden min-h-[400px]">
                    <div className="flex items-center justify-between px-5 py-4 bg-slate-900/50 border-b border-white/5">
                        <div className="flex items-center gap-3">
                            <TerminalIcon className="w-5 h-5 text-primary" />
                            <span className="text-xs font-mono font-bold text-slate-300">Console de Execução</span>
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
                        className="flex-1 p-5 font-mono text-xs overflow-y-auto space-y-1.5 scrollbar-thin scrollbar-thumb-white/10"
                    >
                        {logs.length === 0 && <div className="text-slate-700 italic flex items-center justify-center h-full">Pronto para iniciar tarefas...</div>}
                        {logs.map((log, i) => {
                            const isError = log.includes('[ERRO]');
                            const isInfo = log.includes('[INFO]');
                            const isTqdm = log.includes('|') && log.includes('%');

                            return (
                                <div key={i} className={`
                                    whitespace-pre-wrap break-all border-l-2 pl-3 py-0.5 transition-colors
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
                        <div className="bg-slate-900/80 px-5 py-3 border-t border-white/5 flex items-center justify-between">
                            <span className="text-[10px] text-primary font-bold animate-pulse tracking-widest uppercase">
                                Processando no Servidor
                            </span>
                            <div className="flex gap-1.5">
                                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <FileExplorerModal
                isOpen={explorerOpen}
                onClose={() => setExplorerOpen(false)}
                onSelect={(path) => {
                    setSplitConfig({ ...splitConfig, inputDir: path });
                    setExplorerOpen(false);
                }}
                initialPath={splitConfig.inputDir}
                rootPath={roots.reidentificacoes}
                title="Selecionar Pasta de Resultados (ReID)"
            />
        </div>
    );
}
