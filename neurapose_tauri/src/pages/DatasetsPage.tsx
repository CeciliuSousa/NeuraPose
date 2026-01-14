import { useState, useEffect, useRef } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    Scissors,
    FolderInput,
    FolderOutput,
    PieChart,
    Play,
    Terminal as TerminalIcon,
    RefreshCcw
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { shortenPath } from '../lib/utils';

export default function DatasetsPage() {
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const terminalRef = useRef<HTMLDivElement>(null);

    const [config, setConfig] = useState({
        inputDir: '',
        datasetName: 'meu-dataset',
        outputRoot: '',
        trainSplit: 'treino',
        testSplit: 'teste'
    });

    const [explorerTarget, setExplorerTarget] = useState<'input' | 'output' | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});


    // Load defaults
    useEffect(() => {
        const savedInput = localStorage.getItem('np_split_input');
        const savedOutput = localStorage.getItem('np_split_output');
        const savedName = localStorage.getItem('np_split_name');

        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
                setConfig(prev => ({
                    ...prev,
                    inputDir: savedInput || res.data.paths.reidentificacoes || '',
                    outputRoot: savedOutput || res.data.paths.datasets || '',
                    datasetName: savedName || 'meu-dataset'
                }));
            }
        });
    }, []);

    // Persist settings
    useEffect(() => {
        if (config.inputDir) localStorage.setItem('np_split_input', config.inputDir);
        if (config.outputRoot) localStorage.setItem('np_split_output', config.outputRoot);
        if (config.datasetName) localStorage.setItem('np_split_name', config.datasetName);
    }, [config]);

    // Polling logs
    useEffect(() => {
        let interval: any;
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

    const handleSplit = async () => {
        if (!config.inputDir || !config.datasetName) {
            alert("Preencha a pasta de entrada e o nome do dataset.");
            return;
        }
        setLoading(true);
        setLogs(prev => [...prev, `[INFO] Iniciando split do dataset "${config.datasetName}"...`]);
        try {
            await APIService.splitDataset({
                input_dir_process: config.inputDir,
                dataset_name: config.datasetName,
                output_root: config.outputRoot || undefined,
                train_split: config.trainSplit,
                test_split: config.testSplit
            });
        } catch (error: any) {
            setLoading(false);
            setLogs(prev => [...prev, `[ERRO] ${error.response?.data?.detail || error.message}`]);
        }
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Gerenciar Datasets"
                description="Divida seus dados em conjuntos de Treino e Teste para treinamento do modelo LSTM/TFT."
            />

            <div className="grid gap-6 lg:grid-cols-2">
                {/* Config Panel */}
                <div className="space-y-6">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <Scissors className="w-5 h-5 text-primary" />
                            Split Dataset
                        </h3>

                        <div className="space-y-4">
                            {/* Input Dir */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Pasta de Entrada (ReID com anotações)</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={shortenPath(config.inputDir)}
                                        title={config.inputDir}
                                        onChange={(e) => setConfig(prev => ({ ...prev, inputDir: e.target.value }))}
                                        className="flex-1 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                        placeholder="resultados-reidentificacoes/meus-videos"
                                    />
                                    <button
                                        onClick={() => setExplorerTarget('input')}
                                        className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                                    >
                                        <FolderInput className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>

                            {/* Dataset Name */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Nome do Dataset</label>
                                <input
                                    type="text"
                                    value={config.datasetName}
                                    onChange={(e) => setConfig(prev => ({ ...prev, datasetName: e.target.value }))}
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    placeholder="data-labex-v2"
                                />
                            </div>

                            {/* Output Root */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Pasta de Saída (Datasets)</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={shortenPath(config.outputRoot)}
                                        title={config.outputRoot}
                                        onChange={(e) => setConfig(prev => ({ ...prev, outputRoot: e.target.value }))}
                                        className="flex-1 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                        placeholder="datasets/"
                                    />
                                    <button
                                        onClick={() => setExplorerTarget('output')}
                                        className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                                    >
                                        <FolderOutput className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>

                            {/* Split Names */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Nome Treino</label>
                                    <input
                                        type="text"
                                        value={config.trainSplit}
                                        onChange={(e) => setConfig(prev => ({ ...prev, trainSplit: e.target.value }))}
                                        className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Nome Teste</label>
                                    <input
                                        type="text"
                                        value={config.testSplit}
                                        onChange={(e) => setConfig(prev => ({ ...prev, testSplit: e.target.value }))}
                                        className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                    />
                                </div>
                            </div>

                            <div className="pt-4">
                                <button
                                    onClick={handleSplit}
                                    disabled={loading || !config.inputDir}
                                    className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                >
                                    {loading ? (
                                        <>
                                            <RefreshCcw className="w-4 h-4 animate-spin" />
                                            Processando...
                                        </>
                                    ) : (
                                        <>
                                            <Play className="w-4 h-4" />
                                            Dividir Dataset
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Terminal / Stats Panel */}
                <div className="space-y-4">
                    <div className="rounded-xl border border-border bg-card overflow-hidden">
                        <div className="px-4 py-3 bg-muted/30 border-b border-border flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <TerminalIcon className="w-4 h-4 text-muted-foreground" />
                                <span className="font-medium text-sm">Logs</span>
                            </div>
                            <button
                                onClick={async () => {
                                    setLogs([]);
                                    try { await APIService.clearLogs(); } catch (e) { console.error(e); }
                                }}
                                className="text-[10px] uppercase font-bold text-muted-foreground hover:text-foreground transition-colors"
                            >
                                Limpar
                            </button>
                        </div>
                        <div
                            ref={terminalRef}
                            className="h-[400px] bg-black/90 p-4 font-mono text-xs text-green-400 overflow-y-auto"
                        >
                            {logs.length === 0 ? (
                                <div className="text-muted-foreground italic">Aguardando execução...</div>
                            ) : (
                                logs.map((log, i) => (
                                    <div key={i} className="whitespace-pre-wrap">{log}</div>
                                ))
                            )}
                        </div>
                    </div>

                    <div className="flex items-center justify-center p-8 border border-dashed border-border rounded-xl bg-muted/10">
                        <div className="text-center space-y-2">
                            <PieChart className="w-12 h-12 text-muted-foreground mx-auto" />
                            <p className="font-medium">Visualização de Distribuição</p>
                            <p className="text-xs text-muted-foreground">Execute o split para ver estatísticas nos logs.</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* File Explorer Modal */}
            <FileExplorerModal
                isOpen={explorerTarget !== null}
                onClose={() => setExplorerTarget(null)}
                onSelect={(path) => {
                    if (explorerTarget === 'input') setConfig({ ...config, inputDir: path });
                    if (explorerTarget === 'output') setConfig({ ...config, outputRoot: path });
                    setExplorerTarget(null);
                }}
                initialPath={explorerTarget === 'input' ? config.inputDir : config.outputRoot}
                rootPath={explorerTarget === 'input' ? roots.reidentificacoes : roots.datasets}
                title={explorerTarget === 'input' ? "Selecionar Pasta de ReID" : "Selecionar Pasta Raiz de Datasets"}
            />
        </div>
    );
}
