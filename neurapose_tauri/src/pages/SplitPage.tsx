import { useState, useEffect } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    Scissors,
    FolderInput,
    PieChart,
    Play,
    RefreshCcw
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { Terminal } from '../components/ui/Terminal';
import { StatusMessage } from '../components/ui/StatusMessage';

// Opções de porcentagem para split
const SPLIT_OPTIONS = [
    { value: 70, label: '70% / 30%' },
    { value: 75, label: '75% / 25%' },
    { value: 80, label: '80% / 20%' },
    { value: 85, label: '85% / 15%' },
    { value: 90, label: '90% / 10%' },
];

export default function SplitPage() {
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'processing' } | null>(null);

    const [inputDir, setInputDir] = useState('');
    const [trainPercent, setTrainPercent] = useState(80);
    const [explorerOpen, setExplorerOpen] = useState(false);
    const [roots, setRoots] = useState<Record<string, string>>({});

    // Deriva o nome do dataset do caminho de entrada
    const datasetName = inputDir ? inputDir.replace(/\\/g, '/').split('/').pop() || '' : '';

    // Load defaults
    useEffect(() => {
        // const savedInput = localStorage.getItem('np_split_input'); // Removido para forçar placeholder
        const savedPercent = localStorage.getItem('np_split_percent');

        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
                setInputDir(''); // Força vazio para mostrar placeholder
                if (savedPercent) setTrainPercent(parseInt(savedPercent));
            }
        });
    }, []);

    // Persist settings
    useEffect(() => {
        if (inputDir) localStorage.setItem('np_split_input', inputDir);
        localStorage.setItem('np_split_percent', String(trainPercent));
    }, [inputDir, trainPercent]);

    // Polling logs
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (loading) {
            interval = setInterval(async () => {
                try {
                    const res = await APIService.getLogs('split');
                    setLogs(res.data.logs);

                    const health = await APIService.healthCheck();
                    if (!health.data.processing) {
                        setLoading(false);
                        setMessage({ text: '✅ Split concluído com sucesso!', type: 'success' });
                    }
                } catch (e) { console.error(e); }
            }, 1000);
        }
        return () => { if (interval) clearInterval(interval); };
    }, [loading]);

    const handleSplit = async () => {
        if (!inputDir) {
            alert("Selecione o diretório de entrada com os dados reidentificados.");
            return;
        }
        setLoading(true);
        setLogs(prev => [...prev, `[INFO] Iniciando split do dataset "${datasetName}" (${trainPercent}% treino / ${100 - trainPercent}% teste)...`]);
        try {
            await APIService.splitDataset({
                input_dir_process: inputDir,
                dataset_name: datasetName,
                train_split: 'treino',
                test_split: 'teste',
                train_ratio: trainPercent / 100  // Converte 85% para 0.85
            });
        } catch (error: any) {
            setLoading(false);
            setLogs(prev => [...prev, `[ERRO] ${error.response?.data?.detail || error.message}`]);
        }
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Split Dataset"
                description="Divida seus dados em conjuntos de Treino e Teste para treinamento do modelo."
            />

            <div className="grid gap-6 lg:grid-cols-2">
                {/* Config Panel */}
                <div className="space-y-6">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <Scissors className="w-5 h-5 text-primary" />
                            Configuração do Split
                        </h3>

                        <div className="space-y-5">
                            {/* Input Dir */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Diretório de Entrada (Dados Reidentificados)</label>
                                <div className="flex gap-2">
                                    <div className="flex-1 relative">
                                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                                            <Scissors className="w-4 h-4" />
                                        </div>
                                        <input
                                            type="text"
                                            value={datasetName}
                                            title={inputDir}
                                            readOnly
                                            className="w-full pl-9 pr-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm cursor-pointer truncate"
                                            placeholder="Selecione o diretório para split..."
                                            onClick={() => setExplorerOpen(true)}
                                        />
                                    </div>
                                    <button
                                        onClick={() => setExplorerOpen(true)}
                                        className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                                    >
                                        <FolderInput className="w-4 h-4" />
                                    </button>
                                </div>
                                {datasetName && (
                                    <p className="text-xs text-muted-foreground">
                                        Nome do dataset: <span className="font-mono text-primary">{datasetName}</span>
                                    </p>
                                )}
                            </div>

                            {/* Split Percentage Selector */}
                            <div className="space-y-3">
                                <label className="text-sm font-medium">Proporção do Split</label>
                                <div className="grid grid-cols-5 gap-2">
                                    {SPLIT_OPTIONS.map(opt => (
                                        <button
                                            key={opt.value}
                                            onClick={() => setTrainPercent(opt.value)}
                                            className={`
                                                py-3 px-2 rounded-lg text-xs font-semibold border-2 transition-all
                                                ${trainPercent === opt.value
                                                    ? 'bg-primary/20 border-primary text-primary'
                                                    : 'bg-secondary/30 border-border text-muted-foreground hover:border-primary/50'}
                                            `}
                                        >
                                            {opt.label}
                                        </button>
                                    ))}
                                </div>
                                <div className="flex items-center gap-4 mt-3">
                                    <div className="flex-1 space-y-1">
                                        <div className="flex justify-between text-xs">
                                            <span className="text-green-500 font-semibold">Treino</span>
                                            <span className="font-mono">{trainPercent}%</span>
                                        </div>
                                        <div className="h-2 bg-green-500/30 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-green-500 rounded-full transition-all"
                                                style={{ width: `${trainPercent}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div className="flex-1 space-y-1">
                                        <div className="flex justify-between text-xs">
                                            <span className="text-orange-500 font-semibold">Teste</span>
                                            <span className="font-mono">{100 - trainPercent}%</span>
                                        </div>
                                        <div className="h-2 bg-orange-500/30 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-orange-500 rounded-full transition-all"
                                                style={{ width: `${100 - trainPercent}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="pt-4">
                                <button
                                    onClick={handleSplit}
                                    disabled={loading || !inputDir}
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
                    {/* Status Message */}
                    {message && (
                        <StatusMessage
                            message={message.text}
                            type={message.type}
                            onClose={() => setMessage(null)}
                            autoCloseDelay={5000}
                        />
                    )}

                    {/* Terminal Component */}
                    <Terminal
                        logs={logs}
                        title="Console do Split"
                        height="400px"
                        isLoading={loading}
                        onClear={async () => {
                            setLogs([]);
                            try { await APIService.clearLogs('split'); } catch (e) { console.error(e); }
                        }}
                    />

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
                isOpen={explorerOpen}
                onClose={() => setExplorerOpen(false)}
                onSelect={(path) => {
                    setInputDir(path);
                    setExplorerOpen(false);
                }}
                initialPath={inputDir || roots.reidentificacoes}
                rootPath={roots.reidentificacoes}
                title="Selecionar Diretório de Dados Reidentificados"
            />
        </div>
    );
}
