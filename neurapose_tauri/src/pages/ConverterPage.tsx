import { useState, useEffect } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    FileOutput,
    FolderInput,
    Play,
    RefreshCcw
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { Terminal } from '../components/ui/Terminal';
import { StatusMessage } from '../components/ui/StatusMessage';

// Extensões disponíveis para conversão
const EXTENSION_OPTIONS = [
    { value: '.pt', label: 'PyTorch (.pt) - Recomendado' },
    { value: '.pth', label: 'PyTorch Legacy (.pth)' },
];

export default function ConverterPage() {
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'processing' } | null>(null);

    const [datasetPath, setDatasetPath] = useState('');
    const [extension, setExtension] = useState('.pt');
    const [explorerOpen, setExplorerOpen] = useState(false);
    const [roots, setRoots] = useState<Record<string, string>>({});

    // Nome do dataset derivado do caminho
    const datasetName = datasetPath ? datasetPath.replace(/\\/g, '/').split('/').pop() || '' : '';

    // Load config
    useEffect(() => {
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
            }
        });
    }, []);

    // Polling logs durante conversão
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
                        setMessage({ text: '✅ Conversão concluída! Dataset pronto para treinamento.', type: 'success' });
                    }
                } catch (e) { console.error(e); }
            }, 1000);
        }
        return () => { if (interval) clearInterval(interval); };
    }, [loading]);

    const handleConvert = async () => {
        if (!datasetPath) {
            setMessage({ text: 'Por favor, selecione um dataset para converter.', type: 'error' });
            return;
        }

        setLoading(true);
        setMessage({ text: '⏳ Convertendo JSONs para formato PyTorch...', type: 'processing' });
        setLogs(prev => [...prev, `[INFO] Iniciando conversão do dataset "${datasetName}"...`]);

        try {
            await APIService.convertDataset({
                dataset_path: datasetPath,
                extension: extension
            });
        } catch (error: any) {
            setLoading(false);
            const errMsg = error.response?.data?.detail || error.message;
            setMessage({ text: `❌ Erro: ${errMsg}`, type: 'error' });
            setLogs(prev => [...prev, `[ERRO] ${errMsg}`]);
        }
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Converter para .pt"
                description="Converta JSONs de anotações para formato PyTorch (.pt) para treinamento do modelo LSTM."
            />

            <div className="grid gap-6 lg:grid-cols-2">
                {/* Config Panel */}
                <div className="space-y-6">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <FileOutput className="w-5 h-5 text-primary" />
                            Configuração da Conversão
                        </h3>

                        <div className="space-y-5">
                            {/* Dataset Selection */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Dataset de Entrada</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={datasetName}
                                        title={datasetPath}
                                        readOnly
                                        className="flex-1 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm cursor-pointer"
                                        placeholder="Selecione uma pasta para conversão..."
                                        onClick={() => setExplorerOpen(true)}
                                    />
                                    <button
                                        onClick={() => setExplorerOpen(true)}
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

                            {/* Extension Selection */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Formato de Saída</label>
                                <select
                                    value={extension}
                                    onChange={(e) => setExtension(e.target.value)}
                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm"
                                >
                                    {EXTENSION_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>
                                            {opt.label}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Info Box */}
                            <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-blue-400 mb-2">
                                    O que será gerado
                                </h4>
                                <ul className="text-sm text-blue-300 space-y-1">
                                    <li>• <code className="bg-blue-500/20 px-1 rounded">{datasetName || '<dataset>'}/treino/data/data{extension}</code></li>
                                    <li>• Arquivo de log com frames inválidos</li>
                                    <li>• Debug log com detalhes da conversão</li>
                                </ul>
                            </div>

                            {/* Convert Button */}
                            <div className="pt-4">
                                <button
                                    onClick={handleConvert}
                                    disabled={loading || !datasetPath}
                                    className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                >
                                    {loading ? (
                                        <>
                                            <RefreshCcw className="w-4 h-4 animate-spin" />
                                            Convertendo...
                                        </>
                                    ) : (
                                        <>
                                            <Play className="w-4 h-4" />
                                            Converter Dataset
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Terminal Panel */}
                <div className="space-y-4">
                    {/* Status Message */}
                    {message && (
                        <StatusMessage
                            message={message.text}
                            type={message.type}
                            onClose={() => setMessage(null)}
                            autoCloseDelay={message.type === 'success' ? 5000 : undefined}
                        />
                    )}

                    {/* Terminal */}
                    <Terminal
                        logs={logs}
                        title="Console de Conversão"
                        height="450px"
                        isLoading={loading}
                        onClear={async () => {
                            setLogs([]);
                            try { await APIService.clearLogs(); } catch (e) { console.error(e); }
                        }}
                    />
                </div>
            </div>

            {/* File Explorer Modal */}
            <FileExplorerModal
                isOpen={explorerOpen}
                onClose={() => setExplorerOpen(false)}
                onSelect={(path) => {
                    setDatasetPath(path);
                    setExplorerOpen(false);
                }}
                initialPath={datasetPath || roots.datasets}
                rootPath={roots.datasets}
                title="Selecionar Dataset para Conversão"
            />
        </div>
    );
}
