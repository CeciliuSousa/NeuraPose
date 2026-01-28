import { useState, useEffect } from 'react';
import {
    Video
} from 'lucide-react';
import { APIService } from '../services/api';
import ws from '../services/websocket';

import { FileExplorerModal } from '../components/FileExplorerModal';
import { Terminal } from '../components/ui/Terminal';
import { VideoPreviewPanel } from '../components/ui/VideoPreviewPanel';
import { PreviewToggle } from '../components/ui/PreviewToggle';
import { PathSelector } from '../components/ui/PathSelector';
import { ConfigCard } from '../components/ui/ConfigCard';
import { DeviceSelector } from '../components/ui/DeviceSelector';
import { ProcessControls } from '../components/ui/ProcessControls';
import { PageHeader } from '../components/ui/PageHeader';

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


    // Carregamento inicial de caminhos - apenas UMA VEZ na montagem
    useEffect(() => {
        // Limpa logs e estado residual ao entrar na página para garantir estado fresco
        // Se quisermos persistência, deveria ser apenas se 'loading' fosse true.
        // Como o usuário reportou confusão, melhor limpar sempre que recarregar.
        setLogs([]);
        localStorage.removeItem('np_process_logs');
        localStorage.removeItem('np_process_loading');

        APIService.getConfig().then(res => {
            const data = res.data as any;
            if (data.status === 'success') {
                setRoots(data.paths);
                setConfig(prev => ({
                    ...prev,
                    inputPath: ''
                }));
            }
        }).catch(err => console.error("Erro ao carregar caminhos do backend:", err));
    }, []);

    // Logs via WebSocket & Auto-Stop
    useEffect(() => {
        // Restaurar estado
        const savedConfig = localStorage.getItem('np_process_config');
        if (savedConfig && !config.inputPath) {
            try {
                const parsed = JSON.parse(savedConfig);
                setConfig((prev: any) => ({ ...prev, ...parsed, inputPath: prev.inputPath }));
            } catch (e) { console.error(e); }
        }

        const savedLoading = localStorage.getItem('np_process_loading');
        if (savedLoading === 'true' && !loading) setLoading(true);

        if (loading) {
            localStorage.setItem('np_process_loading', 'true');

            // Conecta ao WebSocket (Usa importação do topo)
            ws.connectLogs('process');
            ws.connectStatus();

            // Listener de Logs
            const handleLogs = (data: any) => {
                // Suporte a payload antigo (array) ou novo (objeto com metadata)
                const newLogs = Array.isArray(data) ? data : (data.logs || []);
                const total = data.total || 0;

                setLogs((prev) => {
                    // Detecção de Full Sync (evita duplicidade) - Reset se receber histórico completo
                    let currentLogs = (newLogs.length >= total && total > 0) ? [] : [...prev];

                    newLogs.forEach((log: string) => {
                        // Modo Lista: Remove \r e sempre adiciona nova linha (Comportamento solicitado)
                        const content = log.replace(/\r/g, '');
                        if (content.trim()) {
                            currentLogs.push(content);
                        }
                    });
                    return currentLogs;
                });
            };

            // Listener de Status
            const handleStatus = (status: any) => {
                setIsPaused(status.is_paused);

                if (!status.is_running && loading) {
                    setLoading(false);
                    localStorage.setItem('np_process_loading', 'false');
                    ws.disconnectLogs();
                }
            };

            ws.events.on('logs', handleLogs);
            ws.events.on('status', handleStatus);

            // Cleanup function
            return () => {
                ws.events.off('logs', handleLogs);
                ws.events.off('status', handleStatus);
            };

        } else {
            localStorage.setItem('np_process_loading', 'false');
        }
    }, [loading]);

    useEffect(() => {
        localStorage.setItem('np_process_config', JSON.stringify(config));
    }, [config]);

    const handleProcess = async () => {
        if (!config.inputPath) {
            alert("Por favor, selecione o diretório de entrada.");
            return;
        }

        // Limpeza MANDATÓRIA antes de iniciar novo processo
        setLogs([`[INFO] Iniciando novo processamento...`]);
        localStorage.removeItem('np_process_logs');
        localStorage.removeItem('np_process_loading');

        setLoading(true);

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
            <PageHeader
                title="Processamento de Vídeo"
                description="Analise vídeos para detectar furtos e anomalias de comportamento em tempo real."
                icon={Video}
            />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left: Configuration */}
                <div className="space-y-6">
                    <ConfigCard title="Configuração de Diretórios">
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

                            <DeviceSelector
                                value={config.device}
                                onChange={(val) => setConfig({ ...config, device: val })}
                            />

                            <div className="space-y-2">
                                <PreviewToggle
                                    checked={config.showPreview}
                                    onChange={(value) => setConfig({ ...config, showPreview: value })}
                                    isProcessing={loading}
                                />
                            </div>

                            <ProcessControls
                                isProcessing={loading}
                                isPaused={isPaused}
                                onStart={handleProcess}
                                onStop={handleStop}
                                onPause={togglePause}
                                canStart={!!config.inputPath}
                                labels={{ start: 'Iniciar Processamento' }}
                            />
                        </div>
                    </ConfigCard>

                    {/* Preview Video */}
                    <VideoPreviewPanel
                        isVisible={loading && config.showPreview}
                        title="Processamento em tempo real"
                    />
                </div>

                {/* Right: Terminal Output */}
                <Terminal
                    logs={logs}
                    title="Console de processamentos"
                    height="550px"
                    width="100%"
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
                showExternalPicker={true}
                title="Selecionar Diretório de Entrada (Vídeos)"
            />
        </div>
    );
}
