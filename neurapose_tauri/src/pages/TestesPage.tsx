import { useState, useEffect, useRef } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    Binary,
    FileCode,
    Database,
} from 'lucide-react';
import { APIService } from '../services/api';
import ws from '../services/websocket';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { Terminal } from '../components/ui/Terminal';
import { StatusMessage } from '../components/ui/StatusMessage';
import { VideoPreviewPanel } from '../components/ui/VideoPreviewPanel';
import { PreviewToggle } from '../components/ui/PreviewToggle';
import { useProcessingStatus } from '../hooks/useProcessingStatus';
import { PathSelector } from '../components/ui/PathSelector';
import { ConfigCard } from '../components/ui/ConfigCard';
import { DeviceSelector } from '../components/ui/DeviceSelector';
import { ProcessControls } from '../components/ui/ProcessControls';

export default function TestesPage() {
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'processing' } | null>(null);

    // Ref para evitar race condition (Grace Period)
    const processStartTimeRef = useRef<number>(0);

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
            const healthData = res.data as any;
            // Só mostra mensagem se for ESTE processo (test)
            if (healthData.processing && healthData.current_process === 'test') {
                setLoading(true);
                setMessage({ text: '⏳ Teste em andamento...', type: 'processing' });
                setPageStatus('test', 'processing');

                // Recarrega logs se rodando
                const savedLogs = localStorage.getItem('np_test_logs');
                if (savedLogs) setLogs(JSON.parse(savedLogs));
            } else {
                // Se NÃO está rodando, garante limpo para não mostrar "Pronto" falso
                setLogs([]);
                localStorage.removeItem('np_test_logs');
                setPageStatus('test', 'idle'); // Reset status visual se hook suportar, ou ignora
            }
        }).catch(() => { });

        APIService.getConfig().then(res => {
            const data = res.data as any; // Cast explicito
            if (data.status === 'success') {
                setRoots(data.paths);
            }
        });

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


    // Logs via WebSocket
    useEffect(() => {
        if (loading) {
            // Conecta ao WebSocket (Top-Level)
            ws.connectLogs('test');
            ws.connectStatus();

            const handleLogs = (data: any) => {
                // Suporte a payload antigo (array) ou novo (objeto)
                const newLogs = Array.isArray(data) ? data : (data.logs || []);
                const total = data.total || 0;

                setLogs((prev) => {
                    // Detecção de Full Sync
                    let currentLogs = (newLogs.length >= total && total > 0) ? [] : [...prev];

                    newLogs.forEach((log: string) => {
                        // Modo Lista: Remove \r 
                        const content = log.replace(/\r/g, '');
                        if (content.trim()) currentLogs.push(content);
                    });

                    // Limita o histórico de logs para evitar travar a UI (DOM Overload)
                    const MAX_LOGS = 300;
                    if (currentLogs.length > MAX_LOGS) {
                        return currentLogs.slice(currentLogs.length - MAX_LOGS);
                    }

                    // Persiste on-the-fly para F5
                    localStorage.setItem('np_test_logs', JSON.stringify(currentLogs));
                    return currentLogs;
                });
            };

            const handleStatus = (status: any) => {
                // GRACE PERIOD: Ignora "is_running: false" se o processo começou há menos de 3 seg
                const elapsed = Date.now() - processStartTimeRef.current;

                if (!status.is_running && loading) {
                    if (elapsed < 3000) return;

                    setLoading(false);
                    setMessage({ text: '✅ Teste concluído! Verifique os resultados.', type: 'success' });
                    setPageStatus('test', 'success');
                    ws.disconnectLogs();
                }
            };

            ws.events.on('logs', handleLogs);
            ws.events.on('status', handleStatus);

            return () => {
                ws.events.off('logs', handleLogs);
                ws.events.off('status', handleStatus);
            };
        }
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

        // LIMPEZA DE ESTADO ANTES DE INICIAR (CRÍTICO)
        setLogs([]);
        localStorage.removeItem('np_test_logs');
        try { await APIService.clearLogs('test'); } catch (e) { console.error(e); }

        processStartTimeRef.current = Date.now();

        setLoading(true);
        setMessage({ text: '⏳ Executando testes de validação...', type: 'processing' });
        setLogs([`[INFO] Iniciando validação de modelo...`]);
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
        <div className="space-y-6">
            <PageHeader
                title="Teste de Modelos"
                description="Execute testes de precisão em modelos treinados e valide o desempenho real."
                icon={Binary}
            />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Configuration Panel */}
                <div className="lg:col-span-12 xl:col-span-6 space-y-6">
                    <ConfigCard title="Setup de Teste" icon={Binary}>
                        <div className="space-y-6 mb-10">
                            {/* Model Selection */}
                            <div className="space-y-2">
                                <PathSelector
                                    label="Modelo Treinado (.pth / .pt)"
                                    value={config.modelPath}
                                    onSelect={() => setExplorerTarget('model')}
                                    placeholder="Selecione o modelo para teste..."
                                    icon={FileCode}
                                />
                            </div>

                            {/* Dataset Selection */}
                            <div className="space-y-2">
                                <PathSelector
                                    label="Dataset de Teste"
                                    value={config.datasetPath}
                                    onSelect={() => setExplorerTarget('dataset')}
                                    placeholder="Selecione o dataset para teste..."
                                    icon={Database}
                                />
                            </div>

                            <DeviceSelector
                                value={config.device}
                                onChange={(val) => setConfig({ ...config, device: val })}
                            />

                            {/* Preview Toggle */}
                            <PreviewToggle
                                checked={config.showPreview}
                                onChange={(value) => setConfig({ ...config, showPreview: value })}
                                isProcessing={loading}
                            />
                        </div>

                        <ProcessControls
                            isProcessing={loading}
                            canStart={!!(config.modelPath && config.datasetPath)}
                            onStart={handleRunTest}
                            onStop={handleStop}
                            labels={{ start: 'Iniciar Bateria de Testes', stop: 'Parar' }}
                            loadingText='Validando...'
                        />
                    </ConfigCard>
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
                        height="550px"
                        width="100%"
                        isLoading={loading}
                        onClear={async () => {
                            setLogs([]);
                            localStorage.removeItem('np_test_logs');
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
