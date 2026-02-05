import { useState, useEffect, useRef } from 'react';

import { Settings, Save, RotateCcw, ShieldAlert, Brain, Scan } from 'lucide-react';
import { PageHeader } from '../components/ui/PageHeader';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { PathSelector } from '../components/ui/PathSelector';

export default function ConfiguracaoPage() {
    const [config, setConfig] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState('');

    // File Explorer State
    const [explorerOpen, setExplorerOpen] = useState(false);
    const [activeKey, setActiveKey] = useState<string | null>(null);
    const [activeRootKey, setActiveRootKey] = useState<string | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});

    // Ref para evitar problemas de closure com roots
    const rootsRef = useRef<Record<string, string>>({});
    useEffect(() => { rootsRef.current = roots; }, [roots]);


    useEffect(() => {
        loadConfig();
        // console.log('[ConfigPage] Iniciando fetch de /config...');
        APIService.getConfig().then(res => {
            // console.log('[ConfigPage] API /config raw response:', res);
            const data = res.data as any;
            // Tenta pegar paths de diferentes formatos de resposta
            const paths = data?.paths || data;
            // console.log('[ConfigPage] Paths extraídos:', paths);
            if (paths && typeof paths === 'object') {
                // Garante que modelos_reid e modelos_pose existam mesmo se o backend não retornar
                const computedPaths = {
                    ...paths,
                    modelos_reid: paths.modelos_reid || (paths.root ? `${paths.root}\\tracker\\weights` : undefined),
                    modelos_pose: paths.modelos_pose || (paths.root ? `${paths.root}\\rtmpose\\modelos` : undefined)
                };
                // console.log('[ConfigPage] Setting roots com:', computedPaths);
                setRoots(computedPaths);
            }
        }).catch(err => console.error('[ConfigPage] Erro ao buscar config:', err));
    }, []);


    const loadConfig = async () => {
        setLoading(true);
        try {
            const res = await APIService.getAllConfig();
            setConfig(res.data);
        } catch (err) {
            console.error("Erro ao carregar config:", err);
            setMessage("Erro ao carregar configurações do backend.");
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        setMessage('');
        try {
            await APIService.updateConfig(config);
            setMessage("Configurações persistidas com sucesso!");
            setTimeout(() => setMessage(''), 3000);
        } catch (err) {
            console.error("Erro ao salvar:", err);
            setMessage("Erro ao salvar configurações.");
        } finally {
            setSaving(false);
        }
    };

    const openExplorer = (key: string, rootKey?: string) => {
        setActiveKey(key);
        setActiveRootKey(rootKey || null);
        setExplorerOpen(true);
    };

    const onPathSelect = (path: string) => {
        if (activeKey) {
            // Para modelos (OSNET/RTMPOSE), extrair path relativo ao root
            // Ex: C:\...\tracker\weights\osnet.pth -> osnet.pth
            // Ex: C:\...\rtmpose\modelos\rtmpose-m_simcc.../end2end.onnx -> rtmpose-m_simcc.../end2end.onnx
            let finalPath = path;
            if (activeRootKey && roots[activeRootKey]) {
                const rootNorm = roots[activeRootKey].replace(/\\/g, '/').replace(/\/$/, '');
                const pathNorm = path.replace(/\\/g, '/');
                if (pathNorm.startsWith(rootNorm)) {
                    // Remove o root path e a barra inicial
                    finalPath = pathNorm.slice(rootNorm.length).replace(/^\//, '');
                }
            }
            setConfig({ ...config, [activeKey]: finalPath });
        }
        setExplorerOpen(false);
        setActiveKey(null);
        setActiveRootKey(null);
    };

    const handleReset = async () => {
        if (!confirm("Isso irá restaurar as configurações originais e excluir seu perfil personalizado. Continuar?")) return;
        setSaving(true);
        try {
            await APIService.resetConfig();
            await loadConfig();
            setMessage("Configurações resetadas com sucesso!");
        } catch (err) {
            console.error("Erro ao resetar:", err);
            setMessage("Erro ao resetar configurações.");
        } finally {
            setSaving(false);
        }
    };

    const handleShutdown = async () => {
        if (!confirm("Isso irá encerrar o servidor e todos os processos. Você precisará reiniciar manualmente o backend. Continuar?")) return;
        try {
            await APIService.shutdown();
            setMessage("Servidor encerrando... Você pode fechar esta aba.");
        } catch (err) {
            console.error("Erro ao desligar:", err);
            setMessage("Erro ao desligar servidor.");
        }
    };

    if (loading) return <div className="flex items-center justify-center h-64 text-primary animate-pulse font-medium text-lg">Carregando configurações...</div>;
    if (!config) return <div className="p-8 text-red-500 font-bold border border-red-500/20 bg-red-500/5 rounded-xl">Erro ao conectar com o backend. Além disso, verifique se o servidor está rodando.</div>;

    const sections = [
        {
            title: "Modelos de IA",
            description: "Modelos usados para detecção, pose e re-identificação.",
            items: [
                { key: "YOLO_MODEL", label: "Modelo YOLO (Detecção)", type: "select", options: ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"] },
                { key: "TRACKER_NAME", label: "Rastreador Ativo", type: "select", options: ["BoTSORT", "DeepOCSORT"] },
                { key: "OSNET_MODEL", label: "Modelo OSNet (Re-ID)", type: "path", rootKey: "modelos_reid" },
                { key: "RTMPOSE_MODEL", label: "Modelo RTMPose (Pose)", type: "path", rootKey: "modelos_pose" },
                { key: "RTMPOSE_INPUT_SIZE", label: "Resol. Entrada RTMPose", type: "select", options: ["256x192", "384x288"] },
                { key: "TEMPORAL_MODEL", label: "Modelo Temporal", type: "select", options: ["lstm", "robust", "pooled", "bilstm", "attention", "tcn", "transformer", "tft", "wavenet"] },
            ]
        },
        {
            title: "Classes de Detecção",
            description: "Nomes das classes para classificação de comportamento.",
            items: [
                { key: "CLASSE1", label: "Classe Negativa (Normal)", type: "text" },
                { key: "CLASSE2", label: "Classe Positiva (Anomalia)", type: "text" },
                { key: "CLASSE2_THRESHOLD", label: "Threshold da Classe Positiva", type: "number", step: 0.01 },
            ]
        },
        {
            title: "Configurações YOLO",
            description: "Parâmetros de detecção de pessoas.",
            items: [
                { key: "DETECTION_CONF", label: "Confiança YOLO (0-1)", type: "number", step: 0.01 },
                { key: "YOLO_IMGSZ", label: "Resolução de Entrada", type: "select", options: ["640", "1280", "1920"] },
                { key: "YOLO_BATCH_SIZE", label: "Batch Size (GPU)", type: "number" },
                { key: "YOLO_SKIP_FRAME_INTERVAL", label: "Intervalo Skip-Frame YOLO (1-3)", type: "select", options: ["1", "2", "3"] },
            ]
        },
        {
            title: "Configurações de Pose",
            description: "Parâmetros de extração de keypoints.",
            items: [
                { key: "POSE_CONF_MIN", label: "Confiança Mínima Keypoint", type: "number", step: 0.01 },
                { key: "EMA_ALPHA", label: "Suavização EMA (Alpha)", type: "number", step: 0.01 },
                { key: "EMA_MIN_CONF", label: "Conf. Mínima para EMA", type: "number", step: 0.01 },
                { key: "RTMPOSE_MAX_BATCH_SIZE", label: "Batch Pose (GPU)", type: "number" },
            ]
        },

        {
            title: `Rastreador (${config.TRACKER_NAME || 'BoTSORT'})`,
            description: "Parâmetros específicos do algoritmo de rastreamento selecionado.",
            items: config.TRACKER_NAME === 'DeepOCSORT' ? [
                // DEEPOCSORT PARAMS
                { key: "det_thresh", label: "Det. Threshold (Conf)", type: "number", step: 0.05 },
                { key: "max_age", label: "Max Age (Frames)", type: "number" },
                { key: "min_hits", label: "Min Hits (Início)", type: "number" },
                { key: "iou_thresh", label: "IoU Threshold", type: "number", step: 0.05 },
                { key: "delta_t", label: "Delta T (Velocidade)", type: "number" },
                { key: "asso_func", label: "Função Associação", type: "select", options: ["iou", "giou", "ciou", "diou", "ct_dist"] },
                { key: "inertia", label: "Inertia (Suavização)", type: "number", step: 0.1 },
                { key: "w_association_emb", label: "Peso ReID (Embedding)", type: "number", step: 0.05 },
                { key: "device", label: "Device", type: "select", options: ["cuda", "cpu"] }
            ] : [
                // BOTSORT PARAMS (Default)
                { key: "track_high_thresh", label: "Track High Thresh", type: "number", step: 0.05 },
                { key: "track_low_thresh", label: "Track Low Thresh", type: "number", step: 0.05 },
                { key: "new_track_thresh", label: "New Track Thresh", type: "number", step: 0.05 },
                { key: "match_thresh", label: "Match (ReID) Thresh", type: "number", step: 0.05 },
                { key: "track_buffer", label: "Track Buffer (Frames)", type: "number" },
                { key: "proximity_thresh", label: "Proximity Thresh", type: "number", step: 0.05 },
                { key: "appearance_thresh", label: "Appearance Thresh", type: "number", step: 0.05 },
                { key: "gmc_method", label: "GMC Method", type: "select", options: ["none", "orb", "sift", "sparseOptFlow"] },
                { key: "fuse_score", label: "Fuse Score (Fusão)", type: "boolean" },
                { key: "with_reid", label: "Ativar ReID (Visual)", type: "boolean" },
                { key: "device", label: "Device", type: "select", options: ["cuda", "cpu"] }
            ]
        },
        {
            title: "Parâmetros de Treinamento",
            description: "Hiperparâmetros usados no treinamento do modelo temporal.",
            items: [
                { key: "TIME_STEPS", label: "Janela Temporal (Frames)", type: "number" },
                { key: "BATCH_SIZE", label: "Tamanho do Lote (Batch)", type: "number" },
                { key: "LEARNING_RATE", label: "Taxa de Aprendizado (LR)", type: "number", step: 0.0001 },
                { key: "EPOCHS", label: "Épocas Padrão", type: "number" },
                { key: "LSTM_DROPOUT", label: "Dropout (Regularização)", type: "number", step: 0.05 },
                { key: "LSTM_HIDDEN_SIZE", label: "Tam. Camada Oculta (Neurônios)", type: "number" },
                { key: "LSTM_NUM_LAYERS", label: "Num. Camadas (Layers)", type: "number" },
                { key: "LSTM_NUM_HEADS", label: "Num. Cabeças (Attention/TFT)", type: "number" },
                { key: "LSTM_KERNEL_SIZE", label: "Kernel Size (TCN)", type: "number" },
            ]
        },
        {
            title: "Parâmetros de Sequência",
            description: "Configurações de frames para processamento de vídeos.",
            items: [
                { key: "MAX_FRAMES_PER_SEQUENCE", label: "Máx. Frames por Sequência", type: "number" },
                { key: "MIN_FRAMES_PER_ID", label: "Mín. Frames por ID", type: "number" },
            ]
        },
        {
            title: "Hardware & Performance",
            description: "Otimizações de GPU e CPU para acelerar o processamento.",
            items: [
                { key: "USE_ASYNC_LOADER", label: "Leitura Assíncrona (Threaded)", type: "boolean" },
                { key: "ASYNC_BUFFER_SIZE", label: "Tamanho Buffer Leitura", type: "number" },
                { key: "USE_TENSORRT", label: "Aceleração TensorRT (.engine)", type: "boolean" },
                { key: "USE_NVENC", label: "NVENC (Decodificação GPU)", type: "boolean" },
                { key: "NVENC_PRESET", label: "Preset NVENC", type: "select", options: ["p1", "p4", "p7"] },
                { key: "USE_FP16", label: "Precisão FP16", type: "boolean" },
            ]
        },
        {
            title: "Filtros Pós-Processamento",
            description: "Filtros para reduzir ruído e falsos positivos.",
            items: [
                { key: "MIN_POSDETECTION_CONF", label: "Conf. Mínima Pós-Detecção", type: "number", step: 0.05 },
                { key: "MIN_POSE_ACTIVITY", label: "Mín. Atividade Pose", type: "number", step: 0.05 },
                { key: "MIN_MEMBER_ACTIVITY", label: "Mín. Variância (Pixels)", type: "number", step: 0.5 },
            ]
        },
    ];


    return (
        <div className="space-y-6">
            <PageHeader
                title="Configurações"
                description="Gestão de perfis e hiperparâmetros do NeuraPose."
                icon={Settings}
            >
                <div className="flex flex-wrap gap-3">
                    <button
                        onClick={handleShutdown}
                        className="flex items-center gap-2 px-5 py-2.5 text-sm bg-red-600/10 text-red-500 hover:bg-red-600 hover:text-white rounded-xl transition-all font-bold border border-red-500/20"
                    >
                        <ShieldAlert className="w-4 h-4" />
                        Desligar
                    </button>
                    <button
                        onClick={handleReset}
                        className="flex items-center gap-2 px-5 py-2.5 text-sm bg-secondary/50 hover:bg-secondary rounded-xl transition-colors font-bold border border-border"
                    >
                        <RotateCcw className="w-4 h-4" />
                        Resetar Perfil
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className={`
                            flex items-center gap-2 px-8 py-2.5 text-sm rounded-xl transition-all font-bold disabled:opacity-50 shadow-xl
                            ${saving ? 'bg-primary/50' : 'bg-primary text-primary-foreground hover:scale-105 active:scale-95 shadow-primary/20'}
                        `}
                    >
                        <Save className="w-4 h-4" />
                        {saving ? 'Persistindo...' : 'Salvar Perfil'}
                    </button>
                </div>
            </PageHeader>

            {message && (
                <div className={`p-4 rounded-xl text-sm flex items-center gap-3 animate-in fade-in slide-in-from-top-2 duration-300 ${message.includes('Erro') ? 'bg-red-500/10 text-red-500 border border-red-500/20' : 'bg-green-500/10 text-green-500 border border-green-500/20'}`}>
                    <ShieldAlert className={`w-5 h-5 ${message.includes('Erro') ? '' : 'hidden'}`} />
                    <span className="font-semibold">{message}</span>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {sections.map((section, sidx) => (
                    <div key={sidx} className="bg-card border border-border rounded-2xl overflow-hidden shadow-md flex flex-col">
                        <div className="px-6 py-5 bg-muted/20 border-b border-border">
                            <h2 className="font-bold text-xl tracking-wide">{section.title}</h2>
                            {section.description && (
                                <p className="text-xs text-muted-foreground mt-1">{section.description}</p>
                            )}
                        </div>
                        <div className="p-6 space-y-6 flex-1">

                            {section.items.map((item) => (
                                <div key={item.key} className="space-y-2.5">
                                    <div className="flex justify-between items-center">
                                        <label className="text-sm font-bold text-muted-foreground uppercase tracking-wider">{item.label}</label>
                                    </div>

                                    {item.type === 'path' ? (
                                        <PathSelector
                                            value={config[item.key] || ''}
                                            onSelect={() => openExplorer(item.key, (item as any).rootKey)}
                                            placeholder="Clique para selecionar..."
                                            icon={item.key.includes('OSNET') ? Brain : Scan}
                                        />
                                    ) : item.type === 'select' ? (
                                        <select
                                            value={config[item.key]}
                                            onChange={(e) => setConfig({ ...config, [item.key]: e.target.value })}
                                            className="w-full bg-background border border-border rounded-xl px-4 py-2.5 outline-none focus:ring-2 focus:ring-primary/40 transition-all text-sm font-medium appearance-none"
                                        >
                                            {(item as any).options?.map((opt: string) => <option key={opt} value={opt}>{opt.toUpperCase()}</option>)}
                                        </select>
                                    ) : item.type === 'boolean' ? (
                                        <select
                                            value={String(config[item.key])}
                                            onChange={(e) => setConfig({ ...config, [item.key]: e.target.value === 'true' })}
                                            className="w-full bg-background border border-border rounded-xl px-4 py-2.5 outline-none focus:ring-2 focus:ring-primary/40 transition-all text-sm font-medium appearance-none"
                                        >
                                            <option value="true">ATIVADO</option>
                                            <option value="false">DESATIVADO</option>
                                        </select>
                                    ) : item.type === 'range' ? (
                                        <div className="flex items-center gap-3">
                                            <input
                                                type="range"
                                                min={(item as any).min}
                                                max={(item as any).max}
                                                step={(item as any).step}
                                                value={config[item.key]}
                                                onChange={(e) => setConfig({ ...config, [item.key]: parseInt(e.target.value) })}
                                                className="flex-1 h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                                                title={(item as any).tooltip}
                                            />
                                            <span className="font-bold text-sm w-8 text-center">{config[item.key]}</span>
                                        </div>
                                    ) : (
                                        <input
                                            type={item.type}
                                            step={(item as any).step}
                                            value={config[item.key]}
                                            onChange={(e) => setConfig({
                                                ...config,
                                                [item.key]: item.type === 'number' ? parseFloat(e.target.value) : e.target.value
                                            })}
                                            className="w-full bg-background border border-border rounded-xl px-4 py-2.5 outline-none focus:ring-2 focus:ring-primary/40 transition-all text-sm font-medium"
                                        />
                                    )}

                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <FileExplorerModal
                isOpen={explorerOpen}
                onClose={() => setExplorerOpen(false)}
                onSelect={onPathSelect}
                initialPath={activeRootKey ? roots[activeRootKey] : (activeKey ? config[activeKey] : '')}
                rootPath={activeRootKey ? roots[activeRootKey] : roots.root}
                title={activeRootKey === 'modelos_reid' ? 'Selecionar Modelo OSNet' : (activeRootKey === 'modelos_pose' ? 'Selecionar Modelo RTMPose' : 'Selecionar Arquivo')}
            />

            <div className="bg-primary/5 border border-primary/10 p-5 rounded-2xl text-xs text-muted-foreground leading-relaxed flex items-start gap-4">
                <ShieldAlert className="w-8 h-8 text-primary/40 shrink-0" />
                <div>
                    <strong>Dica de Segurança:</strong> As alterações feitas aqui são persistidas em seu perfil de usuário (`user_settings.json`).
                    Alguns parâmetros como modelos YOLO ou OSNet podem exigir reinício silencioso do detector em background.
                    Diretórios de sistema e arquivos de código estão ocultos por segurança.
                </div>
            </div>
        </div>
    );
}
