import { useState, useEffect } from 'react';

import { Settings, Save, RotateCcw, ShieldAlert, FolderOpen } from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { shortenPath } from '../lib/utils';

export default function ConfiguracaoPage() {
    const [config, setConfig] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState('');

    // File Explorer State
    const [explorerOpen, setExplorerOpen] = useState(false);
    const [activeKey, setActiveKey] = useState<string | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});


    useEffect(() => {
        loadConfig();
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
            }
        });
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

    const openExplorer = (key: string) => {
        setActiveKey(key);
        setExplorerOpen(true);
    };

    const onPathSelect = (path: string) => {
        if (activeKey) {
            setConfig({ ...config, [activeKey]: path });
        }
        setExplorerOpen(false);
        setActiveKey(null);
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
                { key: "YOLO_MODEL", label: "Modelo YOLO (Detecção)", type: "select", options: ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"] },
                { key: "OSNET_MODEL", label: "Modelo OSNet (Re-ID)", type: "path" },
                { key: "RTMPOSE_MODEL", label: "Modelo RTMPose (Pose)", type: "path" },
                { key: "RTMPOSE_INPUT_SIZE", label: "Resol. Entrada RTMPose", type: "select", options: ["256x192", "384x288"] },
                { key: "TEMPORAL_MODEL", label: "Modelo Temporal", type: "select", options: ["tft", "lstm"] },
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
            ]
        },
        {
            title: "Configurações de Pose",
            description: "Parâmetros de extração de keypoints.",
            items: [
                { key: "POSE_CONF_MIN", label: "Confiança Mínima Keypoint", type: "number", step: 0.01 },
                { key: "EMA_ALPHA", label: "Suavização EMA (Alpha)", type: "number", step: 0.01 },
                { key: "EMA_MIN_CONF", label: "Conf. Mínima para EMA", type: "number", step: 0.01 },
            ]
        },
        {
            title: "Rastreador (BoTSORT)",
            description: "Parâmetros do tracker para manter IDs consistentes.",
            items: [
                { key: "track_high_thresh", label: "Track High Thresh", type: "number", step: 0.05 },
                { key: "track_low_thresh", label: "Track Low Thresh", type: "number", step: 0.05 },
                { key: "new_track_thresh", label: "New Track Thresh", type: "number", step: 0.05 },
                { key: "match_thresh", label: "Match (ReID) Thresh", type: "number", step: 0.05 },
                { key: "track_buffer", label: "Track Buffer (Frames)", type: "number" },
                { key: "proximity_thresh", label: "Proximity Thresh", type: "number", step: 0.05 },
                { key: "appearance_thresh", label: "Appearance Thresh", type: "number", step: 0.05 },
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
            ]
        },
        {
            title: "Parâmetros de Sequência",
            description: "Configurações de frames para processamento de vídeos.",
            items: [
                { key: "MAX_FRAMES_PER_SEQUENCE", label: "Máx. Frames por Sequência", type: "number" },
                { key: "MIN_FRAMES_PER_ID", label: "Mín. Frames por ID", type: "number" },
            ]
        }
    ];


    return (
        <div className="space-y-6 max-w-6xl mx-auto pb-20 px-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-border pb-6 gap-4">
                <div className="flex items-center gap-4">
                    <div className="p-4 bg-primary/10 rounded-2xl shadow-inner">
                        <Settings className="w-10 h-10 text-primary" />
                    </div>
                    <div>
                        <h1 className="text-4xl font-extrabold tracking-tight">Configurações</h1>
                        <p className="text-muted-foreground italic text-lg">Gestão de perfis e hiperparâmetros do NeuraPose.</p>
                    </div>
                </div>
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
            </div>

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
                                        <div className="flex gap-2">
                                            <input
                                                type="text"
                                                value={shortenPath(config[item.key] || '')}
                                                title={config[item.key] || ''}
                                                onChange={(e) => setConfig({ ...config, [item.key]: e.target.value })}
                                                className="flex-1 bg-background border border-border rounded-xl px-4 py-2.5 outline-none focus:ring-2 focus:ring-primary/40 transition-all text-xs font-mono"
                                            />
                                            <button
                                                onClick={() => openExplorer(item.key)}
                                                className="p-2.5 bg-secondary hover:bg-primary/20 hover:text-primary rounded-xl transition-all border border-border"
                                                title="Procurar arquivo/diretório"
                                            >
                                                <FolderOpen className="w-5 h-5" />
                                            </button>
                                        </div>
                                    ) : item.type === 'select' ? (
                                        <select
                                            value={config[item.key]}
                                            onChange={(e) => setConfig({ ...config, [item.key]: e.target.value })}
                                            className="w-full bg-background border border-border rounded-xl px-4 py-2.5 outline-none focus:ring-2 focus:ring-primary/40 transition-all text-sm font-medium appearance-none"
                                        >
                                            {(item as any).options?.map((opt: string) => <option key={opt} value={opt}>{opt.toUpperCase()}</option>)}
                                        </select>
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
                initialPath={activeKey ? config[activeKey] : ''}
                rootPath={roots.root}
                title="Selecionar Arquivo ou Diretório"
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
