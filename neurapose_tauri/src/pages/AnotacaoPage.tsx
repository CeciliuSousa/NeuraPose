import { useState, useEffect, useMemo } from 'react';
import {
    Tag,
    Video as VideoIcon,
    Save,
    RefreshCcw,
    Clock,
    CheckCircle2,
    Pencil,
    Plus,
    Trash2,
    Camera,
    AlertTriangle
} from 'lucide-react';
import { PageHeader } from '../components/ui/PageHeader';
import { APIService } from '../services/api';

import { FileExplorerModal } from '../components/FileExplorerModal';
import { VideoPlayer } from '../components/ui/VideoPlayer';
import { FilterDropdown, FilterOption } from '../components/ui/FilterDropdown';
import { StatusMessage } from '../components/ui/StatusMessage';
import { TerminalModal } from '../components/ui/TerminalModal';
import { PathSelector } from '../components/ui/PathSelector';

interface VideoItem {
    video_id: string;
    video_name: string;
    status: 'anotado' | 'pendente';
    has_json: boolean;
    creation_time: number;
}

export default function AnotacaoPage() {
    // Vídeos e seleção
    const [videos, setVideos] = useState<VideoItem[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<VideoItem | null>(null);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState('');

    // Dados do vídeo selecionado
    const [videoIds, setVideoIds] = useState<{ id: number; frames: number }[]>([]);
    const [annotations, setAnnotations] = useState<Record<string, string>>({});

    // Temporal Mode State
    const [temporalMode, setTemporalMode] = useState(false);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [idIntervals, setIdIntervals] = useState<Record<string, Array<[number, number]>>>({});

    // Editor State (para input manual)
    const [startInput, setStartInput] = useState<string>('');
    const [endInput, setEndInput] = useState<string>('');
    const [activeIdForEdit, setActiveIdForEdit] = useState<string | null>(null);

    // Config
    const [inputPath, setInputPath] = useState('');
    const [roots, setRoots] = useState<Record<string, string>>({});
    const [classe1, setClasse1] = useState('NORMAL');
    const [classe2, setClasse2] = useState('FURTO');

    // Filtro e UI
    const [filterStatus, setFilterStatus] = useState<'all' | 'pending' | 'annotated'>('all');
    const [explorerOpen, setExplorerOpen] = useState(false);

    // Terminal Modal
    const [terminalOpen, setTerminalOpen] = useState(false);
    const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
    const [terminalProcessing, setTerminalProcessing] = useState(false);

    // Stats
    const stats = useMemo(() => {
        let pending = 0;
        let annotated = 0;
        videos.forEach(v => {
            if (v.status === 'anotado') annotated++;
            else pending++;
        });
        return { pending, annotated, total: videos.length };
    }, [videos]);

    // Vídeos filtrados
    const displayVideos = useMemo(() => {
        if (filterStatus === 'all') return videos;
        return videos.filter(v => {
            if (filterStatus === 'pending') return v.status === 'pendente';
            if (filterStatus === 'annotated') return v.status === 'anotado';
            return true;
        });
    }, [videos, filterStatus]);

    // Load inicial
    useEffect(() => {
        APIService.getConfig().then((res: any) => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
                setInputPath(''); // Inicia vazio

                if (res.data.classes) {
                    setClasse1(res.data.classes.classe1 || 'NORMAL');
                    setClasse2(res.data.classes.classe2 || 'FURTO');
                }
            }
        });
    }, []);

    // Reload quando muda inputPath
    useEffect(() => {
        if (inputPath) {
            loadVideos();
        }
    }, [inputPath]);

    const loadVideos = async () => {
        setLoading(true);
        try {
            const res = await APIService.listAnnotationVideos(inputPath || undefined);
            const data = (res as any).data;
            setVideos(data.videos || []);
        } catch (err) {
            console.error(err);
            setMessage("Erro ao carregar vídeos");
        } finally {
            setLoading(false);
        }
    };

    const handleSelectVideo = async (video: VideoItem) => {
        setSelectedVideo(video);
        setVideoIds([]);
        setAnnotations({});
        setIdIntervals({}); // Reset intervals
        setMessage('');

        try {
            const res = await APIService.getAnnotationDetails(video.video_id, inputPath);
            const data = (res as any).data;
            const ids = data.ids || [];
            setVideoIds(ids);

            // Por padrão todos são classe1 (NORMAL)
            const initial: Record<string, string> = {};
            ids.forEach((item: any) => {
                initial[String(item.id)] = classe1;
            });
            setAnnotations(initial);
        } catch (err) {
            console.error(err);
            setMessage("Erro ao carregar dados do vídeo");
        }
    };

    const handleClassChange = (id: string, classe: string) => {
        setAnnotations(prev => ({ ...prev, [id]: classe }));
    };

    const handleAddInterval = (id: string) => {
        const start = parseInt(startInput);
        const end = parseInt(endInput);

        if (isNaN(start) || isNaN(end)) {
            return;
        }

        if (start < 0 || end < start) {
            alert("Intervalo inválido (Início deve ser menor que Fim)");
            return;
        }

        setIdIntervals(prev => {
            const existing = prev[id] || [];
            return { ...prev, [id]: [...existing, [start, end]] };
        });

        setStartInput('');
        setEndInput('');
        setActiveIdForEdit(null);
    };

    const handleRemoveInterval = (id: string, index: number) => {
        setIdIntervals(prev => {
            const existing = prev[id] || [];
            const updated = [...existing];
            updated.splice(index, 1);
            return { ...prev, [id]: updated };
        });
    };

    const handleSave = async () => {
        if (!selectedVideo || !inputPath) return;

        // Validação Modo Temporal
        if (temporalMode) {
            // Verifica se tem furto marcado sem intervalo
            const hasOpenFurto = Object.entries(annotations).some(([id, cls]) => {
                const intervals = idIntervals[id] || [];
                return cls === classe2 && intervals.length === 0;
            });

            if (hasOpenFurto) {
                if (!confirm(`Existem IDs marcados como ${classe2} sem intervalos de tempo definidos. Eles serão salvos como "Vídeo Inteiro". Deseja continuar?`)) {
                    return;
                }
            }
        }

        // Abre modal e inicia estado de processamento
        setTerminalOpen(true);
        setTerminalProcessing(true);
        setTerminalLogs(['[INFO] Iniciando salvamento de anotações...']);
        setSaving(true);
        setMessage('');

        try {
            await APIService.clearLogs('default');

            // Constrói payload híbrido
            let finalAnnotations: any = {};

            Object.keys(annotations).forEach(id => {
                const cls = annotations[id];
                const intervals = idIntervals[id];

                if (temporalMode && cls === classe2 && intervals && intervals.length > 0) {
                    // Modo Complexo (Temporal)
                    finalAnnotations[id] = {
                        classe: cls,
                        intervals: intervals
                    };
                } else {
                    // Modo Simples (String)
                    finalAnnotations[id] = cls;
                }
            });

            // Inicia salvamento
            await APIService.saveAnnotations({
                video_stem: selectedVideo.video_id,
                annotations: finalAnnotations,
                root_path: inputPath
            });

            // Polling de logs durante processamento
            const pollLogs = async () => {
                try {
                    const res = await APIService.getLogs('default');
                    const data = (res as any).data;

                    if (data.logs && data.logs.length > 0) {
                        setTerminalLogs(data.logs);
                    }

                    const health = await APIService.healthCheck();
                    const hData = (health as any).data;

                    if (!hData.processing) {
                        setTerminalProcessing(false);
                        setSaving(false);
                        return; // Para polling
                    }
                    setTimeout(pollLogs, 500);
                } catch (e) {
                    console.error(e);
                }
            };
            pollLogs();

            setTerminalLogs(prev => [...prev, '[OK] Anotações salvas com sucesso!']);
            setTerminalProcessing(false);
            loadVideos();

        } catch (err) {
            console.error(err);
            setTerminalLogs(prev => [...prev, '[ERRO] Falha ao salvar anotações']);
            setTerminalProcessing(false);
            setSaving(false);
        }
    };

    const handleTerminalClose = () => {
        setTerminalOpen(false);
        setMessage('✅ Anotações salvas com sucesso!');
        setTimeout(() => setMessage(''), 5000);
    };

    const getVideoStatusColor = (status: string) => {
        if (status === 'anotado') return 'border-l-4 border-green-500 bg-green-500/10';
        return 'border-l-4 border-yellow-500 bg-yellow-500/10';
    };

    // Opções do filtro  
    const filterOptions: FilterOption[] = useMemo(() => [
        { key: 'all', label: 'Todos os Vídeos', count: stats.total },
        { key: 'pending', label: 'Pendentes', count: stats.pending, color: 'yellow-500' },
        { key: 'annotated', label: 'Anotados', count: stats.annotated, color: 'green-500' },
    ], [stats]);

    const videoSrc = selectedVideo
        ? `http://localhost:8000/reid/video/${selectedVideo.video_id}?root_path=${encodeURIComponent(inputPath)}`
        : '';

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col space-y-6">
            <PageHeader
                title="Anotação de Classes"
                description={`Classificação (Simples ou Temporal) para: ${classe1} / ${classe2}`}
                icon={Pencil}
            >
                <div className="flex items-center gap-4">
                    {/* Toggle Mode */}
                    <div className="flex items-center gap-2 bg-muted/30 px-3 py-1.5 rounded-lg border border-border/50">
                        <span className={`text-xs font-bold uppercase transition-colors ${temporalMode ? 'text-primary' : 'text-muted-foreground'}`}>
                            {temporalMode ? 'Modo Temporal' : 'Modo Simples'}
                        </span>
                        <button
                            onClick={() => setTemporalMode(!temporalMode)}
                            className={`
                                relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1
                                ${temporalMode ? 'bg-primary' : 'bg-input'}
                            `}
                            title="Alternar entre anotação de vídeo inteiro (simples) ou intervalos de tempo (temporal)"
                        >
                            <span
                                className={`
                                    inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform
                                    ${temporalMode ? 'translate-x-4' : 'translate-x-1'}
                                `}
                            />
                        </button>
                    </div>

                    <div className="w-px h-6 bg-border mx-1"></div>

                    <PathSelector
                        value={inputPath}
                        onSelect={() => setExplorerOpen(true)}
                        placeholder="Selecione o diretório..."
                        icon={Tag}
                    />
                    <button
                        onClick={loadVideos}
                        className="p-2 hover:bg-muted rounded-md transition-colors border border-transparent hover:border-border"
                        title="Recarregar Lista"
                    >
                        <RefreshCcw className={`w-4 h-4 text-muted-foreground ${loading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </PageHeader>

            <div className="grid grid-cols-12 gap-6 flex-1 overflow-hidden">
                {/* Lista de Vídeos */}
                <div className="col-span-3 border-r border-border pr-4 overflow-y-auto space-y-4">
                    <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xs font-bold uppercase text-muted-foreground">Fila de Vídeos</h3>
                            <span className="text-[10px] text-muted-foreground font-mono">{displayVideos.length} / {videos.length}</span>
                        </div>
                        <FilterDropdown
                            options={filterOptions}
                            selected={filterStatus}
                            onSelect={(key) => setFilterStatus(key as 'all' | 'pending' | 'annotated')}
                        />
                    </div>

                    <div className="space-y-2">
                        {displayVideos.map(v => (
                            <div
                                key={v.video_id}
                                onClick={() => handleSelectVideo(v)}
                                className={`
                                    p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between
                                    ${getVideoStatusColor(v.status)}
                                    ${selectedVideo?.video_id === v.video_id ? 'ring-2 ring-primary' : 'border-border'}
                                `}
                            >
                                <div className="truncate flex-1">
                                    <p className="font-bold text-sm truncate">{v.video_id}</p>
                                    <p className="text-[10px] text-muted-foreground">
                                        {v.status === 'anotado' ? '✅ ANOTADO' : '⏳ PENDENTE'}
                                    </p>
                                </div>
                                {v.status === 'anotado' && <CheckCircle2 className="w-4 h-4 text-green-500" />}
                                {v.status === 'pendente' && <Clock className="w-4 h-4 text-yellow-500" />}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Player e Anotações */}
                <div className="col-span-9 overflow-y-auto space-y-4">
                    {!selectedVideo ? (
                        <div className="h-full min-h-[400px] border-2 border-dashed border-border rounded-xl flex flex-col items-center justify-center text-muted-foreground p-12 bg-muted/5">
                            <VideoIcon className="w-16 h-16 mb-4 opacity-20" />
                            <p className="text-lg font-medium">Selecione um vídeo para anotar</p>
                        </div>
                    ) : (
                        <>
                            {/* Video Player */}
                            <VideoPlayer
                                key={selectedVideo.video_id}
                                src={videoSrc}
                                fps={30}
                                onFrameChange={setCurrentFrame} // Captura frame para temporal
                            />

                            {/* Mensagem de feedback */}
                            <StatusMessage message={message} onClose={() => setMessage('')} autoCloseDelay={5000} className="mb-4" />

                            {/* Anotação de IDs */}
                            <div className="bg-card border border-border rounded-xl shadow-md">
                                <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                                    <h3 className="font-bold flex items-center gap-2">
                                        <Tag className="w-5 h-5 text-primary" />
                                        Classificação {temporalMode && <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded">TEMPORAL</span>}
                                    </h3>
                                    <button
                                        onClick={handleSave}
                                        disabled={saving || videoIds.length === 0}
                                        className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-bold text-sm hover:brightness-110 active:scale-95 transition-all disabled:opacity-50 flex items-center gap-2 shadow-lg shadow-primary/20"
                                    >
                                        <Save className="w-4 h-4" />
                                        {saving ? 'Salvando...' : 'Salvar'}
                                    </button>
                                </div>

                                <div className="p-6">
                                    {videoIds.length === 0 ? (
                                        <div className="text-center text-muted-foreground py-8">
                                            <p>Carregando IDs do vídeo...</p>
                                        </div>
                                    ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-2 gap-4">
                                            {videoIds.map((item) => (
                                                <div key={item.id} className={`
                                                    p-4 border rounded-xl transition-colors flex flex-col gap-3 relative
                                                    ${annotations[String(item.id)] === classe2 ? 'border-red-500/30 bg-red-500/5' : 'border-border bg-muted/10'}
                                                `}>
                                                    {/* Header do Card */}
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-lg font-extrabold text-primary">ID {item.id}</span>
                                                            <span className="text-[10px] bg-secondary px-2 py-0.5 rounded text-secondary-foreground font-mono">{item.frames} frames</span>
                                                        </div>

                                                        {temporalMode && annotations[String(item.id)] === classe2 && (
                                                            <div className="flex gap-1">
                                                                <button
                                                                    onClick={() => setActiveIdForEdit(activeIdForEdit === String(item.id) ? null : String(item.id))}
                                                                    className={`p-1.5 rounded-md hover:bg-muted ${activeIdForEdit === String(item.id) ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'}`}
                                                                    title="Adicionar Intervalo"
                                                                >
                                                                    <Plus className="w-4 h-4" />
                                                                </button>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {/* Class Buttons */}
                                                    <div className="flex gap-2 p-1 bg-background border border-border rounded-lg">
                                                        <button
                                                            onClick={() => handleClassChange(String(item.id), classe1)}
                                                            className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all ${annotations[String(item.id)] === classe1 ? 'bg-green-500 text-white shadow-sm' : 'hover:bg-muted text-muted-foreground'}`}
                                                        >
                                                            {classe1}
                                                        </button>
                                                        <button
                                                            onClick={() => handleClassChange(String(item.id), classe2)}
                                                            className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all ${annotations[String(item.id)] === classe2 ? 'bg-red-500 text-white shadow-sm' : 'hover:bg-muted text-muted-foreground'}`}
                                                        >
                                                            {classe2}
                                                        </button>
                                                    </div>

                                                    {/* Temporal Editor Area */}
                                                    {temporalMode && annotations[String(item.id)] === classe2 && (
                                                        <div className="mt-2 space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">

                                                            {/* Lista de Intervalos */}
                                                            <div className="flex flex-wrap gap-2">
                                                                {(idIntervals[String(item.id)] || []).map((interval, idx) => (
                                                                    <div key={idx} className="flex items-center gap-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-2 py-1 rounded text-xs font-mono border border-red-200 dark:border-red-800">
                                                                        <Clock className="w-3 h-3" />
                                                                        {interval[0]} - {interval[1]}
                                                                        <button
                                                                            onClick={() => handleRemoveInterval(String(item.id), idx)}
                                                                            className="ml-1 hover:text-red-900 dark:hover:text-red-100"
                                                                        >
                                                                            <Trash2 className="w-3 h-3" />
                                                                        </button>
                                                                    </div>
                                                                ))}
                                                                {(idIntervals[String(item.id)] || []).length === 0 && (
                                                                    <div className="flex items-center gap-2 text-xs text-yellow-600 dark:text-yellow-500 bg-yellow-500/10 px-2 py-1 rounded w-full">
                                                                        <AlertTriangle className="w-3 h-3" />
                                                                        Sem intervalos! (Será todo o vídeo)
                                                                    </div>
                                                                )}
                                                            </div>

                                                            {/* Form de Adição */}
                                                            {activeIdForEdit === String(item.id) && (
                                                                <div className="bg-muted/50 p-2 rounded-lg border border-border space-y-2">
                                                                    <div className="flex items-center gap-2">
                                                                        <div className="flex-1 relative">
                                                                            <input
                                                                                type="number"
                                                                                placeholder="Início"
                                                                                className="w-full h-8 text-xs px-2 rounded border border-border bg-background"
                                                                                value={startInput}
                                                                                onChange={e => setStartInput(e.target.value)}
                                                                            />
                                                                            <button
                                                                                onClick={() => setStartInput(String(currentFrame))}
                                                                                className="absolute right-1 top-1 p-0.5 text-muted-foreground hover:text-primary"
                                                                                title="Capturar frame atual"
                                                                            >
                                                                                <Camera className="w-3 h-3" />
                                                                            </button>
                                                                        </div>
                                                                        <span className="text-muted-foreground">-</span>
                                                                        <div className="flex-1 relative">
                                                                            <input
                                                                                type="number"
                                                                                placeholder="Fim"
                                                                                className="w-full h-8 text-xs px-2 rounded border border-border bg-background"
                                                                                value={endInput}
                                                                                onChange={e => setEndInput(e.target.value)}
                                                                            />
                                                                            <button
                                                                                onClick={() => setEndInput(String(currentFrame))}
                                                                                className="absolute right-1 top-1 p-0.5 text-muted-foreground hover:text-primary"
                                                                                title="Capturar frame atual"
                                                                            >
                                                                                <Camera className="w-3 h-3" />
                                                                            </button>
                                                                        </div>
                                                                    </div>
                                                                    <button
                                                                        onClick={() => handleAddInterval(String(item.id))}
                                                                        className="w-full py-1 bg-primary text-primary-foreground rounded text-xs font-bold hover:opacity-90"
                                                                    >
                                                                        Confirmar Intervalo
                                                                    </button>
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* File Explorer */}
            <FileExplorerModal
                isOpen={explorerOpen}
                onClose={() => setExplorerOpen(false)}
                onSelect={(path) => {
                    setInputPath(path);
                    setExplorerOpen(false);
                }}
                initialPath={roots.reidentificacoes}
                rootPath={roots.reidentificacoes}
                title="Selecionar Diretório de ReID"
                filterFn={(item) => {
                    const normRoot = (roots.reidentificacoes || '').replace(/\\/g, '/').toLowerCase();
                    const normCurrent = (item.currentPath || '').replace(/\\/g, '/').toLowerCase();

                    if (normCurrent === normRoot || normCurrent === normRoot + '/') {
                        return item.is_dir;
                    }

                    const relative = normCurrent.replace(normRoot, '');
                    const depth = relative.split('/').filter(p => p).length;

                    if (depth === 1) return item.name === 'predicoes';

                    return true;
                }}
            />

            {/* Terminal Modal para salvamento */}
            <TerminalModal
                isOpen={terminalOpen}
                title="Console de Anotações"
                logs={terminalLogs}
                isProcessing={terminalProcessing}
                onClose={() => setTerminalOpen(false)}
                autoCloseDelay={3000}
                onAutoClose={handleTerminalClose}
            />
        </div >
    );
}
