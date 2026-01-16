import { useState, useEffect, useMemo } from 'react';
import {
    Tag,
    Video as VideoIcon,
    Save,
    FolderInput,
    RefreshCcw,
    Clock,
    CheckCircle2
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { VideoPlayer } from '../components/ui/VideoPlayer';
import { FilterDropdown, FilterOption } from '../components/ui/FilterDropdown';
import { StatusMessage } from '../components/ui/StatusMessage';
import { TerminalModal } from '../components/ui/TerminalModal';

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
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
                setInputPath(res.data.paths.reidentificacoes || '');

                // Classes do config
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
            // localStorage.setItem('np_annotation_input', inputPath); // Desativado para não persistir
        }
    }, [inputPath]);

    const loadVideos = async () => {
        setLoading(true);
        try {
            const res = await APIService.listAnnotationVideos(inputPath || undefined);
            setVideos(res.data.videos || []);
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
        setMessage('');

        try {
            const res = await APIService.getAnnotationDetails(video.video_id, inputPath);
            const ids = res.data.ids || [];
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

    const handleSave = async () => {
        if (!selectedVideo || !inputPath) return;

        // Abre modal e inicia estado de processamento
        setTerminalOpen(true);
        setTerminalProcessing(true);
        setTerminalLogs(['[INFO] Iniciando salvamento de anotações...']);
        setSaving(true);
        setMessage('');

        try {
            // Limpa logs anteriores no backend
            await APIService.clearLogs();

            // Inicia salvamento
            await APIService.saveAnnotations({
                video_stem: selectedVideo.video_id,
                annotations: annotations,
                root_path: inputPath
            });

            // Polling de logs durante processamento
            const pollLogs = async () => {
                try {
                    const res = await APIService.getLogs();
                    if (res.data.logs && res.data.logs.length > 0) {
                        setTerminalLogs(res.data.logs);
                    }

                    const health = await APIService.healthCheck();
                    if (!health.data.processing) {
                        setTerminalProcessing(false);
                        setSaving(false);
                        return; // Para polling
                    }

                    // Continua polling
                    setTimeout(pollLogs, 500);
                } catch (e) {
                    console.error(e);
                }
            };

            // Inicia polling
            pollLogs();

            // Adiciona mensagem de sucesso aos logs (aparece após o processamento)
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
        <div className="h-[calc(100vh-8rem)] flex flex-col space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border pb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                        <Tag className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight">Anotação de Classes</h1>
                        <p className="text-sm text-muted-foreground">Classifique indivíduos: {classe1} ou {classe2}</p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    {/* Entrada */}
                    <div className="flex items-center gap-2">
                        <input
                            type="text"
                            value={inputPath ? inputPath.replace(/\\/g, '/').split('/').pop() || '' : ''}
                            title={inputPath}
                            readOnly
                            className="w-64 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm cursor-pointer"
                            placeholder="Selecione uma pasta para anotar..."
                            onClick={() => setExplorerOpen(true)}
                        />
                        <button
                            onClick={() => setExplorerOpen(true)}
                            className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                        >
                            <FolderInput className="w-4 h-4" />
                        </button>
                    </div>
                    <button
                        onClick={loadVideos}
                        className="p-2 hover:bg-muted rounded-md transition-colors border border-transparent hover:border-border"
                        title="Recarregar Lista"
                    >
                        <RefreshCcw className={`w-4 h-4 text-muted-foreground ${loading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6 flex-1 overflow-hidden">
                {/* Lista de Vídeos */}
                <div className="col-span-3 border-r border-border pr-4 overflow-y-auto space-y-4">
                    <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xs font-bold uppercase text-muted-foreground">Fila de Vídeos</h3>
                            <span className="text-[10px] text-muted-foreground font-mono">{displayVideos.length} / {videos.length}</span>
                        </div>

                        {/* Filtros Dropdown */}
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
                            <p className="text-sm opacity-60">Os IDs persistentes serão listados abaixo do player.</p>
                        </div>
                    ) : (
                        <>
                            {/* Video Player */}
                            <VideoPlayer key={selectedVideo.video_id} src={videoSrc} fps={30} />

                            {/* Mensagem de feedback */}
                            <StatusMessage message={message} onClose={() => setMessage('')} autoCloseDelay={5000} className="mb-4" />

                            {/* Anotação de IDs */}
                            <div className="bg-card border border-border rounded-xl shadow-md">
                                <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                                    <h3 className="font-bold flex items-center gap-2">
                                        <Tag className="w-5 h-5 text-primary" />
                                        Classificação de Indivíduos - {selectedVideo.video_id}
                                    </h3>
                                    <button
                                        onClick={handleSave}
                                        disabled={saving || videoIds.length === 0}
                                        className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-bold text-sm hover:brightness-110 active:scale-95 transition-all disabled:opacity-50 flex items-center gap-2 shadow-lg shadow-primary/20"
                                    >
                                        <Save className="w-4 h-4" />
                                        {saving ? 'Salvando...' : 'Salvar Anotações'}
                                    </button>
                                </div>

                                <div className="p-6">
                                    {videoIds.length === 0 ? (
                                        <div className="text-center text-muted-foreground py-8">
                                            <p>Carregando IDs do vídeo...</p>
                                        </div>
                                    ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                                            {videoIds.map((item) => (
                                                <div key={item.id} className="p-4 border border-border rounded-xl bg-muted/10 hover:bg-muted/30 transition-colors flex flex-col gap-3">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-lg font-extrabold text-primary">ID {item.id}</span>
                                                        <span className="text-[10px] bg-secondary px-2 py-0.5 rounded text-secondary-foreground font-mono">{item.frames} frames</span>
                                                    </div>

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
                title="Selecionar Pasta de ReID"
                filterFn={(item) => {
                    const normRoot = (roots.reidentificacoes || '').replace(/\\/g, '/').toLowerCase();
                    const normCurrent = (item.currentPath || '').replace(/\\/g, '/').toLowerCase();

                    // Se estamos na raiz, mostrar apenas pastas (datasets)
                    if (normCurrent === normRoot || normCurrent === normRoot + '/') {
                        // Mostrar datasets, ocultar arquivos soltos se houver
                        return item.is_dir;
                    }

                    // Dentro do dataset: Mostrar 'predicoes' (conforme pedido)
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
