import { useState, useEffect, useMemo } from 'react';
import { ScanFace, Save, Trash2, Scissors, ArrowRightLeft, RotateCcw, FileVideo, Pencil } from 'lucide-react';
import { PageHeader } from '../components/ui/PageHeader';
import { APIService, ReIDVideo, ReIDData } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { FilterDropdown, FilterOption } from '../components/ui/FilterDropdown';
import { StatusMessage } from '../components/ui/StatusMessage';
import { TerminalModal } from '../components/ui/TerminalModal';
import { ReidPlayer } from '../components/ReidPlayer';
import { PathSelector } from '../components/ui/PathSelector';


export default function ReidPage() {
    const [loading, setLoading] = useState(false);
    const [videos, setVideos] = useState<ReIDVideo[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<ReIDVideo | null>(null);
    const [reidData, setReidData] = useState<ReIDData | null>(null);
    const [inputPath, setInputPath] = useState('');
    const [filterStatus, setFilterStatus] = useState<'all' | 'pending' | 'processed' | 'excluded'>('all');
    const [roots, setRoots] = useState<Record<string, string>>({}); // Paths from backend
    const [message, setMessage] = useState(''); // Feedback message

    useEffect(() => {
        APIService.getConfig().then(res => {
            const data = res.data as any;
            if (data.status === 'success') {
                setRoots(data.paths);
            }
        });
    }, []);



    // Current Video Edits
    const [swaps, setSwaps] = useState<{ src: number; tgt: number; start: number; end: number }[]>([]);
    const [deletions, setDeletions] = useState<{ id: number; start: number; end: number }[]>([]);
    const [cuts, setCuts] = useState<{ start: number; end: number }[]>([]);
    const [isDeleted, setIsDeleted] = useState(false); // New: Mark video for deletion

    // Batch State
    const [batchAnnotations, setBatchAnnotations] = useState<Record<string, any>>({});

    // Forms
    const [swapForm, setSwapForm] = useState({ src: '', tgt: '', start: '', end: '' });
    const [delForm, setDelForm] = useState({ id: '', start: '', end: '' });
    const [cutForm, setCutForm] = useState({ start: '', end: '' });

    // Explorer
    const [explorerOpen, setExplorerOpen] = useState(false);

    // Terminal Modal
    const [terminalOpen, setTerminalOpen] = useState(false);
    const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
    const [terminalProcessing, setTerminalProcessing] = useState(false);

    // Transform Data for Player - API returns 1-indexed frames
    const playerReidData = useMemo(() => {
        if (!reidData || !reidData.frames) return { frames: {} };

        const frames: Record<string, any[]> = {};
        // Convert 1-based keys from API to 0-based keys for Player
        Object.entries(reidData.frames).forEach(([k, v]) => {
            const frameIdx = parseInt(k) - 1;
            frames[String(frameIdx)] = v;
        });

        return { frames };
    }, [reidData]);



    // Stats & Filtering
    const stats = useMemo(() => {
        let pending = 0;
        let processed = 0;
        let excluded = 0;

        videos.forEach(v => {
            const entry = batchAnnotations[v.id];
            if (!entry) pending++;
            else if (entry.action === 'delete') excluded++;
            else processed++;
        });

        return { pending, processed, excluded, total: videos.length };
    }, [videos, batchAnnotations]);

    const displayVideos = useMemo(() => {
        if (filterStatus === 'all') return videos;
        return videos.filter(v => {
            const entry = batchAnnotations[v.id];
            if (filterStatus === 'pending') return !entry;
            if (filterStatus === 'excluded') return entry?.action === 'delete';
            if (filterStatus === 'processed') return entry && entry.action !== 'delete';
            return true;
        });
    }, [videos, batchAnnotations, filterStatus]);

    // Opções de filtro para o novo componente
    const filterOptions: FilterOption[] = useMemo(() => [
        { key: 'all', label: 'Todos os Vídeos', count: videos.length },
        { key: 'pending', label: 'Pendentes', count: stats.pending, color: 'yellow-500' },
        { key: 'processed', label: 'Re-identificados', count: stats.processed, color: 'green-500' },
        { key: 'excluded', label: 'Removidos', count: stats.excluded, color: 'red-500' },
    ], [videos.length, stats]);

    const loadAgenda = async () => {
        if (!inputPath) return;
        try {
            const res = await APIService.getReidAgenda(inputPath);
            if (res.data.agenda && res.data.agenda.videos) {
                const batch: Record<string, any> = {};
                for (const v of res.data.agenda.videos) {
                    batch[v.video_id] = {
                        video_id: v.video_id,
                        rules: v.swaps?.map((s: any) => ({ src: s.src_id, tgt: s.tgt_id, start: s.frame_start, end: s.frame_end })) || [],
                        deletions: v.deletions?.map((d: any) => ({ id: d.id, start: d.frame_start, end: d.frame_end })) || [],
                        cuts: v.cuts?.map((c: any) => ({ start: c.frame_start, end: c.frame_end })) || [],
                        action: v.action || 'process'
                    };
                }
                setBatchAnnotations(batch);
            } else {
                setBatchAnnotations({});
            }
        } catch (error) {
            console.error(error);
        }
    };

    // Carregar configurações apenas uma vez (já feito no primeiro useEffect)
    // inputPath deve começar vazio para mostrar o placeholder
    useEffect(() => {
        if (inputPath) {
            loadVideos();
            loadAgenda();
        }
    }, [inputPath]);

    // Remover salvamento automático em localStorage - agora usa API
    // useEffect(() => { localStorage.setItem('np_reid_batch', JSON.stringify(batchAnnotations)); }, [batchAnnotations]);

    const loadVideos = async () => {
        setLoading(true);
        try {
            const res = await APIService.listReIDVideos(inputPath || undefined);
            setVideos(res.data.videos);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleSelectVideo = async (video: ReIDVideo) => {
        // Save previous video state if modified
        if (selectedVideo) {
            saveToBatch(selectedVideo.id);
        }

        setSelectedVideo(video);
        setReidData(null);

        // Load annotations from batch if exists
        const saved = batchAnnotations[video.id];
        if (saved) {
            setSwaps(saved.rules || []);
            setDeletions(saved.deletions || []);
            setCuts(saved.cuts || []);
            setIsDeleted(saved.action === 'delete');
        } else {
            // Reset state for new video
            setSwaps([]);
            setDeletions([]);
            setCuts([]);
            setIsDeleted(false);
        }

        try {
            const res = await APIService.getReIDData(video.id, inputPath || undefined);
            setReidData(res.data);
        } catch (error) {
            console.error("Erro ao carregar dados", error);
        }
    };

    const saveToBatch = (videoId: string) => {
        // Only save if there are changes
        const hasChanges = swaps.length > 0 || deletions.length > 0 || cuts.length > 0 || isDeleted;

        if (hasChanges) {
            setBatchAnnotations(prev => ({
                ...prev,
                [videoId]: {
                    video_id: videoId,
                    rules: swaps,
                    deletions: deletions,
                    cuts: cuts,
                    action: isDeleted ? 'delete' : 'process'
                }
            }));
        } else {
            // Remove from batch if no changes (reset to yellow/pending)
            setBatchAnnotations(prev => {
                const copy = { ...prev };
                delete copy[videoId];
                return copy;
            });
        }
    };


    const addSwap = () => {
        if (!swapForm.src || !swapForm.tgt) return;
        const start = swapForm.start ? parseInt(swapForm.start) : 0;
        const end = swapForm.end ? parseInt(swapForm.end) : 999999;
        setSwaps([...swaps, { src: parseInt(swapForm.src), tgt: parseInt(swapForm.tgt), start, end }]);
        setSwapForm({ src: '', tgt: '', start: '', end: '' });
    };

    const addDeletion = () => {
        if (!delForm.id) return;
        const start = delForm.start ? parseInt(delForm.start) : 0;
        const end = delForm.end ? parseInt(delForm.end) : 999999;
        setDeletions([...deletions, { id: parseInt(delForm.id), start, end }]);
        setDelForm({ id: '', start: '', end: '' });
    };

    const addCut = () => {
        if (!cutForm.start && !cutForm.end) return;
        const start = cutForm.start ? parseInt(cutForm.start) : 0;
        const end = cutForm.end ? parseInt(cutForm.end) : 999999;
        setCuts([...cuts, { start, end }]);
        setCutForm({ start: '', end: '' });
    };

    const handleToggleRemoval = async () => {
        if (!selectedVideo || !inputPath) return;

        const newStatus = !isDeleted;
        // Optimistic UI update
        setIsDeleted(newStatus);

        try {
            if (newStatus) {
                // REMOVER: Salva action: 'delete'
                const videoEntry = {
                    video_id: selectedVideo.id,
                    action: 'delete',
                    swaps: [], deletions: [], cuts: []
                };
                await APIService.saveToReidAgenda(inputPath, videoEntry);

                // Update batch status manually cause saveToBatch uses stale state here
                setBatchAnnotations(prev => ({
                    ...prev,
                    [selectedVideo.id]: { ...videoEntry, video_id: selectedVideo.id }
                }));

            } else {
                // RESTAURAR: Remove do JSON (volta a ser pendente)
                await APIService.removeFromReidAgenda(selectedVideo.id, inputPath);

                // Remove from batch
                setBatchAnnotations(prev => {
                    const copy = { ...prev };
                    delete copy[selectedVideo.id];
                    return copy;
                });
            }
        } catch (error) {
            console.error("Erro ao atualizar status de remoção", error);
            setIsDeleted(!newStatus); // Revert
            setMessage("❌ Erro ao salvar status. Tente novamente.");
        }
    };

    const handleSchedule = async () => {
        if (!selectedVideo || !inputPath) return;

        // Monta objeto no formato esperado pela API
        const videoEntry = {
            video_id: selectedVideo.id,
            action: isDeleted ? 'delete' : 'process',
            swaps: swaps.map(s => ({ src_id: s.src, tgt_id: s.tgt, frame_start: s.start, frame_end: s.end })),
            deletions: deletions.map(d => ({ id: d.id, frame_start: d.start, frame_end: d.end })),
            cuts: cuts.map(c => ({ frame_start: c.start, frame_end: c.end }))
        };

        try {
            const res = await APIService.saveToReidAgenda(inputPath, videoEntry);
            const data = res.data as any;
            if (data.status === 'success') {
                // Atualiza estado local também
                saveToBatch(selectedVideo.id);
                setMessage(`✅ Vídeo "${selectedVideo.id}" agendado com sucesso!`);
            } else {
                setMessage('❌ Erro ao salvar agenda.');
            }
        } catch (error) {
            console.error('Erro ao salvar agenda:', error);
            setMessage('❌ Erro ao salvar agenda no servidor.');
        }
    };

    const handleApplyBatch = async () => {
        // Save current open video first
        if (selectedVideo) saveToBatch(selectedVideo.id);

        const list = Object.values(batchAnnotations);
        if (list.length === 0) {
            setMessage("⚠️ Nenhuma alteração agendada para processar.");
            return;
        }

        // Abre modal e inicia estado de processamento
        setTerminalOpen(true);
        setTerminalProcessing(true);
        setTerminalLogs(['[INFO] Iniciando processamento de re-identificação...', `[INFO] ${list.length} vídeos na fila`]);
        setLoading(true);
        setMessage('');

        try {
            // Limpa logs anteriores no backend
            await APIService.clearLogs('process');

            await APIService.batchApplyReID({
                videos: list,
                root_path: inputPath,
            });

            // Polling de logs durante processamento
            const pollLogs = async () => {
                try {
                    const res = await APIService.getLogs('process');
                    const logsData = res.data as any;
                    if (logsData.logs && logsData.logs.length > 0) {
                        setTerminalLogs(logsData.logs);
                    }

                    const health = await APIService.healthCheck();
                    const healthData = health.data as any;
                    if (!healthData.processing) {
                        setTerminalProcessing(false);
                        setLoading(false);
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

            // Adiciona mensagem de sucesso
            setTerminalLogs(prev => [...prev, '[OK] Processamento concluído!']);
            setTerminalProcessing(false);
            setBatchAnnotations({});
            loadVideos();
            loadAgenda();
            setSelectedVideo(null);

        } catch (error) {
            console.error(error);
            setTerminalLogs(prev => [...prev, '[ERRO] Falha ao processar lote']);
            setTerminalProcessing(false);
            setLoading(false);
        }
    };

    const handleTerminalClose = () => {
        setTerminalOpen(false);
        setMessage('✅ Re-identificação processada com sucesso!');
        setTimeout(() => setMessage(''), 5000);
    };

    // Status Color Helper
    const getVideoStatusColor = (vid: string) => {
        const saved = batchAnnotations[vid];
        if (!saved) return 'border-l-4 border-yellow-500/50 opacity-80'; // Pending (Yellowish)
        if (saved.action === 'delete') return 'border-l-4 border-red-500 bg-red-500/10'; // Delete (Red)
        return 'border-l-4 border-green-500 bg-green-500/10'; // Modified (Green)
    };

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col space-y-6">

            <PageHeader
                title="Re-identificação Manual"
                description="Corrija identidades e limpe o dataset."
                icon={ScanFace}
            >
                <div className="flex items-center gap-3">
                    <PathSelector
                        value={inputPath}
                        onSelect={() => setExplorerOpen(true)}
                        placeholder="Selecione o diretório para reidentificar..."
                        icon={ScanFace}
                    />
                    <button
                        onClick={loadVideos}
                        className="p-2 hover:bg-muted rounded-md transition-colors border border-transparent hover:border-border"
                        title="Recarregar Lista"
                    >
                        <RotateCcw className={`w-4 h-4 text-muted-foreground ${loading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </PageHeader>

            <StatusMessage message={message} onClose={() => setMessage('')} autoCloseDelay={5000} />

            <div className="grid grid-cols-12 gap-6 flex-1 overflow-hidden">
                {/* LISTA DE VÍDEOS (Sidebar) */}
                <div className="col-span-3 border-r border-border pr-4 flex flex-col overflow-hidden">
                    {/* Header e Filtro - Fixos */}
                    <div className="flex flex-col gap-2 pb-3 border-b border-border/50 shrink-0">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xs font-bold uppercase text-muted-foreground">Fila de Vídeos</h3>
                            <span className="text-[10px] text-muted-foreground font-mono">{displayVideos.length} / {videos.length}</span>
                        </div>

                        {/* Filtros Dropdown */}
                        <FilterDropdown
                            options={filterOptions}
                            selected={filterStatus}
                            onSelect={(key) => setFilterStatus(key as 'all' | 'pending' | 'processed' | 'excluded')}
                        />
                    </div>

                    {/* Lista de Vídeos - Com Scroll */}
                    <div className="flex-1 overflow-y-auto space-y-2 py-3">
                        {displayVideos.map(v => (
                            <div
                                key={v.id}
                                onClick={() => handleSelectVideo(v)}
                                className={`
                                p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between
                                ${getVideoStatusColor(v.id)}
                                ${selectedVideo?.id === v.id ? 'ring-2 ring-primary' : 'border-border'}
                            `}
                            >
                                <div className="truncate flex-1">
                                    <p className="font-bold text-sm truncate">{v.id}</p>
                                    <p className="text-[10px] text-muted-foreground">
                                        {batchAnnotations[v.id]?.action === 'delete' ? '⛔ PARA EXCLUIR' :
                                            batchAnnotations[v.id] ? '✅ PRONTO PARA SALVAR' : '⚠️ PENDENTE'}
                                    </p>
                                </div>
                                {batchAnnotations[v.id]?.action === 'delete' && <Trash2 className="w-4 h-4 text-red-500" />}
                            </div>
                        ))}
                        {videos.length === 0 && <p className="text-center text-sm text-muted-foreground py-8">Nenhum vídeo encontrado.</p>}
                    </div>
                </div>

                {/* ÁREA DE EDIÇÃO */}
                <div className="col-span-9 flex flex-col h-full overflow-y-auto pr-2 pb-10">
                    {selectedVideo ? (
                        <div className="flex flex-col space-y-6">
                            {/* Toolbar de Ações Rápidas */}
                            <div className="flex items-center justify-between p-2 bg-muted/20 rounded-lg border border-border">
                                <div className="text-xs text-muted-foreground flex items-center gap-2">
                                    <span className="font-bold text-foreground">{selectedVideo.id}</span>
                                    <span>|</span>
                                    <span>{reidData?.id_counts ? Object.keys(reidData.id_counts).length + ' IDs únicos detectados' : 'Carregando stats...'}</span>
                                </div>
                                <button
                                    onClick={handleToggleRemoval}
                                    className={`
                                        px-4 py-2 rounded-lg font-bold text-xs flex items-center gap-2 transition-all
                                        ${isDeleted
                                            ? 'bg-muted text-foreground hover:bg-muted/80'
                                            : 'bg-red-500 text-white hover:bg-red-600 shadow-lg shadow-red-500/20'}
                                    `}
                                >
                                    {isDeleted ? <><RotateCcw className="w-3 h-3" /> Restaurar Vídeo</> : <><Trash2 className="w-3 h-3" /> REMOVER VÍDEO INTEIRO</>}
                                </button>
                            </div>

                            {/* Player Wrapper */}
                            <div className={`
                                relative rounded-xl border-2 bg-black shrink-0
                                ${isDeleted ? 'border-red-500 opacity-50 grayscale pointer-events-none' : 'border-transparent'}
                            `}>
                                <ReidPlayer
                                    key={selectedVideo.id}
                                    src={`http://localhost:8000/reid/video/${selectedVideo.id}?root_path=${encodeURIComponent(inputPath)}`}
                                    reidData={playerReidData}
                                    swaps={swaps}
                                    deletions={deletions}
                                    cuts={cuts}
                                    fps={30}
                                />
                                {isDeleted && (
                                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-50">
                                        <div className="bg-red-600 text-white px-6 py-3 rounded-xl font-bold text-2xl shadow-xl transform -rotate-12 border-4 border-white">
                                            REMOVIDO
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Tabela de IDs Persistentes (NOVO) */}
                            {reidData && reidData.id_counts && (
                                <div className="p-4 bg-muted/10 border border-border rounded-xl">
                                    <h4 className="font-bold text-xs uppercase mb-3 text-muted-foreground flex items-center gap-2">
                                        <ScanFace className="w-4 h-4" /> IDs Persistentes Detectados (Bruto)
                                    </h4>
                                    <div className="flex flex-wrap gap-2">
                                        {Object.entries(reidData.id_counts)
                                            .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
                                            .map(([id, count]) => (
                                                <div key={id} className="flex items-center gap-2 bg-background border border-border px-3 py-1.5 rounded-md text-xs group hover:border-primary transition-colors">
                                                    <span className="font-bold text-primary">ID {id}</span>
                                                    <span className="w-px h-3 bg-border"></span>
                                                    <span className="text-muted-foreground">{count} frames</span>
                                                    {/* Botão rápido para preencher form de exclusão */}
                                                    <button
                                                        onClick={() => setDelForm({ ...delForm, id })}
                                                        className="ml-1 opacity-0 group-hover:opacity-100 text-orange-500 hover:bg-orange-500/10 p-0.5 rounded transition-all"
                                                        title="Preencher para excluir"
                                                    >
                                                        <Trash2 className="w-3 h-3" />
                                                    </button>
                                                </div>
                                            ))}
                                    </div>
                                </div>
                            )}


                            {!isDeleted && (
                                <div className="space-y-6">
                                    {/* Ferramentas de Edição */}
                                    <div className="grid grid-cols-3 gap-4">
                                        {/* Trocas */}
                                        <div className="space-y-3 p-4 bg-blue-500/5 border border-blue-500/20 rounded-xl">
                                            <h4 className="font-bold text-blue-400 text-xs uppercase flex items-center gap-2">
                                                <ArrowRightLeft className="w-4 h-4" /> Corrigir IDs
                                            </h4>
                                            <div className="space-y-2">
                                                <div className="flex gap-2">
                                                    <input placeholder="De (ID)" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={swapForm.src} onChange={e => setSwapForm({ ...swapForm, src: e.target.value })} />
                                                    <input placeholder="Para (ID)" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={swapForm.tgt} onChange={e => setSwapForm({ ...swapForm, tgt: e.target.value })} />
                                                </div>
                                                <div className="flex gap-2">
                                                    <input placeholder="Início" type="number" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={swapForm.start} onChange={e => setSwapForm({ ...swapForm, start: e.target.value })} />
                                                    <input placeholder="Fim" type="number" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={swapForm.end} onChange={e => setSwapForm({ ...swapForm, end: e.target.value })} />
                                                </div>
                                                <button onClick={addSwap} className="w-full py-2 bg-blue-500 text-white text-xs font-bold rounded hover:bg-blue-600">Adicionar Regra</button>
                                            </div>
                                            <div className="space-y-1 max-h-32 overflow-y-auto">
                                                {swaps.map((s, i) => (
                                                    <div key={i} className="text-[10px] bg-background p-1.5 rounded border border-border flex justify-between items-center group">
                                                        <span className="font-mono">{s.src} ➔ {s.tgt} ({s.start}-{s.end})</span>
                                                        <div className="flex items-center gap-1 opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity">
                                                            <button onClick={() => { setSwapForm({ src: s.src.toString(), tgt: s.tgt.toString(), start: s.start.toString(), end: s.end.toString() }); setSwaps(swaps.filter((_, idx) => idx !== i)); }} className="p-1 hover:bg-blue-500/20 text-blue-500 rounded"><Pencil className="w-3 h-3" /></button>
                                                            <button onClick={() => setSwaps(swaps.filter((_, idx) => idx !== i))} className="p-1 hover:bg-red-500/20 text-red-500 rounded"><Trash2 className="w-3 h-3" /></button>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Exclusão de ID */}
                                        <div className="space-y-3 p-4 bg-orange-500/5 border border-orange-500/20 rounded-xl">
                                            <h4 className="font-bold text-orange-400 text-xs uppercase flex items-center gap-2">
                                                <Trash2 className="w-4 h-4" /> Apagar ID (Ruído)
                                            </h4>
                                            <div className="space-y-2">
                                                <input placeholder="ID para remover" className="w-full p-2 text-xs bg-background/50 rounded border border-border" value={delForm.id} onChange={e => setDelForm({ ...delForm, id: e.target.value })} />
                                                <div className="flex gap-2">
                                                    <input placeholder="Início" type="number" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={delForm.start} onChange={e => setDelForm({ ...delForm, start: e.target.value })} />
                                                    <input placeholder="Fim (vazio=tudo)" type="number" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={delForm.end} onChange={e => setDelForm({ ...delForm, end: e.target.value })} />
                                                </div>
                                                <button onClick={addDeletion} className="w-full py-2 bg-orange-500 text-white text-xs font-bold rounded hover:bg-orange-600">Adicionar Exclusão</button>
                                            </div>
                                            <div className="space-y-1 max-h-32 overflow-y-auto">
                                                {deletions.map((d, i) => (
                                                    <div key={i} className="text-[10px] bg-background p-1.5 rounded border border-border flex justify-between items-center group">
                                                        <span className="font-mono">ID {d.id} ({d.start}-{d.end === 999999 ? 'fim' : d.end})</span>
                                                        <div className="flex items-center gap-1 opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity">
                                                            <button onClick={() => { setDelForm({ id: d.id.toString(), start: d.start.toString(), end: d.end === 999999 ? '' : d.end.toString() }); setDeletions(deletions.filter((_, idx) => idx !== i)); }} className="p-1 hover:bg-orange-500/20 text-orange-500 rounded"><Pencil className="w-3 h-3" /></button>
                                                            <button onClick={() => setDeletions(deletions.filter((_, idx) => idx !== i))} className="p-1 hover:bg-red-500/20 text-red-500 rounded"><Trash2 className="w-3 h-3" /></button>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Cortes */}
                                        <div className="space-y-3 p-4 bg-purple-500/5 border border-purple-500/20 rounded-xl">
                                            <h4 className="font-bold text-purple-400 text-xs uppercase flex items-center gap-2">
                                                <Scissors className="w-4 h-4" /> Cortar Vídeo
                                            </h4>
                                            <div className="space-y-2">
                                                <div className="flex gap-2">
                                                    <input placeholder="Início" type="number" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={cutForm.start} onChange={e => setCutForm({ ...cutForm, start: e.target.value })} />
                                                    <input placeholder="Fim" type="number" className="w-1/2 p-2 text-xs bg-background/50 rounded border border-border" value={cutForm.end} onChange={e => setCutForm({ ...cutForm, end: e.target.value })} />
                                                </div>
                                                <button onClick={addCut} className="w-full py-2 bg-purple-500 text-white text-xs font-bold rounded hover:bg-purple-600">Cortar Trecho</button>
                                            </div>
                                            <div className="space-y-1 max-h-32 overflow-y-auto">
                                                {cuts.map((c, i) => (
                                                    <div key={i} className="text-[10px] bg-background p-1.5 rounded border border-border flex justify-between items-center group">
                                                        <span className="font-mono">Cut {c.start}-{c.end}</span>
                                                        <div className="flex items-center gap-1 opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity">
                                                            <button onClick={() => { setCutForm({ start: c.start.toString(), end: c.end.toString() }); setCuts(cuts.filter((_, idx) => idx !== i)); }} className="p-1 hover:bg-purple-500/20 text-purple-500 rounded"><Pencil className="w-3 h-3" /></button>
                                                            <button onClick={() => setCuts(cuts.filter((_, idx) => idx !== i))} className="p-1 hover:bg-red-500/20 text-red-500 rounded"><Trash2 className="w-3 h-3" /></button>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Agendar Button */}
                                    <button
                                        onClick={handleSchedule}
                                        className="w-full py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white font-bold rounded-xl hover:brightness-110 active:scale-[0.98] transition-all shadow-lg shadow-green-600/20 flex items-center justify-center gap-3 text-lg"
                                    >
                                        <Save className="w-5 h-5" />
                                        AGENDAR ALTERAÇÕES DESTE VÍDEO
                                    </button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-50">
                            <FileVideo className="w-16 h-16 mb-4" />
                            <p>Selecione um vídeo da lista ao lado</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Footer Fixo com Contadores e Botão Salvar */}
            <div className="fixed bottom-0 left-64 right-0 bg-card/95 backdrop-blur-sm border-t border-border px-8 py-3 flex items-center justify-between z-40">
                <div className="flex items-center gap-6 text-sm">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
                        <span className="text-muted-foreground">Pendentes:</span>
                        <span className="font-bold text-yellow-500">{stats.pending}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-green-500"></span>
                        <span className="text-muted-foreground">Re-identificados:</span>
                        <span className="font-bold text-green-500">{stats.processed}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-red-500"></span>
                        <span className="text-muted-foreground">Removidos:</span>
                        <span className="font-bold text-red-500">{stats.excluded}</span>
                    </div>
                </div>
                <button
                    onClick={handleApplyBatch}
                    disabled={Object.keys(batchAnnotations).length === 0 || loading}
                    className="px-8 py-3 bg-primary text-primary-foreground text-sm font-bold rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-primary/10 flex items-center gap-2 transition-all"
                >
                    <Save className="w-4 h-4" />
                    {loading ? "Processando..." : "Salvar Alterações"}
                </button>
            </div>

            <FileExplorerModal
                isOpen={explorerOpen}
                onClose={() => setExplorerOpen(false)}
                onSelect={(path) => { setInputPath(path); setExplorerOpen(false); }}
                initialPath={roots.processados}
                rootPath={roots.processados}
                title="Selecionar Diretório de Vídeos/Predições"
                filterFn={(item) => {
                    const normRoot = (roots.processados || '').replace(/\\/g, '/').toLowerCase();
                    const normCurrent = (item.currentPath || '').replace(/\\/g, '/').toLowerCase();

                    // Root: apenas datasets (pastas)
                    if (normCurrent === normRoot || normCurrent === normRoot + '/') {
                        return item.is_dir && !['videos', 'jsons', 'anotacoes'].includes(item.name);
                    }

                    // Inside Dataset: apenas 'predicoes'
                    const relative = normCurrent.replace(normRoot, '');
                    const depth = relative.split('/').filter(p => p).length;

                    if (depth === 1) {
                        return item.name === 'predicoes';
                    }

                    return true;
                }}
            />

            {/* Terminal Modal para processamento */}
            <TerminalModal
                isOpen={terminalOpen}
                title="Console de Re-identificação"
                logs={terminalLogs}
                isProcessing={terminalProcessing}
                onClose={() => setTerminalOpen(false)}
                autoCloseDelay={3000}
                onAutoClose={handleTerminalClose}
            />
        </div >
    );
}
