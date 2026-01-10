'use client';

import { useState, useEffect } from 'react';
import { PageHeader } from '@/components/ui/page-header';
import { Play, ScanFace, Save, Trash2, Scissors, ArrowRightLeft, RefreshCw, FileVideo } from 'lucide-react';
import { APIService, ReIDVideo, ReIDData } from '@/services/api';

export default function ReidPage() {
    const [loading, setLoading] = useState(false);
    const [videos, setVideos] = useState<ReIDVideo[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<ReIDVideo | null>(null);
    const [reidData, setReidData] = useState<ReIDData | null>(null);
    const [rootPath, setRootPath] = useState('');

    // Rules State
    const [swaps, setSwaps] = useState<{ src: number; tgt: number; start: number; end: number }[]>([]);
    const [deletions, setDeletions] = useState<{ id: number; start: number; end: number }[]>([]);
    const [cuts, setCuts] = useState<{ start: number; end: number }[]>([]);

    // Form Inputs
    const [swapForm, setSwapForm] = useState({ src: '', tgt: '', range: '' });
    const [delForm, setDelForm] = useState({ id: '', range: '' });
    const [cutForm, setCutForm] = useState({ range: '' });

    useEffect(() => {
        loadVideos();
    }, []);

    const loadVideos = async () => {
        setLoading(true);
        try {
            const res = await APIService.listReIDVideos(rootPath || undefined);
            setVideos(res.data.videos);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleSelectVideo = async (video: ReIDVideo) => {
        setSelectedVideo(video);
        setReidData(null);
        setSwaps([]);
        setDeletions([]);
        setCuts([]);

        try {
            // Load Data
            const res = await APIService.getReIDData(video.id, rootPath || undefined);
            setReidData(res.data);
        } catch (error) {
            console.error("Erro ao carregar dados do video", error);
        }
    };

    const parseRange = (rangeStr: string): [number, number] => {
        if (!rangeStr.trim()) return [0, 999999];
        const parts = rangeStr.split('-');
        if (parts.length === 2) {
            return [parseInt(parts[0]), parseInt(parts[1])];
        }
        if (parts.length === 1 && !isNaN(parseInt(parts[0]))) {
            const f = parseInt(parts[0]);
            return [f, f];
        }
        return [0, 999999];
    };

    const addSwap = () => {
        const [start, end] = parseRange(swapForm.range);
        setSwaps([...swaps, { src: parseInt(swapForm.src), tgt: parseInt(swapForm.tgt), start, end }]);
        setSwapForm({ src: '', tgt: '', range: '' });
    };

    const addDeletion = () => {
        const [start, end] = parseRange(delForm.range);
        setDeletions([...deletions, { id: parseInt(delForm.id), start, end }]);
        setDelForm({ id: '', range: '' });
    };

    const addCut = () => {
        const [start, end] = parseRange(cutForm.range);
        setCuts([...cuts, { start, end }]);
        setCutForm({ range: '' });
    };

    const handleApply = async () => {
        if (!selectedVideo) return;
        setLoading(true);
        try {
            const payload = {
                rules: swaps,
                deletions: deletions,
                cuts: cuts
            };
            const res = await APIService.applyReIDChanges(selectedVideo.id, payload, rootPath || undefined);
            alert(`Processamento concluído! Novo vídeo em: ${res.data.output_video}`);
        } catch (error) {
            console.error(error);
            alert('Erro ao aplicar mudanças.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col">
            <PageHeader
                title="Re-identificação Manual"
                description="Corrija IDs, remova ruídos e corte trechos indesejados."
            >
                <div className="flex gap-2">
                    <input
                        type="text"
                        placeholder="Pasta personalizada..."
                        className="px-3 py-1 text-sm rounded bg-background border border-border"
                        value={rootPath}
                        onChange={(e) => setRootPath(e.target.value)}
                    />
                    <button
                        onClick={() => loadVideos()}
                        className="p-2 hover:bg-muted rounded transition-colors"
                        title="Recarregar Vídeos"
                    >
                        <RefreshCw className="w-4 h-4" />
                    </button>
                </div>
            </PageHeader>

            <div className="grid grid-cols-12 gap-6 flex-1 overflow-hidden">
                {/* Sidebar: Video List */}
                <div className="col-span-3 border-r border-border pr-6 overflow-y-auto">
                    <h3 className="font-semibold mb-4 text-sm text-muted-foreground uppercase tracking-wider">Vídeos Disponíveis</h3>
                    <div className="space-y-2">
                        {videos.map(v => (
                            <div
                                key={v.id}
                                onClick={() => handleSelectVideo(v)}
                                className={`
                                    p-3 rounded-lg border transition-all cursor-pointer flex items-center gap-3
                                    ${selectedVideo?.id === v.id
                                        ? 'bg-primary/10 border-primary text-primary'
                                        : 'bg-card border-border hover:border-primary/50'}
                                `}
                            >
                                <FileVideo className="w-5 h-5 shrink-0" />
                                <div className="truncate">
                                    <p className="font-medium text-sm truncate">{v.id}</p>
                                    <p className="text-xs text-muted-foreground truncate">{v.video_path ? 'Pronto' : 'Vídeo ausente'}</p>
                                </div>
                            </div>
                        ))}
                        {videos.length === 0 && !loading && (
                            <p className="text-sm text-muted-foreground italic">Nenhum vídeo com JSON encontrado.</p>
                        )}
                        {loading && (
                            <p className="text-sm text-primary animate-pulse">Carregando...</p>
                        )}
                    </div>
                </div>

                {/* Main: Editor */}
                <div className="col-span-9 flex flex-col overflow-y-auto">
                    {selectedVideo ? (
                        <div className="space-y-6">
                            {/* Video Player Area */}
                            <div className="bg-black aspect-video rounded-xl overflow-hidden relative group">
                                <video
                                    src={`/api/reid/video/${selectedVideo.id}?root_path=${encodeURIComponent(rootPath)}`}
                                    controls
                                    className="w-full h-full object-contain"
                                />
                            </div>

                            {/* ID Stats */}
                            <div className="p-4 bg-muted/20 rounded-lg border border-border">
                                <h4 className="font-semibold text-sm mb-2">IDs Mais Frequentes</h4>
                                <div className="flex flex-wrap gap-2">
                                    {reidData && Object.entries(reidData.id_counts).slice(0, 15).map(([id, count]) => (
                                        <span key={id} className="text-xs bg-secondary px-2 py-1 rounded border border-border">
                                            ID {id}: {count} frames
                                        </span>
                                    ))}
                                </div>
                            </div>

                            {/* Actions / Logic Area */}
                            <div className="grid md:grid-cols-3 gap-4">
                                {/* Swap Card */}
                                <div className="p-4 bg-card rounded-xl border border-border space-y-3">
                                    <div className="flex items-center gap-2 font-medium text-blue-400">
                                        <ArrowRightLeft className="w-4 h-4" /> Trocar ID
                                    </div>
                                    <div className="space-y-2 text-sm">
                                        <input
                                            placeholder="ID Original"
                                            className="w-full bg-secondary p-2 rounded"
                                            value={swapForm.src} onChange={e => setSwapForm({ ...swapForm, src: e.target.value })}
                                        />
                                        <input
                                            placeholder="ID Novo"
                                            className="w-full bg-secondary p-2 rounded"
                                            value={swapForm.tgt} onChange={e => setSwapForm({ ...swapForm, tgt: e.target.value })}
                                        />
                                        <input
                                            placeholder="Frame Range (ex: 1-100)"
                                            className="w-full bg-secondary p-2 rounded"
                                            value={swapForm.range} onChange={e => setSwapForm({ ...swapForm, range: e.target.value })}
                                        />
                                        <button onClick={addSwap} className="w-full bg-blue-500/10 text-blue-500 hover:bg-blue-500/20 py-2 rounded">Adicionar Troca</button>
                                    </div>
                                    <ul className="text-xs space-y-1 text-muted-foreground">
                                        {swaps.map((s, i) => (
                                            <li key={i}>{s.src} → {s.tgt} ({s.start}-{s.end})</li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Delete Card */}
                                <div className="p-4 bg-card rounded-xl border border-border space-y-3">
                                    <div className="flex items-center gap-2 font-medium text-red-400">
                                        <Trash2 className="w-4 h-4" /> Excluir ID
                                    </div>
                                    <div className="space-y-2 text-sm">
                                        <input
                                            placeholder="ID para apagar"
                                            className="w-full bg-secondary p-2 rounded"
                                            value={delForm.id} onChange={e => setDelForm({ ...delForm, id: e.target.value })}
                                        />
                                        <input
                                            placeholder="Frame Range (vazio=tudo)"
                                            className="w-full bg-secondary p-2 rounded"
                                            value={delForm.range} onChange={e => setDelForm({ ...delForm, range: e.target.value })}
                                        />
                                        <button onClick={addDeletion} className="w-full bg-red-500/10 text-red-500 hover:bg-red-500/20 py-2 rounded">Adicionar Exclusão</button>
                                    </div>
                                    <ul className="text-xs space-y-1 text-muted-foreground">
                                        {deletions.map((d, i) => (
                                            <li key={i}>ID {d.id} ({d.start}-{d.end})</li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Cut Card */}
                                <div className="p-4 bg-card rounded-xl border border-border space-y-3">
                                    <div className="flex items-center gap-2 font-medium text-amber-400">
                                        <Scissors className="w-4 h-4" /> Cortar Vídeo
                                    </div>
                                    <div className="space-y-2 text-sm">
                                        <input
                                            placeholder="Frame Range (ex: 0-100)"
                                            className="w-full bg-secondary p-2 rounded"
                                            value={cutForm.range} onChange={e => setCutForm({ ...cutForm, range: e.target.value })}
                                        />
                                        <button onClick={addCut} className="w-full bg-amber-500/10 text-amber-500 hover:bg-amber-500/20 py-2 rounded">Adicionar Corte</button>
                                    </div>
                                    <ul className="text-xs space-y-1 text-muted-foreground">
                                        {cuts.map((c, i) => (
                                            <li key={i}>Cut: {c.start}-{c.end}</li>
                                        ))}
                                    </ul>
                                </div>
                            </div>

                            <div className="pt-4 pb-10">
                                <button
                                    onClick={handleApply}
                                    disabled={loading}
                                    className="w-full py-4 bg-primary text-primary-foreground font-bold rounded-xl hover:brightness-110 flex items-center justify-center gap-2 shadow-lg shadow-primary/20"
                                >
                                    <Save className="w-5 h-5" />
                                    {loading ? 'Processando (Pode demorar)...' : 'Aplicar Mudanças e Salvar'}
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-50">
                            <ScanFace className="w-16 h-16 mb-4" />
                            <p>Selecione um vídeo para começar</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
