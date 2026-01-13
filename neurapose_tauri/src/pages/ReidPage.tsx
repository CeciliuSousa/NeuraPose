import { useState, useEffect } from 'react';
import { ScanFace, Save, Trash2, Scissors, ArrowRightLeft, RotateCcw, FileVideo, FolderInput, FolderOutput } from 'lucide-react';
import { APIService, ReIDVideo, ReIDData } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';


export default function ReidPage() {
    const [loading, setLoading] = useState(false);
    const [videos, setVideos] = useState<ReIDVideo[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<ReIDVideo | null>(null);
    const [reidData, setReidData] = useState<ReIDData | null>(null);
    const [inputPath, setInputPath] = useState('');
    const [outputPath, setOutputPath] = useState('');

    // Anotações Atuais (Vídeo Selecionado)
    const [swaps, setSwaps] = useState<{ src: number; tgt: number; start: number; end: number }[]>([]);
    const [deletions, setDeletions] = useState<{ id: number; start: number; end: number }[]>([]);
    const [cuts, setCuts] = useState<{ start: number; end: number }[]>([]);

    // Lote de Anotações (Todos os vídeos)
    const [batchAnnotations, setBatchAnnotations] = useState<Record<string, any>>({});

    // Forms
    const [swapForm, setSwapForm] = useState({ src: '', tgt: '', range: '' });
    const [delForm, setDelForm] = useState({ id: '', range: '' });
    const [cutForm, setCutForm] = useState({ range: '' });

    // FileExplorer state
    const [explorerTarget, setExplorerTarget] = useState<'input' | 'output' | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});


    useEffect(() => {
        // Load settings
        const savedInput = localStorage.getItem('np_reid_input');
        const savedOutput = localStorage.getItem('np_reid_output');
        const savedBatch = localStorage.getItem('np_reid_batch');

        if (savedInput) setInputPath(savedInput);
        if (savedOutput) setOutputPath(savedOutput);
        if (savedBatch) setBatchAnnotations(JSON.parse(savedBatch));

        // Load defaults if empty
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                const { processamentos, reidentificacoes } = res.data.paths;
                setRoots(res.data.paths);
                if (!savedInput) setInputPath(processamentos);
                if (!savedOutput) setOutputPath(reidentificacoes);
            }
        });

        loadVideos();
    }, []);

    useEffect(() => {
        localStorage.setItem('np_reid_input', inputPath);
        localStorage.setItem('np_reid_output', outputPath);
        localStorage.setItem('np_reid_batch', JSON.stringify(batchAnnotations));
    }, [inputPath, outputPath, batchAnnotations]);

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
        if (selectedVideo && (swaps.length > 0 || deletions.length > 0 || cuts.length > 0)) {
            // Auto-save current annotations to batch
            saveToBatch(selectedVideo.id);
        }

        setSelectedVideo(video);
        setReidData(null);

        // Load from batch if exists
        const saved = batchAnnotations[video.id];
        setSwaps(saved?.rules || []);
        setDeletions(saved?.deletions || []);
        setCuts(saved?.cuts || []);

        try {
            const res = await APIService.getReIDData(video.id, inputPath || undefined);
            setReidData(res.data);
        } catch (error) {
            console.error("Erro ao carregar dados do video", error);
        }
    };

    const saveToBatch = (videoId: string) => {
        setBatchAnnotations(prev => ({
            ...prev,
            [videoId]: {
                video_id: videoId,
                rules: swaps,
                deletions: deletions,
                cuts: cuts
            }
        }));
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
    const handleApplyBatch = async () => {
        // Save current one first
        if (selectedVideo) {
            const currentAnnotations = {
                video_id: selectedVideo.id,
                rules: swaps,
                deletions: deletions,
                cuts: cuts
            };
            const updatedBatch = { ...batchAnnotations, [selectedVideo.id]: currentAnnotations };

            const list = Object.values(updatedBatch).filter((v: any) =>
                v.rules.length > 0 || v.deletions.length > 0 || v.cuts.length > 0
            );

            if (list.length === 0) {
                alert("Nenhuma anotação pendente para salvar.");
                return;
            }

            if (!confirm(`Deseja processar ${list.length} vídeos agora? Saída em: ${outputPath}`)) return;

            setLoading(true);
            try {
                await APIService.batchApplyReID({
                    videos: list,
                    root_path: inputPath,
                    output_path: outputPath
                });
                alert("Processamento em lote iniciado no backend!");
                setBatchAnnotations({}); // Clear after starting
            } catch (error) {
                console.error(error);
                alert('Erro ao iniciar processamento em lote.');
            } finally {
                setLoading(false);
            }
        }
    };

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col space-y-4">
            {/* Barra de Configuração Compacta */}
            <div className="bg-card border border-border rounded-xl p-4">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                            <ScanFace className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                            <h1 className="text-lg font-bold">Re-identificação Manual</h1>
                            <p className="text-xs text-muted-foreground">Corrija IDs, remova ruídos e corte trechos.</p>
                        </div>
                    </div>

                    <div className="flex-1" />

                    <div className="flex items-end gap-3">
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase font-bold text-muted-foreground">Entrada</label>
                            <div className="flex gap-1">
                                <input
                                    value={inputPath}
                                    onChange={(e) => setInputPath(e.target.value)}
                                    className="w-48 bg-secondary/50 border border-border text-xs py-2 px-3 rounded-lg"
                                    placeholder="resultados-processamentos/"
                                />
                                <button
                                    onClick={() => setExplorerTarget('input')}
                                    className="p-2 bg-secondary border border-border rounded-lg hover:bg-secondary/80"
                                    title="Selecionar Pasta"
                                >
                                    <FolderInput className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase font-bold text-muted-foreground">Saída</label>
                            <div className="flex gap-1">
                                <input
                                    value={outputPath}
                                    onChange={(e) => setOutputPath(e.target.value)}
                                    className="w-48 bg-secondary/50 border border-border text-xs py-2 px-3 rounded-lg"
                                    placeholder="resultados-reidentificacoes/"
                                />
                                <button
                                    onClick={() => setExplorerTarget('output')}
                                    className="p-2 bg-secondary border border-border rounded-lg hover:bg-secondary/80"
                                    title="Selecionar Pasta"
                                >
                                    <FolderOutput className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                        <button
                            onClick={loadVideos}
                            className="p-2 bg-primary/10 text-primary rounded-lg hover:bg-primary/20"
                            title="Recarregar Vídeos"
                        >
                            <RotateCcw className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>

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
                                    src={`http://localhost:8000/reid/video/${selectedVideo.id}?root_path=${encodeURIComponent(inputPath)}`}
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

                            <div className="pt-4 pb-10 flex gap-4">
                                <button
                                    onClick={() => selectedVideo && saveToBatch(selectedVideo.id)}
                                    className="flex-1 py-4 bg-secondary text-foreground font-semibold rounded-xl hover:bg-muted transition-colors flex items-center justify-center gap-2"
                                >
                                    Agendar para Lote
                                </button>
                                <button
                                    onClick={handleApplyBatch}
                                    disabled={loading}
                                    className="flex-[2] py-4 bg-primary text-primary-foreground font-bold rounded-xl hover:brightness-110 flex items-center justify-center gap-2 shadow-lg shadow-primary/20"
                                >
                                    <Save className="w-5 h-5" />
                                    {loading ? 'Processando Lote...' : 'Salvar Reidentificações (Lote)'}
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-50 space-y-4">
                            <ScanFace className="w-20 h-20" />
                            <div className="text-center">
                                <p className="text-xl font-semibold">Nenhum vídeo selecionado</p>
                                <p className="text-sm">Selecione um vídeo na lateral para iniciar as anotações.</p>
                            </div>
                            {Object.keys(batchAnnotations).length > 0 && (
                                <button
                                    onClick={handleApplyBatch}
                                    className="mt-8 px-6 py-3 bg-primary/20 text-primary border border-primary/30 rounded-lg hover:bg-primary/30 transition-all font-bold"
                                >
                                    Processar {Object.keys(batchAnnotations).length} vídeos pendentes
                                </button>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* File Explorer Modal */}
            <FileExplorerModal
                isOpen={explorerTarget !== null}
                onClose={() => setExplorerTarget(null)}
                onSelect={(path) => {
                    if (explorerTarget === 'input') {
                        setInputPath(path);
                        loadVideos();
                    } else if (explorerTarget === 'output') {
                        setOutputPath(path);
                    }
                    setExplorerTarget(null);
                }}
                initialPath={explorerTarget === 'input' ? inputPath : outputPath}
                rootPath={explorerTarget === 'input' ? roots.processamentos : roots.reidentificacoes}
                title={explorerTarget === 'input' ? 'Selecionar Pasta de Entrada' : 'Selecionar Pasta de Saída'}
            />
        </div>
    );
}
