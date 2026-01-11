import { useState, useEffect } from 'react';
import {
    Tag,
    Video as VideoIcon,
    CheckCircle2,
    AlertCircle,
    Save,
    ChevronRight,
    Play,
    Info
} from 'lucide-react';
import { APIService } from '@/services/api';
import { PageHeader } from '@/components/ui/page-header';

export default function AnnotationPage() {
    const [videos, setVideos] = useState<any[]>([]);
    const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
    const [details, setDetails] = useState<any>(null);
    const [annotations, setAnnotations] = useState<Record<string, string>>({});
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState('');

    useEffect(() => {
        loadVideos();
    }, []);

    const loadVideos = async () => {
        setLoading(true);
        try {
            const res = await APIService.listAnnotationVideos();
            setVideos(res.data.videos);
        } catch (err) {
            console.error(err);
            setMessage("Erro ao carregar lista de vídeos");
        } finally {
            setLoading(false);
        }
    };

    const loadDetails = async (videoId: string) => {
        setSelectedVideo(videoId);
        setDetails(null);
        setAnnotations({});
        try {
            const res = await APIService.getAnnotationDetails(videoId);
            setDetails(res.data);
            // Inicializa anotações se já existirem (fictício por enquanto, o backend retorna 'desconhecido')
            const initial: Record<string, string> = {};
            res.data.ids.forEach((item: any) => {
                initial[item.id] = item.label !== 'desconhecido' ? item.label : 'NORMAL';
            });
            setAnnotations(initial);
        } catch (err) {
            console.error(err);
            setMessage("Erro ao carregar detalhes do vídeo");
        }
    };

    const handleClassChange = (id: string, label: string) => {
        setAnnotations(prev => ({ ...prev, [id]: label }));
    };

    const handleSave = async () => {
        if (!selectedVideo) return;
        setSaving(true);
        try {
            await APIService.saveAnnotations({
                video_stem: selectedVideo,
                annotations: annotations
            });
            setMessage("Anotações salvas com sucesso!");
            loadVideos(); // Refresh list to update status
            setTimeout(() => setMessage(''), 3000);
        } catch (err) {
            console.error(err);
            setMessage("Erro ao salvar anotações");
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Anotação de Classes"
                description="Classifique os indivíduos detectados (NORMAL ou FURTO) para treinar o modelo de detecção de anomalias."
            />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                {/* 1. Lista de Vídeos (Esquerda) */}
                <div className="lg:col-span-3 space-y-4">
                    <div className="bg-card border border-border rounded-xl overflow-hidden shadow-sm">
                        <div className="px-4 py-3 bg-muted/30 border-b border-border font-semibold text-sm flex items-center justify-between">
                            Vídeos Disponíveis
                            <span className="text-[10px] bg-primary/20 text-primary px-2 py-0.5 rounded-full">{videos.length}</span>
                        </div>
                        <div className="max-h-[600px] overflow-y-auto">
                            {loading ? (
                                <div className="p-8 text-center text-muted-foreground animate-pulse text-xs">Carregando...</div>
                            ) : videos.length === 0 ? (
                                <div className="p-8 text-center text-muted-foreground italic text-xs">Nenhum vídeo processado.</div>
                            ) : (
                                <div className="divide-y divide-border">
                                    {videos.map((v) => (
                                        <button
                                            key={v.video_id}
                                            onClick={() => loadDetails(v.video_id)}
                                            className={`w-full text-left px-4 py-3 hover:bg-muted/50 transition-all flex items-center justify-between group ${selectedVideo === v.video_id ? 'bg-primary/10 border-l-4 border-primary' : 'border-l-4 border-transparent'}`}
                                        >
                                            <div className="truncate pr-2">
                                                <div className="text-xs font-bold truncate group-hover:text-primary transition-colors">{v.video_id}</div>
                                                <div className="text-[10px] text-muted-foreground">{v.status === 'anotado' ? '✅ ROLUTADO' : '⏳ PENDENTE'}</div>
                                            </div>
                                            <ChevronRight className={`w-4 h-4 text-muted-foreground transition-transform ${selectedVideo === v.video_id ? 'translate-x-1 text-primary' : ''}`} />
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* 2. Visualização e Anotação (Direita) */}
                <div className="lg:col-span-9">
                    {!selectedVideo ? (
                        <div className="h-full min-h-[400px] border-2 border-dashed border-border rounded-xl flex flex-col items-center justify-center text-muted-foreground p-12 bg-muted/5">
                            <VideoIcon className="w-16 h-16 mb-4 opacity-20" />
                            <p className="text-lg font-medium">Selecione um vídeo para começar a anotar</p>
                            <p className="text-sm opacity-60">Apenas vídeos processados com Re-ID aparecem aqui.</p>
                        </div>
                    ) : (
                        <div className="space-y-6 animate-in fade-in duration-500">

                            {/* Video Player */}
                            <div className="bg-card border border-border rounded-xl overflow-hidden shadow-lg shadow-black/20">
                                <div className="px-4 py-2 bg-slate-900 text-white flex items-center justify-between">
                                    <span className="text-xs font-mono truncate">{selectedVideo}</span>
                                    <div className="flex gap-2">
                                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                        <span className="text-[10px] uppercase font-bold text-slate-400">Preview Mode</span>
                                    </div>
                                </div>
                                <div className="aspect-video bg-black flex items-center justify-center">
                                    <img
                                        src={`/api/reid/video/${selectedVideo}`}
                                        alt="Video Stream"
                                        className="max-h-full max-w-full object-contain"
                                        onError={(e: any) => e.target.src = "https://placehold.co/1280x720/000000/FFFFFF?text=Video+Not+Found"}
                                    />
                                </div>
                            </div>

                            {/* Annotation Form */}
                            <div className="bg-card border border-border rounded-xl shadow-md">
                                <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                                    <h3 className="font-bold flex items-center gap-2">
                                        <Tag className="w-5 h-5 text-primary" />
                                        Classificação de Indivíduos
                                    </h3>
                                    <button
                                        onClick={handleSave}
                                        disabled={saving || !details}
                                        className="px-6 py-2 bg-primary text-primary-foreground rounded-lg font-bold text-sm hover:brightness-110 active:scale-95 transition-all disabled:opacity-50 flex items-center gap-2 shadow-lg shadow-primary/20"
                                    >
                                        <Save className="w-4 h-4" />
                                        {saving ? 'Salvando...' : 'Salvar Anotações'}
                                    </button>
                                </div>

                                <div className="p-6">
                                    {message && (
                                        <div className={`mb-6 p-4 rounded-lg flex items-center gap-3 text-sm font-medium border ${message.includes('sucesso') ? 'bg-green-500/10 text-green-500 border-green-500/20' : 'bg-red-500/10 text-red-500 border-red-500/20'}`}>
                                            {message.includes('sucesso') ? <CheckCircle2 className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                                            {message}
                                        </div>
                                    )}

                                    {!details ? (
                                        <div className="flex flex-col items-center py-12 text-muted-foreground">
                                            <div className="animate-spin mb-4">
                                                <Info className="w-8 h-8" />
                                            </div>
                                            <p>Carregando dados dos indivíduos...</p>
                                        </div>
                                    ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                                            {details.ids.map((item: any) => (
                                                <div key={item.id} className="p-4 border border-border rounded-xl bg-muted/10 hover:bg-muted/30 transition-colors flex flex-col gap-3">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-lg font-extrabold text-primary">ID {item.id}</span>
                                                        <span className="text-[10px] bg-secondary px-2 py-0.5 rounded text-secondary-foreground font-mono">{item.frames} frames</span>
                                                    </div>

                                                    <div className="flex gap-2 p-1 bg-background border border-border rounded-lg">
                                                        <button
                                                            onClick={() => handleClassChange(item.id, 'NORMAL')}
                                                            className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all ${annotations[item.id] === 'NORMAL' ? 'bg-green-500 text-white shadow-sm' : 'hover:bg-muted text-muted-foreground'}`}
                                                        >
                                                            NORMAL
                                                        </button>
                                                        <button
                                                            onClick={() => handleClassChange(item.id, 'FURTO')}
                                                            className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all ${annotations[item.id] === 'FURTO' ? 'bg-red-500 text-white shadow-sm' : 'hover:bg-muted text-muted-foreground'}`}
                                                        >
                                                            FURTO
                                                        </button>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="bg-primary/5 border border-primary/10 p-4 rounded-xl text-[10px] text-muted-foreground leading-relaxed flex gap-3 italic">
                                <Info className="w-4 h-4 text-primary shrink-0" />
                                <div>
                                    <strong>Ajuda:</strong> Assista ao vídeo acima para identificar qual indivíduo (ID) está cometendo a infração.
                                    Apenas IDs com mais de {details?.min_frames || 30} frames são listados para evitar ruído.
                                    As anotações serão usadas para treinar a camada temporal (LSTM/TFT).
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

