import { useState, useEffect } from 'react';
import {
    History,
    Video,
    ScanFace,
    Tag,
    Dumbbell,
    Search,
    Activity,
    RefreshCcw,
    FolderOpen,
    Trash2
} from 'lucide-react';
import { PageHeader } from '../components/ui/PageHeader';
import { APIService } from '../services/api';
import { StatusMessage } from '../components/ui/StatusMessage';

interface HistoryItem {
    id: string;
    name: string;
    path: string;
    type: 'processamento' | 'reid' | 'anotacao' | 'modelo';
    date?: string;
}

export default function HistoricoPage() {
    const [items, setItems] = useState<HistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('');
    const [activeTab, setActiveTab] = useState<'todos' | 'processamento' | 'reid' | 'anotacao' | 'modelo'>('todos');
    const [message, setMessage] = useState<{ text: string, type: 'success' | 'error' | 'info' } | null>(null);

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const configRes = await APIService.getConfig();
            if (configRes.data.status !== 'success') throw new Error('Falha ao carregar config');

            const paths = configRes.data.paths;
            const allItems: HistoryItem[] = [];

            // 1. Processamentos (Vídeos com Pose)
            try {
                const res = await APIService.browse(paths.videos_com_pose);
                res.data.items.forEach(item => {
                    if (!item.is_dir && item.name.endsWith('.mp4')) {
                        allItems.push({
                            id: `proc_${item.name}`,
                            name: item.name,
                            path: item.path,
                            type: 'processamento'
                        });
                    }
                });
            } catch (e) { console.error('Erro ao ler processamentos', e); }

            // 2. Re-identificações (resultados-reidentificacoes)
            try {
                const res = await APIService.browse(paths.reidentificacoes);
                res.data.items.forEach(item => {
                    if (item.is_dir) {
                        allItems.push({
                            id: `reid_${item.name}`,
                            name: item.name,
                            path: item.path,
                            type: 'reid'
                        });
                    }
                });
            } catch (e) { console.error('Erro ao ler reidentificacoes', e); }

            // 3. Anotações (resultados-anotacoes)
            try {
                const res = await APIService.browse(paths.anotacoes);
                res.data.items.forEach(item => {
                    if (!item.is_dir && item.name.endsWith('.json')) {
                        allItems.push({
                            id: `annot_${item.name}`,
                            name: item.name,
                            path: item.path,
                            type: 'anotacao'
                        });
                    }
                });
            } catch (e) { console.error('Erro ao ler anotações', e); }

            // 4. Modelos (modelos-lstm-treinados)
            try {
                const res = await APIService.browse(paths.modelos);
                res.data.items.forEach(item => {
                    if (!item.is_dir && (item.name.endsWith('.pt') || item.name.endsWith('.pth'))) {
                        allItems.push({
                            id: `model_${item.name}`,
                            name: item.name,
                            path: item.path,
                            type: 'modelo'
                        });
                    }
                });
            } catch (e) { console.error('Erro ao ler modelos', e); }

            setItems(allItems);
        } catch (err) {
            setMessage({ text: 'Falha ao carregar histórico.', type: 'error' });
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchHistory();
    }, []);

    const filteredItems = items.filter(item => {
        const matchesSearch = item.name.toLowerCase().includes(filter.toLowerCase());
        const matchesTab = activeTab === 'todos' || item.type === activeTab;
        return matchesSearch && matchesTab;
    });

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'processamento': return <Video className="w-4 h-4 text-blue-500" />;
            case 'reid': return <ScanFace className="w-4 h-4 text-indigo-500" />;
            case 'anotacao': return <Tag className="w-4 h-4 text-cyan-500" />;
            case 'modelo': return <Dumbbell className="w-4 h-4 text-emerald-500" />;
            default: return <History className="w-4 h-4" />;
        }
    };

    return (
        <div className="space-y-6 h-full flex flex-col pb-8">
            <PageHeader
                title="Histórico Global"
                description="Gerencie todos os artefatos gerados pelo NeuraPose"
                icon={History}
            >
                <button
                    onClick={fetchHistory}
                    disabled={loading}
                    className="p-3 hover:bg-muted rounded-xl transition-all border border-transparent hover:border-border active:scale-95"
                    title="Atualizar Histórico"
                >
                    <RefreshCcw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                </button>
            </PageHeader>

            {/* Filtros e Tabs */}
            <div className="flex flex-col md:flex-row gap-4 justify-between items-center bg-card/30 p-4 rounded-2xl border border-border mt-2">
                <div className="flex gap-2 p-1 bg-muted rounded-xl overflow-x-auto w-full md:w-auto">
                    {(['todos', 'processamento', 'reid', 'anotacao', 'modelo'] as const).map(tab => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`
                                px-4 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all whitespace-nowrap
                                ${activeTab === tab ? 'bg-card text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground'}
                            `}
                        >
                            {tab}
                        </button>
                    ))}
                </div>

                <div className="relative w-full md:w-64">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <input
                        type="text"
                        placeholder="Buscar por nome..."
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 bg-card rounded-xl border border-border text-sm focus:border-primary/50 outline-none transition-all"
                    />
                </div>
            </div>

            {/* Listagem */}
            <div className="flex-1 overflow-y-auto min-h-0 pr-2 custom-scrollbar">
                {loading ? (
                    <div className="flex flex-col items-center justify-center h-64 space-y-4">
                        <div className="relative">
                            <div className="w-12 h-12 border-4 border-primary/20 rounded-full animate-pulse" />
                            <div className="absolute inset-0 w-12 h-12 border-4 border-t-primary rounded-full animate-spin" />
                        </div>
                        <p className="text-sm font-medium text-muted-foreground animate-pulse tracking-widest uppercase">Mapeando diretórios...</p>
                    </div>
                ) : filteredItems.length > 0 ? (
                    <div className="grid gap-3">
                        {filteredItems.map(item => (
                            <div
                                key={item.id}
                                className="group flex items-center justify-between p-4 bg-card/40 border border-border rounded-2xl hover:border-primary/30 hover:bg-card/60 transition-all"
                            >
                                <div className="flex items-center gap-4">
                                    <div className={`p-2.5 rounded-xl bg-muted group-hover:bg-primary/5 transition-colors`}>
                                        {getTypeIcon(item.type)}
                                    </div>
                                    <div className="min-w-0">
                                        <h4 className="font-bold text-sm truncate max-w-[400px]" title={item.name}>
                                            {item.name}
                                        </h4>
                                        <p className="text-[10px] text-muted-foreground font-mono truncate" title={item.path}>
                                            {item.path}
                                        </p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-all">
                                    <button
                                        className="p-2 hover:bg-primary/10 rounded-lg text-primary transition-colors flex items-center gap-2 text-xs font-bold"
                                        title="Abrir no Explorer"
                                        onClick={() => {
                                            // Mock de abertura
                                            setMessage({ text: `📍 Localização: ${item.path}`, type: 'info' });
                                        }}
                                    >
                                        <FolderOpen className="w-4 h-4" />
                                        EXPLORER
                                    </button>
                                    <button
                                        className="p-2 hover:bg-red-500/10 rounded-lg text-muted-foreground hover:text-red-500 transition-colors"
                                        title="Remover (não implementado)"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-64 text-center">
                        <div className="p-4 bg-muted rounded-full mb-4">
                            <Activity className="w-8 h-8 text-muted-foreground" />
                        </div>
                        <h3 className="font-bold text-lg">Nenhum resultado encontrado</h3>
                        <p className="text-sm text-muted-foreground max-w-xs">Tente ajustar seus filtros ou execute novas tarefas no sistema.</p>
                    </div>
                )}
            </div>

            <StatusMessage message={message ? message.text : ''} onClose={() => setMessage(null)} type={message?.type} autoCloseDelay={4000} />
        </div>
    );
}
