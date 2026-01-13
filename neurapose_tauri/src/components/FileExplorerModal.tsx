import { useState, useEffect } from 'react';
import { Folder, File, ChevronLeft, X, Check } from 'lucide-react';
import { APIService, BrowseResponse } from '../services/api';

interface FileExplorerModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSelect: (path: string) => void;
    initialPath?: string;
    rootPath?: string;
    title?: string;
}

export function FileExplorerModal({ isOpen, onClose, onSelect, initialPath, rootPath, title = "Selecionar Pasta" }: FileExplorerModalProps) {
    const [currentPath, setCurrentPath] = useState<string>(initialPath || '');
    const [data, setData] = useState<BrowseResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (isOpen) {
            if (rootPath && (!currentPath || !currentPath.startsWith(rootPath))) {
                loadPath(rootPath);
            } else if (!currentPath) {
                APIService.getConfig().then(res => {
                    const defaultPath = res.data.paths?.videos || 'C:\\';
                    loadPath(defaultPath);
                }).catch(() => loadPath('C:\\'));
            } else {
                loadPath(currentPath);
            }
        }
    }, [isOpen]);

    const loadPath = async (path: string) => {
        if (rootPath && !path.startsWith(rootPath)) {
            path = rootPath;
        }

        setLoading(true);
        setError(null);
        try {
            const res = await APIService.browse(path);
            setData(res.data);
            setCurrentPath(res.data.current);
        } catch (err: any) {
            setError(err.response?.data?.detail || "Erro ao carregar pasta");
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    const isAtRoot = rootPath ? currentPath === rootPath || currentPath === rootPath + '\\' || currentPath === rootPath + '/' : false;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
            <div className="bg-card border border-border rounded-xl w-full max-w-2xl flex flex-col max-h-[80vh] shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-border">
                    <h3 className="font-semibold text-lg">{title}</h3>
                    <button onClick={onClose} className="p-1 hover:bg-secondary rounded-md transition-colors">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Path Bar */}
                <div className="p-3 bg-muted/30 flex items-center gap-2 border-b border-border">
                    <button
                        onClick={() => data?.parent && loadPath(data.parent)}
                        disabled={!data?.parent || data.parent === data.current || isAtRoot}
                        className="p-1.5 hover:bg-secondary rounded-md disabled:opacity-30"
                    >
                        <ChevronLeft className="w-4 h-4" />
                    </button>
                    <div className="flex-1 text-xs font-mono bg-background/50 border border-border px-2 py-1.5 rounded truncate">
                        {currentPath}
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-2 min-h-[300px]">
                    {loading ? (
                        <div className="flex items-center justify-center h-full text-muted-foreground animate-pulse">
                            Carregando arquivos...
                        </div>
                    ) : error ? (
                        <div className="flex items-center justify-center h-full text-red-400 p-4 text-center">
                            {error}
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-1">
                            {data?.items.map((item) => (
                                <button
                                    key={item.path}
                                    onClick={() => item.is_dir ? loadPath(item.path) : onSelect(item.path)}
                                    className={`
                                        flex items-center gap-3 p-2.5 rounded-md text-sm text-left transition-colors
                                        hover:bg-primary/10 cursor-pointer
                                    `}
                                >
                                    {item.is_dir ? (
                                        <Folder className="w-4 h-4 text-blue-400 fill-blue-400/20" />
                                    ) : (
                                        <File className="w-4 h-4 text-emerald-400" />
                                    )}
                                    <span className="truncate flex-1 font-mono text-xs">{item.name}</span>
                                </button>

                            ))}
                            {data?.items.length === 0 && (
                                <div className="text-center py-10 text-muted-foreground italic">
                                    Pasta vazia
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-border flex justify-end items-center bg-muted/10">
                    <div className="flex gap-3">

                        <button
                            onClick={onClose}
                            className="px-4 py-2 text-sm font-medium hover:bg-secondary rounded-md transition-colors"
                        >
                            Cancelar
                        </button>
                        <button
                            onClick={() => onSelect(currentPath)}
                            className="px-4 py-2 text-sm font-medium bg-primary text-primary-foreground hover:brightness-110 rounded-md transition-all flex items-center gap-2 shadow-lg shadow-primary/20"
                        >
                            <Check className="w-4 h-4" />
                            Confirmar Seleção
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
