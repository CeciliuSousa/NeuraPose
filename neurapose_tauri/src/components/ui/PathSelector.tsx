import { FolderInput, Video } from 'lucide-react';

interface PathSelectorProps {
    value: string;
    onSelect: () => void;
    placeholder?: string;
    label?: string;
    icon?: React.ElementType; // Ícone opcional para substituir o Video padrão
    readOnly?: boolean;
}

export function PathSelector({
    value,
    onSelect,
    placeholder = "Selecione o diretório...",
    label,
    icon: Icon = Video,
    readOnly = true
}: PathSelectorProps) {

    // Exibe o valor completo (path relativo) ao invés de truncar para só o nome do arquivo
    const displayValue = value ? value.replace(/\\/g, '/') : '';

    return (
        <div className="space-y-2">
            {label && <label className="text-sm font-medium text-muted-foreground">{label}</label>}
            <div className="flex gap-2">
                <div className="flex-1 relative group">
                    <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground group-hover:text-primary transition-colors">
                        <Icon className="w-4 h-4" />
                    </div>
                    <input
                        type="text"
                        value={displayValue}
                        readOnly={readOnly}
                        title={value} // Tooltip com o caminho completo
                        placeholder={placeholder}
                        className="w-full pl-9 bg-background border border-border rounded-md py-2 text-sm outline-none focus:ring-2 focus:ring-primary/50 transition-all font-mono cursor-pointer truncate hover:border-primary/50"
                        onClick={onSelect}
                    />
                </div>
                <button
                    onClick={onSelect}
                    className="px-3 py-2 bg-secondary rounded-md border border-border hover:bg-secondary/80 transition-colors flex items-center justify-center hover:border-primary/50 group"
                    title="Abrir explorador"
                >
                    <FolderInput className="w-4 h-4 group-hover:text-primary transition-colors" />
                </button>
            </div>
        </div>
    );
}
