import { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

export interface FilterOption {
    key: string;
    label: string;
    count: number;
    color?: string; // Cor do indicador (green-500, yellow-500, etc)
}

interface FilterDropdownProps {
    options: FilterOption[];
    selected: string;
    onSelect: (key: string) => void;
    placeholder?: string;
}

export function FilterDropdown({ options, selected, onSelect, placeholder = "Filtrar" }: FilterDropdownProps) {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Fecha dropdown ao clicar fora
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const selectedOption = options.find(o => o.key === selected);

    // Calcula estilos baseados na cor da opção selecionada
    const getButtonStyles = () => {
        if (!selectedOption || !selectedOption.color) {
            return 'bg-muted/30 border-border text-muted-foreground hover:bg-muted hover:text-foreground';
        }
        // Tailwind não suporta interpolação dinâmica confiável sem safelist, 
        // mas como as cores são padrão (green-500, red-500, yellow-500) usadas em outros lugares, deve funcionar.
        // Mapeamento explícito para segurança
        const color = selectedOption.color;
        if (color.includes('green')) return 'bg-green-500/10 border-green-500 text-green-500';
        if (color.includes('yellow')) return 'bg-yellow-500/10 border-yellow-500 text-yellow-500';
        if (color.includes('red')) return 'bg-red-500/10 border-red-500 text-red-500';
        if (color.includes('blue')) return 'bg-blue-500/10 border-blue-500 text-blue-500';

        // Fallback genérico se a cor for passada mas não mapeada acima
        return `bg-${color}/10 border-${color} text-${color}`;
    };

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`
                    w-full flex items-center justify-between p-3 rounded-xl border-2 transition-all group 
                    ${getButtonStyles()}
                `}
            >
                <div className="flex flex-1 items-center justify-between mr-4">
                    <span className="text-xs font-bold uppercase tracking-wider opacity-90 truncate text-left">
                        {selectedOption ? selectedOption.label : placeholder}
                    </span>
                    <span className="text-2xl font-black leading-none ml-2">
                        {selectedOption ? selectedOption.count : ''}
                    </span>
                </div>
                <ChevronDown className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 p-1.5 bg-popover/95 backdrop-blur-xl border border-border rounded-xl shadow-2xl z-50 space-y-1 animate-in fade-in slide-in-from-top-2">
                    {options.map(option => {
                        // Classes de hover dinâmicas
                        let hoverClass = 'hover:bg-muted';
                        if (option.color?.includes('green')) hoverClass = 'hover:bg-green-500/10 hover:text-green-500';
                        else if (option.color?.includes('yellow')) hoverClass = 'hover:bg-yellow-500/10 hover:text-yellow-500';
                        else if (option.color?.includes('red')) hoverClass = 'hover:bg-red-500/10 hover:text-red-500';
                        else if (option.color?.includes('blue')) hoverClass = 'hover:bg-blue-500/10 hover:text-blue-500';

                        return (
                            <button
                                key={option.key}
                                onClick={() => { onSelect(option.key); setIsOpen(false); }}
                                className={`w-full flex items-center justify-between p-2 rounded-lg transition-colors text-left ${hoverClass}`}
                            >
                                <div className="flex items-center gap-2">
                                    {option.color && (
                                        <div className={`w-2 h-2 rounded-full bg-${option.color} shadow-sm`} />
                                    )}
                                    <span className="text-sm font-medium">{option.label}</span>
                                </div>
                                <span className="text-xs font-mono font-bold">{option.count}</span>
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
