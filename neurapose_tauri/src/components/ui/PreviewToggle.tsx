import { APIService } from '../../services/api';

interface PreviewToggleProps {
    checked: boolean;
    onChange: (value: boolean) => void;
    isProcessing?: boolean;  // Se true, chama API para toggle dinâmico
    disabled?: boolean;
}

export function PreviewToggle({ checked, onChange, isProcessing = false, disabled = false }: PreviewToggleProps) {

    const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.checked;
        onChange(newValue);

        // Se já está processando, atualiza no backend dinamicamente
        if (isProcessing) {
            try {
                await APIService.togglePreview(newValue);
            } catch (err) {
                console.error("Erro ao toggle preview:", err);
            }
        }
    };

    return (
        <div className="pt-2">
            <label className={`flex items-center gap-3 cursor-pointer group ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
                <div className="relative">
                    <input
                        type="checkbox"
                        checked={checked}
                        onChange={handleChange}
                        disabled={disabled}
                        className="sr-only peer"
                    />
                    {/* Toggle Background: Cinza escuro OFF, Verde ON */}
                    <div className={`w-12 h-6 rounded-full transition-colors ${checked ? 'bg-green-500' : 'bg-gray-600'}`}></div>
                    {/* Toggle Circle */}
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all shadow-md ${checked ? 'left-7' : 'left-1'}`}></div>
                </div>
                <span className={`text-sm font-medium transition-colors ${checked ? 'text-green-400' : 'text-muted-foreground'}`}>
                    {checked ? '✓ Preview Ativado' : 'Preview Desativado'}
                </span>
            </label>
        </div>
    );
}
