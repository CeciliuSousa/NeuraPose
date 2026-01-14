import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

/**
 * Encurta caminhos absolutos para mostrar apenas a partir de neurapose_backend ou diretório pai
 */
export function shortenPath(fullPath: string): string {
    if (!fullPath) return '';

    // Normaliza barras
    const normalized = fullPath.replace(/\\/g, '/');

    // Tenta encontrar marcadores conhecidos
    const markers = ['neurapose_backend', 'neurapose_tauri', 'neurapose-app'];

    for (const marker of markers) {
        const index = normalized.toLowerCase().indexOf(marker.toLowerCase());
        if (index !== -1) {
            return normalized.substring(index);
        }
    }

    // Fallback: últimas 2 pastas
    const parts = normalized.split('/');
    if (parts.length > 2) {
        return '.../' + parts.slice(-2).join('/');
    }

    return normalized;
}
