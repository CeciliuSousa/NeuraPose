import { invoke } from '@tauri-apps/api/core';

// =========================================================================
// TAURI COMMANDS WRAPPER
// =========================================================================
// Substitui o axios por chamadas diretas ao Rust ("request_python")
// Isso evita o overhead do browser network stack e problemas de CORS.

interface ApiResponse<T> {
    status: number;
    data: T;
    headers: Record<string, string>;
}

async function request<T>(method: string, path: string, body?: any, headers?: Record<string, string>): Promise<{ data: T, status: number, headers: any }> {
    try {
        const response = await invoke<ApiResponse<T>>('request_python', {
            options: {
                method,
                path,
                body: body || null,
                headers: headers || null
            }
        });

        if (response.status >= 400) {
            throw new Error(`Request failed with status ${response.status}: ${JSON.stringify(response.data)}`);
        }

        return {
            data: response.data,
            status: response.status,
            headers: response.headers
        };
    } catch (e) {
        console.error(`[ApiError] ${method} ${path}`, e);
        throw e;
    }
}

// Wrapper style-axios para manter compatibilidade com o resto do código
const api = {
    get: async <T>(path: string, config?: { params?: any }) => {
        let fullPath = path;
        if (config?.params) {
            const qs = new URLSearchParams(config.params).toString();
            if (qs) fullPath += `?${qs}`;
        }
        const res = await request<T>('GET', fullPath);
        return res;
    },
    post: async <T>(path: string, data?: any, config?: { params?: any }) => {
        let fullPath = path;
        if (config?.params) {
            const qs = new URLSearchParams(config.params).toString();
            if (qs) fullPath += `?${qs}`;
        }
        const res = await request<T>('POST', fullPath, data);
        return res;
    },
    put: async <T>(path: string, data?: any) => request<T>('PUT', path, data),
    delete: async <T>(path: string, config?: { params?: any }) => {
        let fullPath = path;
        if (config?.params) {
            const qs = new URLSearchParams(config.params).toString();
            if (qs) fullPath += `?${qs}`;
        }
        const res = await request<T>('DELETE', fullPath);
        return res;
    }
};

export interface ProcessRequest {
    input_path: string;
    dataset_name?: string;
    onnx_path?: string;
    show_preview?: boolean;
    device?: string;
}

export interface TrainRequest {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    model_name: string;
    dataset_name: string;
    temporal_model: string;
}

export interface AnnotationRequest {
    video_stem: string;
    annotations: Record<string, string>;
    root_path: string;  // Pasta raiz do dataset
}

export interface SplitRequest {
    input_dir_process: string;
    dataset_name: string;
    output_root?: string;
    train_split?: string;
    test_split?: string;
    train_ratio?: number; // Porcentagem de treino (0.0 a 1.0)
}

export interface TestRequest {
    model_path?: string;
    dataset_path?: string;
    device?: string;
    show_preview?: boolean;
}

export interface ConvertRequest {
    dataset_path: string;  // Caminho do dataset (datasets/<nome>)
    extension?: string;    // Extensão de saída (.pt, .pth)
    output_name?: string;  // Nome do dataset de saída (opcional, para criar cópia/fork)
}

export interface BrowseResponse {
    current: string;
    parent: string;
    items: {
        name: string;
        path: string;
        is_dir: boolean;
    }[];
}

export interface ReIDVideo {
    id: string;
    json_path: string;
    video_path: string | null;
    processed: boolean;
}

export interface ReIDData {
    video_id: string;
    frames: Record<string, { bbox: number[], id: number }[]>;
    id_counts: Record<string, number>;
}

export const APIService = {
    healthCheck: () => api.get<any>('/health'),
    getConfig: () => api.get('/config'),
    getSystemInfo: () => api.get<any>('/system/info'),

    browse: (path: string) => api.get<BrowseResponse>(`/browse?path=${encodeURIComponent(path)}`),

    startProcessing: (data: ProcessRequest) => api.post('/process', data),

    startTraining: (data: any) => api.post('/train/start', data),
    retrainTraining: (data: any) => api.post('/train/retrain', data),

    listReIDVideos: (rootPath?: string) => api.get<{ videos: ReIDVideo[] }>(`/reid/list`, { params: { root_path: rootPath } }),

    getReIDData: (videoId: string, rootPath?: string) => api.get<ReIDData>(`/reid/${videoId}/data`, { params: { root_path: rootPath } }),

    applyReIDChanges: (videoId: string, data: any, rootPath?: string) => api.post(`/reid/${videoId}/apply`, data, { params: { root_path: rootPath } }),

    getLogs: (category: string = 'default') => api.get<{ logs: string[] }>('/logs', { params: { category } }),
    clearLogs: (category?: string) => api.delete('/logs', { params: { category } }),
    stopTraining: () => api.post('/train/stop'),
    stopTesting: () => api.post('/test/stop'),

    pickFolder: (initialDir?: string) => api.get<{ path: string | null }>('/pick-folder', { params: { initial_dir: initialDir } }),

    // Configuração
    getAllConfig: () => api.get<any>('/config/all'),
    updateConfig: (updates: any) => api.post('/config/update', updates),
    resetConfig: () => api.post('/config/reset'),

    // Controles de Sistema e Processamento
    stopProcess: () => api.post('/process/stop'),
    pauseProcess: () => api.post('/process/pause'),
    resumeProcess: () => api.post('/process/resume'),
    shutdown: () => api.post('/shutdown'),

    // Preview dinâmico
    togglePreview: (enabled: boolean) => api.post('/preview/toggle', null, { params: { enabled } }),
    getPreviewStatus: () => api.get<{ show_preview: boolean, has_frame: boolean }>('/preview/status'),

    // ML Features
    listAnnotationVideos: (rootPath?: string) => api.get<{ videos: any[] }>('/annotate/list', { params: { root_path: rootPath } }),
    getAnnotationDetails: (videoId: string, rootPath?: string) => api.get<any>(`/annotate/${videoId}/details`, { params: { root_path: rootPath } }),
    saveAnnotations: (data: AnnotationRequest) => api.post('/annotate/save', data),

    splitDataset: (data: SplitRequest) => api.post('/dataset/split', data),

    startTesting: (data: TestRequest) => api.post('/test', data),

    // ReID Agenda (persistência em arquivo JSON)
    getReidAgenda: (rootPath?: string) => api.get<{ agenda: any, stats?: any }>('/reid/agenda', { params: { root_path: rootPath } }),
    saveToReidAgenda: (sourceDataset: string, video: any) => api.post('/reid/agenda/save', { source_dataset: sourceDataset, video }),
    removeFromReidAgenda: (videoId: string, rootPath?: string) => api.delete(`/reid/agenda/${videoId}`, { params: { root_path: rootPath } }),

    batchApplyReID: (data: { videos: any[], root_path?: string, output_path?: string }) => api.post('/reid/batch-apply', data),

    // Datasets Manager
    listAllDatasets: () => api.get<{ status: string, data: any }>('/datasets/list'),

    // Conversão de Dataset para .pt
    convertDataset: (data: ConvertRequest) => api.post('/convert/pt', data),
};

export default api;
