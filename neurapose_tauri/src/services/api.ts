import axios from 'axios';

// Para Tauri, o backend roda localmente na porta 8000
// Nota: As rotas do backend NÃO têm prefixo /api
const api = axios.create({
    baseURL: 'http://localhost:8000',
    headers: {
        'Content-Type': 'application/json',
    },
});

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
    output_path?: string;
}

export interface SplitRequest {
    input_dir_process: string;
    dataset_name: string;
    output_root?: string;
    train_split?: string;
    test_split?: string;
}

export interface TestRequest {
    model_path?: string;
    dataset_path?: string;
    device?: string;
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

    startTraining: (data: TrainRequest) => api.post('/train', data),

    listReIDVideos: (rootPath?: string) => api.get<{ videos: ReIDVideo[] }>(`/reid/list`, { params: { root_path: rootPath } }),

    getReIDData: (videoId: string, rootPath?: string) => api.get<ReIDData>(`/reid/${videoId}/data`, { params: { root_path: rootPath } }),

    applyReIDChanges: (videoId: string, data: any, rootPath?: string) => api.post(`/reid/${videoId}/apply`, data, { params: { root_path: rootPath } }),

    getLogs: () => api.get<{ logs: string[] }>('/logs'),
    clearLogs: () => api.delete('/logs'),

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

    // ML Features
    listAnnotationVideos: (rootPath?: string) => api.get<{ videos: any[] }>('/annotate/list', { params: { root_path: rootPath } }),
    getAnnotationDetails: (videoId: string, rootPath?: string) => api.get<any>(`/annotate/${videoId}/details`, { params: { root_path: rootPath } }),
    saveAnnotations: (data: AnnotationRequest) => api.post('/annotate/save', data),

    splitDataset: (data: SplitRequest) => api.post('/dataset/split', data),

    startTesting: (data: TestRequest) => api.post('/test', data),

    batchApplyReID: (data: { videos: any[], root_path?: string, output_path?: string }) => api.post('/reid/batch-apply', data),
};

export default api;
