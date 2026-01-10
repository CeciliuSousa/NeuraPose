'use client';

import { useState, useEffect } from 'react';
import { Settings, Save, RotateCcw, ShieldAlert } from 'lucide-react';
import { APIService } from '@/services/api';

export default function ConfigPage() {
    const [config, setConfig] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState('');

    useEffect(() => {
        loadConfig();
    }, []);

    const loadConfig = async () => {
        setLoading(true);
        try {
            const res = await APIService.getAllConfig();
            setConfig(res.data);
        } catch (err) {
            console.error("Erro ao carregar colnfig:", err);
            setMessage("Erro ao carregar configurações do backend.");
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        setMessage('');
        try {
            await APIService.updateConfig(config);
            setMessage("Configurações salvas com sucesso!");
        } catch (err) {
            console.error("Erro ao salvar:", err);
            setMessage("Erro ao salvar configurações.");
        } finally {
            setSaving(false);
        }
    };

    if (loading) return <div className="flex items-center justify-center h-64">Carregando configurações...</div>;
    if (!config) return <div className="p-8 text-red-500">Erro ao conectar com o backend.</div>;

    const sections = [
        {
            title: "Modelos e Ferramentas",
            items: [
                { key: "YOLO_MODEL", label: "Modelo YOLO (Pessoas)", type: "text" },
                { key: "PROCESSING_DATASET", label: "Dataset de Processamento", type: "text" },
            ]
        },
        {
            title: "Parâmetros de Detecção e Pose",
            items: [
                { key: "DETECTION_CONF", label: "Confiança YOLO (0-1)", type: "number", step: 0.01 },
                { key: "POSE_CONF_MIN", label: "Confiança Pose Min (0-1)", type: "number", step: 0.01 },
                { key: "EMA_ALPHA", label: "Suavização EMA (0-1)", type: "number", step: 0.01 },
                { key: "FURTO_THRESHOLD", label: "Threshold Furto (0-1)", type: "number", step: 0.01 },
            ]
        },
        {
            title: "Parâmetros de Treinamento",
            items: [
                { key: "TIME_STEPS", label: "Janela Temporal (Frames)", type: "number" },
                { key: "BATCH_SIZE", label: "Batch Size", type: "number" },
                { key: "LEARNING_RATE", label: "Learning Rate", type: "number", step: 0.0001 },
                { key: "EPOCHS", label: "Épocas", type: "number" },
            ]
        }
    ];

    return (
        <div className="space-y-6 max-w-4xl mx-auto pb-12">
            <div className="flex items-center justify-between border-b border-border pb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-md">
                        <Settings className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold">Configurações Gerais</h1>
                        <p className="text-sm text-muted-foreground">Gerencie os parâmetros do config_master.py</p>
                    </div>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={loadConfig}
                        className="flex items-center gap-2 px-4 py-2 text-sm bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
                    >
                        <RotateCcw className="w-4 h-4" />
                        Resetar
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex items-center gap-2 px-6 py-2 text-sm bg-primary text-primary-foreground hover:brightness-110 rounded-md transition-all font-medium disabled:opacity-50"
                    >
                        <Save className="w-4 h-4" />
                        {saving ? 'Salvando...' : 'Salvar Alterações'}
                    </button>
                </div>
            </div>

            {message && (
                <div className={`p-4 rounded-md text-sm flex items-center gap-3 ${message.includes('Erro') ? 'bg-red-500/10 text-red-500 border border-red-500/20' : 'bg-green-500/10 text-green-500 border border-green-500/20'}`}>
                    <ShieldAlert className="w-4 h-4" />
                    {message}
                </div>
            )}

            <div className="grid grid-cols-1 gap-6">
                {sections.map((section, sidx) => (
                    <div key={sidx} className="bg-card border border-border rounded-xl overflow-hidden shadow-sm">
                        <div className="px-6 py-4 bg-muted/30 border-b border-border">
                            <h2 className="font-semibold text-lg">{section.title}</h2>
                        </div>
                        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                            {section.items.map((item) => (
                                <div key={item.key} className="space-y-2">
                                    <label className="text-sm font-medium text-muted-foreground">{item.label}</label>
                                    <input
                                        type={item.type}
                                        step={item.step}
                                        value={config[item.key]}
                                        onChange={(e) => setConfig({
                                            ...config,
                                            [item.key]: item.type === 'number' ? parseFloat(e.target.value) : e.target.value
                                        })}
                                        className="w-full bg-background border border-border rounded-md px-3 py-2 outline-none focus:ring-2 focus:ring-primary/50 transition-all text-sm"
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <div className="bg-orange-500/10 border border-orange-500/20 p-4 rounded-xl text-xs text-orange-500 leading-relaxed">
                <strong>Atenção:</strong> Alterar estas configurações afeta diretamente o comportamento do sistema. O backend irá recarregar o arquivo config_master.py automaticamente, mas processos em andamento podem não ser afetados até o próximo reinício do processo.
            </div>
        </div>
    );
}
