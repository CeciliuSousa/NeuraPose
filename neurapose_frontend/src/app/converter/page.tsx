'use client';

import { PageHeader } from '@/components/ui/page-header';
import { FileOutput, RefreshCw, FileCode2 } from 'lucide-react';
import { useState } from 'react';

export default function ConverterPage() {
    const [loading, setLoading] = useState(false);

    return (
        <div>
            <PageHeader
                title="Conversão de Modelos"
                description="Otimize e exporte modelos para produção (.pth -> .pt / ONNX)."
            />

            <div className="max-w-3xl space-y-8">
                <div className="rounded-xl border border-border bg-card p-6">
                    <div className="flex items-center gap-2 mb-6">
                        <RefreshCw className="w-5 h-5 text-primary" />
                        <h3 className="font-semibold text-lg">Configurar Conversão</h3>
                    </div>

                    <div className="grid gap-6">
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Modelo de Entrada (.pth)</label>
                            <div className="flex gap-2">
                                <input type="text" className="flex-1 px-3 py-2 rounded-md bg-secondary/50 border border-border text-sm" placeholder="best.pth" />
                                <button className="px-3 bg-secondary rounded-md border border-border hover:bg-secondary/80">
                                    <FileCode2 className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Formato de Saída</label>
                                <select className="w-full px-3 py-2 rounded-md bg-secondary/50 border border-border text-sm">
                                    <option>TorchScript (.pt)</option>
                                    <option>ONNX (.onnx)</option>
                                    <option>TensorRT (.engine)</option>
                                </select>
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Device</label>
                                <select className="w-full px-3 py-2 rounded-md bg-secondary/50 border border-border text-sm">
                                    <option value="0">GPU 0 (CUDA)</option>
                                    <option value="cpu">CPU</option>
                                </select>
                            </div>
                        </div>

                        <div className="p-4 bg-muted/20 rounded-lg border border-border">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Opções Avançadas</h4>
                            <div className="space-y-2">
                                <label className="flex items-center gap-2">
                                    <input type="checkbox" className="rounded text-primary focus:ring-primary" checked readOnly />
                                    <span className="text-sm">Otimizar para inferência (fuse layers)</span>
                                </label>
                                <label className="flex items-center gap-2">
                                    <input type="checkbox" className="rounded text-primary focus:ring-primary" />
                                    <span className="text-sm">FP16 (Half Precision)</span>
                                </label>
                            </div>
                        </div>

                        <button disabled={loading} className="w-full py-3 bg-primary text-primary-foreground rounded-md font-medium hover:brightness-110 flex items-center justify-center gap-2">
                            <FileOutput className="w-4 h-4" />
                            Converter Agora
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
