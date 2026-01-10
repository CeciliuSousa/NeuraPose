'use client';

import { PageHeader } from '@/components/ui/page-header';
import { TestTube2, Table2, PlayCircle } from 'lucide-react';

export default function TestsPage() {
    return (
        <div>
            <PageHeader
                title="Validação e Testes"
                description="Avalie a precisão do modelo em conjuntos de teste."
            />

            <div className="grid gap-6 lg:grid-cols-3">
                {/* Actions */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <PlayCircle className="w-5 h-5 text-primary" />
                            Novo Teste
                        </h3>
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Modelo</label>
                                <select className="w-full px-3 py-2 rounded-md bg-secondary/50 border border-border text-sm">
                                    <option>yolov8n-pose.pt</option>
                                    <option>custom_best.pt</option>
                                </select>
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Dataset (Test Split)</label>
                                <select className="w-full px-3 py-2 rounded-md bg-secondary/50 border border-border text-sm">
                                    <option>COCO-Pose-Val</option>
                                    <option>MyDataset-Test</option>
                                </select>
                            </div>
                            <button className="w-full py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:brightness-110">
                                Executar Validação
                            </button>
                        </div>
                    </div>
                </div>

                {/* Results */}
                <div className="lg:col-span-2">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="font-semibold text-lg flex items-center gap-2">
                                <Table2 className="w-5 h-5 text-primary" />
                                Resultados Recentes
                            </h3>
                        </div>

                        <div className="relative overflow-x-auto">
                            <table className="w-full text-sm text-left">
                                <thead className="text-xs text-muted-foreground uppercase bg-secondary/30">
                                    <tr>
                                        <th className="px-4 py-3 rounded-l-lg">Modelo</th>
                                        <th className="px-4 py-3">mAP50-95</th>
                                        <th className="px-4 py-3">mAP50</th>
                                        <th className="px-4 py-3">Inference Time</th>
                                        <th className="px-4 py-3 rounded-r-lg">Data</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-border">
                                    <tr className="bg-card hover:bg-muted/10">
                                        <td className="px-4 py-3 font-medium">yolov8n-pose.pt</td>
                                        <td className="px-4 py-3 text-green-500 font-bold">0.654</td>
                                        <td className="px-4 py-3">0.892</td>
                                        <td className="px-4 py-3">2.1ms</td>
                                        <td className="px-4 py-3 text-muted-foreground">{new Date().toLocaleDateString()}</td>
                                    </tr>
                                    <tr className="bg-card hover:bg-muted/10">
                                        <td className="px-4 py-3 font-medium">custom_v1.pt</td>
                                        <td className="px-4 py-3 text-yellow-500 font-bold">0.512</td>
                                        <td className="px-4 py-3">0.710</td>
                                        <td className="px-4 py-3">3.5ms</td>
                                        <td className="px-4 py-3 text-muted-foreground">{new Date().toLocaleDateString()}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
