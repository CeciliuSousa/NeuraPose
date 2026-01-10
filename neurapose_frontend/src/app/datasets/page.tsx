'use client';

import { PageHeader } from '@/components/ui/page-header';
import { Scissors, FolderInput, PieChart } from 'lucide-react';

export default function DatasetsPage() {
    return (
        <div>
            <PageHeader
                title="Gerenciar Datasets"
                description="Divida seus dados em conjuntos de Treino, Teste e Validação."
            />

            <div className="grid gap-8 lg:grid-cols-2">
                <div className="space-y-6">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <Scissors className="w-5 h-5 text-primary" />
                            Split Dataset
                        </h3>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Pasta do Dataset (Imagens + Labels)</label>
                                <div className="flex gap-2">
                                    <input type="text" className="flex-1 px-3 py-2 rounded-md bg-secondary/50 border border-border text-sm" placeholder="C:\Datasets\meu_dataset" />
                                    <button className="p-2 bg-secondary rounded-md border border-border"><FolderInput className="w-4 h-4" /></button>
                                </div>
                            </div>

                            <div className="space-y-4 pt-2">
                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                        <span>Train</span>
                                        <span className="font-mono">70%</span>
                                    </div>
                                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                                        <div className="h-full bg-blue-500 w-[70%]" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                        <span>Validation</span>
                                        <span className="font-mono">20%</span>
                                    </div>
                                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                                        <div className="h-full bg-green-500 w-[20%]" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                        <span>Test</span>
                                        <span className="font-mono">10%</span>
                                    </div>
                                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                                        <div className="h-full bg-orange-500 w-[10%]" />
                                    </div>
                                </div>
                            </div>

                            <div className="pt-4">
                                <button className="w-full py-3 bg-primary text-primary-foreground rounded-md font-medium hover:brightness-110">
                                    Dividir Dataset
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="flex items-center justify-center p-8 border border-dashed border-border rounded-xl bg-muted/10">
                    <div className="text-center space-y-2">
                        <PieChart className="w-12 h-12 text-muted-foreground mx-auto" />
                        <p className="font-medium">Visualização de Distribuição</p>
                        <p className="text-xs text-muted-foreground">Selecione um dataset para ver estatísticas de classes e balanceamento.</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
