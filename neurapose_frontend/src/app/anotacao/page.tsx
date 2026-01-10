'use client';

import { PageHeader } from '@/components/ui/page-header';
import { MousePointer2, BoxSelect, ExternalLink } from 'lucide-react';
import { useState } from 'react';

export default function AnnotationPage() {
    const [launching, setLaunching] = useState<string | null>(null);

    const launchTool = (toolName: string) => {
        setLaunching(toolName);
        // Simulate Backend Launch
        setTimeout(() => setLaunching(null), 2000);
    };

    return (
        <div>
            <PageHeader
                title="Anotação de Dados"
                description="Ferramentas para criar e editar datasets de treinamento (Bounding Boxes e Keypoints)."
            />

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {/* Tool: Custom Annotator */}
                <div className="rounded-xl border border-border bg-card p-6 flex flex-col justify-between">
                    <div>
                        <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                            <BoxSelect className="w-6 h-6 text-primary" />
                        </div>
                        <h3 className="font-semibold text-lg mb-2">NeuraPose Annotator</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                            Ferramenta nativa para correção rápida de keypoints e bounding boxes em vídeos processados.
                        </p>
                    </div>
                    <button
                        onClick={() => launchTool('neurapose')}
                        className="w-full py-2 bg-secondary text-secondary-foreground rounded-md text-sm font-medium hover:bg-secondary/80 flex items-center justify-center gap-2"
                    >
                        {launching === 'neurapose' ? 'Iniciando...' : (
                            <>
                                <ExternalLink className="w-4 h-4" />
                                Abrir Ferramenta
                            </>
                        )}
                    </button>
                </div>

                {/* Tool: LabelImg */}
                <div className="rounded-xl border border-border bg-card p-6 flex flex-col justify-between">
                    <div>
                        <div className="w-12 h-12 rounded-lg bg-blue-500/10 flex items-center justify-center mb-4">
                            <MousePointer2 className="w-6 h-6 text-blue-500" />
                        </div>
                        <h3 className="font-semibold text-lg mb-2">LabelImg</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                            Ferramenta clássica para anotação de Bounding Boxes (formato YOLO/PascalVOC).
                        </p>
                    </div>
                    <button
                        onClick={() => launchTool('labelimg')}
                        className="w-full py-2 bg-secondary text-secondary-foreground rounded-md text-sm font-medium hover:bg-secondary/80 flex items-center justify-center gap-2"
                    >
                        {launching === 'labelimg' ? 'Iniciando...' : (
                            <>
                                <ExternalLink className="w-4 h-4" />
                                Abrir LabelImg
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
