'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard,
    Video,
    ScanFace,
    PenTool,
    Scissors,
    FileOutput,
    Dumbbell,
    TestTube2,
    Settings,
    Menu,
    X
} from 'lucide-react';
import { useState } from 'react';

const menuItems = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Processar Vídeo', href: '/processamento', icon: Video },
    { name: 'Re-identificação', href: '/reid', icon: ScanFace },
    { name: 'Anotação', href: '/anotacao', icon: PenTool },
    { name: 'Split Datasets', href: '/datasets', icon: Scissors },
    { name: 'Converter .pt', href: '/converter', icon: FileOutput },
    { name: 'Treinamento', href: '/treino', icon: Dumbbell },
    { name: 'Testes', href: '/testes', icon: TestTube2 },
    { name: 'Configurações', href: '/configuracao', icon: Settings },
];

export function Sidebar() {
    const pathname = usePathname();
    const [isOpen, setIsOpen] = useState(false);

    return (
        <>
            {/* Mobile Menu Button */}
            <button
                className="md:hidden fixed top-4 left-4 z-50 p-2 bg-secondary rounded-md"
                onClick={() => setIsOpen(!isOpen)}
            >
                {isOpen ? <X /> : <Menu />}
            </button>

            {/* Sidebar Container */}
            <aside className={`
        fixed inset-y-0 left-0 z-40 w-64 bg-card border-r border-border transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
      `}>
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="h-16 flex items-center px-6 border-b border-border">
                        <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-blue-500 bg-clip-text text-transparent">
                            NeuraPose
                        </h1>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 overflow-y-auto py-4">
                        <ul className="space-y-1 px-3">
                            {menuItems.map((item) => {
                                const Icon = item.icon;
                                const isActive = pathname === item.href;

                                return (
                                    <li key={item.href}>
                                        <Link
                                            href={item.href}
                                            className={`
                        flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors
                        ${isActive
                                                    ? 'bg-primary/10 text-primary'
                                                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'}
                      `}
                                        >
                                            <Icon className="w-5 h-5" />
                                            {item.name}
                                        </Link>
                                    </li>
                                );
                            })}
                        </ul>
                    </nav>

                    {/* Footer Status */}
                    <div className="p-4 border-t border-border">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                            Backend: Online
                        </div>
                    </div>
                </div>
            </aside>

            {/* Overlay for mobile */}
            {isOpen && (
                <div
                    className="md:hidden fixed inset-0 z-30 bg-black/50 backdrop-blur-sm"
                    onClick={() => setIsOpen(false)}
                />
            )}
        </>
    );
}
