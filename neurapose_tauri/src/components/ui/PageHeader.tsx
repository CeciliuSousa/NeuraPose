import { LucideIcon } from 'lucide-react';

interface PageHeaderProps {
    title: string;
    description: string;
    icon?: LucideIcon;
    children?: React.ReactNode;
}

export function PageHeader({
    title,
    description,
    icon: Icon,
    children
}: PageHeaderProps) {
    return (
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between pb-6 border-b border-border mb-8">
            <div className="space-y-1">
                <div className="flex items-center gap-3">
                    {Icon && (
                        <div className="p-2 bg-primary/10 rounded-md">
                            <Icon className="w-6 h-6 text-primary" />
                        </div>
                    )}
                    <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
                </div>
                <p className="text-muted-foreground text-sm">
                    {description}
                </p>
            </div>
            {children && (
                <div className="flex items-center gap-2">
                    {children}
                </div>
            )}
        </div>
    );
}
