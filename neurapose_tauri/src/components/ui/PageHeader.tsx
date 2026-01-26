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
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between pb-6 border-b border-border mb-8 shrink-0">
            <div className="flex items-center gap-4">
                {Icon && (
                    <div className="p-3 bg-primary/10 rounded-xl border border-primary/10 shrink-0 shadow-sm">
                        <Icon className="w-8 h-8 text-primary" />
                    </div>
                )}
                <div className="space-y-1">
                    <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground leading-tight">{title}</h2>
                    <p className="text-muted-foreground text-sm md:text-base max-w-2xl">
                        {description}
                    </p>
                </div>
            </div>

            {children && (
                <div className="flex items-center gap-3 self-start md:self-center mt-4 md:mt-0">
                    {children}
                </div>
            )}
        </div>
    );
}
