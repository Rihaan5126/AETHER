import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 text-[11px] font-medium leading-none whitespace-nowrap",
  {
    variants: {
      tone: {
        neutral: "border-line-strong bg-raised text-fg-muted",
        accent: "border-accent/30 bg-accent/10 text-accent",
        pos: "border-pos/30 bg-pos/10 text-pos",
        neg: "border-neg/30 bg-neg/10 text-neg",
        warn: "border-warn/30 bg-warn/10 text-warn",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export function Badge({
  className,
  tone,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
