import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';

export interface TabSpec {
  readonly id: string;
  readonly label: string;
  readonly content: ReactNode;
}

function tabFromHash(tabs: readonly TabSpec[]): string {
  const hash = window.location.hash.slice(1);
  return tabs.some(tab => tab.id === hash) ? hash : tabs[0].id;
}

export function BenchmarkTabs({ tabs }: { tabs: readonly TabSpec[] }) {
  const [active, setActive] = useState(() => tabFromHash(tabs));
  const buttons = useRef(new Map<string, HTMLButtonElement>());

  useEffect(() => {
    const onHash = () => setActive(tabFromHash(tabs));
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [tabs]);

  const select = (id: string, focus: boolean) => {
    setActive(id);
    window.history.replaceState(null, '', `#${id}`);
    if (focus) buttons.current.get(id)?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = tabs.findIndex(tab => tab.id === active);
    const moves: Record<string, number> = {
      ArrowRight: (index + 1) % tabs.length,
      ArrowLeft: (index - 1 + tabs.length) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
    };
    const next = moves[event.key];
    if (next === undefined) return;
    event.preventDefault();
    select(tabs[next].id, true);
  };

  return (
    <div className="hc-tabs">
      <div className="hc-tablist" role="tablist" aria-label="Benchmark families" onKeyDown={onKeyDown}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            ref={node => { if (node) buttons.current.set(tab.id, node); else buttons.current.delete(tab.id); }}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={tab.id === active}
            aria-controls={tab.id}
            tabIndex={tab.id === active ? 0 : -1}
            onClick={() => select(tab.id, false)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map(tab => (
        <div key={tab.id} className="hc-tabpanel" role="tabpanel" id={tab.id} aria-labelledby={`tab-${tab.id}`} tabIndex={0} hidden={tab.id !== active}>
          {tab.content}
        </div>
      ))}
    </div>
  );
}
