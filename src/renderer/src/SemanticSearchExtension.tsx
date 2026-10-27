/**
 * Semantic Search — Inline launcher view.
 *
 * Searches files, photos, video, audio, notes and clipboard history by
 * meaning, using the on-device EmbeddingGemma 2 index kept by the main
 * process (src/main/semantic-search.ts). Matches the Notes / Menu Item
 * search chrome (snippet-view, ExtensionActionFooter, actions overlay).
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  BatteryCharging,
  ChevronDown,
  Clipboard,
  Copy,
  ExternalLink,
  FileText,
  Film,
  FolderOpen,
  Image as ImageIcon,
  Loader2,
  Music,
  RefreshCw,
  ScanSearch,
  Settings,
  StickyNote,
  X,
} from 'lucide-react';
import ExtensionActionFooter from './components/ExtensionActionFooter';
import { useI18n } from './i18n';
import { downloadPercent, formatModelSize } from './settings/SemanticSearchSettings';
import type { SemanticItemKind, SemanticSearchResult, SemanticSearchStatus } from '../types/electron';

interface Action {
  title: string;
  icon?: React.ReactNode;
  shortcut?: string[];
  execute: () => void | Promise<void>;
}

interface SemanticSearchProps {
  onClose: () => void;
}

type Filter = 'all' | SemanticItemKind;

const FILTERS: Filter[] = ['all', 'document', 'image', 'video', 'audio', 'note', 'clipboard'];
const SEARCH_DEBOUNCE_MS = 160;
const SEMANTIC_SEARCH_COMMAND_ID = 'system-semantic-search';

function toFileUrl(filePath: string): string {
  return `file://${encodeURI(filePath)}`;
}

function kindIcon(kind: SemanticItemKind, className: string): React.ReactNode {
  switch (kind) {
    case 'image': return <ImageIcon className={className} />;
    case 'video': return <Film className={className} />;
    case 'audio': return <Music className={className} />;
    case 'note': return <StickyNote className={className} />;
    case 'clipboard': return <Clipboard className={className} />;
    default: return <FileText className={className} />;
  }
}

const SemanticSearch: React.FC<SemanticSearchProps> = ({ onClose }) => {
  const { t } = useI18n();
  const [status, setStatus] = useState<SemanticSearchStatus | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState<Filter>('all');
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [showActions, setShowActions] = useState(false);
  const [includeMediaOnSetup, setIncludeMediaOnSetup] = useState(true);
  const [fileIcons, setFileIcons] = useState<Record<string, string>>({});
  const [brokenThumbnails, setBrokenThumbnails] = useState<Record<string, true>>({});
  const inputRef = useRef<HTMLInputElement>(null);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);
  const searchSeqRef = useRef(0);

  // ─── Status polling (fast while downloading/indexing) ─────────────
  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await window.electron.semanticSearchStatus());
    } catch {}
  }, []);

  const busy = status?.textModel.state === 'downloading'
    || status?.mediaModel.state === 'downloading'
    || status?.index.state === 'indexing';
  useEffect(() => {
    void refreshStatus();
    const timer = setInterval(() => { void refreshStatus(); }, busy ? 800 : 4000);
    return () => clearInterval(timer);
  }, [refreshStatus, busy]);

  useEffect(() => {
    const timer = setTimeout(() => inputRef.current?.focus(), 50);
    return () => clearTimeout(timer);
  }, []);

  // ─── Search ───────────────────────────────────────────────────────
  const ready = Boolean(status?.ready);
  const hasIndex = (status?.index.items ?? 0) > 0;
  useEffect(() => {
    const query = searchQuery.trim();
    const seq = ++searchSeqRef.current;
    if (!ready || !query) {
      setResults([]);
      setSearching(false);
      setSearchError(null);
      return;
    }
    setSearching(true);
    const timer = setTimeout(async () => {
      try {
        const next = await window.electron.semanticSearchQuery(query, {
          limit: 60,
          kinds: filter === 'all' ? undefined : [filter],
        });
        if (searchSeqRef.current !== seq) return;
        setResults(next);
        setSearchError(null);
        setSelectedIndex(0);
      } catch (error) {
        if (searchSeqRef.current !== seq) return;
        setResults([]);
        setSearchError(error instanceof Error ? error.message : String(error));
      } finally {
        if (searchSeqRef.current === seq) setSearching(false);
      }
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // hasIndex: re-run once the first index lands so an early query fills in.
  }, [searchQuery, filter, ready, hasIndex]);

  useEffect(() => {
    if (selectedIndex >= results.length) setSelectedIndex(Math.max(0, results.length - 1));
  }, [results.length, selectedIndex]);

  useEffect(() => {
    itemRefs.current[selectedIndex]?.scrollIntoView({ block: 'nearest' });
  }, [selectedIndex]);

  // System file icons for non-image file results.
  useEffect(() => {
    const missing = results
      .filter((r) => r.path && r.kind !== 'image' && r.kind !== 'clipboard' && !fileIcons[r.path])
      .map((r) => r.path as string);
    if (missing.length === 0) return;
    let cancelled = false;
    void Promise.all(
      missing.slice(0, 30).map(async (p) => [p, await window.electron.getFileIconDataUrl(p, 32).catch(() => null)] as const)
    ).then((pairs) => {
      if (cancelled) return;
      setFileIcons((prev) => {
        const next = { ...prev };
        for (const [p, url] of pairs) if (url) next[p] = url;
        return next;
      });
    });
    return () => { cancelled = true; };
  }, [results, fileIcons]);

  const selected = results[selectedIndex] || null;

  // ─── Actions ──────────────────────────────────────────────────────
  const openResult = useCallback(async (result: SemanticSearchResult | null) => {
    if (!result) return;
    if (result.kind === 'note' && result.noteId) {
      const notes = await window.electron.noteGetAll();
      const note = notes.find((n) => n.id === result.noteId);
      if (note) {
        await window.electron.openNotesWindow('edit', JSON.stringify(note));
        await window.electron.hideWindow();
      }
      return;
    }
    if (result.kind === 'clipboard' && result.clipboardId) {
      await window.electron.clipboardPasteItem(result.clipboardId);
      return;
    }
    if (result.path) {
      await window.electron.execCommand('open', [result.path]);
      await window.electron.hideWindow();
    }
  }, []);

  const revealResult = useCallback(async (result: SemanticSearchResult | null) => {
    if (!result?.path) return;
    await window.electron.execCommand('open', ['-R', result.path]);
    await window.electron.hideWindow();
  }, []);

  const copyResult = useCallback(async (result: SemanticSearchResult | null) => {
    if (!result) return;
    if (result.kind === 'clipboard' && result.clipboardId) {
      await window.electron.clipboardCopyItem(result.clipboardId);
    } else if (result.path) {
      await window.electron.clipboardWrite({ text: result.path });
    }
  }, []);

  const openSettings = useCallback(() => {
    void window.electron.openSettingsTab('ai', { commandName: SEMANTIC_SEARCH_COMMAND_ID });
  }, []);

  const primaryLabel = selected?.kind === 'clipboard'
    ? t('semanticSearch.actions.paste')
    : selected?.kind === 'note'
      ? t('semanticSearch.actions.openNote')
      : t('semanticSearch.actions.open');

  const actions: Action[] = useMemo(() => {
    const list: Action[] = [];
    if (selected) {
      list.push({ title: primaryLabel, icon: <ExternalLink size={14} />, shortcut: ['↩'], execute: () => openResult(selected) });
      if (selected.path) {
        list.push({ title: t('semanticSearch.actions.showInFinder'), icon: <FolderOpen size={14} />, shortcut: ['⌘', '↩'], execute: () => revealResult(selected) });
      }
      if (selected.kind === 'clipboard') {
        list.push({ title: t('semanticSearch.actions.copyText'), icon: <Copy size={14} />, shortcut: ['⌘', '⇧', 'C'], execute: () => copyResult(selected) });
      } else if (selected.path) {
        list.push({ title: t('semanticSearch.actions.copyPath'), icon: <Copy size={14} />, shortcut: ['⌘', '⇧', 'C'], execute: () => copyResult(selected) });
      }
    }
    if (ready) {
      list.push({
        title: t('semanticSearch.actions.reindex'),
        icon: <RefreshCw size={14} />,
        execute: () => { void window.electron.semanticSearchReindex(false).then(setStatus); },
      });
    }
    list.push({ title: t('semanticSearch.actions.settings'), icon: <Settings size={14} />, shortcut: ['⌘', ','], execute: openSettings });
    return list;
  }, [selected, primaryLabel, ready, t, openResult, revealResult, copyResult, openSettings]);

  // ─── Keyboard ─────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (showActions) return; // the overlay handles its own keys
      if (e.key === 'Escape') {
        e.preventDefault();
        if (showFilterMenu) setShowFilterMenu(false);
        else onClose();
        return;
      }
      if (e.key === 'Backspace' && !searchQuery) { e.preventDefault(); onClose(); return; }
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); setShowActions(true); return; }
      if (e.key === ',' && e.metaKey) { e.preventDefault(); openSettings(); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); setSelectedIndex((i) => Math.min(i + 1, results.length - 1)); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); setSelectedIndex((i) => Math.max(0, i - 1)); return; }
      if (e.key === 'Enter' && selected) {
        e.preventDefault();
        if (e.metaKey) void revealResult(selected);
        else void openResult(selected);
        return;
      }
      if (e.metaKey && e.shiftKey && e.key.toLowerCase() === 'c' && selected) { e.preventDefault(); void copyResult(selected); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [showActions, showFilterMenu, searchQuery, results.length, selected, onClose, openSettings, openResult, revealResult, copyResult]);

  // ─── Rendering ────────────────────────────────────────────────────
  const renderLeading = (result: SemanticSearchResult) => {
    const isImage = result.kind === 'image' || (result.kind === 'clipboard' && result.path);
    if (isImage && result.path && !brokenThumbnails[result.path]) {
      return (
        <img
          src={toFileUrl(result.path)}
          alt=""
          draggable={false}
          className="w-8 h-8 rounded object-cover flex-shrink-0 bg-black/20"
          onError={() => setBrokenThumbnails((prev) => ({ ...prev, [result.path as string]: true }))}
        />
      );
    }
    if (result.path && fileIcons[result.path]) {
      return <img src={fileIcons[result.path]} alt="" draggable={false} className="w-8 h-8 object-contain flex-shrink-0" />;
    }
    return (
      <div className="w-8 h-8 rounded bg-[var(--ui-segment-bg)] flex items-center justify-center flex-shrink-0">
        {kindIcon(result.kind, 'w-4 h-4 text-white/50')}
      </div>
    );
  };

  const renderSetup = () => {
    if (!status) return null;
    if (!status.helperAvailable) {
      return <p className="text-amber-300 text-[13px] leading-snug">{t('settings.ai.semanticSearch.helperMissing')}</p>;
    }
    const textDownloading = status.textModel.state === 'downloading';
    const mediaDownloading = status.config.includeMedia && status.mediaModel.state === 'downloading';
    const failed = status.textModel.state === 'error' ? status.textModel : status.mediaModel.state === 'error' ? status.mediaModel : null;
    const downloadBytes = status.textModel.totalBytes + (includeMediaOnSetup ? status.mediaModel.totalBytes : 0);
    return (
      <div className="max-w-[440px] text-center space-y-3">
        <div className="mx-auto w-11 h-11 rounded-xl bg-violet-500/15 flex items-center justify-center">
          <ScanSearch className="w-5 h-5 text-violet-300" />
        </div>
        <h2 className="text-[15px] font-semibold text-[var(--text-primary)]">{t('semanticSearch.setup.title')}</h2>
        <p className="text-[13px] text-[var(--text-muted)] leading-snug">{t('semanticSearch.setup.description')}</p>
        {textDownloading || mediaDownloading ? (
          <div className="space-y-1.5 pt-1">
            <p className="text-[12px] text-[var(--text-secondary)]">
              {t('semanticSearch.setup.downloading', {
                percent: downloadPercent(textDownloading ? status.textModel : status.mediaModel),
              })}
            </p>
            <div className="h-1.5 rounded-full bg-black/20 overflow-hidden">
              <div
                className="h-full bg-violet-400/80 transition-[width] duration-300"
                style={{ width: `${downloadPercent(textDownloading ? status.textModel : status.mediaModel)}%` }}
              />
            </div>
          </div>
        ) : (
          <>
            {!status.config.enabled && (
              <label className="flex items-center justify-center gap-2 text-[12px] text-[var(--text-secondary)] cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={includeMediaOnSetup}
                  onChange={(e) => setIncludeMediaOnSetup(e.target.checked)}
                />
                {t('semanticSearch.setup.includeMedia', { size: formatModelSize(status.mediaModel.totalBytes) })}
              </label>
            )}
            {failed && (
              <p className="text-[12px] text-rose-300">{t('settings.ai.semanticSearch.status.failed', { error: failed.error || '' })}</p>
            )}
            <div className="flex items-center justify-center gap-2 pt-1">
              <button
                type="button"
                className="px-3 py-1.5 rounded-md text-[13px] font-medium bg-violet-500/80 hover:bg-violet-500 text-white transition-colors"
                onClick={async () => {
                  if (!status.config.enabled) {
                    setStatus(await window.electron.semanticSearchUpdateConfig({ enabled: true, includeMedia: includeMediaOnSetup }));
                  } else {
                    setStatus(await window.electron.semanticSearchDownloadModels());
                  }
                }}
              >
                {t('semanticSearch.setup.enable', {
                  size: formatModelSize(status.config.enabled ? status.textModel.totalBytes : downloadBytes),
                })}
              </button>
              <button
                type="button"
                className="px-3 py-1.5 rounded-md text-[13px] font-medium bg-[var(--ui-segment-bg)] border border-[var(--ui-divider)] text-[var(--text-secondary)] hover:bg-[var(--ui-segment-hover-bg)]"
                onClick={openSettings}
              >
                {t('semanticSearch.setup.settings')}
              </button>
            </div>
          </>
        )}
      </div>
    );
  };

  const renderBody = () => {
    if (!status) return null;
    if (!ready) {
      return <div className="flex items-center justify-center h-full px-8">{renderSetup()}</div>;
    }
    if (!searchQuery.trim()) {
      if (!hasIndex && status.index.pausedOnBattery) {
        return (
          <div className="flex flex-col items-center justify-center h-full gap-3 px-10 text-center text-white/45 text-[13px] leading-snug">
            <BatteryCharging className="w-5 h-5 text-amber-300/80" />
            {t('semanticSearch.waitingForPower')}
            <button
              type="button"
              className="px-3 py-1.5 rounded-md text-[12px] font-medium bg-[var(--ui-segment-bg)] border border-[var(--ui-divider)] text-[var(--text-secondary)] hover:bg-[var(--ui-segment-hover-bg)]"
              onClick={() => { void window.electron.semanticSearchReindex(false).then(setStatus); }}
            >
              {t('semanticSearch.indexNow')}
            </button>
          </div>
        );
      }
      return (
        <div className="flex items-center justify-center h-full px-10 text-center text-white/35 text-[13px] leading-snug">
          {hasIndex || status.index.state === 'indexing' ? t('semanticSearch.hint') : t('semanticSearch.emptyIndex')}
        </div>
      );
    }
    if (searchError) {
      return <div className="flex items-center justify-center h-full px-8 text-center text-[#f87171] text-[13px] leading-snug">{searchError}</div>;
    }
    if (results.length === 0) {
      return (
        <div className="flex items-center justify-center h-full text-white/30 text-sm">
          {searching ? t('semanticSearch.searching') : t('semanticSearch.noResults')}
        </div>
      );
    }
    return (
      <div className="p-2 space-y-1">
        {results.map((result, index) => (
          <div
            key={result.id}
            ref={(el) => (itemRefs.current[index] = el)}
            className={`px-2.5 py-2 rounded-md border cursor-pointer transition-colors ${
              index === selectedIndex
                ? 'bg-[var(--launcher-card-selected-bg)] border-[var(--launcher-card-border)]'
                : 'border-transparent hover:bg-[var(--launcher-card-hover-bg)] hover:border-[var(--launcher-card-border)]'
            }`}
            onClick={() => setSelectedIndex(index)}
            onMouseEnter={() => setSelectedIndex(index)}
            onDoubleClick={() => void openResult(result)}
          >
            <div className="flex items-center gap-3 min-w-0">
              {renderLeading(result)}
              <div className="min-w-0 flex-1">
                <div className="text-[13px] text-[var(--text-primary)] font-medium truncate">{result.title}</div>
                <div className="text-[12px] text-[var(--text-subtle)] truncate">
                  {result.preview && result.kind !== 'clipboard' ? result.preview : result.subtitle}
                </div>
              </div>
              <span className="flex-shrink-0 text-[11px] text-[var(--text-disabled)]">{t(`semanticSearch.kinds.${result.kind}`)}</span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const footerStatus = () => {
    if (!status || !ready) return null;
    if (status.index.state === 'indexing') {
      return (
        <>
          <Loader2 className="w-3.5 h-3.5 animate-spin text-white/45" />
          <span className="truncate">
            {t('semanticSearch.indexing', { processed: status.index.processed.toLocaleString(), total: status.index.total.toLocaleString() })}
          </span>
        </>
      );
    }
    return (
      <>
        <ScanSearch className="w-4 h-4 text-white/45" />
        <span className="truncate">{t('semanticSearch.indexed', { count: status.index.items.toLocaleString() })}</span>
        {status.index.pausedOnBattery && (
          <span className="flex items-center gap-1 text-amber-300/80 truncate">
            <BatteryCharging className="w-3.5 h-3.5 flex-shrink-0" />
            {t('semanticSearch.pausedOnBattery')}
          </span>
        )}
      </>
    );
  };

  return (
    <div className="snippet-view flex flex-col h-full">
      {/* ─── Header ─── */}
      <div className="snippet-header drag-region flex h-16 items-center gap-2 px-4">
        <button
          onClick={onClose}
          className="text-white/40 hover:text-white/70 transition-colors flex-shrink-0"
          tabIndex={-1}
          title={t('semanticSearch.back')}
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="relative min-w-0 flex-1">
          <input
            ref={inputRef}
            type="text"
            placeholder={t('semanticSearch.placeholder')}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="min-w-0 w-full bg-transparent border-none outline-none text-white/95 placeholder:text-[color:var(--text-subtle)] text-[15px] font-medium tracking-[0.005em]"
            autoFocus
          />
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {searching && <Loader2 className="w-4 h-4 animate-spin text-white/35" />}
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="text-white/30 hover:text-white/60 transition-colors flex-shrink-0">
              <X className="w-4 h-4" />
            </button>
          )}
          <div className="relative">
            <button
              type="button"
              onClick={() => setShowFilterMenu((v) => !v)}
              className="flex items-center gap-1 px-2 py-1 rounded-md text-[12px] text-white/60 bg-[var(--ui-segment-bg)] border border-[var(--ui-divider)] hover:text-white/85"
            >
              {t(`semanticSearch.filters.${filter}`)}
              <ChevronDown className="w-3 h-3" />
            </button>
            {showFilterMenu && (
              <div className="absolute right-0 top-full mt-1 z-50 min-w-[140px] py-1 rounded-lg border border-[var(--ui-divider)] bg-[var(--card-bg)] shadow-xl backdrop-blur-xl">
                {FILTERS.map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => {
                      setFilter(option);
                      setShowFilterMenu(false);
                      inputRef.current?.focus();
                    }}
                    className={`w-full text-left px-3 py-1.5 text-[12px] transition-colors ${
                      option === filter ? 'text-[var(--text-primary)] bg-white/[0.06]' : 'text-[var(--text-secondary)] hover:bg-white/[0.04]'
                    }`}
                  >
                    {t(`semanticSearch.filters.${option}`)}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ─── Results ─── */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar">{renderBody()}</div>

      {/* ─── Footer ─── */}
      <ExtensionActionFooter
        leftContent={<div className="flex items-center gap-2 min-w-0">{footerStatus()}</div>}
        primaryAction={selected ? { label: primaryLabel, onClick: () => void openResult(selected), shortcut: ['↩'] } : undefined}
        actionsButton={{ label: t('semanticSearch.actions.title'), onClick: () => setShowActions(true), shortcut: ['⌘', 'K'] }}
      />

      {showActions && <SemanticActionsOverlay actions={actions} onClose={() => setShowActions(false)} />}
    </div>
  );
};

// ─── Actions Overlay (same style as Notes / Menu Item search) ────────
const SemanticActionsOverlay: React.FC<{ actions: Action[]; onClose: () => void }> = ({ actions, onClose }) => {
  const { t } = useI18n();
  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? actions.filter((a) => a.title.toLowerCase().includes(q)) : actions;
  }, [actions, query]);

  useEffect(() => { setSelectedIdx(0); }, [query]);
  useEffect(() => { inputRef.current?.focus(); }, []);

  useEffect(() => {
    const run = (action: Action) => {
      onClose();
      void action.execute();
    };
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' || (e.key === 'k' && (e.metaKey || e.ctrlKey))) { e.preventDefault(); e.stopPropagation(); onClose(); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); e.stopPropagation(); setSelectedIdx((i) => Math.min(i + 1, filtered.length - 1)); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); e.stopPropagation(); setSelectedIdx((i) => Math.max(0, i - 1)); return; }
      if (e.key === 'Enter' && filtered[selectedIdx]) { e.preventDefault(); e.stopPropagation(); run(filtered[selectedIdx]); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [filtered, selectedIdx, onClose]);

  return (
    <div className="fixed inset-0 z-[9999]" style={{ background: 'var(--bg-scrim)' }}>
      <div className="absolute inset-0" onClick={onClose} />
      <div
        className="absolute bottom-12 right-3 w-80 max-h-[65vh] overflow-hidden flex flex-col rounded-xl shadow-2xl border border-[var(--border-primary)]"
        style={{ background: 'var(--card-bg)', backdropFilter: 'blur(40px)', WebkitBackdropFilter: 'blur(40px)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-3 py-2.5 border-b border-[var(--ui-divider)]">
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t('semanticSearch.actions.search')}
            className="w-full bg-transparent text-[var(--text-primary)] text-[13px] placeholder:text-[var(--text-subtle)] outline-none"
          />
        </div>
        <div className="flex-1 overflow-y-auto custom-scrollbar py-1">
          {filtered.map((action, idx) => (
            <div
              key={action.title}
              onClick={() => { onClose(); void action.execute(); }}
              onMouseEnter={() => setSelectedIdx(idx)}
              className="flex items-center gap-3 px-3 py-[7px] cursor-pointer transition-colors text-[var(--text-secondary)]"
              style={idx === selectedIdx ? { background: 'rgba(255,255,255,0.08)' } : undefined}
            >
              <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center opacity-60">{action.icon}</span>
              <span className="flex-1 text-[12px]">{action.title}</span>
              {action.shortcut && (
                <span className="flex items-center gap-0.5 flex-shrink-0">
                  {action.shortcut.map((k, ki) => (
                    <kbd key={ki} className="inline-flex items-center justify-center min-w-[20px] h-[18px] px-1 rounded bg-[var(--kbd-bg)] text-[10px] text-[var(--text-subtle)] font-medium">{k}</kbd>
                  ))}
                </span>
              )}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="px-3 py-4 text-center text-[11px] text-[var(--text-disabled)]">{t('semanticSearch.actions.none')}</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SemanticSearch;
