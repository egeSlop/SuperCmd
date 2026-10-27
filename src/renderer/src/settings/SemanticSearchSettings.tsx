/**
 * Settings → AI → Semantic Search.
 *
 * Configures the on-device EmbeddingGemma 2 search: model quantization,
 * media encoder, indexed sources/folders, and index maintenance. All state
 * lives in the main process (semantic-search.ts); this panel polls status.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { BatteryCharging, CheckCircle2, Clapperboard, Database, Download, FolderPlus, HardDrive, Lock, ScanSearch, Search, Server, Trash2, Usb, X } from 'lucide-react';
import type {
  SemanticDriveStatus,
  SemanticModelFileStatus,
  SemanticModelVariant,
  SemanticSearchConfig,
  SemanticSearchStatus,
} from '../../types/electron';
import { useI18n } from '../i18n';

const VARIANT_SIZES: Record<SemanticModelVariant, number> = {
  compact: 175_673_856,
  balanced: 210_055_680,
  precise: 309_855_456,
};
const MEDIA_MODEL_SIZE = 554_821_024;

export function formatModelSize(bytes: number): string {
  return `${Math.round(bytes / 1_000_000)} MB`;
}

export function downloadPercent(status: SemanticModelFileStatus | undefined): number {
  if (!status || !status.totalBytes) return 0;
  return Math.max(0, Math.min(100, Math.round((status.bytesDownloaded / status.totalBytes) * 100)));
}

function formatDriveSize(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
}

function formatTimestamp(ts: number): string {
  const date = new Date(ts);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

const BUTTON_PRIMARY =
  'inline-flex min-h-[30px] items-center justify-center gap-1.5 rounded-md px-3 py-1 text-[0.75rem] font-medium transition-colors bg-[var(--ui-segment-active-bg)] border border-[var(--ui-segment-border)] text-[var(--text-primary)] disabled:opacity-55 disabled:cursor-not-allowed';
const BUTTON_SECONDARY =
  'inline-flex min-h-[30px] items-center justify-center gap-1.5 rounded-md px-3 py-1 text-[0.75rem] font-medium transition-colors bg-[var(--ui-segment-bg)] border border-[var(--ui-divider)] text-[var(--text-secondary)] hover:bg-[var(--ui-segment-hover-bg)] disabled:opacity-55 disabled:cursor-not-allowed';

const Toggle: React.FC<{ enabled: boolean; onToggle: () => void; label: string; disabled?: boolean }> = ({
  enabled,
  onToggle,
  label,
  disabled,
}) => (
  <button
    onClick={onToggle}
    disabled={disabled}
    aria-label={label}
    className={`relative w-10 h-6 rounded-full border transition-colors shrink-0 disabled:opacity-55 ${
      enabled ? 'bg-[var(--accent)] border-[var(--accent-hover)]' : 'bg-[var(--ui-segment-bg)] border-[var(--ui-segment-border)]'
    }`}
  >
    <span
      className={`absolute top-1/2 h-4 w-4 -translate-y-1/2 rounded-full border shadow-sm transition-all bg-[var(--bg-overlay-strong)] border-[var(--ui-segment-border)] ${
        enabled ? 'right-0.5 left-auto' : 'left-0.5 right-auto'
      }`}
    />
  </button>
);

const Row: React.FC<{ icon: React.ReactNode; title: string; description: string; children: React.ReactNode }> = ({
  icon,
  title,
  description,
  children,
}) => (
  <div className="grid gap-3 px-4 py-3.5 md:px-5 md:grid-cols-[220px_minmax(0,1fr)] border-b border-[var(--ui-divider)]">
    <div className="flex items-start gap-2.5">
      <div className="mt-0.5 text-[var(--text-muted)] shrink-0">{icon}</div>
      <div className="min-w-0">
        <h3 className="text-[0.8125rem] font-semibold text-[var(--text-primary)]">{title}</h3>
        <p className="mt-0.5 text-[0.75rem] text-[var(--text-muted)] leading-snug">{description}</p>
      </div>
    </div>
    <div className="min-w-0 flex flex-col justify-center gap-2">{children}</div>
  </div>
);

const SemanticSearchSettings: React.FC = () => {
  const { t } = useI18n();
  const [status, setStatus] = useState<SemanticSearchStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await window.electron.semanticSearchStatus());
    } catch {}
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => { void refresh(); }, 1000);
    return () => clearInterval(timer);
  }, [refresh]);

  const update = useCallback(async (patch: Partial<SemanticSearchConfig>) => {
    setStatus(await window.electron.semanticSearchUpdateConfig(patch));
  }, []);

  // Turning a drive off deletes its index (for every computer), so it asks first.
  const [confirmDriveOff, setConfirmDriveOff] = useState<string | null>(null);
  const [driveError, setDriveError] = useState<string | null>(null);
  const setDriveEnabled = useCallback(async (root: string, enabled: boolean) => {
    const { error, status: next } = await window.electron.semanticSearchSetVolumeEnabled(root, enabled);
    setStatus(next);
    setDriveError(error);
    setConfirmDriveOff(null);
  }, []);
  const breakDriveLock = useCallback(async (root: string) => {
    const { error, status: next } = await window.electron.semanticSearchBreakLock(root);
    setStatus(next);
    setDriveError(error);
  }, []);

  if (!status) return null;
  const { config, textModel, mediaModel, index } = status;
  const volumes = status.volumes ?? [];

  const driveSummary = (drive: SemanticDriveStatus): string => {
    if (!drive.enabled) return t('settings.ai.semanticSearch.drives.notIndexed');
    if (drive.state === 'indexing') {
      return t('settings.ai.semanticSearch.drives.indexing', { processed: index.processed.toLocaleString(), total: index.total.toLocaleString() });
    }
    const parts = [
      t('settings.ai.semanticSearch.drives.summary', { items: (drive.items ?? 0).toLocaleString() }),
      drive.lastIndexedAt
        ? t('settings.ai.semanticSearch.drives.updated', { time: formatTimestamp(drive.lastIndexedAt) })
        : t('settings.ai.semanticSearch.drives.never'),
      drive.storage === 'local' ? t('settings.ai.semanticSearch.drives.onThisMac') : t('settings.ai.semanticSearch.drives.onDrive'),
    ];
    if (!drive.writable) parts.push(t('settings.ai.semanticSearch.drives.searchOnly'));
    return parts.join(' · ');
  };

  const modelStatusLine = (model: SemanticModelFileStatus) => {
    if (model.state === 'downloaded') {
      return <span className="text-[color:var(--status-success)]">{t('settings.ai.semanticSearch.status.ready')}</span>;
    }
    if (model.state === 'downloading') {
      const percent = downloadPercent(model);
      return (
        <div className="space-y-1.5">
          <span className="text-[var(--text-secondary)]">{t('settings.ai.semanticSearch.status.downloading', { percent })}</span>
          <div className="h-1.5 rounded-full bg-black/20 overflow-hidden">
            <div className="h-full bg-emerald-400/80 transition-[width] duration-300" style={{ width: `${percent}%` }} />
          </div>
        </div>
      );
    }
    if (model.state === 'error') {
      return <span className="text-rose-300">{t('settings.ai.semanticSearch.status.failed', { error: model.error || '' })}</span>;
    }
    return <span className="text-amber-300">{t('settings.ai.semanticSearch.status.notDownloaded')}</span>;
  };

  const downloadButton = (model: SemanticModelFileStatus) =>
    model.state === 'downloaded' ? null : (
      <button
        type="button"
        className={BUTTON_PRIMARY}
        disabled={model.state === 'downloading'}
        onClick={() => { void window.electron.semanticSearchDownloadModels().then(setStatus); }}
      >
        <Download className="w-3.5 h-3.5" />
        {model.state === 'error' ? t('settings.ai.semanticSearch.actions.retry') : t('settings.ai.semanticSearch.actions.download')}
      </button>
    );

  return (
    <div>
      {!status.helperAvailable && (
        <div className="px-4 py-2.5 md:px-5 border-b border-[var(--ui-divider)] text-[0.75rem] text-amber-300">
          {t('settings.ai.semanticSearch.helperMissing')}
        </div>
      )}

      <Row
        icon={<ScanSearch className="w-4 h-4" />}
        title={t('settings.ai.semanticSearch.enable.title')}
        description={t('settings.ai.semanticSearch.enable.description')}
      >
        <Toggle
          enabled={config.enabled}
          onToggle={() => { void update({ enabled: !config.enabled }); }}
          label={t('settings.ai.semanticSearch.enable.title')}
        />
      </Row>

      <div className={config.enabled ? '' : 'opacity-65 pointer-events-none select-none'}>
        <Row
          icon={<HardDrive className="w-4 h-4" />}
          title={t('settings.ai.semanticSearch.model.title')}
          description={t('settings.ai.semanticSearch.model.description')}
        >
          <div className="flex flex-wrap gap-1.5">
            {(['compact', 'balanced', 'precise'] as SemanticModelVariant[]).map((variant) => (
              <button
                key={variant}
                type="button"
                onClick={() => { if (variant !== config.modelVariant) void update({ modelVariant: variant }); }}
                className={`px-2.5 py-1 rounded-md text-[0.75rem] font-medium transition-colors border ${
                  config.modelVariant === variant
                    ? 'bg-[var(--ui-segment-active-bg)] text-[var(--text-primary)] border-[var(--ui-segment-border)]'
                    : 'bg-[var(--ui-segment-bg)] text-[var(--text-muted)] border-[var(--ui-divider)] hover:text-[var(--text-secondary)]'
                }`}
              >
                {t(`settings.ai.semanticSearch.model.variants.${variant}`, { size: formatModelSize(VARIANT_SIZES[variant]) })}
              </button>
            ))}
          </div>
          <div className="flex items-center justify-between gap-3 text-[0.75rem]">
            <div className="min-w-0 flex-1">{modelStatusLine(textModel)}</div>
            {downloadButton(textModel)}
          </div>
        </Row>

        <Row
          icon={<Clapperboard className="w-4 h-4" />}
          title={t('settings.ai.semanticSearch.media.title')}
          description={t('settings.ai.semanticSearch.media.description', { size: formatModelSize(MEDIA_MODEL_SIZE) })}
        >
          <Toggle
            enabled={config.includeMedia}
            onToggle={() => { void update({ includeMedia: !config.includeMedia }); }}
            label={t('settings.ai.semanticSearch.media.title')}
          />
          {config.includeMedia && (
            <div className="flex items-center justify-between gap-3 text-[0.75rem]">
              <div className="min-w-0 flex-1">{modelStatusLine(mediaModel)}</div>
              {downloadButton(mediaModel)}
            </div>
          )}
        </Row>

        <Row
          icon={<FolderPlus className="w-4 h-4" />}
          title={t('settings.ai.semanticSearch.sources.title')}
          description={t('settings.ai.semanticSearch.sources.description')}
        >
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[0.75rem] text-[var(--text-secondary)]">
            <label className="flex items-center gap-2">
              <Toggle
                enabled={config.includeNotes}
                onToggle={() => { void update({ includeNotes: !config.includeNotes }); }}
                label={t('settings.ai.semanticSearch.sources.notes')}
              />
              {t('settings.ai.semanticSearch.sources.notes')}
            </label>
            <label className="flex items-center gap-2">
              <Toggle
                enabled={config.includeClipboard}
                onToggle={() => { void update({ includeClipboard: !config.includeClipboard }); }}
                label={t('settings.ai.semanticSearch.sources.clipboard')}
              />
              {t('settings.ai.semanticSearch.sources.clipboard')}
            </label>
          </div>
          <div className="rounded-lg border border-[var(--ui-divider)] divide-y divide-[var(--ui-divider)]">
            {config.folders.length === 0 ? (
              <p className="px-3 py-2 text-[0.75rem] text-[var(--text-muted)]">{t('settings.ai.semanticSearch.folders.empty')}</p>
            ) : (
              config.folders.map((folder) => (
                <div key={folder} className="flex items-center gap-2 px-3 py-1.5 text-[0.75rem] text-[var(--text-secondary)]">
                  <span className="truncate flex-1" title={folder}>{folder}</span>
                  <button
                    type="button"
                    aria-label={t('settings.ai.semanticSearch.folders.remove')}
                    title={t('settings.ai.semanticSearch.folders.remove')}
                    className="text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    onClick={() => { void update({ folders: config.folders.filter((f) => f !== folder) }); }}
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>
          <div>
            <button
              type="button"
              className={BUTTON_SECONDARY}
              onClick={async () => {
                const picked = await window.electron.semanticSearchPickFolders();
                if (picked.length > 0) void update({ folders: [...config.folders, ...picked] });
              }}
            >
              <FolderPlus className="w-3.5 h-3.5" />
              {t('settings.ai.semanticSearch.folders.add')}
            </button>
          </div>
        </Row>

        <Row
          icon={<Usb className="w-4 h-4" />}
          title={t('settings.ai.semanticSearch.drives.title')}
          description={t('settings.ai.semanticSearch.drives.description')}
        >
          <div className="rounded-lg border border-[var(--ui-divider)] divide-y divide-[var(--ui-divider)]">
            {volumes.length === 0 ? (
              <p className="px-3 py-2 text-[0.75rem] text-[var(--text-muted)]">{t('settings.ai.semanticSearch.drives.empty')}</p>
            ) : (
              volumes.map((drive) => (
                <div key={drive.root} className="px-3 py-2 text-[0.75rem]">
                  <div className="flex items-center gap-2">
                    <span className="text-[var(--text-muted)] shrink-0">
                      {drive.network ? <Server className="w-3.5 h-3.5" /> : <Usb className="w-3.5 h-3.5" />}
                    </span>
                    <span className="font-medium text-[var(--text-primary)] truncate" title={drive.root}>{drive.label}</span>
                    <span className="flex-1 min-w-0 truncate text-[var(--text-muted)]">
                      {[drive.fsType, drive.bytesTotal > 0 ? formatDriveSize(drive.bytesTotal) : '', drive.readOnly ? t('settings.ai.semanticSearch.drives.readOnly') : '']
                        .filter(Boolean)
                        .join(' · ')}
                    </span>
                    <Toggle
                      enabled={drive.enabled}
                      onToggle={() => {
                        if (drive.enabled) setConfirmDriveOff(drive.root);
                        else void setDriveEnabled(drive.root, true);
                      }}
                      label={t('settings.ai.semanticSearch.drives.title')}
                    />
                  </div>
                  <p className="mt-1 text-[var(--text-muted)]">{driveSummary(drive)}</p>
                  {drive.incompatible && (
                    <p className="mt-1 text-amber-300">{t('settings.ai.semanticSearch.drives.incompatible')}</p>
                  )}
                  {drive.error && !drive.lock && !drive.incompatible && (
                    <p className="mt-1 text-amber-300">{drive.error}</p>
                  )}
                  {drive.lock && (
                    <div className="mt-1.5 flex flex-wrap items-center gap-2 text-amber-300">
                      <Lock className="w-3.5 h-3.5 shrink-0" />
                      <span className="flex-1 min-w-0">
                        {t('settings.ai.semanticSearch.drives.locked', {
                          owner: drive.lock.owner,
                          time: drive.lock.heartbeat ? formatTimestamp(drive.lock.heartbeat) : '—',
                        })}
                      </span>
                      <button
                        type="button"
                        className={BUTTON_SECONDARY}
                        title={t('settings.ai.semanticSearch.drives.breakLockHint')}
                        onClick={() => { void breakDriveLock(drive.root); }}
                      >
                        {t('settings.ai.semanticSearch.drives.breakLock')}
                      </button>
                    </div>
                  )}
                  {confirmDriveOff === drive.root && (
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <span className="flex-1 min-w-0 text-[var(--text-secondary)]">{t('settings.ai.semanticSearch.drives.confirmOff')}</span>
                      <button
                        type="button"
                        className={`${BUTTON_SECONDARY} hover:text-rose-300`}
                        onClick={() => { void setDriveEnabled(drive.root, false); }}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                        {t('settings.ai.semanticSearch.drives.turnOff')}
                      </button>
                      <button type="button" className={BUTTON_SECONDARY} onClick={() => setConfirmDriveOff(null)}>
                        {t('settings.ai.semanticSearch.drives.cancel')}
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
          {driveError && <p className="text-[0.75rem] text-rose-300">{driveError}</p>}
        </Row>

        <Row
          icon={<Search className="w-4 h-4" />}
          title={t('settings.ai.semanticSearch.launcher.title')}
          description={t('settings.ai.semanticSearch.launcher.description')}
        >
          <Toggle
            enabled={config.showInLauncher}
            onToggle={() => { void update({ showInLauncher: !config.showInLauncher }); }}
            label={t('settings.ai.semanticSearch.launcher.title')}
          />
        </Row>

        <Row
          icon={<BatteryCharging className="w-4 h-4" />}
          title={t('settings.ai.semanticSearch.power.title')}
          description={t('settings.ai.semanticSearch.power.description')}
        >
          <Toggle
            enabled={config.onlyWhileCharging}
            onToggle={() => { void update({ onlyWhileCharging: !config.onlyWhileCharging }); }}
            label={t('settings.ai.semanticSearch.power.title')}
          />
        </Row>

        <Row
          icon={<Database className="w-4 h-4" />}
          title={t('settings.ai.semanticSearch.index.title')}
          description={
            index.lastIndexedAt
              ? t('settings.ai.semanticSearch.index.lastUpdated', { time: formatTimestamp(index.lastIndexedAt) })
              : t('settings.ai.semanticSearch.index.never')
          }
        >
          <div className="text-[0.75rem] text-[var(--text-secondary)]">
            {index.state === 'indexing' ? (
              index.target
                ? t('settings.ai.semanticSearch.index.progressDrive', {
                    processed: index.processed.toLocaleString(),
                    total: index.total.toLocaleString(),
                    drive: index.target,
                  })
                : t('settings.ai.semanticSearch.index.progress', { processed: index.processed.toLocaleString(), total: index.total.toLocaleString() })
            ) : index.state === 'error' ? (
              <span className="text-rose-300">{t('settings.ai.semanticSearch.index.error', { error: index.error || '' })}</span>
            ) : index.pausedOnBattery ? (
              <span className="inline-flex items-center gap-1.5 text-amber-300">
                <BatteryCharging className="w-3.5 h-3.5" />
                {t('settings.ai.semanticSearch.index.pausedOnBattery')}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5">
                {index.items > 0 && <CheckCircle2 className="w-3.5 h-3.5 text-[color:var(--status-success)]" />}
                {t('settings.ai.semanticSearch.index.summary', { items: index.items.toLocaleString(), chunks: index.chunks.toLocaleString() })}
              </span>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className={BUTTON_SECONDARY}
              disabled={!status.ready || index.state === 'indexing'}
              onClick={() => { void window.electron.semanticSearchReindex(false).then(setStatus); }}
            >
              {t('settings.ai.semanticSearch.index.update')}
            </button>
            <button
              type="button"
              className={BUTTON_SECONDARY}
              disabled={!status.ready}
              onClick={() => { void window.electron.semanticSearchReindex(true).then(setStatus); }}
            >
              {t('settings.ai.semanticSearch.index.rebuild')}
            </button>
          </div>
        </Row>
      </div>

      <Row
        icon={<Trash2 className="w-4 h-4" />}
        title={t('settings.ai.semanticSearch.reset.title')}
        description={t('settings.ai.semanticSearch.reset.description')}
      >
        <div>
          <button
            type="button"
            className={`${BUTTON_SECONDARY} hover:text-rose-300`}
            onClick={() => { void window.electron.semanticSearchReset().then(setStatus); }}
          >
            <Trash2 className="w-3.5 h-3.5" />
            {t('settings.ai.semanticSearch.reset.action')}
          </button>
        </div>
      </Row>
    </div>
  );
};

export default SemanticSearchSettings;
