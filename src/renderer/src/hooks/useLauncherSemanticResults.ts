/**
 * useLauncherSemanticResults.ts
 *
 * Files that match the launcher query by meaning (on-device semantic search),
 * shown in their own "By meaning" section below the name-matched files. Only
 * results that clearly stand out from the rest of the index come back, so most
 * app/command queries get none. The main process returns nothing when semantic
 * search is off, its model isn't downloaded, or the launcher section is
 * switched off in settings.
 */

import { useEffect, useRef, useState } from 'react';
import type { SemanticItemKind, SemanticSearchResult } from '../../types/electron';
import { isPathLikeLauncherFileQuery } from '../utils/launcher-file-results';

const SEMANTIC_QUERY_DELAY_MS = 300;
const SEMANTIC_MIN_QUERY_LENGTH = 3;
/** Room for the section's 20 rows after photos below the cutoff and name-matched files drop out. */
const SEMANTIC_RESULT_LIMIT = 30;
/**
 * How far (leave-one-out z-score) a file has to stand out from the rest of
 * the index. Measured on a real index of ~5,700 files: documents above 3.5
 * were nearly always related to the query (12–25 of them for "agreement" or
 * "contract", none for words like "wifi" or "banana"), while with thousands of
 * photos a random one reaches 5.3 for "notes"; real photo matches scored 6–11.
 */
const SEMANTIC_MIN_Z_DOCUMENT = 3.5;
const SEMANTIC_MIN_Z_MEDIA = 5.5;
const SEMANTIC_FILE_KINDS: SemanticItemKind[] = ['document', 'image', 'video', 'audio'];

export function useLauncherSemanticResults({
  searchQuery,
  enabled,
  setFileIcons,
}: {
  searchQuery: string;
  enabled: boolean;
  setFileIcons: (update: (prev: Record<string, string>) => Record<string, string>) => void;
}): SemanticSearchResult[] {
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const requestSeqRef = useRef(0);

  useEffect(() => {
    requestSeqRef.current += 1;
    const requestSeq = requestSeqRef.current;
    const trimmed = searchQuery.trim();
    const eligible =
      enabled &&
      trimmed.length >= SEMANTIC_MIN_QUERY_LENGTH &&
      /\p{L}/u.test(trimmed) &&
      !trimmed.startsWith('!') &&
      !isPathLikeLauncherFileQuery(trimmed);
    if (!eligible) {
      setResults([]);
      return;
    }

    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const found = await window.electron.semanticSearchQuery(trimmed, {
            limit: SEMANTIC_RESULT_LIMIT,
            kinds: SEMANTIC_FILE_KINDS,
            minZ: SEMANTIC_MIN_Z_DOCUMENT,
            launcher: true,
          });
          if (requestSeqRef.current !== requestSeq) return;
          const files = found.filter((result) =>
            Boolean(result.path) && (result.kind === 'document' || result.z >= SEMANTIC_MIN_Z_MEDIA)
          );
          setResults(files);

          const iconEntries = await Promise.all(
            files.map(async (result) => {
              try {
                return [result.path as string, (await window.electron.getFileIconDataUrl(result.path as string, 20)) || ''] as const;
              } catch {
                return [result.path as string, ''] as const;
              }
            })
          );
          if (requestSeqRef.current !== requestSeq) return;
          setFileIcons((prev) => {
            const next = { ...prev };
            for (const [targetPath, icon] of iconEntries) {
              if (icon) next[targetPath] = icon;
            }
            return next;
          });
        } catch (error) {
          console.error('Semantic launcher search failed:', error);
          if (requestSeqRef.current === requestSeq) setResults([]);
        }
      })();
    }, SEMANTIC_QUERY_DELAY_MS);

    return () => {
      window.clearTimeout(timer);
    };
  }, [searchQuery, enabled, setFileIcons]);

  return results;
}
