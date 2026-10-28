import React, {useState} from 'react';
import styles from './styles.module.css';

/**
 * <YouTube id="VIDEO_ID" title="..." channel="..." length="2h 25m" start={0}>
 *   Optional markdown: why this video is worth watching here.
 * </YouTube>
 *
 * Shows a thumbnail until clicked, then loads the player from
 * youtube-nocookie.com, so nothing from YouTube's player loads (and no
 * cookies are set) unless the reader chooses to watch.
 */
export default function YouTube({id, title, channel, length, start, children}) {
  const [playing, setPlaying] = useState(false);
  const params = new URLSearchParams({autoplay: '1', rel: '0'});
  if (start) params.set('start', String(start));
  const watchUrl = `https://www.youtube.com/watch?v=${id}${start ? `&t=${start}s` : ''}`;

  return (
    <figure className={styles.card}>
      <div className={styles.frame}>
        {playing ? (
          <iframe
            src={`https://www.youtube-nocookie.com/embed/${id}?${params}`}
            title={title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          />
        ) : (
          <button
            type="button"
            className={styles.facade}
            onClick={() => setPlaying(true)}
            aria-label={`Play video: ${title}`}>
            <img
              src={`https://i.ytimg.com/vi/${id}/hqdefault.jpg`}
              alt=""
              loading="lazy"
              decoding="async"
              onError={(e) => {
                e.currentTarget.style.visibility = 'hidden';
              }}
            />
            <span className={styles.play} aria-hidden="true" />
          </button>
        )}
      </div>
      <figcaption className={styles.caption}>
        <div className={styles.kind}>Watch</div>
        <div className={styles.title}>
          <a href={watchUrl} target="_blank" rel="noopener noreferrer">
            {title}
          </a>
        </div>
        <div className={styles.meta}>
          {[channel, length].filter(Boolean).join(' · ')}
        </div>
        {children && <div className={styles.note}>{children}</div>}
      </figcaption>
    </figure>
  );
}
