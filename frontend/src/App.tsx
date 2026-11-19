import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  Map as MapLibreMap,
  Marker,
  NavigationControl,
  Popup,
  LngLatBounds,
} from 'maplibre-gl'
import {
  Activity,
  ArrowUpRight,
  Crosshair,
  Database,
  Globe2,
  Layers3,
  Map as MapIcon,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  Wifi,
  X,
} from 'lucide-react'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

type SearchResult = {
  tile_id: string
  score: number
  path?: string | null
  crs?: string | null
  bounds?: {
    left: number
    bottom: number
    right: number
    top: number
  } | null
  center?: {
    longitude: number
    latitude: number
  } | null
  width?: number | null
  height?: number | null
  resolution?: number[] | null
  source?: string | null
}

type SearchResponse = {
  query: string
  indexed_tiles: number
  results: SearchResult[]
}

type HealthResponse = {
  status: string
  model_loaded: boolean
  vector_store_connected: boolean
  indexed_tiles: number
}

function GeoMap({ results }: { results: SearchResult[] }) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const markersRef = useRef<Marker[]>([])

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return

    const map = new MapLibreMap({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap contributors',
          },
        },
        layers: [
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
          },
        ],
      },
      center: [77.0128, 28.9872],
      zoom: 11,
    })

    map.addControl(new NavigationControl(), 'top-right')
    mapRef.current = map

    return () => {
      markersRef.current.forEach((marker) => marker.remove())
      markersRef.current = []
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    markersRef.current.forEach((marker) => marker.remove())
    markersRef.current = []

    const locatedResults = results.filter((result) => result.center)

    locatedResults.forEach((result, index) => {
      const element = document.createElement('div')
      element.className = 'map-marker'
      element.textContent = String(index + 1)

      const marker = new Marker({ element })
        .setLngLat([
          result.center!.longitude,
          result.center!.latitude,
        ])
        .setPopup(
          new Popup({ offset: 20 }).setHTML(
            `<strong>${result.tile_id}</strong><br/>Similarity: ${(result.score * 100).toFixed(1)}%`,
          ),
        )
        .addTo(map)

      markersRef.current.push(marker)
    })

    if (locatedResults.length > 0) {
      const bounds = new LngLatBounds()

      locatedResults.forEach((result) => {
        bounds.extend([
          result.center!.longitude,
          result.center!.latitude,
        ])
      })

      map.fitBounds(bounds, {
        padding: 90,
        maxZoom: 14,
        duration: 900,
      })
    }
  }, [results])

  return <div ref={mapContainer} className="map-container" />
}

function App() {
  const [query, setQuery] = useState('river surrounded by vegetation')
  const [results, setResults] = useState<SearchResult[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [indexedTiles, setIndexedTiles] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [searchedQuery, setSearchedQuery] = useState('')

  useEffect(() => {
    let active = true

    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/health`)
        if (!response.ok) throw new Error('API unavailable')
        const data: HealthResponse = await response.json()

        if (active) {
          setHealth(data)
          setIndexedTiles(data.indexed_tiles)
        }
      } catch {
        if (active) setHealth(null)
      }
    }

    checkHealth()
    const interval = window.setInterval(checkHealth, 10000)

    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [])

  const runSearch = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault()

    if (!query.trim()) {
      setError('Enter a search query first.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          limit: 10,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Search request failed.')
      }

      const searchData = data as SearchResponse
      setResults(searchData.results)
      setIndexedTiles(searchData.indexed_tiles)
      setSearchedQuery(searchData.query)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not connect to GeoSentinel API.',
      )
    } finally {
      setLoading(false)
    }
  }

  const apiOnline = health?.status === 'ok'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Globe2 size={23} />
          </div>
          <div>
            <h1>GEOSENTINEL</h1>
            <span>EARTH INTELLIGENCE</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="navigation">
          <button className="nav-item active">
            <Globe2 size={18} /> Explore
          </button>
          <button className="nav-item">
            <Layers3 size={18} /> Indexed tiles
          </button>
          <button className="nav-item">
            <Activity size={18} /> Activity
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="system-card">
            <div className="system-heading">
              <ShieldCheck size={16} />
              SYSTEM STATUS
            </div>
            <div className="system-row">
              <span>API service</span>
              <span className={apiOnline ? 'status-dot online' : 'status-dot'} />
            </div>
            <div className="system-row">
              <span>RemoteCLIP</span>
              <span className={health?.model_loaded ? 'status-dot online' : 'status-dot'} />
            </div>
            <div className="system-row">
              <span>Qdrant store</span>
              <span className={health?.vector_store_connected ? 'status-dot online' : 'status-dot'} />
            </div>
          </div>

          <button className="nav-item settings-button">
            <Settings2 size={18} /> Settings
          </button>
          <div className="version">GEOSENTINEL v0.1.0 · LOCAL</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="slash">/</span>
            <strong>Explore</strong>
          </div>
          <div className="topbar-right">
            <span className="environment">
              <span className="status-dot online" />
              LOCAL ENVIRONMENT
            </span>
            <button className="icon-button" aria-label="Settings">
              <Settings2 size={18} />
            </button>
          </div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow">
              <Sparkles size={14} /> MULTIMODAL GEOINTELLIGENCE
            </div>
            <h2>Explore the Earth.</h2>
            <p>Search satellite imagery with natural language and discover meaningful locations.</p>
          </div>
          <div className="heading-status">
            <Wifi size={15} />
            {apiOnline ? 'Backend connected' : 'Backend disconnected'}
          </div>
        </section>

        <section className="metrics">
          <div className="metric-card">
            <div className="metric-icon"><Database size={18} /></div>
            <div>
              <span>INDEXED TILES</span>
              <strong>{indexedTiles}</strong>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-icon"><Target size={18} /></div>
            <div>
              <span>SEARCH RESULTS</span>
              <strong>{results.length}</strong>
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-icon"><Activity size={18} /></div>
            <div>
              <span>MODEL STATUS</span>
              <strong className="metric-small">
                {health?.model_loaded ? 'READY' : 'OFFLINE'}
              </strong>
            </div>
          </div>
        </section>

        <section className="workspace">
          <aside className="search-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">DISCOVERY</span>
                <h3>AI Search</h3>
              </div>
              <SlidersHorizontal size={17} />
            </div>

            <p className="panel-description">
              Describe the landscape or feature you're looking for.
            </p>

            <form onSubmit={runSearch}>
              <label htmlFor="search-query">SEARCH QUERY</label>
              <div className="search-input-wrap">
                <Search size={17} />
                <input
                  id="search-query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="e.g. river surrounded by vegetation"
                />
                {query && (
                  <button
                    type="button"
                    className="clear-button"
                    onClick={() => setQuery('')}
                    aria-label="Clear search"
                  >
                    <X size={15} />
                  </button>
                )}
              </div>

              <button className="search-button" type="submit" disabled={loading}>
                {loading ? (
                  <><span className="spinner" /> Searching...</>
                ) : (
                  <><Sparkles size={16} /> Search imagery <ArrowUpRight size={16} /></>
                )}
              </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            <div className="suggestions">
              <span className="section-kicker">TRY A SEARCH</span>
              {[
                'River surrounded by vegetation',
                'Dense urban settlement',
                'Agricultural fields',
                'Water bodies and wetlands',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  className="suggestion"
                  onClick={() => setQuery(suggestion.toLowerCase())}
                >
                  <Search size={13} />
                  {suggestion}
                </button>
              ))}
            </div>

            <div className="search-note">
              <Sparkles size={15} />
              <span>Powered by RemoteCLIP semantic image embeddings.</span>
            </div>
          </aside>

          <section className="map-panel">
            <div className="map-toolbar">
              <div className="map-title">
                <MapIcon size={17} />
                <strong>Geospatial Viewer</strong>
                <span className="map-tag">EPSG:4326</span>
              </div>
              <button className="icon-button" title="Map layers">
                <Layers3 size={17} />
              </button>
            </div>

            <div className="map-wrap">
              <GeoMap results={results} />
              <div className="map-overlay-label">
                <Crosshair size={13} />
                TILE LOCATION VIEW
              </div>
              <div className="map-legend">
                <span className="legend-marker" />
                Search matches
              </div>
            </div>

            <div className="map-footer">
              <span>
                <span className="status-dot online" />
                Map renderer active
              </span>
              <span>{results.filter((result) => result.center).length} mapped locations</span>
            </div>
          </section>

          <aside className="results-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">INTELLIGENCE</span>
                <h3>Search Results</h3>
              </div>
              <span className="result-count">{results.length}</span>
            </div>

            {searchedQuery && (
              <div className="searched-query">
                <span>QUERY</span>
                <p>{searchedQuery}</p>
              </div>
            )}

            {results.length === 0 ? (
              <div className="empty-results">
                <div className="empty-icon"><Search size={22} /></div>
                <strong>No results yet</strong>
                <p>Run an AI search to discover matching satellite tiles.</p>
              </div>
            ) : (
              <div className="result-list">
                {results.map((result, index) => (
                  <article className="result-card" key={result.tile_id + index}>
                    <div className="result-top">
                      <div className="result-number">{String(index + 1).padStart(2, '0')}</div>
                      <div className="result-name">
                        <strong>{result.tile_id}</strong>
                        <span>{result.crs || 'CRS unavailable'}</span>
                      </div>
                      <span className="score">
                        {(result.score * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="score-track">
                      <div style={{ width: `${Math.max(0, Math.min(100, result.score * 100))}%` }} />
                    </div>

                    {result.center && (
                      <div className="coordinates">
                        <Crosshair size={12} />
                        {result.center.latitude.toFixed(5)}, {result.center.longitude.toFixed(5)}
                      </div>
                    )}

                    <div className="result-meta">
                      {result.width && result.height && (
                        <span>{result.width} × {result.height} px</span>
                      )}
                      {result.source && <span>{result.source}</span>}
                    </div>
                    {result.path && (
                      <div className="result-path" title={result.path}>
                        {result.path}
                      </div>
                    )}
                  </article>
                ))}
              </div>
            )}

            <div className="results-footer">
              <Database size={14} />
              {indexedTiles} tiles in local vector index
            </div>
          </aside>
        </section>

        <footer className="page-footer">
          <span>GEOSENTINEL · EARTH INTELLIGENCE PLATFORM</span>
          <span>BUILT FOR SPATIAL DISCOVERY</span>
        </footer>
      </main>
    </div>
  )
}

export default App
