import { MapContainer, TileLayer, GeoJSON, Popup } from 'react-leaflet'
import { useState, useEffect } from 'react'
import 'leaflet/dist/leaflet.css'

const ALBEMARLE_COUNTY_CENTER = [38.03, -78.48]

const STATUS_OPTIONS = ['unchanged', 'modified', 'demolished']

function MapView() {
    const [buildings, setBuildings] = useState(null)
    const [annotations, setAnnotations] = useState({})

    // tracks which single building's popup is currently open, or null if none
    const [selectedFeature, setSelectedFeature] = useState(null)

    // where the click actually happened, so the popup opens at the right spot
    // regardless of how the map has been panned/zoomed
    const [clickPosition, setClickPosition] = useState(null)

    // in-progress form values for whichever popup is open right now
    const [draftStatus, setDraftStatus] = useState('unchanged')
    const [draftNote, setDraftNote] = useState('')

    function getAnnotation(footprintId) {
        return annotations[footprintId]
    }

    function saveAnnotation(footprintId, { status, note }) {
        setAnnotations({
            ...annotations,
            [footprintId]: { status, note },
        })
    }

    function onEachBuilding(feature, layer) {
        layer.on('click', (e) => {
            const footprintId = feature.properties.footprint_id
            const existing = getAnnotation(footprintId)

            // pre-fill the form: existing annotation's values if there is one, defaults otherwise
            setDraftStatus(existing ? existing.status : 'unchanged')
            setDraftNote(existing ? existing.note : '')
            setSelectedFeature(feature)
            setClickPosition(e.latlng) // where the click happened, in map coordinates
        })
    }

    function handleSubmit() {
        saveAnnotation(selectedFeature.properties.footprint_id, {
            status: draftStatus,
            note: draftNote,
        })
        setSelectedFeature(null) // close the popup after saving
    }

    useEffect(() => {
        async function loadBuildings() {
            const response = await fetch('/albemarle-buildings.geojson')
            const data = await response.json()
            setBuildings(data)
        }
        loadBuildings()
    }, [])

    return (
        <MapContainer center={ALBEMARLE_COUNTY_CENTER} zoom={15} style={{ height: '500px', width: '100%' }} preferCanvas>
            <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="Tiles &copy; Esri"
            />
            {buildings && <GeoJSON data={buildings} onEachFeature={onEachBuilding} />}

            {selectedFeature && clickPosition && (
                <Popup
                    position={clickPosition}
                    eventHandlers={{ remove: () => setSelectedFeature(null) }}
                >
                    <div>
                        <b>Data Release:</b> {selectedFeature.properties.release}<br />
                        <b>Capture Date:</b> {selectedFeature.properties.capture_dates_range || 'No capture date recorded'}<br />
                        <hr />
                        <label>
                            Status:
                            <select value={draftStatus} onChange={(e) => setDraftStatus(e.target.value)}>
                                {STATUS_OPTIONS.map((option) => (
                                    <option key={option} value={option}>{option}</option>
                                ))}
                            </select>
                        </label>
                        <br />
                        <label>
                            Note:
                            <input
                                type="text"
                                value={draftNote}
                                onChange={(e) => setDraftNote(e.target.value)}
                            />
                        </label>
                        <br />
                        <button onClick={handleSubmit}>Save</button>
                    </div>
                </Popup>
            )}
        </MapContainer>
    )
}

export default MapView