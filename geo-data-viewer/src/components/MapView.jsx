// This pulls in the two components we need from react-leaflet (both are named exports so we use {})
import { MapContainer, TileLayer } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

const ALBEMARLE_COUNTY_CENTER = [38.03, -78.48]

function MapView() {
  return (
    <MapContainer center={ALBEMARLE_COUNTY_CENTER} zoom={12} style = {{ height: '500px', width: '100%' }}>
        {/* TileLayer pulls actual map images from OpenStreetMap's tile servers */}
        <TileLayer url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
    </MapContainer>
  )
}

export default MapView