// This pulls in the two components we need from react-leaflet (both are named exports so we use {})
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import sites from '../data/albemarle-sites-mini.json'

const ALBEMARLE_COUNTY_CENTER = [38.03, -78.48]

function getLatLng(feature) {
  const [lng, lat] = feature.geometry.coordinates  // GeoJSON order: [lng, lat]
  return [lat, lng]  // Leaflet order: [lat, lng]
}

function MapView() {
    console.log(sites)
    console.log(sites.features.length)
    console.log(sites.features[0].label, sites.features[0].geometry.coordinates)
  return (
    <MapContainer center={ALBEMARLE_COUNTY_CENTER} zoom={12} style = {{ height: '500px', width: '100%' }}>
        {/* TileLayer pulls actual map images from OpenStreetMap's tile servers */}
        <TileLayer url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {sites.features.map((feature) => {
            const position = getLatLng(feature)  // calling helper
            
            return (
                <Marker key={feature.id} position={position}>
                <Popup>
                    <p>{feature.properties["Site Category"]}</p>
                    <p>{feature.properties["Time Period"]}</p>
                </Popup>
                </Marker>
        )
        })}
    </MapContainer>
  )
}

export default MapView