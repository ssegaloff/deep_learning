// This pulls in the two components we need from react-leaflet (both are named exports so we use {})
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet'
import { useState, useEffect } from 'react'
import 'leaflet/dist/leaflet.css'

const ALBEMARLE_COUNTY_CENTER = [38.03, -78.48]

function onEachBuilding(feature, layer) {
    // pull out the two properties
    const release = feature.properties.release
    const captureDate = feature.properties.capture_dates_range

    // handle the case when captureDate is blank/missing
    const captureDateDisplay = captureDate ? captureDate : "No capture date recorded (earlier data release)"
    
    // build the popup content as a plain HTML string
    //    (this is NOT JSX — think template literal with <b>, <br>, etc.)
    const popupContent = `
        <b>Data Release:</b> ${release}<br>
        <b>Capture Date:</b> ${captureDateDisplay}
    `

    // attach it to this specific layer
    layer.bindPopup(popupContent)
}

function MapView() {
  
    const [buildings, setBuildings] = useState(null)

    useEffect(() => {
        // useEffect's own callback func can't be declared as async directly react expects
        // it to return nothing or a clean-up func. an async func always returns a Promise
        // instead. so we have to define a separate async func inside the effect
        async function loadBuildings() {
            // fetch() starts an HTTP request and returns a Promise (a placeholder for the value that it eventually returns)
            // await pauses the execution of the function until the Promise resolves and returns the
            // resolved value, which is here a Response object
            const response = await fetch('/albemarle-buildings.geojson')
            // response is metadata about the HTTP request, not the json data itself
            // .json() reads and parses the HTTP response body which is itself asynchronous
            // so it needs its own await
            const data = await response.json()
            // now we have the parsed geojson object. calling setBuildings() will update the buildings state
            // and tell react to re-render the component with the new data
            setBuildings(data)
        }
        // we call loadBuildings() with no await because we are no longer inside an async func
        // we are in useEffect()'s callback. await is only legal syntax inside a function marked async
        // we don't need to wait for it to finish here anyway, we just want to start it and let
        // setBuildings() (called later once data arrives) to handle the rest of the updating of UI when its ready
        loadBuildings()
    }, [])
    // the empty array here is the "dependencies array". it tells react this effect should run exactly once
    // right after the component's first render, and never again. without it, the effect would re-run every time
    // the component re-renders which would mean re-fetching in an infinite loop

    return (
    <MapContainer center={ALBEMARLE_COUNTY_CENTER} zoom={15} style = {{ height: '500px', width: '100%' }} preferCanvas >
        {/* TileLayer pulls actual map images from OpenStreetMap's tile servers */}
        <TileLayer 
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attribution="Tiles &copy; Esri"/>
        {buildings && <GeoJSON data={buildings} onEachFeature={onEachBuilding} />}
    </MapContainer>
  )
}

export default MapView