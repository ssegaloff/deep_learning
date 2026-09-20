import './App.css'
import MapView from './components/MapView' // Import the MapView component (which is the default export)

function App() {

  return (
    <>
      <section id="center">
        <div>
          <h1>Geographic Data Viewer</h1>
          <p>
            This is a geographic data viewer displaying building footprint data for Albemarle County, Virginia, sourced from Microsoft's US Building Footprints dataset (extracted from satellite/aerial imagery via computer vision, licensed under the Open Data Commons Open Database License). Footprint capture dates range from approximately 2012–2020 depending on region and were not verified to align with the satellite base imagery's own capture dates — a limitation acceptable for viewing, but one that would need reconciling before use in model training. Buildings are rendered directly over live satellite imagery (Esri World Imagery).
          </p>
        </div>
        <div>
          <MapView />
        </div>
      </section>
    </>
  )
}

export default App
