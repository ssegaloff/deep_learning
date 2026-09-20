import './App.css'
import MapView from './components/MapView' // Import the MapView component (which is the default export)

function App() {

  return (
    <>
      <section id="center">
        <div>
          <h1>Geographic Data Viewer</h1>
          <p>
            This is a geographic data viewer that displays Albemarle County, Virginia archaeological site records from DINAA, sourced via Open Context, with coordinates generalized to ~20km grid cells rather than exact locations, plus attributes like Time Period and Site Category.
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
