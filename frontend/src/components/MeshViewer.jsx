import { Suspense } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { meshUrl } from '../api'
import ErrorBoundary from './ErrorBoundary'

function Shape({ id }) {
  // suspends until the file arrives, and caches by url, so revisiting an object doesn't refetch it
  const geometry = useLoader(PLYLoader, meshUrl(id))
  return (
    // modelnet is z-up and three.js is y-up; this maps +z to +y for display only, the points and models are untouched
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      {/* flat: one normal per triangle, computed in the shader, so cad edges stay crisp and the ply needs no normals */}
      <meshStandardMaterial color="#b8b2c6" flatShading />
    </mesh>
  )
}

export default function MeshViewer({ id }) {
  return (
    <div className="viewer">
      {/* meshes arrive normalized to the unit sphere, so one fixed camera ~2.8 away fits every object (45° fov needs ≥ 2.61) */}
      <Canvas camera={{ position: [1.8, 1.2, 1.8], fov: 45 }}>
        <hemisphereLight args={['#ffffff', '#666677', 1.2]} />
        <directionalLight position={[3, 4, 2]} intensity={1.5} />
        {/* inside the canvas and keyed by id: a failed mesh replaces only the shape, while the canvas, its webgl context, and the camera stay mounted */}
        {id ? (
          // inside the canvas and keyed by id: a failed mesh replaces only the shape, while the canvas, its webgl context, and the camera stay mounted
          <ErrorBoundary
            key={id}
            fallback={(error) => <Html center className="status error">Could not load mesh: {error.message}</Html>}
          >
            {/* the loading fallback also renders inside the canvas, so it uses drei's Html overlay rather than plain jsx */}
            <Suspense fallback={<Html center>Loading mesh…</Html>}>
              <Shape id={id} />
            </Suspense>
          </ErrorBoundary>
        ) : (
          // the hint renders inside the canvas too, so "nothing selected" doesn't unmount it
          <Html center className="muted">Pick an object.</Html>
        )}
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  )
}
