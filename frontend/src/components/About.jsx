// shown in the results panel while nothing is selected, so a first-time visitor sees what the data is before using it
export default function About() {
  return (
    <div className="about">
      <h2>About</h2>
      <p>
        This viewer shows ModelNet10: 4,899 CAD meshes of household objects in 10 classes (Wu et al., <i>3D ShapeNets</i>,
        CVPR 2015). The models were collected from online CAD repositories; labels were checked by Mechanical Turk workers and
        then by the dataset's authors.
      </p>
      <p>
        Pick an object to see its mesh, then <strong>Classify</strong> it with two networks trained here: a one-hidden-layer
        MLP on a 16³ voxel grid, and PointNet (Qi et al., CVPR 2017) on 1,024 surface points. The <strong>Show</strong> menu
        lists only the objects a model gets wrong.
      </p>
      <p className="muted">
        Limitations: every mesh is scaled to the unit sphere before the models see it, so absolute size is gone (a dresser and
        a night stand differ mostly in size). The test split was never used for training or model selection. The percentages
        are softmax confidences, not calibrated probabilities.
      </p>
    </div>
  )
}
