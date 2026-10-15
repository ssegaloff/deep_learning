// display names for the backend's architecture keys; shared so the filter menu and the results table agree
export const MODEL_LABELS = { mlp: 'Voxel MLP', pointnet: 'PointNet' }

// which objects each list filter keeps, given one object's predicted labels and its true class
export const FILTERS = {
  all: { label: 'All objects', keep: () => true },
  // every model, not two named ones, so a third model can't silently fall outside it
  any: { label: 'Any model wrong', keep: (pred, truth) => Object.values(pred).some((c) => c !== truth) },
  pointnet: { label: `${MODEL_LABELS.pointnet} wrong`, keep: (pred, truth) => pred.pointnet !== truth },
  mlp: { label: `${MODEL_LABELS.mlp} wrong`, keep: (pred, truth) => pred.mlp !== truth },
}
